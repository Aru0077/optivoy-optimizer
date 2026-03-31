from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

from fastapi import FastAPI, HTTPException
from ortools.sat.python import cp_model
from pydantic import BaseModel, Field


app = FastAPI(title="Optivoy Optimizer", version="0.1.0")

PACE_MODE_WINDOWS: dict[str, tuple[str, str, int]] = {
    "light": ("10:00", "18:00", 480),
    "standard": ("09:00", "20:00", 660),
    "compact": ("08:00", "21:00", 780),
}


class CoordinateIn(BaseModel):
    latitude: float | None = None
    longitude: float | None = None


class PointIn(BaseModel):
    id: str = Field(min_length=1, max_length=64)
    pointType: Literal["spot", "shopping", "restaurant"]
    suggestedDurationMinutes: int = Field(ge=10, le=720)
    latitude: float | None = None
    longitude: float | None = None


class HotelIn(BaseModel):
    id: str = Field(min_length=1, max_length=64)
    latitude: float | None = None
    longitude: float | None = None


class DistanceMatrixRowIn(BaseModel):
    fromPointId: str = Field(min_length=1, max_length=64)
    toPointId: str = Field(min_length=1, max_length=64)
    transitMinutes: int = Field(ge=1, le=1440)
    drivingMinutes: int = Field(ge=1, le=1440)
    walkingMeters: int = Field(ge=0, le=200000)
    distanceKm: float = Field(ge=0, le=2000)
    transitSummary: str | None = None


class DistanceMatrixIn(BaseModel):
    rows: list[DistanceMatrixRowIn] = Field(default_factory=list, max_length=300000)


class SolveRequest(BaseModel):
    city: str | None = Field(default=None, min_length=1, max_length=120)
    province: str | None = Field(default=None, min_length=1, max_length=120)
    arrivalAirport: CoordinateIn | None = None
    departureAirport: CoordinateIn | None = None
    arrivalAirportCode: str | None = Field(default=None, pattern=r"^[A-Z]{3}$")
    departureAirportCode: str | None = Field(default=None, pattern=r"^[A-Z]{3}$")
    arrivalAirportId: str | None = Field(default=None, min_length=1, max_length=64)
    departureAirportId: str | None = Field(default=None, min_length=1, max_length=64)
    arrivalDateTime: datetime
    airportBufferMinutes: int = Field(default=90, ge=60, le=120)
    points: list[PointIn] = Field(min_length=1, max_length=200)
    hotels: list[HotelIn] = Field(min_length=1, max_length=200)
    distanceMatrix: DistanceMatrixIn = Field(default_factory=DistanceMatrixIn)
    paceMode: Literal["light", "standard", "compact"] = "standard"
    hotelMode: Literal["single", "multi"] = "multi"
    mealPolicy: Literal["auto", "off"] = "auto"
    objective: Literal["min_days", "min_transit", "min_days_then_transit"] = (
        "min_days_then_transit"
    )
    maxDays: int = Field(default=14, ge=1, le=14)
    timeLimitSeconds: float = Field(default=2.5, ge=0.2, le=30.0)


class DayPlan(BaseModel):
    dayNumber: int
    date: str
    pointIds: list[str]
    hotelId: str


class SolveResponse(BaseModel):
    tripDays: int
    solverStatus: Literal["OPTIMAL", "FEASIBLE"]
    objective: str
    days: list[DayPlan]


@dataclass(frozen=True)
class Coord:
    lat: float | None
    lng: float | None


@dataclass(frozen=True)
class Anchor:
    point_id: str | None
    coord: Coord


def haversine_km(a: Coord, b: Coord) -> float:
    if a.lat is None or a.lng is None or b.lat is None or b.lng is None:
        return 8.0
    to_rad = math.pi / 180.0
    dlat = (b.lat - a.lat) * to_rad
    dlng = (b.lng - a.lng) * to_rad
    aa = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(a.lat * to_rad)
        * math.cos(b.lat * to_rad)
        * math.sin(dlng / 2.0) ** 2
    )
    return 6371.0 * 2.0 * math.atan2(math.sqrt(aa), math.sqrt(1.0 - aa))


def fallback_transit_minutes(a: Coord, b: Coord) -> int:
    km = haversine_km(a, b)
    driving = (km / 25.0) * 60.0
    return max(8, int(round(driving + 8)))


def matrix_transit_minutes(
    from_id: str | None,
    from_coord: Coord,
    to_id: str | None,
    to_coord: Coord,
    lookup: dict[tuple[str, str], int],
) -> int:
    if from_id and to_id:
        direct = lookup.get((from_id, to_id))
        if direct is not None and direct > 0:
            return direct
        reverse = lookup.get((to_id, from_id))
        if reverse is not None and reverse > 0:
            return reverse
    return fallback_transit_minutes(from_coord, to_coord)


def route_cost(
    order: list[int],
    points: list[PointIn],
    start: Anchor | None,
    lookup: dict[tuple[str, str], int],
) -> int:
    total = 0
    prev = start
    for idx in order:
        current_point = points[idx]
        current_anchor = Anchor(
            point_id=current_point.id,
            coord=Coord(current_point.latitude, current_point.longitude),
        )
        if prev is not None:
            total += matrix_transit_minutes(
                prev.point_id,
                prev.coord,
                current_anchor.point_id,
                current_anchor.coord,
                lookup,
            )
        prev = current_anchor
    return total


def best_day_order(
    point_indexes: list[int],
    points: list[PointIn],
    start: Anchor | None,
    lookup: dict[tuple[str, str], int],
) -> list[int]:
    if len(point_indexes) <= 1:
        return point_indexes

    # Day size is usually small; brute force gives exact best order.
    if len(point_indexes) <= 8:
        best = point_indexes
        best_cost = None
        for perm in itertools.permutations(point_indexes):
            cost = route_cost(list(perm), points, start, lookup)
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best = list(perm)
        return best

    # Fallback heuristic for larger day sets.
    remaining = set(point_indexes)
    ordered: list[int] = []

    if start is None:
        first = min(
            remaining,
            key=lambda idx: sum(
                matrix_transit_minutes(
                    points[idx].id,
                    Coord(points[idx].latitude, points[idx].longitude),
                    points[j].id,
                    Coord(points[j].latitude, points[j].longitude),
                    lookup,
                )
                for j in remaining
                if j != idx
            ),
        )
    else:
        first = min(
            remaining,
            key=lambda idx: matrix_transit_minutes(
                start.point_id,
                start.coord,
                points[idx].id,
                Coord(points[idx].latitude, points[idx].longitude),
                lookup,
            ),
        )

    ordered.append(first)
    remaining.remove(first)
    current = Anchor(
        point_id=points[first].id,
        coord=Coord(points[first].latitude, points[first].longitude),
    )

    while remaining:
        next_idx = min(
            remaining,
            key=lambda idx: matrix_transit_minutes(
                current.point_id,
                current.coord,
                points[idx].id,
                Coord(points[idx].latitude, points[idx].longitude),
                lookup,
            ),
        )
        ordered.append(next_idx)
        remaining.remove(next_idx)
        current = Anchor(
            point_id=points[next_idx].id,
            coord=Coord(points[next_idx].latitude, points[next_idx].longitude),
        )

    return ordered


def choose_single_hotel(
    hotels: list[HotelIn],
    points: list[PointIn],
    lookup: dict[tuple[str, str], int],
) -> str:
    all_indexes = list(range(len(points)))
    return choose_day_hotel(hotels, points, all_indexes, None, None, lookup)


def choose_day_hotel(
    hotels: list[HotelIn],
    points: list[PointIn],
    day_indexes: list[int],
    start_anchor: Anchor | None,
    next_day_first_anchor: Anchor | None,
    lookup: dict[tuple[str, str], int],
) -> str:
    if not day_indexes:
        return hotels[0].id

    best_id = hotels[0].id
    best_cost = None
    for hotel in hotels:
        hotel_anchor = Anchor(hotel.id, Coord(hotel.latitude, hotel.longitude))
        first_point = points[day_indexes[0]]
        last_point = points[day_indexes[-1]]

        cost = matrix_transit_minutes(
            last_point.id,
            Coord(last_point.latitude, last_point.longitude),
            hotel_anchor.point_id,
            hotel_anchor.coord,
            lookup,
        )

        if next_day_first_anchor is not None:
            cost += matrix_transit_minutes(
                hotel_anchor.point_id,
                hotel_anchor.coord,
                next_day_first_anchor.point_id,
                next_day_first_anchor.coord,
                lookup,
            )

        if start_anchor is not None:
            cost += matrix_transit_minutes(
                start_anchor.point_id,
                start_anchor.coord,
                first_point.id,
                Coord(first_point.latitude, first_point.longitude),
                lookup,
            )

        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_id = hotel.id
    return best_id


def coord_from_point(point: PointIn) -> Coord:
    return Coord(point.latitude, point.longitude)


def coord_from_hotel(hotel: HotelIn) -> Coord:
    return Coord(hotel.latitude, hotel.longitude)


def parse_hhmm(value: str) -> tuple[int, int]:
    hour_str, minute_str = value.split(":", 1)
    return int(hour_str), int(minute_str)


def first_day_budget_minutes(
    arrival_dt: datetime,
    buffer_minutes: int,
    pace_start: str,
    pace_end: str,
    pace_minutes: int,
    meal_overhead: int,
) -> int:
    start_hour, start_minute = parse_hhmm(pace_start)
    end_hour, end_minute = parse_hhmm(pace_end)

    pace_start_dt = arrival_dt.replace(
        hour=start_hour,
        minute=start_minute,
        second=0,
        microsecond=0,
    )
    pace_end_dt = arrival_dt.replace(
        hour=end_hour,
        minute=end_minute,
        second=0,
        microsecond=0,
    )

    arrival_with_buffer = arrival_dt + timedelta(minutes=buffer_minutes)
    usable_start = max(arrival_with_buffer, pace_start_dt)
    raw_usable = int((pace_end_dt - usable_start).total_seconds() // 60)

    if raw_usable <= 0:
        return 0

    clipped = min(raw_usable, pace_minutes)
    return max(0, clipped - meal_overhead)


def build_transit_lookup(rows: list[DistanceMatrixRowIn]) -> dict[tuple[str, str], int]:
    lookup: dict[tuple[str, str], int] = {}
    for row in rows:
        key = (row.fromPointId, row.toPointId)
        current = lookup.get(key)
        if current is None or row.transitMinutes < current:
            lookup[key] = row.transitMinutes
    return lookup


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/solve", response_model=SolveResponse)
def solve(input_data: SolveRequest) -> SolveResponse:
    pace_start, pace_end, pace_minutes = PACE_MODE_WINDOWS[input_data.paceMode]
    has_restaurant_point = any(p.pointType == "restaurant" for p in input_data.points)
    meal_overhead = 45 if input_data.mealPolicy == "auto" and not has_restaurant_point else 0
    daily_budget = pace_minutes - meal_overhead
    if daily_budget < 120:
        raise HTTPException(status_code=400, detail="daily playable minutes too low")

    first_day_budget = first_day_budget_minutes(
        input_data.arrivalDateTime,
        input_data.airportBufferMinutes,
        pace_start,
        pace_end,
        pace_minutes,
        meal_overhead,
    )

    point_loads = [p.suggestedDurationMinutes + 12 for p in input_data.points]
    if any(load > daily_budget for load in point_loads):
        raise HTTPException(
            status_code=400,
            detail="at least one point duration exceeds daily budget",
        )

    model = cp_model.CpModel()
    n = len(input_data.points)

    # Add one optional calendar day slot to absorb arrival-day buffer impact.
    d = min(len(input_data.points) + 1, input_data.maxDays + 1)

    x: list[list[cp_model.IntVar]] = []
    for i in range(n):
        row = [model.NewBoolVar(f"x_{i}_{day}") for day in range(d)]
        x.append(row)
        model.Add(sum(row) == 1)

    day_budgets = [daily_budget for _ in range(d)]
    day_budgets[0] = first_day_budget

    used = [model.NewBoolVar(f"used_{day}") for day in range(d)]
    for day in range(d):
        model.Add(
            sum(point_loads[i] * x[i][day] for i in range(n)) <= day_budgets[day] * used[day]
        )
        model.Add(sum(x[i][day] for i in range(n)) >= used[day])

    # day0 can be rest day; day1+ must stay contiguous.
    for day in range(1, d - 1):
        model.Add(used[day] >= used[day + 1])

    model.Add(sum(used) <= input_data.maxDays)

    transit_lookup = build_transit_lookup(input_data.distanceMatrix.rows)

    used_days_expr = sum(used)
    day_pack_penalty = sum((day + 1) * x[i][day] for i in range(n) for day in range(d))

    # Encourage geographically compact daily clusters using cached transit matrix.
    pair_weight_terms: list[cp_model.LinearExpr] = []
    if input_data.objective in ("min_transit", "min_days_then_transit") and n <= 60:
        for i in range(n):
            for j in range(i + 1, n):
                pair_cost = matrix_transit_minutes(
                    input_data.points[i].id,
                    coord_from_point(input_data.points[i]),
                    input_data.points[j].id,
                    coord_from_point(input_data.points[j]),
                    transit_lookup,
                )
                for day in range(d):
                    y = model.NewBoolVar(f"pair_{i}_{j}_{day}")
                    model.Add(y <= x[i][day])
                    model.Add(y <= x[j][day])
                    model.Add(y >= x[i][day] + x[j][day] - 1)
                    pair_weight_terms.append(y * pair_cost)
    transit_proxy_penalty = sum(pair_weight_terms)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = input_data.timeLimitSeconds
    solver.parameters.num_search_workers = 8

    lexicographic_first_phase_status: int | None = None
    if input_data.objective == "min_days":
        model.Minimize(used_days_expr * 10000 + day_pack_penalty)
        status = solver.Solve(model)
    elif input_data.objective == "min_transit":
        model.Minimize(used_days_expr * 10000 + transit_proxy_penalty + day_pack_penalty)
        status = solver.Solve(model)
    else:
        # Lexicographic:
        # phase-1 minimize trip days, phase-2 fix minimum days and minimize transit proxy.
        model.Minimize(used_days_expr)
        lexicographic_first_phase_status = solver.Solve(model)
        if lexicographic_first_phase_status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            raise HTTPException(status_code=503, detail="solver failed to find feasible plan")

        minimum_days = sum(solver.Value(day_used) for day_used in used)
        model.Add(used_days_expr == minimum_days)
        model.Minimize(transit_proxy_penalty + day_pack_penalty)
        status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise HTTPException(status_code=503, detail="solver failed to find feasible plan")

    used_day_indexes = [day for day in range(d) if solver.Value(used[day]) == 1]
    used_days = len(used_day_indexes)
    if used_days == 0:
        raise HTTPException(status_code=503, detail="solver returned empty day set")

    day_indexes: list[list[int]] = [[] for _ in range(used_days)]
    day_position_by_index = {day_idx: pos for pos, day_idx in enumerate(used_day_indexes)}
    for i in range(n):
        assigned_day = None
        for day in range(d):
            if solver.Value(x[i][day]) == 1:
                assigned_day = day
                break
        if assigned_day is None:
            raise HTTPException(status_code=503, detail="solver assignment is invalid")
        pos = day_position_by_index.get(assigned_day)
        if pos is None:
            raise HTTPException(status_code=503, detail="solver used-day mapping is invalid")
        day_indexes[pos].append(i)

    arrival_anchor = None
    if input_data.arrivalAirport is not None:
        arrival_anchor = Anchor(
            point_id=input_data.arrivalAirportId,
            coord=Coord(
                input_data.arrivalAirport.latitude,
                input_data.arrivalAirport.longitude,
            ),
        )

    # First pass order (for hotel selection).
    day_orders: list[list[int]] = []
    for pos in range(used_days):
        start_for_day = arrival_anchor if pos == 0 else None
        order = best_day_order(day_indexes[pos], input_data.points, start_for_day, transit_lookup)
        day_orders.append(order)

    hotel_by_id = {hotel.id: hotel for hotel in input_data.hotels}
    day_hotels: list[str] = []
    if input_data.hotelMode == "single":
        hotel_id = choose_single_hotel(input_data.hotels, input_data.points, transit_lookup)
        day_hotels = [hotel_id for _ in range(used_days)]
    else:
        for pos in range(used_days):
            next_first_anchor = None
            if pos + 1 < used_days and day_orders[pos + 1]:
                next_point = input_data.points[day_orders[pos + 1][0]]
                next_first_anchor = Anchor(
                    point_id=next_point.id,
                    coord=coord_from_point(next_point),
                )

            start_anchor = (
                arrival_anchor
                if pos == 0
                else Anchor(
                    point_id=day_hotels[pos - 1],
                    coord=coord_from_hotel(hotel_by_id[day_hotels[pos - 1]]),
                )
            )

            day_hotels.append(
                choose_day_hotel(
                    input_data.hotels,
                    input_data.points,
                    day_orders[pos],
                    start_anchor,
                    next_first_anchor,
                    transit_lookup,
                )
            )

    # Second pass order (day2+ starts from previous night hotel).
    refined_orders: list[list[int]] = []
    for pos in range(used_days):
        if pos == 0:
            start_anchor = arrival_anchor
        else:
            previous_hotel = hotel_by_id.get(day_hotels[pos - 1])
            start_anchor = (
                Anchor(day_hotels[pos - 1], coord_from_hotel(previous_hotel))
                if previous_hotel
                else None
            )
        refined_orders.append(
            best_day_order(day_indexes[pos], input_data.points, start_anchor, transit_lookup)
        )

    start_date = input_data.arrivalDateTime.date()
    plans: list[DayPlan] = []
    for pos in range(used_days):
        calendar_index = used_day_indexes[pos]
        plans.append(
            DayPlan(
                dayNumber=pos + 1,
                date=(start_date + timedelta(days=calendar_index)).isoformat(),
                pointIds=[input_data.points[idx].id for idx in refined_orders[pos]],
                hotelId=day_hotels[pos],
            )
        )

    solver_status: Literal["OPTIMAL", "FEASIBLE"] = "OPTIMAL"
    if status != cp_model.OPTIMAL:
        solver_status = "FEASIBLE"
    if (
        input_data.objective == "min_days_then_transit"
        and lexicographic_first_phase_status != cp_model.OPTIMAL
    ):
        solver_status = "FEASIBLE"

    return SolveResponse(
        tripDays=used_days,
        solverStatus=solver_status,
        objective=input_data.objective,
        days=plans,
    )
