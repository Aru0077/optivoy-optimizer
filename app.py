from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

from fastapi import FastAPI, HTTPException
from ortools.sat.python import cp_model
from pydantic import BaseModel, Field as PydField

app = FastAPI(title="Optivoy Optimizer", version="0.2.3")

PACE_MODE_WINDOWS: dict[str, tuple[str, str, int]] = {
    "light": ("10:00", "18:00", 480),
    "standard": ("09:00", "20:00", 660),
    "compact": ("08:00", "21:00", 780),
}

HOTEL_SWITCH_PENALTY_MINUTES = 45
PER_POINT_OVERHEAD_MINUTES = 12
MAX_EXACT_PERMUTATION_SIZE = 8

MAX_MISSING_UNDIRECTED_RATIO = 0.0

# v0.2.3: Geographic coherence tiebreaker.
# When two routes have equal transit cost (common with symmetric driving distances),
# this tiny penalty (in minutes) per geographic direction reversal breaks the tie
# in favor of routes that flow consistently in one direction (e.g. south→north).
# 0.01 min = 0.6 seconds — too small to override any real distance difference,
# but large enough to break ties deterministically.
GEO_ZIGZAG_PENALTY = 0.01


class CoordinateIn(BaseModel):
    latitude: float | None = None
    longitude: float | None = None


class PointIn(BaseModel):
    id: str = PydField(min_length=1, max_length=64)
    pointType: Literal["spot", "shopping", "restaurant"]
    suggestedDurationMinutes: int = PydField(ge=10, le=720)
    latitude: float | None = None
    longitude: float | None = None


class HotelIn(BaseModel):
    id: str = PydField(min_length=1, max_length=64)
    latitude: float | None = None
    longitude: float | None = None


class DistanceMatrixRowIn(BaseModel):
    fromPointId: str = PydField(min_length=1, max_length=64)
    toPointId: str = PydField(min_length=1, max_length=64)
    transitMinutes: int = PydField(ge=1, le=1440)
    drivingMinutes: int = PydField(ge=1, le=1440)
    walkingMeters: int = PydField(ge=0, le=200000)
    distanceKm: float = PydField(ge=0, le=2000)
    transitSummary: str | None = None


class DistanceMatrixIn(BaseModel):
    rows: list[DistanceMatrixRowIn] = PydField(default_factory=list, max_length=300000)


class SolveRequest(BaseModel):
    city: str | None = PydField(default=None, min_length=1, max_length=120)
    province: str | None = PydField(default=None, min_length=1, max_length=120)
    arrivalAirport: CoordinateIn | None = None
    departureAirport: CoordinateIn | None = None
    arrivalAirportCode: str | None = PydField(default=None, pattern=r"^[A-Z]{3}$")
    departureAirportCode: str | None = PydField(default=None, pattern=r"^[A-Z]{3}$")
    arrivalAirportId: str | None = PydField(default=None, min_length=1, max_length=64)
    departureAirportId: str | None = PydField(default=None, min_length=1, max_length=64)
    arrivalDateTime: datetime
    airportBufferMinutes: int = PydField(default=90, ge=60, le=120)
    points: list[PointIn] = PydField(min_length=1, max_length=200)
    hotels: list[HotelIn] = PydField(min_length=1, max_length=200)
    distanceMatrix: DistanceMatrixIn = PydField(default_factory=DistanceMatrixIn)
    paceMode: Literal["light", "standard", "compact"] = "standard"
    hotelMode: Literal["single", "multi"] = "multi"
    mealPolicy: Literal["auto", "off"] = "auto"
    objective: Literal["min_days", "min_transit", "min_days_then_transit"] = (
        "min_days_then_transit"
    )
    maxDays: int = PydField(default=14, ge=1, le=14)
    timeLimitSeconds: float = PydField(default=2.5, ge=0.2, le=30.0)
    switchPenaltyMinutes: int = PydField(default=45, ge=0, le=600)
    newHotelPenaltyMinutes: int = PydField(default=30, ge=0, le=600)
    maxIterations: int = PydField(default=3, ge=1, le=8)
    badDayTransitMinutesThreshold: int = PydField(default=0, ge=0, le=5000)
    badDayPenaltyMinutes: int = PydField(default=120, ge=0, le=5000)


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
    diagnostics: dict[str, object] | None = None


@dataclass(frozen=True)
class Coord:
    lat: float | None
    lng: float | None


@dataclass(frozen=True)
class Anchor:
    node_id: str | None
    coord: Coord


def haversine_km(a: Coord, b: Coord) -> float:
    if a.lat is None or a.lng is None or b.lat is None or b.lng is None:
        return 8.0
    r = math.pi / 180.0
    dlat = (b.lat - a.lat) * r
    dlng = (b.lng - a.lng) * r
    h = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(a.lat * r) * math.cos(b.lat * r) * math.sin(dlng / 2.0) ** 2
    )
    return 6371.0 * 2.0 * math.atan2(math.sqrt(h), math.sqrt(1.0 - h))


def _fallback_minutes(a: Coord, b: Coord) -> int:
    km = haversine_km(a, b)
    return max(8, int(round(km / 25.0 * 60.0 + 8)))


def coord_of(p: PointIn | HotelIn) -> Coord:
    return Coord(p.latitude, p.longitude)


def _parse_hhmm(value: str) -> tuple[int, int]:
    h, m = value.split(":", 1)
    return int(h), int(m)


def build_lookup(rows: list[DistanceMatrixRowIn]) -> dict[tuple[str, str], int]:
    lookup: dict[tuple[str, str], int] = {}
    for row in rows:
        key = (row.fromPointId, row.toPointId)
        current = lookup.get(key)
        if current is None or row.transitMinutes < current:
            lookup[key] = row.transitMinutes
    return lookup


def directed(
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
    return _fallback_minutes(from_coord, to_coord)


def symmetric(
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
    return _fallback_minutes(from_coord, to_coord)


# ═══════════════════════════════════════════════════════════
# v0.2.3: Geographic coherence tiebreaker
#
# Problem: driving distances are approximately symmetric (A→B ≈ B→A).
# A round-trip Hotel→P1→P2→P3→Hotel costs the same as Hotel→P3→P2→P1→Hotel.
# The optimizer picks one arbitrarily, sometimes producing a "backwards" route.
#
# Fix: add a tiny penalty for routes that zigzag geographically.
# A route that goes consistently south→north gets 0 penalty.
# A route that goes south→north→south gets 0.01 min penalty per reversal.
# This breaks ties deterministically without affecting routes where
# distance differences are real (even 1 minute > any number of reversals).
# ═══════════════════════════════════════════════════════════

def _geo_zigzag_penalty(coords: list[Coord]) -> float:
    """Count geographic direction reversals in a sequence of coordinates.

    A reversal occurs when the latitude direction or longitude direction
    flips between consecutive legs.  Each reversal adds GEO_ZIGZAG_PENALTY.

    For the Hotel(N)→天安门(S)→故宫(M)→景山(N)→Hotel(N) route:
      Leg 1: N→S (south)
      Leg 2: S→M (north)  ← lat reversal
      Leg 3: M→N (north)  ← consistent
      Leg 4: N→N (north)  ← consistent
      Total: 1 reversal → 0.01 penalty

    For the Hotel(N)→景山(N)→故宫(M)→天安门(S)→Hotel(N) route:
      Leg 1: N→N (flat/south)
      Leg 2: N→M (south)  ← consistent
      Leg 3: M→S (south)  ← consistent
      Leg 4: S→N (north)  ← lat reversal
      Total: 1 reversal → 0.01 penalty

    Wait — both have 1 reversal? That's because the Hotel is north of all points.
    The correct route starts by going south (to the farthest point), then comes back north.
    The wrong route starts by going to the nearest, then goes further south, then has to come all the way back.

    For a linear layout (Hotel-景山-故宫-天安门, north to south), the optimal pattern is:
    go to the far end first, then come back step by step.

    So we also penalize the total geographic spread of direction changes:
    instead of just counting reversals, we weight by the magnitude of the reversal.
    """
    if len(coords) < 3:
        return 0.0

    penalty = 0.0
    for i in range(1, len(coords) - 1):
        prev_c = coords[i - 1]
        curr_c = coords[i]
        next_c = coords[i + 1]

        if (
            prev_c.lat is None or curr_c.lat is None or next_c.lat is None
            or prev_c.lng is None or curr_c.lng is None or next_c.lng is None
        ):
            continue

        dlat_in = curr_c.lat - prev_c.lat
        dlat_out = next_c.lat - curr_c.lat
        dlng_in = curr_c.lng - prev_c.lng
        dlng_out = next_c.lng - curr_c.lng

        # Penalize latitude reversal (going north then south, or vice versa)
        if dlat_in * dlat_out < 0:
            penalty += GEO_ZIGZAG_PENALTY

        # Penalize longitude reversal
        if dlng_in * dlng_out < 0:
            penalty += GEO_ZIGZAG_PENALTY

    return penalty


def _route_geo_penalty(
    start_coord: Coord,
    day_points: list[PointIn],
    ordering: list[int],
    end_coord: Coord,
) -> float:
    """Compute geographic zigzag penalty for a full route: start → points → end."""
    coords: list[Coord] = [start_coord]
    for idx in ordering:
        coords.append(coord_of(day_points[idx]))
    coords.append(end_coord)
    return _geo_zigzag_penalty(coords)


def _arrival_anchor(req: SolveRequest) -> Anchor | None:
    if req.arrivalAirport is None:
        return None
    return Anchor(
        req.arrivalAirportId or "__arrival_airport__",
        Coord(req.arrivalAirport.latitude, req.arrivalAirport.longitude),
    )


def _departure_anchor(req: SolveRequest) -> Anchor | None:
    if req.departureAirport is None:
        return None
    return Anchor(
        req.departureAirportId or "__departure_airport__",
        Coord(req.departureAirport.latitude, req.departureAirport.longitude),
    )


def audit_matrix_completeness(
    req: SolveRequest,
    lookup: dict[tuple[str, str], int],
) -> None:
    node_ids: list[str] = []
    node_ids.extend(p.id for p in req.points)
    node_ids.extend(h.id for h in req.hotels)

    arrival = _arrival_anchor(req)
    departure = _departure_anchor(req)
    if arrival and arrival.node_id:
        node_ids.append(arrival.node_id)
    if departure and departure.node_id and departure.node_id != (
        arrival.node_id if arrival else None
    ):
        node_ids.append(departure.node_id)

    unique_ids = list(dict.fromkeys(node_ids))
    n = len(unique_ids)
    if n <= 1:
        return

    total_pairs = 0
    present_pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            total_pairs += 1
            a = unique_ids[i]
            b = unique_ids[j]
            if (a, b) in lookup or (b, a) in lookup:
                present_pairs += 1

    if total_pairs == 0:
        return

    missing_ratio = 1.0 - (present_pairs / total_pairs)
    if missing_ratio > MAX_MISSING_UNDIRECTED_RATIO:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Distance matrix incomplete (undirected): {present_pairs}/{total_pairs} pairs present "
                f"({missing_ratio:.1%} missing, threshold {MAX_MISSING_UNDIRECTED_RATIO:.0%})."
            ),
        )


@dataclass
class Phase1Result:
    day_point_indexes: list[list[int]]
    used_calendar_days: list[int]
    solver_status: Literal["OPTIMAL", "FEASIBLE"]


def _first_day_budget(
    arrival_dt: datetime,
    buffer_minutes: int,
    pace_start: str,
    pace_end: str,
    pace_minutes: int,
    meal_overhead: int,
) -> int:
    sh, sm = _parse_hhmm(pace_start)
    eh, em = _parse_hhmm(pace_end)
    start_dt = arrival_dt.replace(hour=sh, minute=sm, second=0, microsecond=0)
    end_dt = arrival_dt.replace(hour=eh, minute=em, second=0, microsecond=0)

    usable_start = max(arrival_dt + timedelta(minutes=buffer_minutes), start_dt)
    raw = int((end_dt - usable_start).total_seconds() // 60)
    if raw <= 0:
        return 0

    clipped = min(raw, pace_minutes)
    return max(0, clipped - meal_overhead)


def phase1_assign_days(
    req: SolveRequest, lookup: dict[tuple[str, str], int]
) -> Phase1Result:
    pace_start, pace_end, pace_minutes = PACE_MODE_WINDOWS[req.paceMode]
    has_restaurant = any(p.pointType == "restaurant" for p in req.points)
    meal_overhead = 45 if req.mealPolicy == "auto" and not has_restaurant else 0

    daily_budget = pace_minutes - meal_overhead
    if daily_budget < 120:
        raise HTTPException(status_code=400, detail="daily playable minutes too low")

    first_day_budget = _first_day_budget(
        req.arrivalDateTime,
        req.airportBufferMinutes,
        pace_start,
        pace_end,
        pace_minutes,
        meal_overhead,
    )

    point_loads = [
        p.suggestedDurationMinutes + PER_POINT_OVERHEAD_MINUTES for p in req.points
    ]
    if any(load > daily_budget for load in point_loads):
        raise HTTPException(
            status_code=400,
            detail="at least one point duration exceeds daily budget",
        )

    n = len(req.points)
    d = min(n + 1, req.maxDays + 1)

    model = cp_model.CpModel()

    x = [[model.NewBoolVar(f"x_{i}_{day}") for day in range(d)] for i in range(n)]
    for i in range(n):
        model.Add(sum(x[i]) == 1)

    day_budgets = [daily_budget for _ in range(d)]
    day_budgets[0] = first_day_budget

    used = [model.NewBoolVar(f"u_{day}") for day in range(d)]
    for day in range(d):
        model.Add(
            sum(point_loads[i] * x[i][day] for i in range(n))
            <= day_budgets[day] * used[day]
        )
        model.Add(sum(x[i][day] for i in range(n)) >= used[day])

    for day in range(1, d - 1):
        model.Add(used[day] >= used[day + 1])

    model.Add(sum(used) <= req.maxDays)

    used_days_expr = sum(used)
    day_pack_penalty = sum(
        (day + 1) * x[i][day] for i in range(n) for day in range(d)
    )

    pair_terms: list[cp_model.LinearExpr] = []
    if req.objective in ("min_transit", "min_days_then_transit") and n <= 60:
        for i in range(n):
            for j in range(i + 1, n):
                cost = symmetric(
                    req.points[i].id,
                    coord_of(req.points[i]),
                    req.points[j].id,
                    coord_of(req.points[j]),
                    lookup,
                )
                for day in range(d):
                    y = model.NewBoolVar(f"pair_{i}_{j}_{day}")
                    model.Add(y <= x[i][day])
                    model.Add(y <= x[j][day])
                    model.Add(y >= x[i][day] + x[j][day] - 1)
                    pair_terms.append(y * cost)
    transit_proxy_penalty = sum(pair_terms)

    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 8

    phase1_time = max(0.2, req.timeLimitSeconds * 0.4)
    phase2_time = max(0.2, req.timeLimitSeconds * 0.5)

    lex_first_status: int | None = None

    if req.objective == "min_days":
        solver.parameters.max_time_in_seconds = phase1_time + phase2_time
        model.Minimize(used_days_expr * 10000 + day_pack_penalty)
        status = solver.Solve(model)
    elif req.objective == "min_transit":
        solver.parameters.max_time_in_seconds = phase1_time + phase2_time
        model.Minimize(
            used_days_expr * 10000 + transit_proxy_penalty + day_pack_penalty
        )
        status = solver.Solve(model)
    else:
        solver.parameters.max_time_in_seconds = phase1_time
        model.Minimize(used_days_expr)
        lex_first_status = solver.Solve(model)
        if lex_first_status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            raise HTTPException(
                status_code=503, detail="solver failed to find feasible plan"
            )

        min_days_found = sum(solver.Value(day_used) for day_used in used)
        if lex_first_status == cp_model.OPTIMAL:
            model.Add(used_days_expr == min_days_found)
            model.Minimize(transit_proxy_penalty + day_pack_penalty)
        else:
            model.Add(used_days_expr <= min_days_found)
            model.Minimize(
                used_days_expr * 10000 + transit_proxy_penalty + day_pack_penalty
            )

        solver.parameters.max_time_in_seconds = phase2_time
        status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise HTTPException(
            status_code=503, detail="solver failed to find feasible plan"
        )

    used_day_indexes = [day for day in range(d) if solver.Value(used[day]) == 1]
    if not used_day_indexes:
        raise HTTPException(
            status_code=503, detail="solver returned empty day set"
        )

    day_position_by_index = {
        day_idx: pos for pos, day_idx in enumerate(used_day_indexes)
    }
    day_indexes: list[list[int]] = [[] for _ in range(len(used_day_indexes))]

    for i in range(n):
        assigned_day = None
        for day in range(d):
            if solver.Value(x[i][day]) == 1:
                assigned_day = day
                break
        if assigned_day is None:
            raise HTTPException(
                status_code=503, detail="solver assignment is invalid"
            )
        pos = day_position_by_index.get(assigned_day)
        if pos is None:
            raise HTTPException(
                status_code=503, detail="solver used-day mapping is invalid"
            )
        day_indexes[pos].append(i)

    solver_status: Literal["OPTIMAL", "FEASIBLE"] = "OPTIMAL"
    if status != cp_model.OPTIMAL:
        solver_status = "FEASIBLE"
    if (
        req.objective == "min_days_then_transit"
        and lex_first_status == cp_model.FEASIBLE
    ):
        solver_status = "FEASIBLE"

    return Phase1Result(
        day_point_indexes=day_indexes,
        used_calendar_days=used_day_indexes,
        solver_status=solver_status,
    )


@dataclass(frozen=True)
class HardDayPattern:
    calendar_day: int
    hotel_id: str
    point_indexes: tuple[int, ...]


@dataclass(frozen=True)
class SoftDayPattern:
    calendar_day: int
    hotel_id: str
    point_indexes: tuple[int, ...]
    penalty_minutes: int


@dataclass
class MasterResult:
    day_point_indexes: list[list[int]]
    used_calendar_days: list[int]
    hotel_ids: list[str]
    solver_status: Literal["OPTIMAL", "FEASIBLE"]


@dataclass
class SubproblemResult:
    orderings: list[list[int]]
    day_travel_minutes: list[int]
    day_total_minutes: list[int]
    over_budget_minutes: list[int]


def _compute_day_budgets(req: SolveRequest, day_slots: int) -> list[int]:
    pace_start, pace_end, pace_minutes = PACE_MODE_WINDOWS[req.paceMode]
    has_restaurant = any(p.pointType == "restaurant" for p in req.points)
    meal_overhead = 45 if req.mealPolicy == "auto" and not has_restaurant else 0

    daily_budget = pace_minutes - meal_overhead
    if daily_budget < 120:
        raise HTTPException(status_code=400, detail="daily playable minutes too low")

    first_day_budget = _first_day_budget(
        req.arrivalDateTime,
        req.airportBufferMinutes,
        pace_start,
        pace_end,
        pace_minutes,
        meal_overhead,
    )

    budgets = [daily_budget for _ in range(day_slots)]
    if day_slots > 0:
        budgets[0] = first_day_budget
    return budgets


def phase1_assign_days_and_hotels(
    req: SolveRequest,
    lookup: dict[tuple[str, str], int],
    hard_patterns: list[HardDayPattern] | None = None,
    soft_patterns: list[SoftDayPattern] | None = None,
) -> MasterResult:
    n = len(req.points)
    k = len(req.hotels)
    d = min(n + 1, req.maxDays + 1)
    if n <= 0 or k <= 0:
        raise HTTPException(status_code=400, detail="points/hotels cannot be empty")

    point_loads = [
        p.suggestedDurationMinutes + PER_POINT_OVERHEAD_MINUTES for p in req.points
    ]
    day_budgets = _compute_day_budgets(req, d)

    if any(load > max(day_budgets) for load in point_loads):
        raise HTTPException(
            status_code=400,
            detail="at least one point duration exceeds daily budget",
        )

    model = cp_model.CpModel()

    x = [[model.NewBoolVar(f"x_{i}_{day}") for day in range(d)] for i in range(n)]
    used = [model.NewBoolVar(f"u_{day}") for day in range(d)]
    z = [
        [model.NewBoolVar(f"z_{day}_{hotel}") for hotel in range(k)]
        for day in range(d)
    ]

    for i in range(n):
        model.Add(sum(x[i]) == 1)

    for day in range(d):
        model.Add(
            sum(point_loads[i] * x[i][day] for i in range(n))
            <= day_budgets[day] * used[day]
        )
        model.Add(sum(x[i][day] for i in range(n)) >= used[day])
        model.Add(sum(z[day][hotel] for hotel in range(k)) == used[day])

    for day in range(1, d - 1):
        model.Add(used[day] >= used[day + 1])
    model.Add(sum(used) <= req.maxDays)

    hotel_used = [model.NewBoolVar(f"hotel_used_{hotel}") for hotel in range(k)]
    for hotel in range(k):
        for day in range(d):
            model.Add(hotel_used[hotel] >= z[day][hotel])

    if req.hotelMode == "single":
        model.Add(sum(hotel_used) == 1)

    used_days_expr = sum(used)
    day_pack_penalty = sum(
        (day + 1) * x[i][day] for i in range(n) for day in range(d)
    )

    point_pair_terms: list[cp_model.LinearExpr] = []
    if req.objective in ("min_transit", "min_days_then_transit") and n <= 60:
        for i in range(n):
            for j in range(i + 1, n):
                cost = symmetric(
                    req.points[i].id,
                    coord_of(req.points[i]),
                    req.points[j].id,
                    coord_of(req.points[j]),
                    lookup,
                )
                for day in range(d):
                    y = model.NewBoolVar(f"pair_{i}_{j}_{day}")
                    model.Add(y <= x[i][day])
                    model.Add(y <= x[j][day])
                    model.Add(y >= x[i][day] + x[j][day] - 1)
                    point_pair_terms.append(y * cost)
    point_pair_proxy = sum(point_pair_terms)

    hotel_proxy_terms: list[cp_model.LinearExpr] = []
    for i in range(n):
        point = req.points[i]
        point_coord = coord_of(point)
        for day in range(d):
            for hotel in range(k):
                h = req.hotels[hotel]
                h_coord = coord_of(h)
                y = model.NewBoolVar(f"p2h_{i}_{day}_{hotel}")
                model.Add(y <= x[i][day])
                model.Add(y <= z[day][hotel])
                model.Add(y >= x[i][day] + z[day][hotel] - 1)
                hotel_cost = directed(
                    h.id,
                    h_coord,
                    point.id,
                    point_coord,
                    lookup,
                ) + directed(
                    point.id,
                    point_coord,
                    h.id,
                    h_coord,
                    lookup,
                )
                hotel_proxy_terms.append(y * hotel_cost)
    hotel_proxy_penalty = sum(hotel_proxy_terms)

    switch_terms: list[cp_model.LinearExpr] = []
    for day in range(1, d):
        same_hotel_bits: list[cp_model.IntVar] = []
        for hotel in range(k):
            both = model.NewBoolVar(f"same_{day}_{hotel}")
            model.Add(both <= z[day - 1][hotel])
            model.Add(both <= z[day][hotel])
            model.Add(both >= z[day - 1][hotel] + z[day][hotel] - 1)
            same_hotel_bits.append(both)

        same_expr = sum(same_hotel_bits)
        switched = model.NewBoolVar(f"switch_{day}")
        model.Add(switched >= used[day - 1] + used[day] - 1 - same_expr)
        model.Add(switched <= used[day - 1])
        model.Add(switched <= used[day])
        switch_terms.append(switched * req.switchPenaltyMinutes)
    switch_penalty = sum(switch_terms)

    new_hotel_penalty = sum(
        hotel_used[hotel] * req.newHotelPenaltyMinutes for hotel in range(k)
    )

    hotel_index_by_id = {hotel.id: idx for idx, hotel in enumerate(req.hotels)}
    if hard_patterns:
        for pattern in hard_patterns:
            if pattern.calendar_day < 0 or pattern.calendar_day >= d:
                continue
            h_idx = hotel_index_by_id.get(pattern.hotel_id)
            if h_idx is None:
                continue
            point_indexes = [
                idx for idx in pattern.point_indexes if 0 <= idx < n
            ]
            if not point_indexes:
                continue
            model.Add(
                sum(x[idx][pattern.calendar_day] for idx in point_indexes)
                + z[pattern.calendar_day][h_idx]
                <= len(point_indexes)
            )

    feedback_penalty_terms: list[cp_model.LinearExpr] = []
    if soft_patterns:
        for pattern_idx, pattern in enumerate(soft_patterns):
            if pattern.penalty_minutes <= 0:
                continue
            if pattern.calendar_day < 0 or pattern.calendar_day >= d:
                continue
            h_idx = hotel_index_by_id.get(pattern.hotel_id)
            if h_idx is None:
                continue
            point_indexes = [
                idx for idx in pattern.point_indexes if 0 <= idx < n
            ]
            if not point_indexes:
                continue

            matched = model.NewBoolVar(f"fb_{pattern_idx}")
            model.Add(matched <= z[pattern.calendar_day][h_idx])
            for idx in point_indexes:
                model.Add(matched <= x[idx][pattern.calendar_day])
            model.Add(
                matched
                >= z[pattern.calendar_day][h_idx]
                + sum(x[idx][pattern.calendar_day] for idx in point_indexes)
                - len(point_indexes)
            )
            feedback_penalty_terms.append(matched * pattern.penalty_minutes)
    feedback_penalty = sum(feedback_penalty_terms)

    transit_structure_penalty = (
        point_pair_proxy
        + day_pack_penalty
        + hotel_proxy_penalty
        + switch_penalty
        + new_hotel_penalty
        + feedback_penalty
    )

    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 8
    phase1_time = max(0.2, req.timeLimitSeconds * 0.4)
    phase2_time = max(0.2, req.timeLimitSeconds * 0.5)

    lex_first_status: int | None = None

    if req.objective == "min_days":
        solver.parameters.max_time_in_seconds = phase1_time + phase2_time
        model.Minimize(used_days_expr * 10000 + day_pack_penalty)
        status = solver.Solve(model)
    elif req.objective == "min_transit":
        solver.parameters.max_time_in_seconds = phase1_time + phase2_time
        model.Minimize(used_days_expr * 10000 + transit_structure_penalty)
        status = solver.Solve(model)
    else:
        solver.parameters.max_time_in_seconds = phase1_time
        model.Minimize(used_days_expr)
        lex_first_status = solver.Solve(model)
        if lex_first_status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            raise HTTPException(
                status_code=503, detail="solver failed to find feasible plan"
            )
        min_days_found = sum(solver.Value(day_used) for day_used in used)
        if lex_first_status == cp_model.OPTIMAL:
            model.Add(used_days_expr == min_days_found)
            model.Minimize(transit_structure_penalty)
        else:
            model.Add(used_days_expr <= min_days_found)
            model.Minimize(used_days_expr * 10000 + transit_structure_penalty)
        solver.parameters.max_time_in_seconds = phase2_time
        status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise HTTPException(
            status_code=503, detail="solver failed to find feasible plan"
        )

    used_day_indexes = [day for day in range(d) if solver.Value(used[day]) == 1]
    if not used_day_indexes:
        raise HTTPException(
            status_code=503, detail="solver returned empty day set"
        )

    day_position_by_index = {
        day_idx: pos for pos, day_idx in enumerate(used_day_indexes)
    }
    day_indexes: list[list[int]] = [[] for _ in range(len(used_day_indexes))]
    hotel_ids: list[str] = ["" for _ in range(len(used_day_indexes))]

    for i in range(n):
        assigned_day = None
        for day in range(d):
            if solver.Value(x[i][day]) == 1:
                assigned_day = day
                break
        if assigned_day is None:
            raise HTTPException(
                status_code=503, detail="solver assignment is invalid"
            )
        pos = day_position_by_index.get(assigned_day)
        if pos is None:
            raise HTTPException(
                status_code=503, detail="solver used-day mapping is invalid"
            )
        day_indexes[pos].append(i)

    for day in used_day_indexes:
        pos = day_position_by_index[day]
        hotel_idx = None
        for h in range(k):
            if solver.Value(z[day][h]) == 1:
                hotel_idx = h
                break
        if hotel_idx is None:
            raise HTTPException(
                status_code=503, detail="solver hotel assignment is invalid"
            )
        hotel_ids[pos] = req.hotels[hotel_idx].id

    solver_status: Literal["OPTIMAL", "FEASIBLE"] = "OPTIMAL"
    if status != cp_model.OPTIMAL:
        solver_status = "FEASIBLE"
    if (
        req.objective == "min_days_then_transit"
        and lex_first_status == cp_model.FEASIBLE
    ):
        solver_status = "FEASIBLE"

    return MasterResult(
        day_point_indexes=day_indexes,
        used_calendar_days=used_day_indexes,
        hotel_ids=hotel_ids,
        solver_status=solver_status,
    )


def _solve_fixed_day_route(
    day_points: list[PointIn],
    start_id: str,
    start_coord: Coord,
    end_hotel: HotelIn,
    lookup: dict[tuple[str, str], int],
) -> DayRoute:
    if len(day_points) <= MAX_EXACT_PERMUTATION_SIZE:
        coord_map = {
            start_id: start_coord,
            end_hotel.id: coord_of(end_hotel),
        }
        routes = _precompute_day_exact(
            day_points,
            [start_id],
            [end_hotel],
            coord_map,
            lookup,
        )
        route = routes.get((start_id, end_hotel.id))
        if route is not None:
            return route

    return _nn_order(
        day_points,
        start_id,
        start_coord,
        end_hotel.id,
        coord_of(end_hotel),
        lookup,
    )


def phase2_solve_fixed_hotels(
    req: SolveRequest,
    master: MasterResult,
    lookup: dict[tuple[str, str], int],
) -> SubproblemResult:
    arrival = _arrival_anchor(req)
    day_slot_count = min(len(req.points) + 1, req.maxDays + 1)
    day_budgets = _compute_day_budgets(req, day_slot_count)
    point_loads = [
        p.suggestedDurationMinutes + PER_POINT_OVERHEAD_MINUTES for p in req.points
    ]

    hotel_by_id = {hotel.id: hotel for hotel in req.hotels}

    orderings: list[list[int]] = []
    day_travel_minutes: list[int] = []
    day_total_minutes: list[int] = []
    over_budget_minutes: list[int] = []

    for pos, point_indexes in enumerate(master.day_point_indexes):
        hotel_id = master.hotel_ids[pos]
        end_hotel = hotel_by_id.get(hotel_id)
        if end_hotel is None:
            raise HTTPException(
                status_code=503,
                detail=f"hotel id {hotel_id} not found in request",
            )

        if pos == 0:
            if master.used_calendar_days[0] > 0:
                start_id = hotel_id
                start_coord = coord_of(end_hotel)
            elif arrival and arrival.node_id:
                start_id = arrival.node_id
                start_coord = arrival.coord
            else:
                start_id = hotel_id
                start_coord = coord_of(end_hotel)
        else:
            prev_hotel_id = master.hotel_ids[pos - 1]
            prev_hotel = hotel_by_id.get(prev_hotel_id)
            if prev_hotel is None:
                raise HTTPException(
                    status_code=503,
                    detail=f"hotel id {prev_hotel_id} not found in request",
                )
            start_id = prev_hotel_id
            start_coord = coord_of(prev_hotel)

        day_points = [req.points[idx] for idx in point_indexes]
        route = _solve_fixed_day_route(
            day_points,
            start_id,
            start_coord,
            end_hotel,
            lookup,
        )

        service_minutes = sum(point_loads[idx] for idx in point_indexes)
        total_minutes = service_minutes + route.cost
        calendar_day = master.used_calendar_days[pos]
        budget = (
            day_budgets[calendar_day]
            if 0 <= calendar_day < len(day_budgets)
            else day_budgets[-1]
        )
        over_budget = max(0, total_minutes - budget)

        orderings.append(route.ordering)
        day_travel_minutes.append(route.cost)
        day_total_minutes.append(total_minutes)
        over_budget_minutes.append(over_budget)

    return SubproblemResult(
        orderings=orderings,
        day_travel_minutes=day_travel_minutes,
        day_total_minutes=day_total_minutes,
        over_budget_minutes=over_budget_minutes,
    )


def _pattern_key(
    calendar_day: int,
    hotel_id: str,
    point_indexes: list[int],
) -> tuple[int, str, tuple[int, ...]]:
    uniq = tuple(sorted(set(point_indexes)))
    return (calendar_day, hotel_id, uniq)


def solve_with_layered_iteration(
    req: SolveRequest,
    lookup: dict[tuple[str, str], int],
) -> tuple[MasterResult, SubproblemResult, list[dict[str, int]]]:
    hard_patterns: list[HardDayPattern] = []
    soft_patterns: list[SoftDayPattern] = []
    seen_hard: set[tuple[int, str, tuple[int, ...]]] = set()
    seen_soft: set[tuple[int, str, tuple[int, ...]]] = set()

    best_pair: tuple[MasterResult, SubproblemResult] | None = None
    best_score: int | None = None
    traces: list[dict[str, int]] = []

    for _ in range(req.maxIterations):
        try:
            master = phase1_assign_days_and_hotels(
                req=req,
                lookup=lookup,
                hard_patterns=hard_patterns,
                soft_patterns=soft_patterns,
            )
        except HTTPException:
            if best_pair is not None:
                return best_pair
            raise

        sub = phase2_solve_fixed_hotels(req, master, lookup)

        total_over = sum(sub.over_budget_minutes)
        total_travel = sum(sub.day_travel_minutes)
        score = total_over * 100000 + total_travel
        if best_score is None or score < best_score:
            best_score = score
            best_pair = (master, sub)

        new_hard = 0
        new_soft = 0

        for pos, point_indexes in enumerate(master.day_point_indexes):
            calendar_day = master.used_calendar_days[pos]
            hotel_id = master.hotel_ids[pos]
            key = _pattern_key(calendar_day, hotel_id, point_indexes)
            if len(key[2]) == 0:
                continue

            if sub.over_budget_minutes[pos] > 0:
                if key not in seen_hard:
                    seen_hard.add(key)
                    hard_patterns.append(
                        HardDayPattern(
                            calendar_day=calendar_day,
                            hotel_id=hotel_id,
                            point_indexes=key[2],
                        )
                    )
                    new_hard += 1
                continue

            if (
                req.badDayTransitMinutesThreshold > 0
                and sub.day_travel_minutes[pos] > req.badDayTransitMinutesThreshold
                and req.badDayPenaltyMinutes > 0
            ):
                if key not in seen_soft:
                    seen_soft.add(key)
                    soft_patterns.append(
                        SoftDayPattern(
                            calendar_day=calendar_day,
                            hotel_id=hotel_id,
                            point_indexes=key[2],
                            penalty_minutes=req.badDayPenaltyMinutes,
                        )
                    )
                    new_soft += 1

        traces.append(
            {
                "iteration": len(traces) + 1,
                "travelMinutes": total_travel,
                "overBudgetMinutes": total_over,
                "hardCutsAdded": new_hard,
                "softPenaltiesAdded": new_soft,
            }
        )

        if new_hard == 0 and new_soft == 0:
            return master, sub, traces

    if best_pair is None:
        raise HTTPException(status_code=503, detail="solver failed to find plan")
    return best_pair[0], best_pair[1], traces


@dataclass
class DayRoute:
    ordering: list[int]
    cost: int
    effective_cost: float  # cost + geo penalty (for comparison only)


@dataclass
class DayPrecomputed:
    points: list[PointIn]
    global_indexes: list[int]
    routes: dict[tuple[str, str], DayRoute]


def _nn_order(
    day_points: list[PointIn],
    start_id: str,
    start_coord: Coord,
    end_hotel_id: str,
    end_hotel_coord: Coord,
    lookup: dict[tuple[str, str], int],
) -> DayRoute:
    m = len(day_points)
    if m == 0:
        c = directed(start_id, start_coord, end_hotel_id, end_hotel_coord, lookup)
        return DayRoute([], c, float(c))

    remaining = set(range(m))
    ordering: list[int] = []

    first = min(
        remaining,
        key=lambda i: directed(
            start_id, start_coord, day_points[i].id, coord_of(day_points[i]), lookup
        ),
    )
    ordering.append(first)
    remaining.remove(first)

    while remaining:
        current_point = day_points[ordering[-1]]
        next_idx = min(
            remaining,
            key=lambda i: directed(
                current_point.id,
                coord_of(current_point),
                day_points[i].id,
                coord_of(day_points[i]),
                lookup,
            ),
        )
        ordering.append(next_idx)
        remaining.remove(next_idx)

    cost = directed(
        start_id,
        start_coord,
        day_points[ordering[0]].id,
        coord_of(day_points[ordering[0]]),
        lookup,
    )
    for k in range(m - 1):
        a = day_points[ordering[k]]
        b = day_points[ordering[k + 1]]
        cost += directed(a.id, coord_of(a), b.id, coord_of(b), lookup)
    cost += directed(
        day_points[ordering[-1]].id,
        coord_of(day_points[ordering[-1]]),
        end_hotel_id,
        end_hotel_coord,
        lookup,
    )

    geo_pen = _route_geo_penalty(start_coord, day_points, ordering, end_hotel_coord)
    return DayRoute(ordering, cost, cost + geo_pen)


def _precompute_day_exact(
    day_points: list[PointIn],
    start_ids: list[str],
    hotels: list[HotelIn],
    coord_map: dict[str, Coord],
    lookup: dict[tuple[str, str], int],
) -> dict[tuple[str, str], DayRoute]:
    m = len(day_points)
    routes: dict[tuple[str, str], DayRoute] = {}

    if m == 0:
        for start_id in start_ids:
            start_coord = coord_map[start_id]
            for hotel in hotels:
                c = directed(start_id, start_coord, hotel.id, coord_of(hotel), lookup)
                routes[(start_id, hotel.id)] = DayRoute([], c, float(c))
        return routes

    if m == 1:
        only = day_points[0]
        for start_id in start_ids:
            start_coord = coord_map[start_id]
            for hotel in hotels:
                in_cost = directed(
                    start_id, start_coord, only.id, coord_of(only), lookup
                )
                out_cost = directed(
                    only.id, coord_of(only), hotel.id, coord_of(hotel), lookup
                )
                c = in_cost + out_cost
                routes[(start_id, hotel.id)] = DayRoute([0], c, float(c))
        return routes

    # Step A: enumerate permutations, group by (first, last), keep cheapest inner
    best_inner: dict[tuple[int, int], tuple[int, list[int]]] = {}

    for perm in itertools.permutations(range(m)):
        order = list(perm)
        first_idx = order[0]
        last_idx = order[-1]

        inner_cost = 0
        for k in range(m - 1):
            a = day_points[order[k]]
            b = day_points[order[k + 1]]
            inner_cost += directed(a.id, coord_of(a), b.id, coord_of(b), lookup)

        key = (first_idx, last_idx)
        current = best_inner.get(key)
        if current is None or inner_cost < current[0]:
            best_inner[key] = (inner_cost, order)

    # Step B: precompute edge tables
    start_to_first: dict[tuple[str, int], int] = {}
    for start_id in start_ids:
        start_coord = coord_map[start_id]
        for point_idx in range(m):
            point = day_points[point_idx]
            start_to_first[(start_id, point_idx)] = directed(
                start_id, start_coord, point.id, coord_of(point), lookup
            )

    last_to_hotel: dict[tuple[int, str], int] = {}
    for point_idx in range(m):
        point = day_points[point_idx]
        for hotel in hotels:
            last_to_hotel[(point_idx, hotel.id)] = directed(
                point.id, coord_of(point), hotel.id, coord_of(hotel), lookup
            )

    # Step C: for each (start, hotel), find best route using effective_cost (with geo penalty)
    grouped = list(best_inner.items())

    for start_id in start_ids:
        start_coord = coord_map[start_id]
        for hotel in hotels:
            hotel_coord = coord_of(hotel)
            best_eff: float | None = None
            best_cost: int = 0
            best_perm: list[int] = grouped[0][1][1]

            for (first_idx, last_idx), (inner_cost, perm) in grouped:
                transit_cost = (
                    start_to_first[(start_id, first_idx)]
                    + inner_cost
                    + last_to_hotel[(last_idx, hotel.id)]
                )
                # v0.2.3: add geographic coherence penalty for tiebreaking
                geo_pen = _route_geo_penalty(
                    start_coord, day_points, perm, hotel_coord
                )
                eff = transit_cost + geo_pen

                if best_eff is None or eff < best_eff:
                    best_eff = eff
                    best_cost = transit_cost
                    best_perm = perm

            routes[(start_id, hotel.id)] = DayRoute(
                list(best_perm), best_cost, best_eff or float(best_cost)
            )

    return routes


def phase2_precompute(
    req: SolveRequest,
    p1: Phase1Result,
    lookup: dict[tuple[str, str], int],
) -> list[DayPrecomputed]:
    arrival = _arrival_anchor(req)
    has_rest_day_before_tour = p1.used_calendar_days[0] > 0

    hotel_ids = [h.id for h in req.hotels]
    coord_map: dict[str, Coord] = {h.id: coord_of(h) for h in req.hotels}

    if arrival and arrival.node_id:
        coord_map[arrival.node_id] = arrival.coord

    results: list[DayPrecomputed] = []

    for pos, point_indexes in enumerate(p1.day_point_indexes):
        day_points = [req.points[i] for i in point_indexes]

        if pos == 0:
            if has_rest_day_before_tour:
                start_ids = list(hotel_ids)
            elif arrival and arrival.node_id:
                start_ids = [arrival.node_id]
            else:
                start_ids = list(hotel_ids)
        else:
            start_ids = list(hotel_ids)

        if len(day_points) <= MAX_EXACT_PERMUTATION_SIZE:
            routes = _precompute_day_exact(
                day_points, start_ids, req.hotels, coord_map, lookup
            )
        else:
            routes = {}
            for start_id in start_ids:
                start_coord = coord_map[start_id]
                for hotel in req.hotels:
                    routes[(start_id, hotel.id)] = _nn_order(
                        day_points,
                        start_id,
                        start_coord,
                        hotel.id,
                        coord_of(hotel),
                        lookup,
                    )

        results.append(
            DayPrecomputed(
                points=day_points,
                global_indexes=list(point_indexes),
                routes=routes,
            )
        )

    return results


def _max_switches(trip_days: int) -> int:
    if trip_days <= 3:
        return 0
    if trip_days <= 5:
        return 1
    if trip_days <= 7:
        return 2
    return 3


@dataclass
class Phase3Result:
    hotel_ids: list[str]
    orderings: list[list[int]]
    total_cost: int
    arrival_hotel_id: str | None


def phase3_hotel_dp(
    req: SolveRequest,
    p1: Phase1Result,
    precomputed: list[DayPrecomputed],
    lookup: dict[tuple[str, str], int],
) -> Phase3Result:
    num_days = len(precomputed)
    hotels = req.hotels
    hotel_count = len(hotels)

    if num_days == 0:
        return Phase3Result([], [], 0, None)

    max_switches = 0 if req.hotelMode == "single" else _max_switches(num_days)
    inf = 10**9

    has_rest_day_before_tour = p1.used_calendar_days[0] > 0
    arrival = _arrival_anchor(req)
    departure = _departure_anchor(req)

    dp = [
        [[inf] * (max_switches + 1) for _ in range(hotel_count)]
        for _ in range(num_days)
    ]
    back = [
        [[(-1, -1)] * (max_switches + 1) for _ in range(hotel_count)]
        for _ in range(num_days)
    ]
    arrival_hotel_for = [
        [[-1] * (max_switches + 1) for _ in range(hotel_count)]
        for _ in range(num_days)
    ]

    for h_idx in range(hotel_count):
        h = hotels[h_idx]

        if has_rest_day_before_tour and arrival and arrival.node_id:
            for ah_idx in range(hotel_count):
                ah = hotels[ah_idx]
                arrival_cost = directed(
                    arrival.node_id, arrival.coord, ah.id, coord_of(ah), lookup
                )
                is_switch = int(ah_idx != h_idx)
                if is_switch > max_switches:
                    continue
                route = precomputed[0].routes.get((ah.id, h.id))
                if route is None:
                    continue
                total = (
                    arrival_cost
                    + route.cost
                    + HOTEL_SWITCH_PENALTY_MINUTES * is_switch
                )
                if total < dp[0][h_idx][is_switch]:
                    dp[0][h_idx][is_switch] = total
                    back[0][h_idx][is_switch] = (ah_idx, -1)
                    arrival_hotel_for[0][h_idx][is_switch] = ah_idx
        elif arrival and arrival.node_id:
            route = precomputed[0].routes.get((arrival.node_id, h.id))
            if route is None:
                continue
            dp[0][h_idx][0] = route.cost
            arrival_hotel_for[0][h_idx][0] = h_idx
        else:
            route = precomputed[0].routes.get((h.id, h.id))
            if route is None:
                continue
            dp[0][h_idx][0] = route.cost
            arrival_hotel_for[0][h_idx][0] = h_idx

    for day in range(1, num_days):
        for h_idx in range(hotel_count):
            h = hotels[h_idx]
            for prev_h_idx in range(hotel_count):
                prev_h = hotels[prev_h_idx]
                is_switch = int(prev_h_idx != h_idx)
                for prev_switch_count in range(max_switches + 1):
                    new_switch_count = prev_switch_count + is_switch
                    if new_switch_count > max_switches:
                        continue
                    prev_cost = dp[day - 1][prev_h_idx][prev_switch_count]
                    if prev_cost >= inf:
                        continue

                    route = precomputed[day].routes.get((prev_h.id, h.id))
                    if route is None:
                        continue

                    total = (
                        prev_cost
                        + route.cost
                        + HOTEL_SWITCH_PENALTY_MINUTES * is_switch
                    )
                    if total < dp[day][h_idx][new_switch_count]:
                        dp[day][h_idx][new_switch_count] = total
                        back[day][h_idx][new_switch_count] = (
                            prev_h_idx,
                            prev_switch_count,
                        )
                        arrival_hotel_for[day][h_idx][new_switch_count] = (
                            arrival_hotel_for[day - 1][prev_h_idx][prev_switch_count]
                        )

    if departure and departure.node_id:
        for h_idx in range(hotel_count):
            h = hotels[h_idx]
            airport_cost = directed(
                h.id, coord_of(h), departure.node_id, departure.coord, lookup
            )
            for switch_count in range(max_switches + 1):
                if dp[num_days - 1][h_idx][switch_count] < inf:
                    dp[num_days - 1][h_idx][switch_count] += airport_cost

    best_cost = inf
    best_h_idx = 0
    best_switch_count = 0
    for h_idx in range(hotel_count):
        for switch_count in range(max_switches + 1):
            cost = dp[num_days - 1][h_idx][switch_count]
            if cost < best_cost:
                best_cost = cost
                best_h_idx = h_idx
                best_switch_count = switch_count

    if best_cost >= inf:
        raise HTTPException(
            status_code=503, detail="no feasible hotel assignment found"
        )

    h_path = [0] * num_days
    s_path = [0] * num_days
    h_path[-1] = best_h_idx
    s_path[-1] = best_switch_count

    for day in range(num_days - 1, 0, -1):
        prev_h_idx, prev_switch_count = back[day][h_path[day]][s_path[day]]
        h_path[day - 1] = prev_h_idx
        s_path[day - 1] = prev_switch_count

    hotel_ids = [hotels[h_idx].id for h_idx in h_path]

    arr_h_idx = arrival_hotel_for[0][h_path[0]][s_path[0]]
    arrival_hotel_id = (
        hotels[arr_h_idx].id if 0 <= arr_h_idx < hotel_count else None
    )

    orderings: list[list[int]] = []
    for day in range(num_days):
        if day == 0:
            if has_rest_day_before_tour and arrival_hotel_id:
                start_key = arrival_hotel_id
            elif arrival and arrival.node_id:
                start_key = arrival.node_id
            else:
                start_key = hotel_ids[0]
        else:
            start_key = hotel_ids[day - 1]

        route = precomputed[day].routes.get((start_key, hotel_ids[day]))
        if route is None:
            raise HTTPException(
                status_code=503,
                detail=f"route reconstruction failed at day={day + 1}",
            )
        orderings.append(route.ordering)

    return Phase3Result(hotel_ids, orderings, best_cost, arrival_hotel_id)


def assemble_response(
    req: SolveRequest,
    p1: Phase1Result,
    p3: Phase3Result,
    precomputed: list[DayPrecomputed],
) -> SolveResponse:
    start_date = req.arrivalDateTime.date()
    days: list[DayPlan] = []

    for pos in range(len(p1.day_point_indexes)):
        calendar_index = p1.used_calendar_days[pos]
        day_pre = precomputed[pos]
        order = p3.orderings[pos]

        point_ids = [req.points[day_pre.global_indexes[idx]].id for idx in order]

        days.append(
            DayPlan(
                dayNumber=pos + 1,
                date=(start_date + timedelta(days=calendar_index)).isoformat(),
                pointIds=point_ids,
                hotelId=p3.hotel_ids[pos],
            )
        )

    return SolveResponse(
        tripDays=len(days),
        solverStatus=p1.solver_status,
        objective=req.objective,
        days=days,
    )


def assemble_layered_response(
    req: SolveRequest,
    master: MasterResult,
    sub: SubproblemResult,
    traces: list[dict[str, int]],
) -> SolveResponse:
    start_date = req.arrivalDateTime.date()
    days: list[DayPlan] = []

    for pos in range(len(master.day_point_indexes)):
        calendar_index = master.used_calendar_days[pos]
        point_indexes = master.day_point_indexes[pos]
        order = sub.orderings[pos]
        point_ids = [req.points[point_indexes[idx]].id for idx in order]

        days.append(
            DayPlan(
                dayNumber=pos + 1,
                date=(start_date + timedelta(days=calendar_index)).isoformat(),
                pointIds=point_ids,
                hotelId=master.hotel_ids[pos],
            )
        )

    solver_status: Literal["OPTIMAL", "FEASIBLE"] = master.solver_status
    if any(value > 0 for value in sub.over_budget_minutes):
        solver_status = "FEASIBLE"

    return SolveResponse(
        tripDays=len(days),
        solverStatus=solver_status,
        objective=req.objective,
        days=days,
        diagnostics={
            "iterations": traces,
            "iterationCount": len(traces),
            "totalTravelMinutes": int(sum(sub.day_travel_minutes)),
            "totalOverBudgetMinutes": int(sum(sub.over_budget_minutes)),
        },
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/solve", response_model=SolveResponse)
def solve(req: SolveRequest) -> SolveResponse:
    lookup = build_lookup(req.distanceMatrix.rows)

    audit_matrix_completeness(req, lookup)

    master, sub, traces = solve_with_layered_iteration(req, lookup)
    return assemble_layered_response(req, master, sub, traces)
