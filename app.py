from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

from fastapi import FastAPI, HTTPException
from ortools.sat.python import cp_model
from pydantic import BaseModel, Field as PydField

app = FastAPI(title="Optivoy Optimizer", version="0.3.0")

PACE_MODE_WINDOWS: dict[str, tuple[str, str, int]] = {
    "light": ("10:00", "18:00", 480),
    "standard": ("09:00", "20:00", 660),
    "compact": ("08:00", "21:00", 780),
}

PER_POINT_OVERHEAD_MINUTES = 12
MAX_EXACT_PERMUTATION_SIZE = 8
GEO_ZIGZAG_PENALTY = 0.01

# ── 酒店切换阈值（公里）─────────────────────────────────────
# 两天点位质心距离超过此值，鼓励换酒店；低于此值偏向留同一家
HOTEL_SWITCH_DISTANCE_KM = 15.0
HOTEL_SWITCH_PENALTY_MINUTES = 45


# ═══════════════════════════════════════════════════════════
# Pydantic 模型（与 v0.2.3 保持一致）
# ═══════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════
# 几何工具
# ═══════════════════════════════════════════════════════════

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
    return directed(from_id, from_coord, to_id, to_coord, lookup)


def _geo_zigzag_penalty(coords: list[Coord]) -> float:
    if len(coords) < 3:
        return 0.0
    penalty = 0.0
    for i in range(1, len(coords) - 1):
        prev_c, curr_c, next_c = coords[i - 1], coords[i], coords[i + 1]
        if any(c.lat is None or c.lng is None for c in (prev_c, curr_c, next_c)):
            continue
        dlat_in = curr_c.lat - prev_c.lat  # type: ignore[operator]
        dlat_out = next_c.lat - curr_c.lat  # type: ignore[operator]
        dlng_in = curr_c.lng - prev_c.lng  # type: ignore[operator]
        dlng_out = next_c.lng - curr_c.lng  # type: ignore[operator]
        if dlat_in * dlat_out < 0:
            penalty += GEO_ZIGZAG_PENALTY
        if dlng_in * dlng_out < 0:
            penalty += GEO_ZIGZAG_PENALTY
    return penalty


def _route_geo_penalty(
    start_coord: Coord,
    day_points: list[PointIn],
    ordering: list[int],
    end_coord: Coord,
) -> float:
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


def _centroid(coords: list[Coord]) -> Coord:
    """计算多个坐标的质心（忽略 None）"""
    valid = [(c.lat, c.lng) for c in coords if c.lat is not None and c.lng is not None]
    if not valid:
        return Coord(None, None)
    avg_lat = sum(v[0] for v in valid) / len(valid)
    avg_lng = sum(v[1] for v in valid) / len(valid)
    return Coord(avg_lat, avg_lng)


# ═══════════════════════════════════════════════════════════
# 矩阵完整性审计
# ═══════════════════════════════════════════════════════════

MAX_MISSING_UNDIRECTED_RATIO = 0.0


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
            a, b = unique_ids[i], unique_ids[j]
            if (a, b) in lookup or (b, a) in lookup:
                present_pairs += 1

    if total_pairs == 0:
        return

    missing_ratio = 1.0 - (present_pairs / total_pairs)
    if missing_ratio > MAX_MISSING_UNDIRECTED_RATIO:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Distance matrix incomplete: {present_pairs}/{total_pairs} pairs "
                f"({missing_ratio:.1%} missing)."
            ),
        )


# ═══════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════

def _compute_day_budgets(req: SolveRequest, day_slots: int) -> list[int]:
    pace_start, pace_end, pace_minutes = PACE_MODE_WINDOWS[req.paceMode]
    has_restaurant = any(p.pointType == "restaurant" for p in req.points)
    meal_overhead = 45 if req.mealPolicy == "auto" and not has_restaurant else 0
    daily_budget = pace_minutes - meal_overhead
    if daily_budget < 120:
        raise HTTPException(status_code=400, detail="daily playable minutes too low")

    budgets = [daily_budget] * day_slots
    if day_slots > 0:
        budgets[0] = _first_day_budget(
            req.arrivalDateTime,
            req.airportBufferMinutes,
            pace_start,
            pace_end,
            pace_minutes,
            meal_overhead,
        )
    return budgets


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


# ═══════════════════════════════════════════════════════════
# Stage 1: 纯点位分天（不含酒店）
# ═══════════════════════════════════════════════════════════

@dataclass
class Stage1Result:
    """每天的点位索引列表，以及对应的日历偏移"""
    day_point_indexes: list[list[int]]   # pos → [point_index, ...]
    used_calendar_days: list[int]         # pos → calendar day offset
    solver_status: Literal["OPTIMAL", "FEASIBLE"]


def stage1_assign_points_to_days(
    req: SolveRequest,
    lookup: dict[tuple[str, str], int],
) -> Stage1Result:
    """
    Stage 1: 纯 CP-SAT 地理聚类。
    目标：最少天数，然后最小化每天内点位间通勤代理和。
    完全不涉及酒店——酒店由 Stage 2 独立决定。
    """
    n = len(req.points)
    d = min(n + 1, req.maxDays + 1)

    pace_start, pace_end, pace_minutes = PACE_MODE_WINDOWS[req.paceMode]
    has_restaurant = any(p.pointType == "restaurant" for p in req.points)
    meal_overhead = 45 if req.mealPolicy == "auto" and not has_restaurant else 0
    daily_budget = pace_minutes - meal_overhead
    if daily_budget < 120:
        raise HTTPException(status_code=400, detail="daily playable minutes too low")

    first_budget = _first_day_budget(
        req.arrivalDateTime,
        req.airportBufferMinutes,
        pace_start, pace_end,
        pace_minutes, meal_overhead,
    )

    point_loads = [p.suggestedDurationMinutes + PER_POINT_OVERHEAD_MINUTES for p in req.points]
    if any(load > daily_budget for load in point_loads):
        raise HTTPException(status_code=400, detail="at least one point duration exceeds daily budget")

    model = cp_model.CpModel()

    x = [[model.NewBoolVar(f"x_{i}_{day}") for day in range(d)] for i in range(n)]
    for i in range(n):
        model.Add(sum(x[i]) == 1)

    used = [model.NewBoolVar(f"u_{day}") for day in range(d)]
    day_budgets = [daily_budget] * d
    day_budgets[0] = first_budget

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
    # 天序打包惩罚：鼓励点位靠前安排，减少空洞
    day_pack_penalty = sum(
        (day + 1) * x[i][day] for i in range(n) for day in range(d)
    )

    # 核心：同天点位间通勤代理（点对点，与酒店无关）
    pair_terms: list = []
    if req.objective in ("min_transit", "min_days_then_transit") and n <= 80:
        for i in range(n):
            for j in range(i + 1, n):
                cost = symmetric(
                    req.points[i].id, coord_of(req.points[i]),
                    req.points[j].id, coord_of(req.points[j]),
                    lookup,
                )
                for day in range(d):
                    y = model.NewBoolVar(f"pair_{i}_{j}_{day}")
                    model.Add(y <= x[i][day])
                    model.Add(y <= x[j][day])
                    model.Add(y >= x[i][day] + x[j][day] - 1)
                    pair_terms.append(y * cost)
    transit_proxy = sum(pair_terms)

    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 8
    t1 = max(0.2, req.timeLimitSeconds * 0.45)
    t2 = max(0.2, req.timeLimitSeconds * 0.45)

    lex_first_status: int | None = None

    if req.objective == "min_days":
        solver.parameters.max_time_in_seconds = t1 + t2
        model.Minimize(used_days_expr * 10000 + day_pack_penalty)
        status = solver.Solve(model)
    elif req.objective == "min_transit":
        solver.parameters.max_time_in_seconds = t1 + t2
        model.Minimize(used_days_expr * 10000 + transit_proxy + day_pack_penalty)
        status = solver.Solve(model)
    else:  # min_days_then_transit
        solver.parameters.max_time_in_seconds = t1
        model.Minimize(used_days_expr)
        lex_first_status = solver.Solve(model)
        if lex_first_status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            raise HTTPException(status_code=503, detail="solver failed phase 1a")

        min_days_found = sum(solver.Value(u) for u in used)
        if lex_first_status == cp_model.OPTIMAL:
            model.Add(used_days_expr == min_days_found)
            model.Minimize(transit_proxy + day_pack_penalty)
        else:
            model.Add(used_days_expr <= min_days_found)
            model.Minimize(used_days_expr * 10000 + transit_proxy + day_pack_penalty)

        solver.parameters.max_time_in_seconds = t2
        status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise HTTPException(status_code=503, detail="solver failed to find feasible plan")

    used_day_indexes = [day for day in range(d) if solver.Value(used[day]) == 1]
    if not used_day_indexes:
        raise HTTPException(status_code=503, detail="solver returned empty day set")

    day_position_by_index = {day_idx: pos for pos, day_idx in enumerate(used_day_indexes)}
    day_indexes: list[list[int]] = [[] for _ in range(len(used_day_indexes))]

    for i in range(n):
        assigned_day = next(
            (day for day in range(d) if solver.Value(x[i][day]) == 1), None
        )
        if assigned_day is None:
            raise HTTPException(status_code=503, detail="solver assignment invalid")
        pos = day_position_by_index.get(assigned_day)
        if pos is None:
            raise HTTPException(status_code=503, detail="solver day mapping invalid")
        day_indexes[pos].append(i)

    solver_status: Literal["OPTIMAL", "FEASIBLE"] = "OPTIMAL"
    if status != cp_model.OPTIMAL:
        solver_status = "FEASIBLE"
    if req.objective == "min_days_then_transit" and lex_first_status == cp_model.FEASIBLE:
        solver_status = "FEASIBLE"

    return Stage1Result(
        day_point_indexes=day_indexes,
        used_calendar_days=used_day_indexes,
        solver_status=solver_status,
    )


# ═══════════════════════════════════════════════════════════
# Stage 2: 基于天质心的酒店分配
# ═══════════════════════════════════════════════════════════

@dataclass
class Stage2Result:
    hotel_ids: list[str]


def _day_centroid(point_indexes: list[int], req: SolveRequest) -> Coord:
    """计算某天所有点位的地理质心"""
    coords = [coord_of(req.points[i]) for i in point_indexes]
    return _centroid(coords)


def _hotel_to_day_cost(
    hotel: HotelIn,
    point_indexes: list[int],
    req: SolveRequest,
    lookup: dict[tuple[str, str], int],
) -> float:
    """
    计算酒店对某天点位集合的服务成本。
    使用"酒店到质心距离"作为代理，同时加权实际通勤矩阵数据。
    成本 = sum(directed(hotel→point)) + sum(directed(point→hotel))
    近似实际路线 hotel→p1→...→hotel 中酒店段的双向成本。
    """
    total = 0.0
    h_coord = coord_of(hotel)
    for i in point_indexes:
        p = req.points[i]
        p_coord = coord_of(p)
        total += directed(hotel.id, h_coord, p.id, p_coord, lookup)
        total += directed(p.id, p_coord, hotel.id, h_coord, lookup)
    return total


def stage2_assign_hotels(
    req: SolveRequest,
    s1: Stage1Result,
    lookup: dict[tuple[str, str], int],
) -> Stage2Result:
    """
    Stage 2: 解耦酒店分配。
    策略：
    1. 计算每天所有点位的地理质心
    2. 若 hotelMode == 'single'：找对所有天总成本最低的单一酒店
    3. 若 hotelMode == 'multi'：
       a. 为每天独立选最优酒店（最小化当天路线成本）
       b. 应用连续天切换惩罚（与酒店切换距离成正比）
       c. 用 DP 在"保留同一家" vs "切换到更好的" 之间权衡
    """
    num_days = len(s1.day_point_indexes)
    hotels = req.hotels
    k = len(hotels)

    if num_days == 0 or k == 0:
        raise HTTPException(status_code=503, detail="no days or hotels")

    # 预计算每个 (天, 酒店) 的成本
    day_hotel_cost: list[list[float]] = []
    for pos in range(num_days):
        row: list[float] = []
        for hotel in hotels:
            cost = _hotel_to_day_cost(hotel, s1.day_point_indexes[pos], req, lookup)
            row.append(cost)
        day_hotel_cost.append(row)

    if req.hotelMode == "single":
        # 选总成本最低的单一酒店
        best_h = min(range(k), key=lambda h: sum(day_hotel_cost[pos][h] for pos in range(num_days)))
        return Stage2Result(hotel_ids=[hotels[best_h].id] * num_days)

    # multi 模式：DP
    # state: (pos, last_hotel_idx) → min_total_cost
    INF = float("inf")
    dp: list[list[float]] = [[INF] * k for _ in range(num_days)]
    back: list[list[int]] = [[-1] * k for _ in range(num_days)]

    # 初始化第 0 天
    for h in range(k):
        dp[0][h] = day_hotel_cost[0][h]

    # 酒店切换惩罚（分钟）：来自 req 参数，可由调用方配置
    switch_penalty = float(req.switchPenaltyMinutes)  # 默认 45 分钟

    # 用质心间距离动态决定是否值得换酒店
    # 如果两天质心距离 > HOTEL_SWITCH_DISTANCE_KM，换酒店的惩罚折半（距离远，换酒店合理）
    day_centroids = [
        _day_centroid(s1.day_point_indexes[pos], req)
        for pos in range(num_days)
    ]

    for pos in range(1, num_days):
        centroid_dist = haversine_km(day_centroids[pos - 1], day_centroids[pos])
        # 两天质心距离越远，切换酒店的惩罚越低
        switch_factor = 0.3 if centroid_dist >= HOTEL_SWITCH_DISTANCE_KM else 1.0

        for h in range(k):
            best_prev_cost = INF
            best_prev_h = -1
            for prev_h in range(k):
                if dp[pos - 1][prev_h] >= INF:
                    continue
                is_switch = int(prev_h != h)
                cost = dp[pos - 1][prev_h] + is_switch * switch_penalty * switch_factor
                if cost < best_prev_cost:
                    best_prev_cost = cost
                    best_prev_h = prev_h
            if best_prev_cost < INF:
                dp[pos][h] = best_prev_cost + day_hotel_cost[pos][h]
                back[pos][h] = best_prev_h

    # 回溯最优路径
    last_h = min(range(k), key=lambda h: dp[num_days - 1][h])
    if dp[num_days - 1][last_h] >= INF:
        # 降级：每天选最近酒店
        hotel_ids = [
            hotels[min(range(k), key=lambda h: day_hotel_cost[pos][h])].id
            for pos in range(num_days)
        ]
        return Stage2Result(hotel_ids=hotel_ids)

    path = [-1] * num_days
    path[-1] = last_h
    for pos in range(num_days - 1, 0, -1):
        path[pos - 1] = back[pos][path[pos]]

    return Stage2Result(hotel_ids=[hotels[h].id for h in path])


# ═══════════════════════════════════════════════════════════
# Stage 3: 日内 TSP 排序（与原版相同）
# ═══════════════════════════════════════════════════════════

@dataclass
class DayRoute:
    ordering: list[int]
    cost: int
    effective_cost: float


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
                current_point.id, coord_of(current_point),
                day_points[i].id, coord_of(day_points[i]),
                lookup,
            ),
        )
        ordering.append(next_idx)
        remaining.remove(next_idx)

    cost = directed(start_id, start_coord, day_points[ordering[0]].id, coord_of(day_points[ordering[0]]), lookup)
    for k_idx in range(m - 1):
        a, b = day_points[ordering[k_idx]], day_points[ordering[k_idx + 1]]
        cost += directed(a.id, coord_of(a), b.id, coord_of(b), lookup)
    cost += directed(day_points[ordering[-1]].id, coord_of(day_points[ordering[-1]]), end_hotel_id, end_hotel_coord, lookup)

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
                in_cost = directed(start_id, start_coord, only.id, coord_of(only), lookup)
                out_cost = directed(only.id, coord_of(only), hotel.id, coord_of(hotel), lookup)
                c = in_cost + out_cost
                routes[(start_id, hotel.id)] = DayRoute([0], c, float(c))
        return routes

    # 枚举所有排列，按 (first, last) 分组保留最优内路线
    best_inner: dict[tuple[int, int], tuple[int, list[int]]] = {}
    for perm in itertools.permutations(range(m)):
        order = list(perm)
        inner_cost = sum(
            directed(
                day_points[order[k_idx]].id, coord_of(day_points[order[k_idx]]),
                day_points[order[k_idx + 1]].id, coord_of(day_points[order[k_idx + 1]]),
                lookup,
            )
            for k_idx in range(m - 1)
        )
        key = (order[0], order[-1])
        current = best_inner.get(key)
        if current is None or inner_cost < current[0]:
            best_inner[key] = (inner_cost, order)

    start_to_first: dict[tuple[str, int], int] = {
        (start_id, point_idx): directed(
            start_id, coord_map[start_id],
            day_points[point_idx].id, coord_of(day_points[point_idx]),
            lookup,
        )
        for start_id in start_ids
        for point_idx in range(m)
    }
    last_to_hotel: dict[tuple[int, str], int] = {
        (point_idx, hotel.id): directed(
            day_points[point_idx].id, coord_of(day_points[point_idx]),
            hotel.id, coord_of(hotel), lookup,
        )
        for point_idx in range(m)
        for hotel in hotels
    }

    for start_id in start_ids:
        start_coord = coord_map[start_id]
        for hotel in hotels:
            hotel_coord = coord_of(hotel)
            best_eff: float | None = None
            best_cost: int = 0
            best_perm: list[int] = next(iter(best_inner.values()))[1]

            for (first_idx, last_idx), (inner_cost, perm) in best_inner.items():
                transit_cost = (
                    start_to_first[(start_id, first_idx)]
                    + inner_cost
                    + last_to_hotel[(last_idx, hotel.id)]
                )
                geo_pen = _route_geo_penalty(start_coord, day_points, perm, hotel_coord)
                eff = transit_cost + geo_pen
                if best_eff is None or eff < best_eff:
                    best_eff = eff
                    best_cost = transit_cost
                    best_perm = perm

            routes[(start_id, hotel.id)] = DayRoute(list(best_perm), best_cost, best_eff or float(best_cost))

    return routes


def stage3_order_points(
    req: SolveRequest,
    s1: Stage1Result,
    s2: Stage2Result,
    lookup: dict[tuple[str, str], int],
) -> list[tuple[list[int], int]]:
    """
    Stage 3: 给每天选出最优点位排序。
    返回 [(ordering, travel_minutes), ...] per day
    """
    arrival = _arrival_anchor(req)
    hotel_by_id = {h.id: h for h in req.hotels}
    coord_map: dict[str, Coord] = {h.id: coord_of(h) for h in req.hotels}
    if arrival and arrival.node_id:
        coord_map[arrival.node_id] = arrival.coord

    results: list[tuple[list[int], int]] = []

    for pos, point_indexes in enumerate(s1.day_point_indexes):
        hotel_id = s2.hotel_ids[pos]
        hotel = hotel_by_id[hotel_id]
        day_points = [req.points[i] for i in point_indexes]

        # 确定当天出发地
        if pos == 0:
            if s1.used_calendar_days[0] > 0:
                start_id = hotel_id
                start_coord = coord_of(hotel)
            elif arrival and arrival.node_id:
                start_id = arrival.node_id
                start_coord = arrival.coord
            else:
                start_id = hotel_id
                start_coord = coord_of(hotel)
        else:
            prev_hotel_id = s2.hotel_ids[pos - 1]
            prev_hotel = hotel_by_id[prev_hotel_id]
            start_id = prev_hotel_id
            start_coord = coord_of(prev_hotel)

        if len(day_points) <= MAX_EXACT_PERMUTATION_SIZE:
            routes = _precompute_day_exact(
                day_points,
                [start_id],
                [hotel],
                coord_map,
                lookup,
            )
            route = routes.get((start_id, hotel_id))
            if route is None:
                route = _nn_order(day_points, start_id, start_coord, hotel_id, coord_of(hotel), lookup)
        else:
            route = _nn_order(day_points, start_id, start_coord, hotel_id, coord_of(hotel), lookup)

        results.append((route.ordering, route.cost))

    return results


# ═══════════════════════════════════════════════════════════
# 主求解流程（三段解耦）
# ═══════════════════════════════════════════════════════════

def solve_three_stage(
    req: SolveRequest,
    lookup: dict[tuple[str, str], int],
) -> SolveResponse:
    # Stage 1: 纯点位地理聚类
    s1 = stage1_assign_points_to_days(req, lookup)

    # Stage 2: 基于质心的酒店分配
    s2 = stage2_assign_hotels(req, s1, lookup)

    # Stage 3: 日内最优排序
    orderings = stage3_order_points(req, s1, s2, lookup)

    # 组装响应
    start_date = req.arrivalDateTime.date()
    days: list[DayPlan] = []
    for pos in range(len(s1.day_point_indexes)):
        calendar_index = s1.used_calendar_days[pos]
        point_indexes = s1.day_point_indexes[pos]
        ordering, travel_minutes = orderings[pos]
        point_ids = [req.points[point_indexes[idx]].id for idx in ordering]
        days.append(DayPlan(
            dayNumber=pos + 1,
            date=(start_date + timedelta(days=calendar_index)).isoformat(),
            pointIds=point_ids,
            hotelId=s2.hotel_ids[pos],
        ))

    return SolveResponse(
        tripDays=len(days),
        solverStatus=s1.solver_status,
        objective=req.objective,
        days=days,
        diagnostics={
            "totalTravelMinutes": int(sum(t for _, t in orderings)),
        },
    )


# ═══════════════════════════════════════════════════════════
# HTTP 入口
# ═══════════════════════════════════════════════════════════

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/solve", response_model=SolveResponse)
def solve(req: SolveRequest) -> SolveResponse:
    lookup = build_lookup(req.distanceMatrix.rows)
    audit_matrix_completeness(req, lookup)
    return solve_three_stage(req, lookup)