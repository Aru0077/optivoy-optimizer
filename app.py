from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field as PydField

app = FastAPI(title="Optivoy Optimizer", version="0.8.0")


PACE_MODE_WINDOWS: dict[str, tuple[str, str, int]] = {
    "light": ("10:00", "18:00", 480),
    "standard": ("09:00", "20:00", 660),
    "compact": ("08:00", "21:00", 780),
}

WALK_THRESHOLD_METERS = 1500
WALK_SPEED_M_PER_MIN = 83
MAX_EXACT_PERMUTATION_SIZE = 8
MAX_POINTS_PER_DAY = 8
TRAVEL_OVERHEAD_PER_POINT = 30
LUNCH_OVERHEAD_MINUTES = 45
LUNCH_INSERT_AFTER_MINUTES = 11 * 60 + 30
PER_POINT_OVERHEAD_MINUTES = 12
GEO_ZIGZAG_PENALTY = 8.0


class CoordinateIn(BaseModel):
    latitude: float | None = None
    longitude: float | None = None


class PlanningTimeRangeIn(BaseModel):
    start: str = PydField(pattern=r"^([01]\d|2[0-3]):[0-5]\d$")
    end: str = PydField(pattern=r"^([01]\d|2[0-3]):[0-5]\d$")


class OpeningHoursRuleIn(BaseModel):
    weekday: int = PydField(ge=0, le=6)
    periods: list[PlanningTimeRangeIn] = PydField(min_length=1, max_length=8)


class QueueProfileIn(BaseModel):
    weekdayMinutes: int | None = PydField(default=None, ge=0, le=1440)
    weekendMinutes: int | None = PydField(default=None, ge=0, le=1440)
    holidayMinutes: int | None = PydField(default=None, ge=0, le=1440)


class PointIn(BaseModel):
    id: str = PydField(min_length=1, max_length=64)
    pointType: Literal["spot", "shopping"]
    suggestedDurationMinutes: int = PydField(ge=10, le=720)
    latitude: float | None = None
    longitude: float | None = None
    arrivalAnchor: CoordinateIn | None = None
    departureAnchor: CoordinateIn | None = None
    openingHoursJson: list[OpeningHoursRuleIn] = PydField(default_factory=list)
    specialClosureDates: list[date] = PydField(default_factory=list)
    lastEntryTime: str | None = PydField(default=None, pattern=r"^([01]\d|2[0-3]):[0-5]\d$")
    hasFoodCourt: bool = False
    queueProfileJson: QueueProfileIn | None = None


class HotelIn(BaseModel):
    id: str = PydField(min_length=1, max_length=64)
    latitude: float | None = None
    longitude: float | None = None
    arrivalAnchor: CoordinateIn | None = None
    departureAnchor: CoordinateIn | None = None
    checkInTime: str | None = PydField(default=None, pattern=r"^([01]\d|2[0-3]):[0-5]\d$")
    checkOutTime: str | None = PydField(default=None, pattern=r"^([01]\d|2[0-3]):[0-5]\d$")


class DistanceMatrixRowIn(BaseModel):
    fromPointId: str = PydField(min_length=1, max_length=64)
    toPointId: str = PydField(min_length=1, max_length=64)
    transitMinutes: int = PydField(ge=1, le=1440)
    drivingMinutes: int = PydField(ge=1, le=1440)
    walkingMeters: int = PydField(ge=0, le=200000)
    walkingMinutes: int | None = PydField(default=None, ge=0, le=2880)
    distanceKm: float = PydField(ge=0, le=2000)
    transitSummary: str | None = None


class DistanceMatrixIn(BaseModel):
    rows: list[DistanceMatrixRowIn] = PydField(default_factory=list, max_length=300000)


class SolveRequest(BaseModel):
    city: str | None = PydField(default=None, min_length=1, max_length=120)
    province: str | None = PydField(default=None, min_length=1, max_length=120)
    startDate: date
    points: list[PointIn] = PydField(min_length=1, max_length=200)
    hotels: list[HotelIn] = PydField(min_length=1, max_length=200)
    distanceMatrix: DistanceMatrixIn = PydField(default_factory=DistanceMatrixIn)
    paceMode: Literal["light", "standard", "compact"] = "standard"
    hotelStrategy: Literal["single", "smart"] = "smart"
    mealPolicy: Literal["auto", "off"] = "auto"
    transportPreference: Literal["transit_first", "driving_first", "mixed"] = "mixed"
    maxIntradayDrivingMinutes: int = PydField(default=120, ge=30, le=480)


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
    arrival: Coord
    departure: Coord


@dataclass(frozen=True)
class EdgeChoice:
    minutes: int
    mode: Literal["walking", "transit", "driving"]
    used_fallback: bool
    summary: str | None = None


@dataclass(frozen=True)
class DayWindow:
    calendar_index: int
    date_value: date
    start_minutes: int
    end_minutes: int


@dataclass
class DayDiagnostics:
    travel_minutes: int
    queue_minutes: int
    lunch_break_minutes: int
    lunch_break_before_point_id: str | None
    hotel_switch: bool
    fallback_edges_used: int
    transport_modes: dict[str, int]
    window_wait_minutes: int
    route_effective_cost: float


@dataclass
class EvaluatedRoute:
    ordering: list[int]
    point_ids: list[str]
    travel_minutes: int
    effective_cost: float
    diagnostics: DayDiagnostics
    end_time_minutes: int


@dataclass
class PointPrepared:
    point: PointIn
    arrival_coord: Coord
    departure_coord: Coord


@dataclass
class HotelPrepared:
    hotel: HotelIn
    arrival_coord: Coord
    departure_coord: Coord


@dataclass
class LookupMaps:
    transit: dict[tuple[str, str], int]
    driving: dict[tuple[str, str], int]
    walking_meters: dict[tuple[str, str], int]
    walking_minutes: dict[tuple[str, str], int]
    distance_km: dict[tuple[str, str], float]
    transit_summary: dict[tuple[str, str], str]


def parse_hhmm(value: str) -> int:
    hour, minute = value.split(":", 1)
    return int(hour) * 60 + int(minute)


def haversine_km(a: Coord, b: Coord) -> float:
    if None in (a.lat, a.lng, b.lat, b.lng):
        return 8.0
    r = math.pi / 180.0
    dlat = (b.lat - a.lat) * r  # type: ignore[operator]
    dlng = (b.lng - a.lng) * r  # type: ignore[operator]
    h = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(a.lat * r) * math.cos(b.lat * r) * math.sin(dlng / 2.0) ** 2  # type: ignore[operator]
    )
    return 6371.0 * 2.0 * math.atan2(math.sqrt(h), math.sqrt(max(1e-9, 1.0 - h)))


def fallback_distance_km(a: Coord, b: Coord) -> float:
    return max(0.1, haversine_km(a, b))


def fallback_driving_minutes(a: Coord, b: Coord) -> int:
    return max(8, int(round(fallback_distance_km(a, b) / 25.0 * 60.0 + 8)))


def fallback_transit_minutes(a: Coord, b: Coord) -> int:
    return max(10, int(round(fallback_distance_km(a, b) / 18.0 * 60.0 + 12)))


def fallback_walking_minutes(a: Coord, b: Coord) -> int:
    return max(3, int(round(fallback_distance_km(a, b) / 4.5 * 60.0)))


def fallback_walking_meters(a: Coord, b: Coord) -> int:
    return max(100, int(round(fallback_distance_km(a, b) * 1000)))


def point_arrival_coord(point: PointIn) -> Coord:
    if point.arrivalAnchor and point.arrivalAnchor.latitude is not None and point.arrivalAnchor.longitude is not None:
        return Coord(point.arrivalAnchor.latitude, point.arrivalAnchor.longitude)
    return Coord(point.latitude, point.longitude)


def point_departure_coord(point: PointIn) -> Coord:
    if point.departureAnchor and point.departureAnchor.latitude is not None and point.departureAnchor.longitude is not None:
        return Coord(point.departureAnchor.latitude, point.departureAnchor.longitude)
    return Coord(point.latitude, point.longitude)


def hotel_arrival_coord(hotel: HotelIn) -> Coord:
    if hotel.arrivalAnchor and hotel.arrivalAnchor.latitude is not None and hotel.arrivalAnchor.longitude is not None:
        return Coord(hotel.arrivalAnchor.latitude, hotel.arrivalAnchor.longitude)
    return Coord(hotel.latitude, hotel.longitude)


def hotel_departure_coord(hotel: HotelIn) -> Coord:
    if hotel.departureAnchor and hotel.departureAnchor.latitude is not None and hotel.departureAnchor.longitude is not None:
        return Coord(hotel.departureAnchor.latitude, hotel.departureAnchor.longitude)
    return Coord(hotel.latitude, hotel.longitude)


def centroid(coords: list[Coord]) -> Coord:
    valid = [(c.lat, c.lng) for c in coords if c.lat is not None and c.lng is not None]
    if not valid:
        return Coord(None, None)
    return Coord(
        sum(value[0] for value in valid) / len(valid),
        sum(value[1] for value in valid) / len(valid),
    )


def geo_route_penalty(coords: list[Coord]) -> float:
    if len(coords) < 3:
        return 0.0
    penalty = 0.0
    for index in range(1, len(coords) - 1):
        prev_c, curr_c, next_c = coords[index - 1], coords[index], coords[index + 1]
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


def is_weekend(day_value: date) -> bool:
    return day_value.weekday() >= 5


def queue_minutes_for_day(point: PointIn, day_value: date) -> int:
    profile = point.queueProfileJson
    if profile is None:
        return 0
    if is_weekend(day_value) and profile.weekendMinutes is not None:
        return profile.weekendMinutes
    if profile.weekdayMinutes is not None:
        return profile.weekdayMinutes
    if profile.holidayMinutes is not None:
        return profile.holidayMinutes
    return 0


def weekday_periods(point: PointIn, day_value: date) -> list[tuple[int, int]]:
    weekday = day_value.weekday()
    periods: list[tuple[int, int]] = []
    for rule in point.openingHoursJson:
        if rule.weekday != weekday:
            continue
        for period in rule.periods:
            periods.append((parse_hhmm(period.start), parse_hhmm(period.end)))
    return sorted(periods)


def day_is_closed(point: PointIn, day_value: date) -> bool:
    if day_value in point.specialClosureDates:
        return True
    return len(weekday_periods(point, day_value)) == 0


def validate_request(req: SolveRequest) -> None:
    if req.transportPreference not in ("transit_first", "driving_first", "mixed"):
        raise HTTPException(status_code=400, detail="unsupported transportPreference")
    if req.hotelStrategy == "single" and len(req.hotels) == 0:
        raise HTTPException(status_code=400, detail="single hotel mode requires hotels")


def filter_hotels(req: SolveRequest) -> list[HotelPrepared]:
    prepared = [
        HotelPrepared(
            hotel=item,
            arrival_coord=hotel_arrival_coord(item),
            departure_coord=hotel_departure_coord(item),
        )
        for item in req.hotels
    ]
    if not prepared:
        raise HTTPException(status_code=400, detail="no eligible hotels provided")
    return prepared


def prepare_points(req: SolveRequest) -> list[PointPrepared]:
    prepared = [
        PointPrepared(
            point=item,
            arrival_coord=point_arrival_coord(item),
            departure_coord=point_departure_coord(item),
        )
        for item in req.points
    ]
    for item in prepared:
        if item.arrival_coord.lat is None or item.arrival_coord.lng is None:
            raise HTTPException(status_code=400, detail=f"point {item.point.id} missing arrival coordinate")
        if item.departure_coord.lat is None or item.departure_coord.lng is None:
            raise HTTPException(status_code=400, detail=f"point {item.point.id} missing departure coordinate")
    return prepared

def build_lookup_maps(rows: list[DistanceMatrixRowIn]) -> LookupMaps:
    transit: dict[tuple[str, str], int] = {}
    driving: dict[tuple[str, str], int] = {}
    walking_meters: dict[tuple[str, str], int] = {}
    walking_minutes: dict[tuple[str, str], int] = {}
    distance_km: dict[tuple[str, str], float] = {}
    transit_summary: dict[tuple[str, str], str] = {}

    for row in rows:
        key = (row.fromPointId, row.toPointId)
        if key not in transit or row.transitMinutes < transit[key]:
            transit[key] = row.transitMinutes
        if key not in driving or row.drivingMinutes < driving[key]:
            driving[key] = row.drivingMinutes
        if key not in walking_meters or row.walkingMeters < walking_meters[key]:
            walking_meters[key] = row.walkingMeters
        row_walking_minutes = row.walkingMinutes or max(1, int(round(row.walkingMeters / WALK_SPEED_M_PER_MIN)))
        if key not in walking_minutes or row_walking_minutes < walking_minutes[key]:
            walking_minutes[key] = row_walking_minutes
        if key not in distance_km or row.distanceKm < distance_km[key]:
            distance_km[key] = row.distanceKm
        if row.transitSummary:
            transit_summary[key] = row.transitSummary

    return LookupMaps(
        transit=transit,
        driving=driving,
        walking_meters=walking_meters,
        walking_minutes=walking_minutes,
        distance_km=distance_km,
        transit_summary=transit_summary,
    )


def lookup_mode_minutes(
    mode: Literal["walking", "transit", "driving"],
    from_id: str | None,
    from_coord: Coord,
    to_id: str | None,
    to_coord: Coord,
    lookups: LookupMaps,
) -> tuple[int, bool, str | None]:
    if from_id and to_id:
        key = (from_id, to_id)
        reverse = (to_id, from_id)
        if mode == "walking":
            value = lookups.walking_minutes.get(key) or lookups.walking_minutes.get(reverse)
            if value is not None:
                return value, False, None
        elif mode == "transit":
            value = lookups.transit.get(key) or lookups.transit.get(reverse)
            if value is not None:
                return value, False, lookups.transit_summary.get(key) or lookups.transit_summary.get(reverse)
        else:
            value = lookups.driving.get(key) or lookups.driving.get(reverse)
            if value is not None:
                return value, False, None

    if mode == "walking":
        return fallback_walking_minutes(from_coord, to_coord), True, None
    if mode == "transit":
        return fallback_transit_minutes(from_coord, to_coord), True, None
    return fallback_driving_minutes(from_coord, to_coord), True, None


def lookup_walking_distance(
    from_id: str | None,
    from_coord: Coord,
    to_id: str | None,
    to_coord: Coord,
    lookups: LookupMaps,
) -> int:
    if from_id and to_id:
        return (
            lookups.walking_meters.get((from_id, to_id))
            or lookups.walking_meters.get((to_id, from_id))
            or fallback_walking_meters(from_coord, to_coord)
        )
    return fallback_walking_meters(from_coord, to_coord)


def lookup_travel_choice(
    from_id: str | None,
    from_coord: Coord,
    to_id: str | None,
    to_coord: Coord,
    req: SolveRequest,
    lookups: LookupMaps,
) -> EdgeChoice:
    walking_distance = lookup_walking_distance(from_id, from_coord, to_id, to_coord, lookups)
    walking_minutes, walking_fallback, _ = lookup_mode_minutes("walking", from_id, from_coord, to_id, to_coord, lookups)
    transit_minutes, transit_fallback, transit_summary = lookup_mode_minutes("transit", from_id, from_coord, to_id, to_coord, lookups)
    driving_minutes, driving_fallback, _ = lookup_mode_minutes("driving", from_id, from_coord, to_id, to_coord, lookups)

    if 0 < walking_distance <= WALK_THRESHOLD_METERS:
        return EdgeChoice(minutes=walking_minutes, mode="walking", used_fallback=walking_fallback)
    if req.transportPreference == "transit_first":
        return EdgeChoice(minutes=transit_minutes, mode="transit", used_fallback=transit_fallback, summary=transit_summary)
    if req.transportPreference == "driving_first":
        return EdgeChoice(minutes=driving_minutes, mode="driving", used_fallback=driving_fallback)
    if transit_minutes <= driving_minutes:
        return EdgeChoice(minutes=transit_minutes, mode="transit", used_fallback=transit_fallback, summary=transit_summary)
    return EdgeChoice(minutes=driving_minutes, mode="driving", used_fallback=driving_fallback)


def directed_driving_minutes(
    from_id: str | None,
    from_coord: Coord,
    to_id: str | None,
    to_coord: Coord,
    lookups: LookupMaps,
) -> int:
    minutes, _, _ = lookup_mode_minutes("driving", from_id, from_coord, to_id, to_coord, lookups)
    return minutes


def build_day_window(req: SolveRequest, calendar_index: int) -> DayWindow:
    pace_start, pace_end, _ = PACE_MODE_WINDOWS[req.paceMode]
    return DayWindow(
        calendar_index=calendar_index,
        date_value=req.startDate + timedelta(days=calendar_index),
        start_minutes=parse_hhmm(pace_start),
        end_minutes=parse_hhmm(pace_end),
    )


def point_day_load(point: PointIn, day_value: date) -> int:
    return point.suggestedDurationMinutes + PER_POINT_OVERHEAD_MINUTES + queue_minutes_for_day(point, day_value)


def point_can_be_on_day(point: PointIn, window: DayWindow) -> bool:
    if window.end_minutes <= window.start_minutes:
        return False
    if day_is_closed(point, window.date_value):
        return False
    if point.pointType == "spot" and not point.lastEntryTime:
        return False
    # Check lastEntryTime: if day starts at or after lastEntry, this spot can never be entered
    if point.lastEntryTime:
        last_entry = parse_hhmm(point.lastEntryTime)
        if window.start_minutes >= last_entry:
            return False
    # Check that the point can actually complete within some opening period on this day
    load = point_day_load(point, window.date_value)
    periods = weekday_periods(point, window.date_value)
    if periods:
        last_entry_min = parse_hhmm(point.lastEntryTime) if point.lastEntryTime else None
        for period_start, period_end in periods:
            earliest_start = max(window.start_minutes, period_start)
            if last_entry_min is not None and earliest_start > last_entry_min:
                continue
            if earliest_start + load <= min(period_end, window.end_minutes):
                return True
        return False
    return load <= (window.end_minutes - window.start_minutes)


def effective_day_capacity(req: SolveRequest, n_points: int, has_food: bool) -> int:
    base = PACE_MODE_WINDOWS[req.paceMode][2]
    if req.mealPolicy == "auto" and not has_food:
        base -= LUNCH_OVERHEAD_MINUTES
    base -= (n_points + 1) * TRAVEL_OVERHEAD_PER_POINT
    return max(120, base)


def cluster_has_food(cluster: list[int], points: list[PointPrepared]) -> bool:
    if any(points[index].point.hasFoodCourt for index in cluster):
        return True
    return all(points[index].point.suggestedDurationMinutes >= 240 for index in cluster)


def cluster_weight(
    from_index: int,
    to_index: int,
    points: list[PointPrepared],
    lookups: LookupMaps,
) -> float:
    from_point = points[from_index]
    to_point = points[to_index]
    walking_distance = lookup_walking_distance(
        from_point.point.id,
        from_point.departure_coord,
        to_point.point.id,
        to_point.arrival_coord,
        lookups,
    )
    if 0 < walking_distance <= WALK_THRESHOLD_METERS:
        return float(walking_distance)
    return float(
        directed_driving_minutes(
            from_point.point.id,
            from_point.departure_coord,
            to_point.point.id,
            to_point.arrival_coord,
            lookups,
        )
    )


def pick_seed(candidates: list[int], points: list[PointPrepared], lookups: LookupMaps) -> int:
    best_index = candidates[0]
    best_cost = float("inf")
    for index in candidates:
        total = 0
        for other in candidates:
            if other == index:
                continue
            total += cluster_weight(index, other, points, lookups)
        if total < best_cost:
            best_cost = total
            best_index = index
    return best_index


def pick_nearest_to_cluster(
    candidates: list[int],
    cluster: list[int],
    points: list[PointPrepared],
    lookups: LookupMaps,
) -> int:
    best_index = candidates[0]
    best_cost = float("inf")
    for index in candidates:
        total = 0.0
        for cluster_index in cluster:
            total += cluster_weight(cluster_index, index, points, lookups)
        avg = total / max(1, len(cluster))
        if avg < best_cost:
            best_cost = avg
            best_index = index
    return best_index


def cluster_points_to_days(
    req: SolveRequest,
    points: list[PointPrepared],
    lookups: LookupMaps,
) -> list[tuple[DayWindow, list[int]]]:
    unassigned = list(range(len(points)))
    day_clusters: list[tuple[DayWindow, list[int]]] = []
    calendar_index = 0
    max_calendar_span = max(366, len(points) * 30)

    while unassigned:
        if calendar_index >= max_calendar_span:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "POINTS_CANNOT_BE_SCHEDULED",
                    "message": "remaining points cannot be scheduled within a reasonable calendar span",
                    "remainingPointIds": [points[index].point.id for index in unassigned],
                },
            )

        window = build_day_window(req, calendar_index)
        available_today = [index for index in unassigned if point_can_be_on_day(points[index].point, window)]
        if not available_today:
            calendar_index += 1
            continue

        seed = pick_seed(available_today, points, lookups)
        cluster = [seed]
        unassigned.remove(seed)
        cluster_load = point_day_load(points[seed].point, window.date_value)

        while True:
            candidates = [index for index in unassigned if point_can_be_on_day(points[index].point, window)]
            if not candidates:
                break
            if len(cluster) >= MAX_POINTS_PER_DAY:
                break
            next_index = pick_nearest_to_cluster(candidates, cluster, points, lookups)
            has_food = cluster_has_food(cluster + [next_index], points)
            capacity = effective_day_capacity(req, len(cluster) + 1, has_food)
            next_load = point_day_load(points[next_index].point, window.date_value)
            if cluster_load + next_load > capacity:
                break
            cluster.append(next_index)
            unassigned.remove(next_index)
            cluster_load += next_load

        if cluster:
            day_clusters.append((window, cluster))
        calendar_index += 1

    return day_clusters


def find_nearest_hotel(
    point_indexes: list[int],
    points: list[PointPrepared],
    hotels: list[HotelPrepared],
    lookups: LookupMaps,
) -> int:
    best_hotel = 0
    best_cost = float("inf")
    for hotel_index, hotel in enumerate(hotels):
        total = 0.0
        for point_index in point_indexes:
            to_point = directed_driving_minutes(
                hotel.hotel.id,
                hotel.departure_coord,
                points[point_index].point.id,
                points[point_index].arrival_coord,
                lookups,
            )
            from_point = directed_driving_minutes(
                points[point_index].point.id,
                points[point_index].departure_coord,
                hotel.hotel.id,
                hotel.arrival_coord,
                lookups,
            )
            total += to_point + from_point
        avg = total / max(1, len(point_indexes))
        if avg < best_cost:
            best_cost = avg
            best_hotel = hotel_index
    return best_hotel


def average_one_way_driving_minutes(
    hotel_index: int,
    point_indexes: list[int],
    points: list[PointPrepared],
    hotels: list[HotelPrepared],
    lookups: LookupMaps,
) -> float:
    hotel = hotels[hotel_index]
    total = 0.0
    for point_index in point_indexes:
        total += directed_driving_minutes(
            hotel.hotel.id,
            hotel.departure_coord,
            points[point_index].point.id,
            points[point_index].arrival_coord,
            lookups,
        )
    return total / max(1, len(point_indexes))


def assign_hotel_per_day(
    day_clusters: list[tuple[DayWindow, list[int]]],
    points: list[PointPrepared],
    hotels: list[HotelPrepared],
    lookups: LookupMaps,
    req: SolveRequest,
) -> list[int]:
    clusters_only = [cluster for _, cluster in day_clusters]
    if req.hotelStrategy == "single":
        all_point_indexes = [index for cluster in clusters_only for index in cluster]
        best = find_nearest_hotel(all_point_indexes, points, hotels, lookups)
        return [best] * len(clusters_only)

    if not clusters_only:
        return []

    all_point_indexes = [index for cluster in clusters_only for index in cluster]
    current_hotel = find_nearest_hotel(all_point_indexes, points, hotels, lookups)
    assignments: list[int] = []

    for cluster in clusters_only:
        best_for_day = find_nearest_hotel(cluster, points, hotels, lookups)
        current_avg = average_one_way_driving_minutes(current_hotel, cluster, points, hotels, lookups)
        best_avg = average_one_way_driving_minutes(best_for_day, cluster, points, hotels, lookups)
        if current_avg > 40.0 and best_avg + 20.0 < current_avg:
            current_hotel = best_for_day
        assignments.append(current_hotel)

    return assignments


def opening_period_for_arrival(point: PointIn, day_value: date, arrival_minutes: int) -> tuple[int, int] | None:
    periods = weekday_periods(point, day_value)
    last_entry = parse_hhmm(point.lastEntryTime) if point.lastEntryTime else None
    for period_start, period_end in periods:
        start_candidate = max(arrival_minutes, period_start)
        if last_entry is not None and start_candidate > last_entry:
            continue
        return start_candidate, period_end
    return None


def extract_http_detail_message(detail: object) -> str:
    if isinstance(detail, dict):
        for key in ("message", "reason", "detail"):
            value = detail.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return str(detail)
    if isinstance(detail, str):
        return detail
    return str(detail)


def build_no_feasible_ordering_detail(route_points: list[PointPrepared], failure_messages: list[str]) -> dict[str, object]:
    counts: dict[str, int] = {}
    for message in failure_messages:
        counts[message] = counts.get(message, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return {
        "code": "NO_FEASIBLE_ORDERING_FOR_DAY",
        "message": "no feasible ordering for day",
        "reason": ranked[0][0] if ranked else "unknown ordering failure",
        "orderingAttempts": len(failure_messages),
        "pointIds": [item.point.id for item in route_points],
        "reasonSummary": [{"reason": reason, "count": count} for reason, count in ranked[:5]],
    }


def evaluate_order(
    req: SolveRequest,
    points: list[PointPrepared],
    hotels: list[HotelPrepared],
    day_window: DayWindow,
    hotel_index: int,
    day_point_indexes: list[int],
    ordering: list[int],
    lookups: LookupMaps,
    start_anchor: Anchor,
    previous_hotel_index: int | None,
    is_last_day: bool,
) -> EvaluatedRoute:
    hotel = hotels[hotel_index]
    route_points = [points[day_point_indexes[index]] for index in ordering]
    day_end_limit = day_window.end_minutes

    current_time = day_window.start_minutes
    current_id = start_anchor.node_id
    current_coord = start_anchor.departure
    transport_modes = {"walking": 0, "transit": 0, "driving": 0}
    queue_total = 0
    travel_total = 0
    wait_total = 0
    fallback_edges = 0
    lunch_inserted = False
    lunch_break_minutes = 0
    lunch_break_before_point_id: str | None = None
    day_has_food = any(route_point.point.hasFoodCourt for route_point in route_points) or all(
        route_point.point.suggestedDurationMinutes >= 240 for route_point in route_points
    )

    for route_point in route_points:
        point = route_point.point
        choice = lookup_travel_choice(current_id, current_coord, point.id, route_point.arrival_coord, req, lookups)

        transport_modes[choice.mode] += 1
        travel_total += choice.minutes
        fallback_edges += int(choice.used_fallback)
        current_time += choice.minutes

        if (
            not lunch_inserted
            and req.mealPolicy == "auto"
            and not day_has_food
            and current_time >= LUNCH_INSERT_AFTER_MINUTES
        ):
            lunch_end = current_time + LUNCH_OVERHEAD_MINUTES
            if lunch_end > day_end_limit:
                raise HTTPException(status_code=400, detail="lunch break exceeds day window")
            current_time = lunch_end
            lunch_inserted = True
            lunch_break_minutes = LUNCH_OVERHEAD_MINUTES
            lunch_break_before_point_id = point.id

        if day_is_closed(point, day_window.date_value):
            raise HTTPException(status_code=400, detail=f"point {point.id} closed on scheduled day")

        open_period = opening_period_for_arrival(point, day_window.date_value, current_time)
        if open_period is None:
            raise HTTPException(status_code=400, detail=f"point {point.id} has no feasible time window")

        service_start, service_window_end = open_period
        wait_total += max(0, service_start - current_time)
        current_time = service_start

        queue_minutes = queue_minutes_for_day(point, day_window.date_value)
        queue_total += queue_minutes
        service_end = current_time + point.suggestedDurationMinutes + queue_minutes
        if service_end > service_window_end or service_end > day_end_limit:
            raise HTTPException(status_code=400, detail=f"point {point.id} exceeds open window")

        current_time = service_end
        current_id = point.id
        current_coord = route_point.departure_coord

    hotel_choice = lookup_travel_choice(current_id, current_coord, hotel.hotel.id, hotel.arrival_coord, req, lookups)
    transport_modes[hotel_choice.mode] += 1
    travel_total += hotel_choice.minutes
    fallback_edges += int(hotel_choice.used_fallback)
    current_time += hotel_choice.minutes

    if previous_hotel_index is not None and previous_hotel_index != hotel_index and hotel.hotel.checkInTime is not None:
        check_in_minutes = parse_hhmm(hotel.hotel.checkInTime)
        if current_time < check_in_minutes:
            wait_total += check_in_minutes - current_time
            current_time = check_in_minutes

    if current_time > day_end_limit:
        raise HTTPException(status_code=400, detail="day route exceeds day window")

    route_coords = [start_anchor.departure] + [item.arrival_coord for item in route_points] + [hotel.arrival_coord]
    effective_cost = float(travel_total) + geo_route_penalty(route_coords) + float(wait_total) * 0.2

    return EvaluatedRoute(
        ordering=ordering,
        point_ids=[item.point.id for item in route_points],
        travel_minutes=travel_total,
        effective_cost=effective_cost,
        diagnostics=DayDiagnostics(
            travel_minutes=travel_total,
            queue_minutes=queue_total,
            lunch_break_minutes=lunch_break_minutes,
            lunch_break_before_point_id=lunch_break_before_point_id,
            hotel_switch=previous_hotel_index is not None and previous_hotel_index != hotel_index,
            fallback_edges_used=fallback_edges,
            transport_modes=transport_modes,
            window_wait_minutes=wait_total,
            route_effective_cost=effective_cost,
        ),
        end_time_minutes=current_time,
    )


def solve_day_route(
    req: SolveRequest,
    points: list[PointPrepared],
    hotels: list[HotelPrepared],
    day_window: DayWindow,
    hotel_index: int,
    day_point_indexes: list[int],
    start_anchor: Anchor,
    previous_hotel_index: int | None,
    is_last_day: bool,
    lookups: LookupMaps,
) -> EvaluatedRoute:
    if not day_point_indexes:
        return evaluate_order(
            req,
            points,
            hotels,
            day_window,
            hotel_index,
            day_point_indexes,
            [],
            lookups,
            start_anchor,
            previous_hotel_index,
            is_last_day,
        )

    best: EvaluatedRoute | None = None
    failures: list[str] = []
    route_points = [points[day_point_indexes[idx]] for idx in range(len(day_point_indexes))]
    for ordering in itertools.permutations(range(len(day_point_indexes))):
        try:
            candidate = evaluate_order(
                req,
                points,
                hotels,
                day_window,
                hotel_index,
                day_point_indexes,
                list(ordering),
                lookups,
                start_anchor,
                previous_hotel_index,
                is_last_day,
            )
        except HTTPException as exc:
            failures.append(extract_http_detail_message(exc.detail))
            continue
        if best is None or candidate.effective_cost < best.effective_cost:
            best = candidate
    if best is None:
        raise HTTPException(status_code=400, detail=build_no_feasible_ordering_detail(route_points, failures))
    return best


def build_start_anchor(
    hotels: list[HotelPrepared],
    hotel_index: int,
    previous_hotel_index: int | None,
) -> Anchor:
    if previous_hotel_index is None:
        hotel = hotels[hotel_index]
        return Anchor(hotel.hotel.id, hotel.departure_coord, hotel.departure_coord)
    previous_hotel = hotels[previous_hotel_index]
    return Anchor(previous_hotel.hotel.id, previous_hotel.departure_coord, previous_hotel.departure_coord)


def build_solver_diagnostics(
    req: SolveRequest,
    day_clusters: list[tuple[DayWindow, list[int]]],
    hotel_indexes: list[int],
    points: list[PointPrepared],
    hotels: list[HotelPrepared],
    day_routes: list[EvaluatedRoute],
) -> dict[str, object]:
    total_queue = sum(route.diagnostics.queue_minutes for route in day_routes)
    total_travel = sum(route.diagnostics.travel_minutes for route in day_routes)
    total_lunch_break = sum(route.diagnostics.lunch_break_minutes for route in day_routes)
    total_fallback = sum(route.diagnostics.fallback_edges_used for route in day_routes)
    hotel_switches = sum(1 for route in day_routes if route.diagnostics.hotel_switch)

    return {
        "transportPreference": req.transportPreference,
        "tripWindowDays": [window.date_value.isoformat() for window, _ in day_clusters],
        "hotelSwitches": hotel_switches,
        "totalTravelMinutes": total_travel,
        "totalQueueMinutes": total_queue,
        "totalLunchBreakMinutes": total_lunch_break,
        "fallbackEdgesUsed": total_fallback,
        "days": [
            {
                "dayNumber": index + 1,
                "date": window.date_value.isoformat(),
                "pointIds": [points[point_index].point.id for point_index in cluster],
                "hotelId": hotels[hotel_indexes[index]].hotel.id,
                "travelMinutes": route.diagnostics.travel_minutes,
                "queueMinutes": route.diagnostics.queue_minutes,
                "lunchBreakMinutes": route.diagnostics.lunch_break_minutes,
                "lunchBreakBeforePointId": route.diagnostics.lunch_break_before_point_id,
                "hotelSwitch": route.diagnostics.hotel_switch,
                "fallbackEdgesUsed": route.diagnostics.fallback_edges_used,
                "transportModes": route.diagnostics.transport_modes,
                "windowWaitMinutes": route.diagnostics.window_wait_minutes,
            }
            for index, ((window, cluster), route) in enumerate(zip(day_clusters, day_routes))
        ],
    }


def solve_itinerary(req: SolveRequest) -> SolveResponse:
    validate_request(req)
    points = prepare_points(req)
    hotels = filter_hotels(req)
    lookups = build_lookup_maps(req.distanceMatrix.rows)
    day_clusters = cluster_points_to_days(req, points, lookups)
    hotel_per_day = assign_hotel_per_day(day_clusters, points, hotels, lookups, req)

    days: list[DayPlan] = []
    routes: list[EvaluatedRoute] = []

    for day_position, (window, cluster) in enumerate(day_clusters):
        hotel_index = hotel_per_day[day_position]
        previous_hotel_index = hotel_per_day[day_position - 1] if day_position > 0 else None
        start = build_start_anchor(hotels, hotel_index, previous_hotel_index)
        is_last = day_position == len(day_clusters) - 1

        try:
            route = solve_day_route(
                req,
                points,
                hotels,
                window,
                hotel_index,
                cluster,
                start,
                previous_hotel_index,
                is_last,
                lookups,
            )
        except HTTPException as exc:
            if exc.status_code != 400:
                raise
            req_no_meal = req.model_copy(update={"mealPolicy": "off"})
            try:
                route = solve_day_route(
                    req_no_meal,
                    points,
                    hotels,
                    window,
                    hotel_index,
                    cluster,
                    start,
                    previous_hotel_index,
                    is_last,
                    lookups,
                )
            except HTTPException:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "code": "NO_FEASIBLE_DAY_ROUTE",
                        "message": "no feasible itinerary for scheduled day",
                        "dayNumber": day_position + 1,
                        "date": window.date_value.isoformat(),
                        "pointIds": [points[index].point.id for index in cluster],
                        "hotelId": hotels[hotel_index].hotel.id,
                        "reason": extract_http_detail_message(exc.detail),
                        "optimizerDetail": exc.detail,
                    },
                ) from exc

        routes.append(route)
        days.append(
            DayPlan(
                dayNumber=day_position + 1,
                date=window.date_value.isoformat(),
                pointIds=route.point_ids,
                hotelId=hotels[hotel_index].hotel.id,
            )
        )

    return SolveResponse(
        tripDays=len(days),
        solverStatus="FEASIBLE",
        objective="min_days_then_transport",
        days=days,
        diagnostics=build_solver_diagnostics(req, day_clusters, hotel_per_day, points, hotels, routes),
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": app.version}


@app.post("/solve", response_model=SolveResponse)
def solve(req: SolveRequest) -> SolveResponse:
    return solve_itinerary(req)
