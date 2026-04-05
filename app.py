from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Literal

import numpy as np

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field as PydField

try:
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2

    ORTOOLS_AVAILABLE = True
except ImportError:  # pragma: no cover - deployment env should provide ortools
    pywrapcp = None  # type: ignore[assignment]
    routing_enums_pb2 = None  # type: ignore[assignment]
    ORTOOLS_AVAILABLE = False

app = FastAPI(title="Optivoy Optimizer", version="0.9.0")


PACE_MODE_WINDOWS: dict[str, tuple[str, str, int]] = {
    "light": ("10:00", "18:00", 480),
    "standard": ("09:00", "20:00", 660),
    "compact": ("08:00", "21:00", 780),
}

WALK_THRESHOLD_METERS = 1500
WALK_SPEED_M_PER_MIN = 83
MAX_EXACT_PERMUTATION_SIZE = 8
MAX_POINTS_PER_DAY = 8
LUNCH_OVERHEAD_MINUTES = 45
LUNCH_INSERT_AFTER_MINUTES = 11 * 60 + 30
PER_POINT_OVERHEAD_MINUTES = 12
GEO_ZIGZAG_PENALTY = 8.0
HOTEL_SWITCH_TRIGGER_MINUTES = 60.0
HOTEL_SWITCH_IMPROVEMENT_MINUTES = 30.0
OPTIMIZATION_ROUNDS = 2


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
    staminaFactor: float = PydField(default=1.0, ge=0.5, le=3.0)
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


@dataclass
class MatrixStore:
    node_ids: list[str]
    id_to_idx: dict[str, int]
    driving: np.ndarray
    transit: np.ndarray
    walking_meters: np.ndarray
    walking_minutes: np.ndarray
    is_fallback: np.ndarray


@dataclass(frozen=True)
class CityProfile:
    walk_threshold_meters: int
    hotel_switch_trigger_min: float
    hotel_switch_improve_min: float
    max_points_per_day: int
    stamina_factor: float
    day_capacity_buffer: float


@dataclass(frozen=True)
class IterationFeedback:
    hotel_per_day: list[int]
    point_day_by_index: dict[int, int]
    point_order_by_index: dict[int, int]


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


def build_matrix_store(
    points: list[PointPrepared],
    hotels: list[HotelPrepared],
    rows: list[DistanceMatrixRowIn],
) -> MatrixStore:
    node_ids: list[str] = []
    id_to_coord: dict[str, tuple[float, float]] = {}

    for point in points:
        node_ids.append(point.point.id)
        id_to_coord[point.point.id] = (
            point.departure_coord.lat or 0.0,
            point.departure_coord.lng or 0.0,
        )
    for hotel in hotels:
        node_ids.append(hotel.hotel.id)
        id_to_coord[hotel.hotel.id] = (
            hotel.departure_coord.lat or 0.0,
            hotel.departure_coord.lng or 0.0,
        )

    node_ids = list(dict.fromkeys(node_ids))
    id_to_idx = {node_id: index for index, node_id in enumerate(node_ids)}
    node_count = len(node_ids)
    is_fallback = np.ones((node_count, node_count), dtype=bool)

    coords = np.array([id_to_coord[node_id] for node_id in node_ids], dtype=np.float64)
    lats = np.radians(coords[:, 0:1])
    lngs = np.radians(coords[:, 1:2])
    dlat = lats - lats.T
    dlng = lngs - lngs.T
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lats) * np.cos(lats.T) * np.sin(dlng / 2.0) ** 2
    dist_km = 6371.0 * 2.0 * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(1e-9, 1.0 - a)))

    driving = np.maximum(8, np.rint(dist_km / 25.0 * 60.0 + 8.0).astype(np.int32))
    transit = np.maximum(10, np.rint(dist_km / 18.0 * 60.0 + 12.0).astype(np.int32))
    walking_meters = np.maximum(100, np.rint(dist_km * 1000.0).astype(np.int32))
    walking_minutes = np.maximum(3, np.rint(dist_km / 4.5 * 60.0).astype(np.int32))

    for row in rows:
        if row.fromPointId not in id_to_idx or row.toPointId not in id_to_idx:
            continue
        from_index = id_to_idx[row.fromPointId]
        to_index = id_to_idx[row.toPointId]
        driving[from_index, to_index] = row.drivingMinutes
        transit[from_index, to_index] = row.transitMinutes
        walking_meters[from_index, to_index] = row.walkingMeters
        walking_minutes[from_index, to_index] = row.walkingMinutes or max(
            1,
            int(round(row.walkingMeters / WALK_SPEED_M_PER_MIN)),
        )
        is_fallback[from_index, to_index] = False

    np.fill_diagonal(driving, 0)
    np.fill_diagonal(transit, 0)
    np.fill_diagonal(walking_meters, 0)
    np.fill_diagonal(walking_minutes, 0)
    np.fill_diagonal(is_fallback, False)

    return MatrixStore(
        node_ids=node_ids,
        id_to_idx=id_to_idx,
        driving=driving,
        transit=transit,
        walking_meters=walking_meters,
        walking_minutes=walking_minutes,
        is_fallback=is_fallback,
    )


def detect_city_profile(
    city: str | None,
    points: list[PointPrepared],
    matrix: MatrixStore,
) -> CityProfile:
    city_name = (city or "").strip()
    if city_name in CITY_PROFILES:
        return CITY_PROFILES[city_name]
    for key in CITY_PROFILES:
        if key == "default":
            continue
        if city_name and (key in city_name or city_name in key):
            return CITY_PROFILES[key]

    if len(points) < 3:
        return CITY_PROFILES["default"]

    point_indexes = np.array([matrix.id_to_idx[item.point.id] for item in points], dtype=np.int32)
    sub_driving = matrix.driving[np.ix_(point_indexes, point_indexes)]
    mean_driving = float(sub_driving[sub_driving > 0].mean()) if (sub_driving > 0).any() else 30.0
    sub_walking = matrix.walking_meters[np.ix_(point_indexes, point_indexes)]
    mean_walking = float(sub_walking[sub_walking > 0].mean()) if (sub_walking > 0).any() else 2000.0

    if mean_driving > 60.0:
        return CityProfile(1500, 70.0, 35.0, 8, 1.0, 0.85)
    if mean_walking < 600.0:
        return CityProfile(600, 25.0, 10.0, 6, 1.1, 0.85)
    if mean_driving < 20.0:
        return CityProfile(1200, 30.0, 15.0, 7, 1.0, 0.90)
    return CITY_PROFILES["default"]

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


CITY_PROFILES: dict[str, CityProfile] = {
    "张家界": CityProfile(300, 20.0, 10.0, 4, 1.4, 0.80),
    "丽江": CityProfile(800, 30.0, 15.0, 6, 1.1, 0.85),
    "三亚": CityProfile(2000, 40.0, 20.0, 7, 1.0, 0.90),
    "拉萨": CityProfile(1000, 35.0, 15.0, 5, 1.5, 0.75),
    "重庆": CityProfile(1000, 45.0, 20.0, 7, 1.2, 0.85),
    "上海": CityProfile(1200, 45.0, 25.0, 8, 1.0, 0.90),
    "北京": CityProfile(1500, 60.0, 30.0, 8, 1.0, 0.90),
    "成都": CityProfile(1500, 40.0, 20.0, 8, 1.0, 0.90),
    "西安": CityProfile(1200, 35.0, 15.0, 8, 1.0, 0.90),
    "桂林": CityProfile(1000, 30.0, 15.0, 6, 1.0, 0.85),
    "default": CityProfile(1500, 60.0, 30.0, 8, 1.0, 0.90),
}


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


def point_day_load_v25(point: PointIn, day_value: date, profile: CityProfile) -> int:
    return point_visit_minutes_v25(point, profile) + PER_POINT_OVERHEAD_MINUTES + queue_minutes_for_day(point, day_value)


def point_visit_minutes_v25(point: PointIn, profile: CityProfile) -> int:
    spot_factor = point.staminaFactor if point.pointType == "spot" else 1.0
    return max(10, int(round(point.suggestedDurationMinutes * profile.stamina_factor * spot_factor)))


def effective_day_capacity_v25(req: SolveRequest, profile: CityProfile, has_food: bool) -> int:
    base = PACE_MODE_WINDOWS[req.paceMode][2]
    if req.mealPolicy == "auto" and not has_food:
        base -= LUNCH_OVERHEAD_MINUTES
    return max(120, int(base * profile.day_capacity_buffer))


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


def point_can_be_on_day_v25(point: PointIn, window: DayWindow, profile: CityProfile) -> bool:
    if window.end_minutes <= window.start_minutes:
        return False
    if day_is_closed(point, window.date_value):
        return False
    if point.pointType == "spot" and not point.lastEntryTime:
        return False
    if point.lastEntryTime:
        last_entry = parse_hhmm(point.lastEntryTime)
        if window.start_minutes >= last_entry:
            return False

    load = point_day_load_v25(point, window.date_value, profile)
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


def effective_day_capacity(req: SolveRequest, has_food: bool) -> int:
    base = PACE_MODE_WINDOWS[req.paceMode][2]
    if req.mealPolicy == "auto" and not has_food:
        base -= LUNCH_OVERHEAD_MINUTES
    return max(120, base)


def cluster_service_load(cluster: list[int], points: list[PointPrepared], day_value: date) -> int:
    return sum(point_day_load(points[index].point, day_value) for index in cluster)


def route_leg_minutes(
    req: SolveRequest,
    lookups: LookupMaps,
    from_id: str | None,
    from_coord: Coord,
    to_id: str | None,
    to_coord: Coord,
) -> int:
    return lookup_travel_choice(from_id, from_coord, to_id, to_coord, req, lookups).minutes


def estimate_route_travel_for_sequence(
    req: SolveRequest,
    sequence: list[int],
    points: list[PointPrepared],
    lookups: LookupMaps,
    start_anchor: Anchor | None,
    end_anchor: Anchor | None,
) -> int:
    total = 0

    if start_anchor and sequence:
        first_point = points[sequence[0]]
        total += route_leg_minutes(
            req,
            lookups,
            start_anchor.node_id,
            start_anchor.departure,
            first_point.point.id,
            first_point.arrival_coord,
        )

    for current_index, next_index in zip(sequence, sequence[1:]):
        current_point = points[current_index]
        next_point = points[next_index]
        total += route_leg_minutes(
            req,
            lookups,
            current_point.point.id,
            current_point.departure_coord,
            next_point.point.id,
            next_point.arrival_coord,
        )

    if end_anchor and sequence:
        last_point = points[sequence[-1]]
        total += route_leg_minutes(
            req,
            lookups,
            last_point.point.id,
            last_point.departure_coord,
            end_anchor.node_id,
            end_anchor.arrival,
        )

    return total


def choose_seed_for_estimate(
    req: SolveRequest,
    point_indexes: list[int],
    points: list[PointPrepared],
    lookups: LookupMaps,
    start_anchor: Anchor | None,
    end_anchor: Anchor | None,
) -> int:
    if len(point_indexes) == 1:
        return point_indexes[0]

    best_index = point_indexes[0]
    best_cost = float("inf")
    for point_index in point_indexes:
        cost = estimate_route_travel_for_sequence(
            req,
            [point_index],
            points,
            lookups,
            start_anchor,
            end_anchor,
        )
        if cost < best_cost:
            best_cost = cost
            best_index = point_index
    return best_index


def estimate_cluster_route_minutes(
    req: SolveRequest,
    cluster: list[int],
    points: list[PointPrepared],
    lookups: LookupMaps,
    start_anchor: Anchor | None,
    end_anchor: Anchor | None,
) -> int:
    if not cluster:
        return 0

    seed = choose_seed_for_estimate(req, cluster, points, lookups, start_anchor, end_anchor)
    ordering = [seed]
    remaining = [index for index in cluster if index != seed]

    while remaining:
        best_candidate = remaining[0]
        best_ordering: list[int] | None = None
        best_cost = float("inf")

        for candidate in remaining:
            for insert_position in range(len(ordering) + 1):
                candidate_ordering = ordering[:insert_position] + [candidate] + ordering[insert_position:]
                candidate_cost = estimate_route_travel_for_sequence(
                    req,
                    candidate_ordering,
                    points,
                    lookups,
                    start_anchor,
                    end_anchor,
                )
                if candidate_cost < best_cost:
                    best_cost = candidate_cost
                    best_candidate = candidate
                    best_ordering = candidate_ordering

        ordering = best_ordering if best_ordering is not None else ordering + [best_candidate]
        remaining.remove(best_candidate)

    return estimate_route_travel_for_sequence(req, ordering, points, lookups, start_anchor, end_anchor)


def cluster_has_food(cluster: list[int], points: list[PointPrepared]) -> bool:
    if any(points[index].point.hasFoodCourt for index in cluster):
        return True
    return all(points[index].point.suggestedDurationMinutes >= 240 for index in cluster)


def find_nearest_hotel_v25(
    point_indexes: list[int],
    points: list[PointPrepared],
    hotels: list[HotelPrepared],
    matrix: MatrixStore,
) -> int:
    if not point_indexes:
        return 0
    point_idxes = np.array([matrix.id_to_idx[points[index].point.id] for index in point_indexes], dtype=np.int32)
    best_hotel = 0
    best_cost = float("inf")
    for hotel_index, hotel in enumerate(hotels):
        hotel_matrix_index = matrix.id_to_idx[hotel.hotel.id]
        avg_round_trip = float(
            (matrix.driving[hotel_matrix_index, point_idxes] + matrix.driving[point_idxes, hotel_matrix_index]).mean()
        )
        if avg_round_trip < best_cost:
            best_cost = avg_round_trip
            best_hotel = hotel_index
    return best_hotel


def matrix_leg_minutes_v25(
    req: SolveRequest,
    matrix: MatrixStore,
    profile: CityProfile,
    from_id: str,
    to_id: str,
) -> int:
    from_index = matrix.id_to_idx.get(from_id)
    to_index = matrix.id_to_idx.get(to_id)
    if from_index is None or to_index is None:
        return 30
    walking_distance = int(matrix.walking_meters[from_index, to_index])
    if 0 < walking_distance <= profile.walk_threshold_meters:
        return int(matrix.walking_minutes[from_index, to_index])
    transit_minutes = int(matrix.transit[from_index, to_index])
    driving_minutes = int(matrix.driving[from_index, to_index])
    if req.transportPreference == "transit_first":
        return transit_minutes
    if req.transportPreference == "driving_first":
        return driving_minutes
    return min(transit_minutes, driving_minutes)


def compute_edge_weight_v25(
    from_point: PointPrepared,
    to_point: PointPrepared,
    matrix: MatrixStore,
    profile: CityProfile,
) -> float:
    from_index = matrix.id_to_idx[from_point.point.id]
    to_index = matrix.id_to_idx[to_point.point.id]
    walking_distance = int(matrix.walking_meters[from_index, to_index])
    if 0 < walking_distance <= profile.walk_threshold_meters:
        return float(matrix.walking_minutes[from_index, to_index]) * 0.7
    return float(matrix.driving[from_index, to_index])


def pick_seed_v25(
    candidates: list[int],
    points: list[PointPrepared],
    matrix: MatrixStore,
    profile: CityProfile,
) -> int:
    candidate_indexes = np.array([matrix.id_to_idx[points[index].point.id] for index in candidates], dtype=np.int32)
    walking_neighbor_counts: dict[int, int] = {}
    for array_pos, candidate in enumerate(candidates):
        current_index = candidate_indexes[array_pos]
        other_indexes = candidate_indexes[candidate_indexes != current_index]
        if len(other_indexes) == 0:
            walking_neighbor_counts[candidate] = 0
            continue
        distances = matrix.walking_meters[current_index, other_indexes]
        walking_neighbor_counts[candidate] = int(
            np.sum((distances > 0) & (distances <= profile.walk_threshold_meters)),
        )

    max_neighbors = max(walking_neighbor_counts.values())
    top_candidates = (
        [candidate for candidate in candidates if walking_neighbor_counts[candidate] == max_neighbors]
        if max_neighbors > 0
        else candidates
    )

    best_candidate = top_candidates[0]
    best_cost = float("inf")
    top_indexes = np.array([matrix.id_to_idx[points[index].point.id] for index in top_candidates], dtype=np.int32)
    for array_pos, candidate in enumerate(top_candidates):
        current_index = top_indexes[array_pos]
        other_indexes = top_indexes[top_indexes != current_index]
        if len(other_indexes) == 0:
            return candidate
        walking_minutes = matrix.walking_minutes[current_index, other_indexes].astype(np.float32)
        walking_distances = matrix.walking_meters[current_index, other_indexes]
        driving_minutes = matrix.driving[current_index, other_indexes].astype(np.float32)
        weights = np.where(
            (walking_distances > 0) & (walking_distances <= profile.walk_threshold_meters),
            walking_minutes * 0.7,
            driving_minutes,
        )
        total = float(weights.sum())
        if total < best_cost:
            best_cost = total
            best_candidate = candidate
    return best_candidate


def average_weight_to_cluster_v25(
    candidate_index: int,
    cluster: list[int],
    points: list[PointPrepared],
    matrix: MatrixStore,
    profile: CityProfile,
) -> float:
    if not cluster:
        return 0.0
    total = sum(
        compute_edge_weight_v25(points[cluster_index], points[candidate_index], matrix, profile)
        for cluster_index in cluster
    )
    return total / len(cluster)


def estimate_sequence_travel_minutes_v25(
    sequence: list[int],
    points: list[PointPrepared],
    hotels: list[HotelPrepared],
    hotel_index: int,
    matrix: MatrixStore,
    req: SolveRequest,
    profile: CityProfile,
    previous_hotel_index: int | None = None,
) -> int:
    if not sequence:
        return 0
    hotel_id = hotels[hotel_index].hotel.id
    start_id = hotels[previous_hotel_index].hotel.id if previous_hotel_index is not None else hotel_id
    route_ids = [start_id]
    remaining = [points[index].point.id for index in sequence]

    while remaining:
        best_cost_increase = float("inf")
        best_insert_position = 1
        best_candidate_id = remaining[0]
        for candidate_id in remaining:
            for insert_position in range(1, len(route_ids) + 1):
                previous_id = route_ids[insert_position - 1]
                next_id = route_ids[insert_position] if insert_position < len(route_ids) else hotel_id
                cost_increase = (
                    matrix_leg_minutes_v25(req, matrix, profile, previous_id, candidate_id)
                    + matrix_leg_minutes_v25(req, matrix, profile, candidate_id, next_id)
                    - matrix_leg_minutes_v25(req, matrix, profile, previous_id, next_id)
                )
                if cost_increase < best_cost_increase:
                    best_cost_increase = cost_increase
                    best_insert_position = insert_position
                    best_candidate_id = candidate_id
        route_ids.insert(best_insert_position, best_candidate_id)
        remaining.remove(best_candidate_id)

    total = 0
    for current_id, next_id in zip(route_ids, route_ids[1:]):
        total += matrix_leg_minutes_v25(req, matrix, profile, current_id, next_id)
    total += matrix_leg_minutes_v25(req, matrix, profile, route_ids[-1], hotel_id)
    return total


def resolve_hotel_for_day_v25(
    day_position: int,
    cluster: list[int],
    points: list[PointPrepared],
    hotels: list[HotelPrepared],
    matrix: MatrixStore,
    feedback: IterationFeedback | None,
) -> int:
    if feedback is not None and day_position < len(feedback.hotel_per_day):
        return feedback.hotel_per_day[day_position]
    return find_nearest_hotel_v25(cluster, points, hotels, matrix)


def try_swap_for_candidate_v25(
    candidate_index: int,
    cluster: list[int],
    points: list[PointPrepared],
    hotels: list[HotelPrepared],
    hotel_index: int,
    matrix: MatrixStore,
    req: SolveRequest,
    profile: CityProfile,
    window: DayWindow,
    capacity: int,
    previous_hotel_index: int | None,
) -> tuple[list[int], int] | None:
    original_travel = estimate_sequence_travel_minutes_v25(
        cluster,
        points,
        hotels,
        hotel_index,
        matrix,
        req,
        profile,
        previous_hotel_index,
    )
    for existing_index in list(cluster):
        trial_cluster = [index for index in cluster if index != existing_index] + [candidate_index]
        service_load = sum(
            point_day_load_v25(points[index].point, window.date_value, profile)
            for index in trial_cluster
        )
        travel_load = estimate_sequence_travel_minutes_v25(
            trial_cluster,
            points,
            hotels,
            hotel_index,
            matrix,
            req,
            profile,
            previous_hotel_index,
        )
        if service_load + travel_load <= capacity and original_travel - travel_load >= 5:
            return trial_cluster, existing_index
    return None


def cluster_points_to_days_v25(
    req: SolveRequest,
    points: list[PointPrepared],
    hotels: list[HotelPrepared],
    matrix: MatrixStore,
    profile: CityProfile,
    feedback: IterationFeedback | None = None,
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
        day_position = len(day_clusters)
        available = [index for index in unassigned if point_can_be_on_day_v25(points[index].point, window, profile)]
        if not available:
            calendar_index += 1
            continue

        seed = pick_seed_v25(available, points, matrix, profile)
        cluster = [seed]
        unassigned.remove(seed)
        hotel_index = resolve_hotel_for_day_v25(day_position, cluster, points, hotels, matrix, feedback)
        previous_hotel_index = (
            feedback.hotel_per_day[day_position - 1]
            if feedback is not None and day_position > 0 and day_position - 1 < len(feedback.hotel_per_day)
            else None
        )

        while True:
            candidates = [index for index in unassigned if point_can_be_on_day_v25(points[index].point, window, profile)]
            if not candidates or len(cluster) >= profile.max_points_per_day:
                break
            scored_candidates = sorted(
                candidates,
                key=lambda candidate: average_weight_to_cluster_v25(candidate, cluster, points, matrix, profile),
            )
            added = False
            for candidate_index in scored_candidates:
                trial_cluster = cluster + [candidate_index]
                service_load = sum(
                    point_day_load_v25(points[index].point, window.date_value, profile)
                    for index in trial_cluster
                )
                travel_load = estimate_sequence_travel_minutes_v25(
                    trial_cluster,
                    points,
                    hotels,
                    hotel_index,
                    matrix,
                    req,
                    profile,
                    previous_hotel_index,
                )
                capacity = effective_day_capacity_v25(req, profile, cluster_has_food(trial_cluster, points))
                if service_load + travel_load <= capacity:
                    cluster.append(candidate_index)
                    unassigned.remove(candidate_index)
                    added = True
                    break
                swapped = try_swap_for_candidate_v25(
                    candidate_index,
                    cluster,
                    points,
                    hotels,
                    hotel_index,
                    matrix,
                    req,
                    profile,
                    window,
                    capacity,
                    previous_hotel_index,
                )
                if swapped is not None:
                    new_cluster, evicted_index = swapped
                    unassigned.remove(candidate_index)
                    unassigned.append(evicted_index)
                    cluster[:] = new_cluster
                    added = True
                    break
            if not added:
                break

        day_clusters.append((window, cluster))
        calendar_index += 1

    return day_clusters


def assign_hotel_per_day_v25(
    day_clusters: list[tuple[DayWindow, list[int]]],
    points: list[PointPrepared],
    hotels: list[HotelPrepared],
    matrix: MatrixStore,
    req: SolveRequest,
    profile: CityProfile,
    feedback: IterationFeedback | None = None,
) -> list[int]:
    clusters_only = [cluster for _, cluster in day_clusters]
    if not clusters_only:
        return []
    if req.hotelStrategy == "single":
        all_indexes = [index for cluster in clusters_only for index in cluster]
        best_hotel = find_nearest_hotel_v25(all_indexes, points, hotels, matrix)
        return [best_hotel] * len(clusters_only)

    hotel_count = len(hotels)
    day_count = len(clusters_only)
    hotel_day_avg = np.zeros((hotel_count, day_count), dtype=np.float32)
    hotel_matrix_indexes = np.array([matrix.id_to_idx[item.hotel.id] for item in hotels], dtype=np.int32)

    for day_index, cluster in enumerate(clusters_only):
        if not cluster:
            continue
        point_indexes = np.array([matrix.id_to_idx[points[index].point.id] for index in cluster], dtype=np.int32)
        hotel_day_avg[:, day_index] = matrix.driving[np.ix_(hotel_matrix_indexes, point_indexes)].mean(axis=1)

    initial_hotel = int(np.argmin(hotel_day_avg.sum(axis=1)))
    if feedback is not None and feedback.hotel_per_day:
        initial_hotel = feedback.hotel_per_day[0]

    current_hotel = initial_hotel
    assignments: list[int] = []
    for day_index in range(day_count):
        best_for_day = int(np.argmin(hotel_day_avg[:, day_index]))
        incumbent = current_hotel
        if feedback is not None and day_index < len(feedback.hotel_per_day):
            prior = feedback.hotel_per_day[day_index]
            if hotel_day_avg[prior, day_index] < hotel_day_avg[current_hotel, day_index]:
                incumbent = prior
        current_avg = float(hotel_day_avg[incumbent, day_index])
        best_avg = float(hotel_day_avg[best_for_day, day_index])
        if (
            current_avg > profile.hotel_switch_trigger_min
            and best_avg + profile.hotel_switch_improve_min < current_avg
        ):
            incumbent = best_for_day
        current_hotel = incumbent
        assignments.append(current_hotel)
    return assignments


def local_search_improvement_v25(
    day_clusters: list[tuple[DayWindow, list[int]]],
    points: list[PointPrepared],
    hotels: list[HotelPrepared],
    hotel_per_day: list[int],
    matrix: MatrixStore,
    req: SolveRequest,
    profile: CityProfile,
    max_iterations: int = 50,
) -> tuple[list[tuple[DayWindow, list[int]]], list[int]]:
    best_clusters = [list(cluster) for _, cluster in day_clusters]
    best_hotels = list(hotel_per_day)
    windows = [window for window, _ in day_clusters]

    def day_cost(cluster: list[int], hotel_index: int, window: DayWindow, previous_hotel_index: int | None) -> int:
        return estimate_sequence_travel_minutes_v25(
            cluster,
            points,
            hotels,
            hotel_index,
            matrix,
            req,
            profile,
            previous_hotel_index,
        )

    def check_capacity(cluster: list[int], window: DayWindow, hotel_index: int, previous_hotel_index: int | None) -> bool:
        if len(cluster) > profile.max_points_per_day:
            return False
        service = sum(point_day_load_v25(points[index].point, window.date_value, profile) for index in cluster)
        travel = day_cost(cluster, hotel_index, window, previous_hotel_index)
        capacity = effective_day_capacity_v25(req, profile, cluster_has_food(cluster, points))
        return service + travel <= capacity

    improved = True
    iterations = 0
    while improved and iterations < max_iterations:
        improved = False
        iterations += 1
        day_count = len(best_clusters)

        for source_day in range(day_count):
            for point_index in list(best_clusters[source_day]):
                point = points[point_index]
                for target_day in range(day_count):
                    if source_day == target_day:
                        continue
                    if not point_can_be_on_day_v25(point.point, windows[target_day], profile):
                        continue
                    new_source_cluster = [index for index in best_clusters[source_day] if index != point_index]
                    new_target_cluster = best_clusters[target_day] + [point_index]
                    previous_target_hotel = best_hotels[target_day - 1] if target_day > 0 else None
                    if not check_capacity(new_target_cluster, windows[target_day], best_hotels[target_day], previous_target_hotel):
                        continue
                    old_cost = (
                        day_cost(best_clusters[source_day], best_hotels[source_day], windows[source_day], best_hotels[source_day - 1] if source_day > 0 else None)
                        + day_cost(best_clusters[target_day], best_hotels[target_day], windows[target_day], previous_target_hotel)
                    )
                    new_source_hotel = (
                        find_nearest_hotel_v25(new_source_cluster, points, hotels, matrix)
                        if new_source_cluster
                        else best_hotels[source_day]
                    )
                    new_cost = (
                        day_cost(new_source_cluster, new_source_hotel, windows[source_day], best_hotels[source_day - 1] if source_day > 0 else None)
                        + day_cost(new_target_cluster, best_hotels[target_day], windows[target_day], previous_target_hotel)
                    )
                    if new_cost < old_cost - 5:
                        best_clusters[source_day] = new_source_cluster
                        best_clusters[target_day] = new_target_cluster
                        best_hotels[source_day] = new_source_hotel
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue

        for first_day in range(day_count):
            for second_day in range(first_day + 1, day_count):
                previous_first_hotel = best_hotels[first_day - 1] if first_day > 0 else None
                previous_second_hotel = best_hotels[second_day - 1] if second_day > 0 else None
                for first_point in list(best_clusters[first_day]):
                    for second_point in list(best_clusters[second_day]):
                        if not point_can_be_on_day_v25(points[first_point].point, windows[second_day], profile):
                            continue
                        if not point_can_be_on_day_v25(points[second_point].point, windows[first_day], profile):
                            continue
                        new_first_cluster = [index for index in best_clusters[first_day] if index != first_point] + [second_point]
                        new_second_cluster = [index for index in best_clusters[second_day] if index != second_point] + [first_point]
                        if not check_capacity(new_first_cluster, windows[first_day], best_hotels[first_day], previous_first_hotel):
                            continue
                        if not check_capacity(new_second_cluster, windows[second_day], best_hotels[second_day], previous_second_hotel):
                            continue
                        old_cost = (
                            day_cost(best_clusters[first_day], best_hotels[first_day], windows[first_day], previous_first_hotel)
                            + day_cost(best_clusters[second_day], best_hotels[second_day], windows[second_day], previous_second_hotel)
                        )
                        new_cost = (
                            day_cost(new_first_cluster, best_hotels[first_day], windows[first_day], previous_first_hotel)
                            + day_cost(new_second_cluster, best_hotels[second_day], windows[second_day], previous_second_hotel)
                        )
                        if new_cost < old_cost - 5:
                            best_clusters[first_day] = new_first_cluster
                            best_clusters[second_day] = new_second_cluster
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break

    return [(windows[index], best_clusters[index]) for index in range(len(best_clusters))], best_hotels


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


def prioritize_candidates_for_day(
    candidates: list[int],
    day_position: int,
    feedback: IterationFeedback | None,
) -> list[int]:
    if feedback is None:
        return candidates

    same_day = [index for index in candidates if feedback.point_day_by_index.get(index) == day_position]
    if not same_day:
        return candidates

    return sorted(
        same_day,
        key=lambda index: feedback.point_order_by_index.get(index, 10**9),
    )


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
    hotels: list[HotelPrepared],
    lookups: LookupMaps,
    feedback: IterationFeedback | None = None,
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
        day_position = len(day_clusters)
        available_today = [
            index for index in unassigned if point_can_be_on_day(points[index].point, window)
        ]
        if not available_today:
            calendar_index += 1
            continue

        prioritized_available = prioritize_candidates_for_day(available_today, day_position, feedback)
        seed = pick_seed(prioritized_available, points, lookups)
        cluster = [seed]
        unassigned.remove(seed)

        while True:
            candidates = [
                index for index in unassigned if point_can_be_on_day(points[index].point, window)
            ]
            if not candidates:
                break
            if len(cluster) >= MAX_POINTS_PER_DAY:
                break
            prioritized_candidates = prioritize_candidates_for_day(candidates, day_position, feedback)
            next_index = pick_nearest_to_cluster(prioritized_candidates, cluster, points, lookups)
            candidate_cluster = cluster + [next_index]
            has_food = cluster_has_food(candidate_cluster, points)
            capacity = effective_day_capacity(req, has_food)
            feedback_hotel_index = (
                feedback.hotel_per_day[day_position]
                if feedback is not None and day_position < len(feedback.hotel_per_day)
                else find_nearest_hotel(candidate_cluster, points, hotels, lookups)
            )
            previous_feedback_hotel_index = (
                feedback.hotel_per_day[day_position - 1]
                if feedback is not None and day_position > 0 and day_position - 1 < len(feedback.hotel_per_day)
                else None
            )
            start_anchor = build_start_anchor(hotels, feedback_hotel_index, previous_feedback_hotel_index)
            end_anchor = Anchor(
                hotels[feedback_hotel_index].hotel.id,
                hotels[feedback_hotel_index].arrival_coord,
                hotels[feedback_hotel_index].departure_coord,
            )
            candidate_service_load = cluster_service_load(candidate_cluster, points, window.date_value)
            candidate_travel_load = estimate_cluster_route_minutes(
                req,
                candidate_cluster,
                points,
                lookups,
                start_anchor,
                end_anchor,
            )
            if candidate_service_load + candidate_travel_load > capacity:
                if prioritized_candidates != candidates:
                    fallback_candidates = [index for index in candidates if index not in prioritized_candidates]
                    if fallback_candidates:
                        prioritized_candidates = fallback_candidates
                        next_index = pick_nearest_to_cluster(prioritized_candidates, cluster, points, lookups)
                        candidate_cluster = cluster + [next_index]
                        has_food = cluster_has_food(candidate_cluster, points)
                        capacity = effective_day_capacity(req, has_food)
                        feedback_hotel_index = (
                            feedback.hotel_per_day[day_position]
                            if feedback is not None and day_position < len(feedback.hotel_per_day)
                            else find_nearest_hotel(candidate_cluster, points, hotels, lookups)
                        )
                        start_anchor = build_start_anchor(hotels, feedback_hotel_index, previous_feedback_hotel_index)
                        end_anchor = Anchor(
                            hotels[feedback_hotel_index].hotel.id,
                            hotels[feedback_hotel_index].arrival_coord,
                            hotels[feedback_hotel_index].departure_coord,
                        )
                        candidate_service_load = cluster_service_load(candidate_cluster, points, window.date_value)
                        candidate_travel_load = estimate_cluster_route_minutes(
                            req,
                            candidate_cluster,
                            points,
                            lookups,
                            start_anchor,
                            end_anchor,
                        )
                if candidate_service_load + candidate_travel_load > capacity:
                    break
            cluster.append(next_index)
            unassigned.remove(next_index)

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
    feedback: IterationFeedback | None = None,
) -> list[int]:
    clusters_only = [cluster for _, cluster in day_clusters]
    if req.hotelStrategy == "single":
        all_point_indexes = [index for cluster in clusters_only for index in cluster]
        best = find_nearest_hotel(all_point_indexes, points, hotels, lookups)
        return [best] * len(clusters_only)

    if not clusters_only:
        return []

    all_point_indexes = [index for cluster in clusters_only for index in cluster]
    current_hotel = (
        feedback.hotel_per_day[0]
        if feedback is not None and feedback.hotel_per_day
        else find_nearest_hotel(all_point_indexes, points, hotels, lookups)
    )
    assignments: list[int] = []

    for day_position, cluster in enumerate(clusters_only):
        best_for_day = find_nearest_hotel(cluster, points, hotels, lookups)
        incumbent_hotel = current_hotel
        if feedback is not None and day_position < len(feedback.hotel_per_day):
            prior_round_hotel = feedback.hotel_per_day[day_position]
            prior_avg = average_one_way_driving_minutes(prior_round_hotel, cluster, points, hotels, lookups)
            current_avg = average_one_way_driving_minutes(current_hotel, cluster, points, hotels, lookups)
            if prior_avg < current_avg:
                incumbent_hotel = prior_round_hotel

        current_avg = average_one_way_driving_minutes(incumbent_hotel, cluster, points, hotels, lookups)
        best_avg = average_one_way_driving_minutes(best_for_day, cluster, points, hotels, lookups)
        if (
            current_avg > HOTEL_SWITCH_TRIGGER_MINUTES
            and best_avg + HOTEL_SWITCH_IMPROVEMENT_MINUTES < current_avg
        ):
            incumbent_hotel = best_for_day
        current_hotel = incumbent_hotel
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
    profile: CityProfile,
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
        service_end = current_time + point_visit_minutes_v25(point, profile) + queue_minutes
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
    profile: CityProfile,
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
            profile,
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
                profile,
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


def verify_and_build_route_v25(
    req: SolveRequest,
    points: list[PointPrepared],
    hotels: list[HotelPrepared],
    profile: CityProfile,
    day_window: DayWindow,
    hotel_index: int,
    cluster: list[int],
    ordered_ids: list[str],
    lookups: LookupMaps,
    previous_hotel_index: int | None,
    is_last_day: bool,
) -> EvaluatedRoute:
    id_to_cluster_pos = {points[index].point.id: pos for pos, index in enumerate(cluster)}
    ordering = [id_to_cluster_pos[point_id] for point_id in ordered_ids if point_id in id_to_cluster_pos]
    start_anchor = build_start_anchor(hotels, hotel_index, previous_hotel_index)
    return evaluate_order(
        req,
        points,
        hotels,
        profile,
        day_window,
        hotel_index,
        cluster,
        ordering,
        lookups,
        start_anchor,
        previous_hotel_index,
        is_last_day,
    )


def rank_hotel_candidates_v25(
    cluster: list[int],
    points: list[PointPrepared],
    hotels: list[HotelPrepared],
    matrix: MatrixStore,
    preferred_hotel_index: int,
    previous_hotel_index: int | None,
    limit: int = 12,
) -> list[int]:
    if not cluster:
        return [preferred_hotel_index]
    point_indexes = np.array([matrix.id_to_idx[points[index].point.id] for index in cluster], dtype=np.int32)
    scores: list[tuple[float, int]] = []
    for hotel_index, hotel in enumerate(hotels):
        hotel_matrix_index = matrix.id_to_idx[hotel.hotel.id]
        avg_round_trip = float(
            (matrix.driving[hotel_matrix_index, point_indexes] + matrix.driving[point_indexes, hotel_matrix_index]).mean()
        )
        scores.append((avg_round_trip, hotel_index))
    scores.sort(key=lambda item: item[0])
    ranked = [hotel_index for _, hotel_index in scores[: max(1, min(limit, len(scores)))]]
    for special in (preferred_hotel_index, previous_hotel_index):
        if special is not None and special not in ranked:
            ranked.append(special)
    return ranked


def build_routing_order_for_hotel_v25(
    req: SolveRequest,
    points: list[PointPrepared],
    hotels: list[HotelPrepared],
    cluster: list[int],
    hotel_index: int,
    lookups: LookupMaps,
    previous_hotel_index: int | None,
) -> list[str] | None:
    if not ORTOOLS_AVAILABLE or not cluster:
        return None

    start_hotel = hotels[previous_hotel_index] if previous_hotel_index is not None else hotels[hotel_index]
    end_hotel = hotels[hotel_index]

    node_ids = [start_hotel.hotel.id] + [points[index].point.id for index in cluster] + [end_hotel.hotel.id]
    node_departures = [start_hotel.departure_coord] + [points[index].departure_coord for index in cluster] + [end_hotel.departure_coord]
    node_arrivals = [start_hotel.arrival_coord] + [points[index].arrival_coord for index in cluster] + [end_hotel.arrival_coord]

    node_count = len(node_ids)
    travel_matrix: list[list[int]] = [[0] * node_count for _ in range(node_count)]
    for from_index in range(node_count):
        for to_index in range(node_count):
            if from_index == to_index:
                continue
            if to_index == 0 or from_index == node_count - 1:
                travel_matrix[from_index][to_index] = 10**6
                continue
            choice = lookup_travel_choice(
                node_ids[from_index],
                node_departures[from_index],
                node_ids[to_index],
                node_arrivals[to_index],
                req,
                lookups,
            )
            travel_matrix[from_index][to_index] = choice.minutes

    manager = pywrapcp.RoutingIndexManager(node_count, 1, [0], [node_count - 1])
    routing = pywrapcp.RoutingModel(manager)

    def transit_callback(from_index: int, to_index: int) -> int:
        return travel_matrix[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]

    transit_callback_index = routing.RegisterTransitCallback(transit_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.time_limit.FromMilliseconds(500)
    search_parameters.log_search = False

    solution = routing.SolveWithParameters(search_parameters)
    if solution is None:
        return None

    index = routing.Start(0)
    ordered_point_ids: list[str] = []
    while not routing.IsEnd(index):
        node = manager.IndexToNode(index)
        if 0 < node < node_count - 1:
            ordered_point_ids.append(node_ids[node])
        index = solution.Value(routing.NextVar(index))
    return ordered_point_ids


def solve_day_with_ortools_v25(
    req: SolveRequest,
    points: list[PointPrepared],
    hotels: list[HotelPrepared],
    profile: CityProfile,
    day_window: DayWindow,
    cluster: list[int],
    matrix: MatrixStore,
    lookups: LookupMaps,
    preferred_hotel_index: int,
    previous_hotel_index: int | None,
    is_last_day: bool,
) -> tuple[int, EvaluatedRoute]:
    if not cluster:
        route = verify_and_build_route_v25(
            req,
            points,
            hotels,
            profile,
            day_window,
            preferred_hotel_index,
            cluster,
            [],
            lookups,
            previous_hotel_index,
            is_last_day,
        )
        return preferred_hotel_index, route

    candidate_hotels = (
        [preferred_hotel_index]
        if req.hotelStrategy == "single"
        else rank_hotel_candidates_v25(
            cluster,
            points,
            hotels,
            matrix,
            preferred_hotel_index,
            previous_hotel_index,
        )
    )

    best_route: EvaluatedRoute | None = None
    best_hotel_index: int | None = None
    failures: list[str] = []

    for hotel_index in candidate_hotels:
        ordered_ids = build_routing_order_for_hotel_v25(
            req,
            points,
            hotels,
            cluster,
            hotel_index,
            lookups,
            previous_hotel_index,
        )
        if ordered_ids is None:
            continue
        try:
            route = verify_and_build_route_v25(
                req,
                points,
                hotels,
                profile,
                day_window,
                hotel_index,
                cluster,
                ordered_ids,
                lookups,
                previous_hotel_index,
                is_last_day,
            )
        except HTTPException as exc:
            failures.append(extract_http_detail_message(exc.detail))
            continue
        if best_route is None or route.effective_cost < best_route.effective_cost:
            best_route = route
            best_hotel_index = hotel_index

    exact_candidates = candidate_hotels if len(cluster) <= MAX_EXACT_PERMUTATION_SIZE else []
    for hotel_index in exact_candidates:
        try:
            route = solve_day_route(
                req,
                points,
                hotels,
                profile,
                day_window,
                hotel_index,
                cluster,
                build_start_anchor(hotels, hotel_index, previous_hotel_index),
                previous_hotel_index,
                is_last_day,
                lookups,
            )
        except HTTPException as exc:
            failures.append(extract_http_detail_message(exc.detail))
            continue
        if best_route is None or route.effective_cost < best_route.effective_cost:
            best_route = route
            best_hotel_index = hotel_index

    if best_route is not None and best_hotel_index is not None:
        return best_hotel_index, best_route

    fallback_candidates = candidate_hotels or [preferred_hotel_index]
    for hotel_index in fallback_candidates:
        try:
            route = solve_day_route(
                req,
                points,
                hotels,
                profile,
                day_window,
                hotel_index,
                cluster,
                build_start_anchor(hotels, hotel_index, previous_hotel_index),
                previous_hotel_index,
                is_last_day,
                lookups,
            )
        except HTTPException as exc:
            failures.append(extract_http_detail_message(exc.detail))
            continue
        if best_route is None or route.effective_cost < best_route.effective_cost:
            best_route = route
            best_hotel_index = hotel_index

    if best_route is None or best_hotel_index is None:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "NO_FEASIBLE_DAY_ROUTE",
                "message": "no feasible itinerary for scheduled day",
                "date": day_window.date_value.isoformat(),
                "pointIds": [points[index].point.id for index in cluster],
                "reasonSummary": failures[:5],
            },
        )
    return best_hotel_index, best_route


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
        "optimizationRounds": OPTIMIZATION_ROUNDS,
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
                "pointIds": route.point_ids,
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


def solution_score_v25(
    routes: list[EvaluatedRoute],
    points: list[PointPrepared],
    day_count: int,
    profile: CityProfile,
) -> float:
    total_travel = sum(route.diagnostics.travel_minutes for route in routes)
    hotel_switches = sum(1 for route in routes if route.diagnostics.hotel_switch)
    point_duration_by_id = {item.point.id: point_visit_minutes_v25(item.point, profile) for item in points}
    single_point_penalty = 0.0
    for route in routes:
        if len(route.point_ids) == 1:
            if point_duration_by_id.get(route.point_ids[0], 0) < 360:
                single_point_penalty += 120.0
    return total_travel + hotel_switches * profile.hotel_switch_improve_min + single_point_penalty + day_count * 10.0


def validate_solution_universally_v25(
    result: SolveResponse,
    req: SolveRequest,
    points: list[PointPrepared],
    hotels: list[HotelPrepared],
    lookups: LookupMaps,
    profile: CityProfile,
) -> list[str]:
    warnings: list[str] = []
    point_index_by_id = {item.point.id: index for index, item in enumerate(points)}
    hotel_index_by_id = {item.hotel.id: index for index, item in enumerate(hotels)}
    previous_hotel_index: int | None = None

    for day in result.days:
        window = build_day_window(req, day.dayNumber - 1)
        cluster = [point_index_by_id[point_id] for point_id in day.pointIds if point_id in point_index_by_id]
        hotel_index = hotel_index_by_id.get(day.hotelId)
        if hotel_index is None:
            warnings.append(f"day {day.dayNumber}: unknown hotel {day.hotelId}")
            continue
        ordering_map = {points[index].point.id: pos for pos, index in enumerate(cluster)}
        ordering = [ordering_map[point_id] for point_id in day.pointIds if point_id in ordering_map]
        start_anchor = build_start_anchor(hotels, hotel_index, previous_hotel_index)
        try:
            evaluate_order(
                req,
                points,
                hotels,
                profile,
                window,
                hotel_index,
                cluster,
                ordering,
                lookups,
                start_anchor,
                previous_hotel_index,
                day.dayNumber == result.tripDays,
            )
        except HTTPException as exc:
            warnings.append(f"day {day.dayNumber}: {extract_http_detail_message(exc.detail)}")
        previous_hotel_index = hotel_index
    return warnings


def build_iteration_feedback(
    day_clusters: list[tuple[DayWindow, list[int]]],
    hotel_per_day: list[int],
    day_routes: list[EvaluatedRoute],
) -> IterationFeedback:
    point_day_by_index: dict[int, int] = {}
    point_order_by_index: dict[int, int] = {}

    for day_position, ((_, cluster), route) in enumerate(zip(day_clusters, day_routes)):
        ordered_cluster = [cluster[position] for position in route.ordering]
        for point_index in cluster:
            point_day_by_index[point_index] = day_position
        for order_position, point_index in enumerate(ordered_cluster):
            point_order_by_index[point_index] = order_position

    return IterationFeedback(
        hotel_per_day=list(hotel_per_day),
        point_day_by_index=point_day_by_index,
        point_order_by_index=point_order_by_index,
    )


def solution_round_score(day_routes: list[EvaluatedRoute]) -> float:
    total_cost = sum(route.effective_cost for route in day_routes)
    hotel_switches = sum(1 for route in day_routes if route.diagnostics.hotel_switch)
    return total_cost + hotel_switches * HOTEL_SWITCH_IMPROVEMENT_MINUTES


def execute_planning_round(
    req: SolveRequest,
    points: list[PointPrepared],
    hotels: list[HotelPrepared],
    lookups: LookupMaps,
    feedback: IterationFeedback | None,
) -> tuple[list[tuple[DayWindow, list[int]]], list[int], list[EvaluatedRoute]]:
    profile = CITY_PROFILES["default"]
    day_clusters = cluster_points_to_days(req, points, hotels, lookups, feedback)
    hotel_per_day = assign_hotel_per_day(day_clusters, points, hotels, lookups, req, feedback)
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
                profile,
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
                    profile,
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

    return day_clusters, hotel_per_day, routes


def solve_itinerary(req: SolveRequest) -> SolveResponse:
    validate_request(req)
    points = prepare_points(req)
    hotels = filter_hotels(req)
    lookups = build_lookup_maps(req.distanceMatrix.rows)
    matrix = build_matrix_store(points, hotels, req.distanceMatrix.rows)
    profile = detect_city_profile(req.city, points, matrix)
    feedback: IterationFeedback | None = None
    best_day_clusters: list[tuple[DayWindow, list[int]]] | None = None
    best_hotel_per_day: list[int] | None = None
    best_routes: list[EvaluatedRoute] | None = None
    best_score: float | None = None

    for _ in range(OPTIMIZATION_ROUNDS):
        day_clusters = cluster_points_to_days_v25(
            req,
            points,
            hotels,
            matrix,
            profile,
            feedback,
        )
        hotel_per_day = assign_hotel_per_day_v25(
            day_clusters,
            points,
            hotels,
            matrix,
            req,
            profile,
            feedback,
        )
        day_clusters, hotel_per_day = local_search_improvement_v25(
            day_clusters,
            points,
            hotels,
            hotel_per_day,
            matrix,
            req,
            profile,
        )

        routes: list[EvaluatedRoute] = []
        final_hotel_per_day: list[int] = []
        for day_position, (window, cluster) in enumerate(day_clusters):
            previous_hotel_index = final_hotel_per_day[day_position - 1] if day_position > 0 else None
            is_last_day = day_position == len(day_clusters) - 1
            preferred_hotel_index = hotel_per_day[day_position]
            hotel_index, route = solve_day_with_ortools_v25(
                req,
                points,
                hotels,
                profile,
                window,
                cluster,
                matrix,
                lookups,
                preferred_hotel_index,
                previous_hotel_index,
                is_last_day,
            )
            routes.append(route)
            final_hotel_per_day.append(hotel_index)

        score = solution_score_v25(
            routes,
            points,
            len(day_clusters),
            profile,
        )
        if best_score is None or score < best_score:
            best_score = score
            best_day_clusters = day_clusters
            best_hotel_per_day = final_hotel_per_day
            best_routes = routes
        feedback = build_iteration_feedback(day_clusters, final_hotel_per_day, routes)

    if best_day_clusters is None or best_hotel_per_day is None or best_routes is None:
        raise HTTPException(status_code=500, detail="optimizer failed to produce itinerary")

    days: list[DayPlan] = []
    for day_position, ((window, _cluster), route) in enumerate(zip(best_day_clusters, best_routes)):
        hotel_index = best_hotel_per_day[day_position]
        days.append(
            DayPlan(
                dayNumber=day_position + 1,
                date=window.date_value.isoformat(),
                pointIds=route.point_ids,
                hotelId=hotels[hotel_index].hotel.id,
            )
        )

    result = SolveResponse(
        tripDays=len(days),
        solverStatus="FEASIBLE",
        objective="min_travel_then_hotel_switches",
        days=days,
        diagnostics=build_solver_diagnostics(
            req,
            best_day_clusters,
            best_hotel_per_day,
            points,
            hotels,
            best_routes,
        ),
    )
    if result.diagnostics is not None:
        result.diagnostics["cityProfile"] = {
            "walkThresholdMeters": profile.walk_threshold_meters,
            "hotelSwitchTriggerMin": profile.hotel_switch_trigger_min,
            "hotelSwitchImproveMin": profile.hotel_switch_improve_min,
            "maxPointsPerDay": profile.max_points_per_day,
            "staminaFactor": profile.stamina_factor,
            "dayCapacityBuffer": profile.day_capacity_buffer,
        }
        warnings = validate_solution_universally_v25(result, req, points, hotels, lookups, profile)
        if warnings:
            result.diagnostics["validationWarnings"] = warnings
    return result


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": app.version}


@app.post("/solve", response_model=SolveResponse)
def solve(req: SolveRequest) -> SolveResponse:
    return solve_itinerary(req)
