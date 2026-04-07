from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
CASES_DIR = ROOT / "regression_cases"


class RegressionFailure(Exception):
    pass


def load_optimizer_module():
    module_name = "optivoy_optimizer_app"
    spec = importlib.util.spec_from_file_location(module_name, ROOT / "app.py")
    if spec is None or spec.loader is None:
        raise RegressionFailure("unable to load optimizer module spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as exc:  # pragma: no cover - runtime guard
        raise RegressionFailure(
            f"missing dependency '{exc.name}', install requirements.txt before running regressions",
        ) from exc
    return module


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise RegressionFailure(message)


def load_case(case_path: Path) -> dict[str, Any]:
    return json.loads(case_path.read_text(encoding="utf-8"))


def normalize_result_day(day: Any) -> dict[str, Any]:
    return {
        "dayNumber": day.dayNumber,
        "date": day.date.isoformat() if hasattr(day.date, "isoformat") else str(day.date),
        "pointIds": list(day.pointIds),
        "hotelId": day.hotelId,
        "dayType": getattr(day, "dayType", None),
        "blankReason": getattr(day, "blankReason", None),
    }


def validate_smart_hotel_rules(result_days: list[dict[str, Any]]) -> None:
    hotel_ids = [item["hotelId"] for item in result_days]
    trip_days = len(hotel_ids)
    distinct_hotels = len(set(hotel_ids))
    max_distinct_hotels = max(1, (trip_days + 1) // 2)
    assert_true(
        distinct_hotels <= max_distinct_hotels,
        f"distinct hotel count {distinct_hotels} exceeds limit {max_distinct_hotels}",
    )

    index = 0
    while index < trip_days:
        next_index = index + 1
        while next_index < trip_days and hotel_ids[next_index] == hotel_ids[index]:
            next_index += 1
        run_length = next_index - index
        is_last_segment = next_index == trip_days
        if not is_last_segment:
            assert_true(
                run_length >= 2,
                f"hotel segment starting day {index + 1} has run length {run_length}, expected >= 2",
            )
        index = next_index


def validate_case(module: Any, case_path: Path) -> None:
    case = load_case(case_path)
    request = module.SolveRequest.model_validate(case["request"])
    result = module.solve_itinerary_exact(request)
    expectations = case.get("expect", {})

    result_days = [normalize_result_day(day) for day in result.days]
    diagnostics = result.diagnostics or {}
    diagnostics_days = {}
    for item in diagnostics.get("days", []):
        if isinstance(item, dict) and isinstance(item.get("dayNumber"), int):
            diagnostics_days[item["dayNumber"]] = item

    if "solverStatus" in expectations:
        assert_true(
            result.solverStatus == expectations["solverStatus"],
            f"solverStatus={result.solverStatus}, expected {expectations['solverStatus']}",
        )
    if "tripDays" in expectations:
        assert_true(
            result.tripDays == expectations["tripDays"],
            f"tripDays={result.tripDays}, expected {expectations['tripDays']}",
        )
    if "blankDays" in expectations:
        assert_true(
            diagnostics.get("blankDays") == expectations["blankDays"],
            f"blankDays={diagnostics.get('blankDays')}, expected {expectations['blankDays']}",
        )
    if "dayTypesByDay" in expectations:
        for day_number_str, expected_value in expectations["dayTypesByDay"].items():
            day_number = int(day_number_str)
            actual = result_days[day_number - 1]["dayType"]
            assert_true(
                actual == expected_value,
                f"day {day_number} dayType={actual}, expected {expected_value}",
            )
    if "blankReasonByDay" in expectations:
        for day_number_str, expected_value in expectations["blankReasonByDay"].items():
            day_number = int(day_number_str)
            actual = result_days[day_number - 1]["blankReason"]
            assert_true(
                actual == expected_value,
                f"day {day_number} blankReason={actual}, expected {expected_value}",
            )
    if "queueMinutesByDay" in expectations:
        for day_number_str, expected_value in expectations["queueMinutesByDay"].items():
            day_number = int(day_number_str)
            actual = diagnostics_days.get(day_number, {}).get("queueMinutes")
            assert_true(
                actual == expected_value,
                f"day {day_number} queueMinutes={actual}, expected {expected_value}",
            )
    if expectations.get("requireSmartHotelRules"):
        validate_smart_hotel_rules(result_days)
    if "minHotelSwitches" in expectations:
        actual_switches = int(diagnostics.get("hotelSwitches") or 0)
        assert_true(
            actual_switches >= expectations["minHotelSwitches"],
            f"hotelSwitches={actual_switches}, expected >= {expectations['minHotelSwitches']}",
        )
    if expectations.get("requireHotelTransferOnSwitchDays"):
        switched_days: list[int] = []
        previous_hotel_id: str | None = None
        for item in result_days:
            if previous_hotel_id is not None and item["hotelId"] != previous_hotel_id:
                switched_days.append(item["dayNumber"])
            previous_hotel_id = item["hotelId"]
        assert_true(switched_days, "expected at least one hotel switch day")
        for day_number in switched_days:
            actual = diagnostics_days.get(day_number, {}).get("hotelTransferMinutes")
            assert_true(
                isinstance(actual, int) and actual > 0,
                f"day {day_number} hotelTransferMinutes={actual}, expected > 0",
            )


def main() -> int:
    case_paths = sorted(CASES_DIR.glob("*.json"))
    filters = set(sys.argv[1:])
    if filters:
        case_paths = [path for path in case_paths if path.name in filters]
    if not case_paths:
        print("no regression cases found", file=sys.stderr)
        return 1

    try:
        module = load_optimizer_module()
    except RegressionFailure as exc:
        print(f"[regression] setup failed: {exc}", file=sys.stderr)
        return 2

    passed = 0
    for case_path in case_paths:
        try:
            validate_case(module, case_path)
            print(f"[regression] PASS {case_path.name}", flush=True)
            passed += 1
        except Exception as exc:  # pragma: no cover - command-line reporting
            print(f"[regression] FAIL {case_path.name}: {exc}", file=sys.stderr, flush=True)
            return 1

    print(f"[regression] all {passed} cases passed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
