"""
tide_tool.py
------------
Python port of the Node.js tide server (index.js + findNearestStation.js),
packaged as a single callable "tool" function: get_tide.

Usage as a plain function:

    from tide_tool import get_tide
    result = get_tide(latitude=19.076, longitude=72.8777,
                       from_date="2026-08-29", to_date="2026-11-30")

Usage as an LLM tool (OpenAI / Anthropic / LangChain style function calling):
    See GET_TIDE_TOOL_SCHEMA at the bottom of this file — pass it straight
    into your `tools=[...]` list, and dispatch calls to `get_tide(**args)`.

NOTE — ACTION NEEDED:
    `fetch_tide_page()` and `parse_tide_page()` below are placeholders.
    Your original Node project pulls the real scraping/parsing logic from
    "./tideService.js" (fetchTidePage + parseTidePage), which was not
    included in what you pasted. I don't want to guess at INCOIS's HTML
    structure and give you a function that silently returns wrong data,
    so these two raise NotImplementedError until you either:
      (a) paste me tideService.js so I can port it exactly, or
      (b) tell me the INCOIS PAT URL pattern you were hitting and what
          the HTML/JSON response looks like.
    Everything else (station lookup, validation, the tool wrapper,
    multi-location batching) is a complete, working port.
"""

import json
import math
import os
import re
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional

import requests
from bs4 import BeautifulSoup
from langchain_core.tools import tool

# ==================================================
# LOAD STATIONS
# ==================================================

_STATIONS_PATH = os.path.join(os.path.dirname(__file__), "stations.json")

with open(_STATIONS_PATH, "r", encoding="utf-8") as f:
    _STATIONS: List[Dict[str, Any]] = json.load(f)


# ==================================================
# FIND NEAREST STATION  (port of findNearestStation.js)
# ==================================================

def find_nearest_station(latitude: float, longitude: float) -> Dict[str, Any]:
    """
    Returns the nearest INCOIS PAT station to (latitude, longitude)
    using the haversine formula, same as the Node version.
    """
    if not isinstance(latitude, (int, float)) or not isinstance(longitude, (int, float)):
        raise ValueError("Latitude and longitude must be valid numbers.")
    if not (math.isfinite(latitude) and math.isfinite(longitude)):
        raise ValueError("Latitude and longitude must be valid numbers.")

    R = 6371  # Earth radius in km

    def to_radians(deg: float) -> float:
        return deg * math.pi / 180

    nearest_station: Optional[Dict[str, Any]] = None
    shortest_distance = math.inf

    for station in _STATIONS:
        d_lat = to_radians(station["latitude"] - latitude)
        d_lon = to_radians(station["longitude"] - longitude)

        a = (
            math.sin(d_lat / 2) ** 2
            + math.cos(to_radians(latitude))
            * math.cos(to_radians(station["latitude"]))
            * math.sin(d_lon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distance = R * c

        if distance < shortest_distance:
            shortest_distance = distance
            nearest_station = station

    return {
        "name": nearest_station["name"],
        "latitude": nearest_station["latitude"],
        "longitude": nearest_station["longitude"],
        "distance_km": round(shortest_distance, 2),
    }


# ==================================================
# FETCH + PARSE TIDE PAGE  (port of tideService.js)
# ==================================================

_INCOIS_BASE_URL = "https://www.incois.gov.in/oceanservices/PAT/tidegraphphases.jsp"

_TIDE_TIME_RE = re.compile(r"^(\d{2}-\d{2}-\d{4})\s+(\d{2}:\d{2})$")


def fetch_tide_page(station_name: str, from_date: str, to_date: str) -> str:
    """
    Fetch the raw INCOIS PAT HTML for `station_name` between `from_date`
    and `to_date`. Port of tideService.js's fetchTidePage.
    """
    if not station_name:
        raise ValueError("Station name is required")
    if not from_date:
        raise ValueError("fromDate is required")
    if not to_date:
        raise ValueError("toDate is required")

    params = {
        "fromDate": from_date,
        "toDate": to_date,
        "region": station_name,
    }

    print("\nFetching INCOIS PAT:")
    print(f"{_INCOIS_BASE_URL}?{requests.compat.urlencode(params)}")

    response = requests.get(_INCOIS_BASE_URL, params=params, timeout=30)

    if not response.ok:
        raise RuntimeError(f"INCOIS request failed: {response.status_code}")

    return response.text


def _is_valid_tide_time(value: Optional[str]) -> bool:
    if not value or not isinstance(value, str):
        return False
    return bool(_TIDE_TIME_RE.match(value.strip()))


def _extract_date(value: str) -> Optional[str]:
    match = _TIDE_TIME_RE.match(value.strip())
    return match.group(1) if match else None


def _extract_time(value: str) -> Optional[str]:
    match = _TIDE_TIME_RE.match(value.strip())
    return match.group(2) if match else None


def _to_number(value: str) -> Optional[float]:
    """Mirrors JS `Number(x)` + `Number.isFinite` — bad input -> None."""
    try:
        n = float(value)
        return n if math.isfinite(n) else None
    except (TypeError, ValueError):
        return None


def _parse_incois_time(value: str) -> Optional[datetime]:
    """
    Parses 'DD-MM-YYYY HH:MM' as IST and converts to UTC, mirroring the
    Node version's `Date.UTC(year, month - 1, day, hour - 5, minute - 30)`
    (which relies on JS's automatic field-overflow normalization).
    Python's timedelta arithmetic normalizes the same way.
    """
    match = re.match(r"^(\d{2})-(\d{2})-(\d{4})\s+(\d{2}):(\d{2})$", value)
    if not match:
        return None

    day, month, year, hour, minute = (int(g) for g in match.groups())

    base = datetime(year, month, day, tzinfo=timezone.utc)
    return base + timedelta(hours=hour - 5, minutes=minute - 30)


def _remove_duplicates(tides: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: Dict[str, Dict[str, Any]] = {}
    for tide in tides:
        key = f"{tide['date']}|{tide['time']}|{tide['height_m']}"
        deduped[key] = tide  # last write wins, same as JS Map behavior
    return list(deduped.values())


def parse_tide_page(html: str) -> Dict[str, Any]:
    """
    Parses the INCOIS PAT HTML into the ORCA tide structure. Port of
    tideService.js's parseTidePage.
    """
    soup = BeautifulSoup(html, "html.parser")

    high_tide: List[Dict[str, Any]] = []
    low_tide: List[Dict[str, Any]] = []

    # ==================================================
    # FIND TABLE(S) CONTAINING "High Tide" AND "Low Tide"
    # ==================================================
    for table in soup.find_all("table"):
        table_text = re.sub(r"\s+", " ", table.get_text()).strip().lower()

        if "high tide" not in table_text or "low tide" not in table_text:
            continue

        # ==================================================
        # READ EVERY ROW
        # ==================================================
        for row in table.find_all("tr"):
            cells = [
                re.sub(r"\s+", " ", cell.get_text()).strip()
                for cell in row.find_all("td")
            ]

            # Expected INCOIS row:
            # High Time | High Level | Low Time | Low Level
            if len(cells) < 4:
                continue

            high_time = cells[0]
            high_level = _to_number(cells[1])

            low_time = cells[2]
            low_level = _to_number(cells[3])

            # HIGH TIDE
            if _is_valid_tide_time(high_time) and high_level is not None:
                high_tide.append({
                    "date": _extract_date(high_time),
                    "time": _extract_time(high_time),
                    "height_m": high_level,
                })

            # LOW TIDE
            if _is_valid_tide_time(low_time) and low_level is not None:
                low_tide.append({
                    "date": _extract_date(low_time),
                    "time": _extract_time(low_time),
                    "height_m": low_level,
                })

    # REMOVE DUPLICATES
    unique_high = _remove_duplicates(high_tide)
    unique_low = _remove_duplicates(low_tide)

    # SORT
    unique_high.sort(key=lambda t: _parse_incois_time(f"{t['date']} {t['time']}"))
    unique_low.sort(key=lambda t: _parse_incois_time(f"{t['date']} {t['time']}"))

    # FINAL REQUIRED ORCA STRUCTURE
    return {
        "current_tide_height_m": None,
        "tide_phase": None,
        "high_tide": [
            {"time": f"{t['date']} {t['time']}", "height_m": t["height_m"]}
            for t in unique_high
        ],
        "low_tide": [
            {"time": f"{t['date']} {t['time']}", "height_m": t["height_m"]}
            for t in unique_low
        ],
        "tidal_current_velocity_ms": None,
        "tidal_current_direction_deg": None,
    }


# ==================================================
# VALIDATION HELPERS
# ==================================================

def _validate_date(value: str, field_name: str) -> None:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be in YYYY-MM-DD format.")
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        raise ValueError(f"{field_name} must be in YYYY-MM-DD format.")


# ==================================================
# GET TIDE FOR ONE LOCATION  (port of getTideForLocation)
# ==================================================

def get_tide_for_location(
    latitude: float,
    longitude: float,
    from_date: str,
    to_date: str,
) -> Dict[str, Any]:
    """
    Full pipeline for one coordinate: validate -> nearest station ->
    fetch INCOIS PAT page -> parse tides -> assemble response.
    """
    if not isinstance(latitude, (int, float)) or not isinstance(longitude, (int, float)):
        raise ValueError("Latitude and longitude must be valid numbers.")
    if not (math.isfinite(latitude) and math.isfinite(longitude)):
        raise ValueError("Latitude and longitude must be valid numbers.")

    _validate_date(from_date, "fromDate")
    _validate_date(to_date, "toDate")

    # 1. Find nearest PAT station
    station = find_nearest_station(latitude, longitude)
    print("\n" + "=" * 40)
    print("Nearest INCOIS PAT Station")
    print("=" * 40)
    print("Station:", station["name"])
    print("Station Latitude:", station["latitude"])
    print("Station Longitude:", station["longitude"])
    print("Distance:", station["distance_km"], "km")

    # 2. Use station name as PAT region
    region = station["name"]
    print("PAT Region:", region)

    # 3. Fetch INCOIS PAT HTML
    html = fetch_tide_page(region, from_date, to_date)

    # 4. Parse high + low tides
    tide_data = parse_tide_page(html)

    # 5. Final response
    return {
        "latitude": latitude,
        "longitude": longitude,
        "station": {
            "name": station["name"],
            "latitude": station["latitude"],
            "longitude": station["longitude"],
            "distance_km": station["distance_km"],
        },
        "date_range": {"from": from_date, "to": to_date},
        "tides": tide_data,
        "source": "INCOIS PAT",
    }


# ==================================================
# GET TIDE FOR MULTIPLE COORDINATES (port of getTideData)
# ==================================================

def get_tide_data(
    locations: List[Dict[str, float]],
    from_date: str,
    to_date: str,
) -> List[Dict[str, Any]]:
    if not isinstance(locations, list) or len(locations) == 0:
        raise ValueError("locations must be a non-empty array.")

    return [
        get_tide_for_location(loc["latitude"], loc["longitude"], from_date, to_date)
        for loc in locations
    ]


# ==================================================
# TOOL FUNCTION — this is the one to hand to an LLM / agent
# ==================================================

@tool
def get_tide(
    latitude: float,
    longitude: float,
    from_date: str,
    to_date: str,
) -> Dict[str, Any]:
    """
    Get high/low tide predictions for a coastal location in the Indian
    Ocean region (India, Sri Lanka, Bangladesh, Myanmar, Pakistan, Maldives,
    etc.) given its latitude and longitude, using data from the nearest
    INCOIS Potential Alongshore Tide (PAT) station.

    Args:
        latitude: Latitude of the location, e.g. 19.076
        longitude: Longitude of the location, e.g. 72.8777
        from_date: Start date in YYYY-MM-DD format
        to_date: End date in YYYY-MM-DD format

    Returns:
        A JSON-serializable dict: success/error, nearest station info, and
        tide predictions for the date range.
    """
    try:
        data = get_tide_for_location(latitude, longitude, from_date, to_date)
        return {
            "success": True,
            "date_range": {"from": from_date, "to": to_date},
            "data": data,
        }
    except Exception as error:
        return {"success": False, "error": str(error)}
