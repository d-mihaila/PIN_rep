import os
import json
import csv
import pandas as pd
import reverse_geocoder as rg
from geopy.distance import geodesic
# from geolocation import * # for the extract location thingy 
from pathlib import Path
from typing import Dict, Any, Tuple, List
import geopandas as gpd


def fetch_idmc_displacements_by_admin(
    event_start_date: str,
    iso3: str,
    matched_polys: gpd.GeoDataFrame,
    out_json_path: str,
    *,
    past_look_years: int = 0,
    past_look_months: int = 0,
    past_look_weeks: int = 0,
    idmc_raw_path: str = "/eos/jeodpp/home/users/mihadar/data/IDMC/idmc_displacements_raw.json",
    predicate: str = "within",
) -> Tuple[Dict[str, Any], pd.DataFrame]:

    if matched_polys is None or matched_polys.empty:
        raise ValueError("matched_polys is empty — no admin regions to filter within.")

    anchor_dt = pd.to_datetime(event_start_date)
    lookback_start_dt = anchor_dt - pd.DateOffset(
        years=past_look_years,
        months=past_look_months,
        weeks=past_look_weeks
    )

    def is_in_range(date_str: Any) -> bool:
        if not date_str or date_str == "x":
            return False
        try:
            dt = pd.to_datetime(date_str)
            return lookback_start_dt <= dt <= anchor_dt
        except Exception:
            return False

    with open(idmc_raw_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    if not isinstance(raw_data, list):
        raise ValueError("Expected idmc_displacements_raw.json to be a LIST of events.")

    candidates = []
    for entry in raw_data:
        if entry.get("iso3") != iso3:
            continue

        disp_start = entry.get("displacement_start_date")
        disp_end = entry.get("displacement_end_date")

        if not (is_in_range(disp_start) or is_in_range(disp_end)):
            continue

        lat = entry.get("latitude")
        lon = entry.get("longitude")
        if lat is None or lon is None:
            continue

        try:
            lat = float(lat)
            lon = float(lon)
        except (TypeError, ValueError):
            continue

        candidates.append({"_entry": entry, "lat": lat, "lon": lon})

    if not candidates:
        out = {
            "meta": {
                "iso3": iso3,
                "start_date": str(lookback_start_dt.date()),
                "end_date": str(anchor_dt.date()),
                "predicate": predicate,
                "n_events_time_filtered": 0,
                "n_events_with_coords": 0,
                "n_events_inside_regions": 0,
            },
            "events": []
        }
        Path(out_json_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        return out, pd.DataFrame(columns=["polygon_id", "n_events"])

    pts_df = pd.DataFrame(candidates)
    pts_gdf = gpd.GeoDataFrame(
        pts_df,
        geometry=gpd.points_from_xy(pts_df["lon"], pts_df["lat"]),
        crs="EPSG:4326"
    )

    # ---- polygons CRS ----
    polys = matched_polys.copy()

    # ✅ FIX: avoid 'cannot insert GID_2' during sjoin
    polys = polys.reset_index(drop=True)
    polys.index.name = None

    if polys.crs is None:
        polys = polys.set_crs(epsg=4326)
    elif polys.crs.to_epsg() != 4326:
        polys = polys.to_crs(epsg=4326)

    poly_id_col = None
    for c in ["UID", "GID_5", "GID_4", "GID_3", "GID_2", "GID_1"]:
        if c in polys.columns:
            poly_id_col = c
            break

    polys_for_join = polys[[poly_id_col, "geometry"]].copy() if poly_id_col else polys[["geometry"]].copy()

    joined = gpd.sjoin(pts_gdf, polys_for_join, how="inner", predicate=predicate)

    matched_entries = []
    for _, r in joined.iterrows():
        e = r["_entry"]
        if poly_id_col:
            e = dict(e)
            e["_matched_polygon_id"] = r.get(poly_id_col)
        matched_entries.append(e)

    def _event_key(e: dict):
        for k in ["id", "event_id", "record_id"]:
            if k in e:
                return (k, e[k])
        return ("_fallback", json.dumps(e, sort_keys=True, ensure_ascii=False))

    uniq = {}
    for e in matched_entries:
        uniq[_event_key(e)] = e
    matched_entries = list(uniq.values())

    if poly_id_col:
        counts_df = (
            pd.DataFrame(matched_entries)
            .dropna(subset=["_matched_polygon_id"])
            .groupby("_matched_polygon_id")
            .size()
            .reset_index(name="n_events")
            .rename(columns={"_matched_polygon_id": "polygon_id"})
            .sort_values("n_events", ascending=False)
        )
    else:
        counts_df = pd.DataFrame(columns=["polygon_id", "n_events"])

    out = {
        "meta": {
            "iso3": iso3,
            "start_date": str(lookback_start_dt.date()),
            "end_date": str(anchor_dt.date()),
            "predicate": predicate,
            "n_events_time_filtered": len(candidates),
            "n_events_with_coords": len(candidates),
            "n_events_inside_regions": len(matched_entries),
            "polygon_id_col": poly_id_col,
        },
        "events": matched_entries
    }

    Path(out_json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    return out, counts_df


## extract 
def extract_lat_lon_figure_from_idmc_json(idmc_json_path, dedupe_on="id"):
    """
    Extracts lat/lon/figure from IDMC JSON structure like:
      {
        "meta": {...},
        "events": [
          {"id": ..., "latitude": ..., "longitude": ..., "figure": ..., ...},
          ...
        ]
      }

    Parameters
    ----------
    idmc_json_path : str
    dedupe_on : str or None
        If provided and that key exists in events (e.g. "id"), duplicates are dropped.

    Returns
    -------
    list[dict]: [{"lat": float, "lon": float, "figure": float, "event_id": any}, ...]
    """
    with open(idmc_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    events = data.get("events", []) or []

    rows = []
    seen = set()

    for ev in events:
        lat = ev.get("latitude", None)
        lon = ev.get("longitude", None)
        fig = ev.get("figure", 0)

        event_id = ev.get(dedupe_on) if dedupe_on else None
        if dedupe_on and event_id is not None:
            if event_id in seen:
                continue
            seen.add(event_id)

        try:
            lat = float(lat)
            lon = float(lon)
        except (TypeError, ValueError):
            continue

        try:
            figure = float(fig) if fig is not None else 0.0
        except (TypeError, ValueError):
            figure = 0.0

        rows.append({"lat": lat, "lon": lon, "figure": figure, "event_id": event_id})

    return rows