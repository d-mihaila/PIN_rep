import pandas as pd
import json
from collections import defaultdict
from datetime import datetime
from tabulate import tabulate
from pathlib import Path
import math
import os
from geopy.distance import geodesic
import csv
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import geopandas as gpd
import math
import matplotlib.pyplot as plt


def post_process_acled_events(filtered_events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Post-process ACLED events after time-window filtering (and before spatial join):

      1) Drop events with fatalities == 0
      2) Drop events with time_precision < 0.8 OR geo_precision < 1
      3) Deduplicate by event_id_cnty, keeping the event with the highest timestamp

    Returns: cleaned list of event dicts
    """

    def _to_float(x, default=None):
        if x is None or x == "Nui":
            return default
        try:
            return float(x)
        except (TypeError, ValueError):
            return default

    # ---- 1) fatalities filter (strict) ----
    kept = []
    for e in filtered_events:
        fatalities = _to_float(e.get("fatalities"), default=0.0)
        if fatalities == 0:
            continue
        kept.append(e)

    # ---- 2) precision filters ----
    kept2 = []
    for e in kept:
        time_prec = _to_float(e.get("time_precision"), default=None)
        geo_prec  = _to_float(e.get("geo_precision"), default=None)

        # if missing precision, treat as failing (discard)
        if time_prec is None or geo_prec is None:
            continue

        if time_prec < 0.8 or geo_prec < 1.0:
            continue

        kept2.append(e)

    # ---- 3) deduplicate by event_id_cnty (keep max timestamp) ----
    best_by_id: Dict[str, Dict[str, Any]] = {}
    no_id_events: List[Dict[str, Any]] = []

    for e in kept2:
        eid = e.get("event_id_cnty")
        if not eid or eid == "Nui":
            # Can't dedup reliably without id; keep all of these
            no_id_events.append(e)
            continue

        ts = _to_float(e.get("timestamp"), default=-math.inf)

        if eid not in best_by_id:
            best_by_id[eid] = e
        else:
            prev_ts = _to_float(best_by_id[eid].get("timestamp"), default=-math.inf)
            if ts > prev_ts:
                best_by_id[eid] = e

    return list(best_by_id.values()) + no_id_events

def filter_acled_events_by_admin(
    summary_df: pd.DataFrame,
    matched_poly: gpd.GeoDataFrame,
    acled_json_path: str,
    out_json_path: str,
    *,
    iso3: str,
    start_date: str,
    past_look_years: int = 0,
    past_look_months: int = 0,
    past_look_weeks: int = 0,
    # ✅ defaults that match your SDN.json
    date_field: str = "event_date",
    lat_field: str = "latitude",
    lon_field: str = "longitude",
    # (kept for backwards compatibility; ignored if lat_field/lon_field exist)
    coord_path: Tuple[str, str] = ("location", "coordinates"),
    polygon_join_key: Optional[str] = None,
    predicate: str = "within",
) -> Tuple[Dict[str, Any], pd.DataFrame, str]:
    """
    Filter ACLED events within the matched admin polygons (GADM), per query.

    Always returns 3 values:
      (output_dict, counts_df, out_json_path)
    """

    if summary_df is None or summary_df.empty:
        raise ValueError("summary_df is empty (no matched locations).")
    if matched_poly is None or matched_poly.empty:
        raise ValueError("matched_poly is empty (no matched polygons).")

    # --- time window ---
    anchor_dt = pd.to_datetime(start_date)
    lookback_start_dt = anchor_dt - pd.DateOffset(
        years=past_look_years, months=past_look_months, weeks=past_look_weeks
    )

    # --- polygon join key (ADMIN-2 default) ---
    polygon_join_key = polygon_join_key or "GID_2"

    if polygon_join_key not in summary_df.columns:
        raise ValueError(f"{polygon_join_key} not found in summary_df")

    if polygon_join_key not in matched_poly.columns:
        raise ValueError(f"{polygon_join_key} not found in matched_poly")


    # --- CRS: ACLED coords are EPSG:4326 ---
    polys = matched_poly.copy()

    polys = polys.reset_index(drop=True)
    polys.index.name = None

    if polys.crs is None:
        polys = polys.set_crs(epsg=4326)
    elif polys.crs.to_epsg() != 4326:
        polys = polys.to_crs(epsg=4326)

    # --- load ACLED ---
    with open(acled_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Your current JSON format: {"events": [...]}
    events = data.get("events", [])
    if not events:
        out = {
            "meta": {
                "iso3": iso3,
                "start_date": str(lookback_start_dt.date()),
                "end_date": str(anchor_dt.date()),
                "n_events_considered": 0,
                "note": "No events found in input ACLED file under key 'events'."
            },
            "per_query": {}
        }
        Path(out_json_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        return out, pd.DataFrame(columns=["query", "n_polygons", "n_events"]), out_json_path

    # --- filter events by time window ---
    filtered_events = []
    for e in events:
        raw_date = e.get(date_field)
        if not raw_date or raw_date == "Nui":
            continue
        dt = pd.to_datetime(raw_date, errors="coerce")
        if pd.isna(dt):
            continue
        if dt > anchor_dt or dt < lookback_start_dt:
            continue
        filtered_events.append(e)
        
    print('events found in the Admin2 regions: ', len(filtered_events))
    # put postprocessing 
    filtered_events = post_process_acled_events(filtered_events)
    print(f'out of which {len(filtered_events)} had fatalities')

    # --- build points geodataframe ---
    ev_rows = []
    for idx, e in enumerate(filtered_events):
        # Preferred: explicit lat/lon fields
        ev_lat = e.get(lat_field)
        ev_lon = e.get(lon_field)

        # Fallback: nested coord_path if lat/lon not present
        if (ev_lat is None or ev_lon is None) or (ev_lat == "Nui" or ev_lon == "Nui"):
            coords = None
            try:
                coords = e.get(coord_path[0], {}).get(coord_path[1])
            except Exception:
                coords = None
            if isinstance(coords, list) and len(coords) == 2:
                ev_lat, ev_lon = coords[0], coords[1]

        if ev_lat in (None, "Nui") or ev_lon in (None, "Nui"):
            continue

        try:
            ev_lat = float(ev_lat)
            ev_lon = float(ev_lon)
        except (TypeError, ValueError):
            continue

        ev_rows.append({
            "_event_idx": idx,
            "event_date": e.get(date_field),
            "lat": ev_lat,
            "lon": ev_lon,
            "_event_obj": e,
        })

    if not ev_rows:
        out = {
            "meta": {
                "iso3": iso3,
                "start_date": str(lookback_start_dt.date()),
                "end_date": str(anchor_dt.date()),
                "n_events_considered": len(filtered_events),
                "n_events_with_coords": 0,
                "note": "No events had usable coordinates in the time window (check lat/lon fields)."
            },
            "per_query": {}
        }
        Path(out_json_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        return out, pd.DataFrame(columns=["query", "n_polygons", "n_events"]), out_json_path

    ev_df = pd.DataFrame(ev_rows)
    ev_gdf = gpd.GeoDataFrame(
        ev_df,
        geometry=gpd.points_from_xy(ev_df["lon"], ev_df["lat"]),
        crs="EPSG:4326"
    )

    # --- spatial join: event point -> polygon ---
    joined = gpd.sjoin(
        ev_gdf,
        polys[[polygon_join_key, "geometry"]],
        how="inner",
        predicate=predicate
    )

    # --- polygon id -> events ---
    poly_to_events: Dict[Any, List[dict]] = {}
    for _, r in joined.iterrows():
        pid = r[polygon_join_key]
        poly_to_events.setdefault(pid, []).append(r["_event_obj"])

    # --- per-query grouping ---
    per_query = {}
    counts = []

    def _event_key(e: dict):
        for k in ["event_id_cnty", "event_id_no_cnty", "data_id", "id"]:
            if k in e:
                return (k, e[k])
        return ("_fallback", json.dumps(e, sort_keys=True, ensure_ascii=False))

    for query, grp in summary_df.groupby("query"):
        poly_ids = grp[polygon_join_key].dropna().unique().tolist()

        events_for_query = []
        for pid in poly_ids:
            events_for_query.extend(poly_to_events.get(pid, []))

        # de-dup
        uniq = {}
        for e in events_for_query:
            uniq[_event_key(e)] = e
        events_for_query = list(uniq.values())

        per_query[query] = {
            "n_polygons": len(poly_ids),
            "polygons": poly_ids,
            "events": events_for_query
        }
        counts.append({"query": query, "n_polygons": len(poly_ids), "n_events": len(events_for_query)})

    out = {
        "meta": {
            "iso3": iso3,
            "start_date": str(lookback_start_dt.date()),
            "end_date": str(anchor_dt.date()),
            "predicate": predicate,
            "polygon_join_key": polygon_join_key,
            "n_events_total_in_file": len(events),
            "n_events_in_time_window": len(filtered_events),
            "n_events_with_coords": len(ev_rows),
            "n_events_inside_any_polygon": int(len(joined)),
        },
        "per_query": per_query
    }

    Path(out_json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    counts_df = pd.DataFrame(counts).sort_values("n_events", ascending=False)

    return out, counts_df, out_json_path


def extract_lat_lon_fatalities_from_acled(acled_json_path):
    """
    Extract lat/lon/fatalities from ACLED filtered JSON:
      {"meta": {...}, "per_query": {"<query>": {"events": [...]}, ...}}

    Returns:
        list[dict]: [{"lat": float, "lon": float, "fatalities": float, "event_id": str|None}, ...]
    """
    with open(acled_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    per_query = data.get("per_query", {}) or {}

    rows = []
    seen_ids = set()

    for _, block in per_query.items():
        events = (block or {}).get("events", []) or []
        # print('number of events is', len(events))
        for ev in events:
            lat = ev.get("latitude", None)
            lon = ev.get("longitude", None)
            fat = ev.get("fatalities", 0)
            
            # print(lat, lon, fat)

            # ---- parse lat/lon FIRST ----
            if lat in (None, "Nui") or lon in (None, "Nui"):
                continue
            try:
                lat_f = float(lat)
                lon_f = float(lon)
            except (TypeError, ValueError):
                continue

            # ---- parse fatalities ----
            try:
                fatalities = float(fat) if fat not in (None, "Nui") else 0.0
            except (TypeError, ValueError):
                fatalities = 0.0

            # ---- NOW dedupe (after we know row is valid) ----
            event_id = ev.get("event_id_cnty", None)
            if event_id not in (None, "Nui"):
                if event_id in seen_ids:
                    continue
                seen_ids.add(event_id)

            rows.append({"lat": lat_f, "lon": lon_f, "fatalities": fatalities, "event_id": event_id})

    return rows



def plot_acled_6mo_windows_per_query(
    json_path: str,
    event_start_date,
    months_back: int = 6,
    query_key: str = "per_query",
    figsize=(10, 6),
):
    """
    Plots rolling monthly ACLED summaries per query with:
      - identical y-axis scales across ALL plots
      - x-axis = window start date (eg: 15 October)

    Returns:
      summary_by_query : dict[str, pd.DataFrame]
    """

    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON path does not exist: {json_path}")

    start = pd.to_datetime(event_start_date).normalize()

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if query_key not in data or not isinstance(data[query_key], dict):
        raise ValueError(f"Expected dict at key '{query_key}'")

    per_query = data[query_key]

    # ----- Build rolling windows -----
    boundaries = [start + pd.DateOffset(months=-k) for k in range(months_back, -1, -1)]
    windows = [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]

    summary_by_query = {}

    # ----- FIRST PASS: compute summaries + global maxima -----

    global_max_events = 0
    global_max_fatalities = 0

    for query_name, payload in per_query.items():
        events = payload.get("events", [])

        if not events:
            print(f"[ACLED] Query '{query_name}' has no events in JSON. Skipping.")
            continue

        df = pd.DataFrame(events)

        if df.empty or "event_date" not in df.columns:
            print(f"[ACLED] Query '{query_name}' has no usable event_date. Skipping.")
            continue

        df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
        df["fatalities"] = pd.to_numeric(df.get("fatalities", 0), errors="coerce").fillna(0)
        df = df.dropna(subset=["event_date"])

        rows = []

        for w_start, w_end in windows:

            if w_end == start:
                mask = (df["event_date"] >= w_start) & (df["event_date"] <= w_end)
            else:
                mask = (df["event_date"] >= w_start) & (df["event_date"] < w_end)

            sub = df.loc[mask]

            n_events = int(len(sub))
            fatalities = float(sub["fatalities"].sum()) if len(sub) else 0.0

            rows.append(
                {
                    "window_start": w_start,
                    "window_end": w_end,
                    "n_events": n_events,
                    "fatalities": fatalities,
                }
            )

        summary = pd.DataFrame(rows)

        if summary["n_events"].sum() == 0:
            print(f"[ACLED] Query '{query_name}' has no events in last {months_back} windows. Skipping.")
            continue

        # Update global maxima
        global_max_events = max(global_max_events, summary["n_events"].max())
        global_max_fatalities = max(global_max_fatalities, summary["fatalities"].max())

        summary_by_query[query_name] = summary

    if not summary_by_query:
        print("No queries contained usable data — nothing to plot.")
        return {}

    print("\nGLOBAL SCALE INFO")
    print("-----------------")
    print(f"Max events in any window: {global_max_events}")
    print(f"Max fatalities in any window: {global_max_fatalities}\n")

    # ----- SECOND PASS: plotting with fixed scales -----

    for query_name, summary in summary_by_query.items():

        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

        x = range(len(summary))

        xticklabels = summary["window_start"].dt.strftime("%d %B")

        # Events plot
        axes[0].plot(x, summary["n_events"], marker="o")
        axes[0].set_ylabel("# Events")
        axes[0].set_ylim(0, global_max_events)
        axes[0].set_title(f"ACLED — last {months_back} months — {query_name}")

        # Fatalities plot
        axes[1].plot(x, summary["fatalities"], marker="o")
        axes[1].set_ylabel("# Fatalities")
        axes[1].set_xlabel("Window start date")
        axes[1].set_ylim(0, global_max_fatalities)

        axes[1].set_xticks(list(x))
        axes[1].set_xticklabels(xticklabels, rotation=45, ha="right")

        plt.tight_layout()
        plt.show()

    return summary_by_query
