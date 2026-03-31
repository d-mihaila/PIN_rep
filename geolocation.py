# file to work with acled 
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

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import geopandas as gpd
from shapely.geometry import Point

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.lines as mlines
import matplotlib.dates as mdates
import matplotlib.ticker as mticker


path_full_acled = "/eos/jeodpp/home/users/mihadar/data/ACLED/ACLED_events_by_country.json"
path_matched_events = "/eos/jeodpp/home/users/mihadar/data/ACLED/matched_events.json"


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
    date_field: str = "event_date",
    lat_field: str = "latitude",
    lon_field: str = "longitude",
    fatalities_field: str = "fatalities",
    population_best_field: str = "population_best",
    coord_path: Tuple[str, str] = ("location", "coordinates"),
    polygon_join_key: Optional[str] = None,
    predicate: str = "within",
    debug: bool = True,          # ← flip to False once happy
) -> Tuple[Dict[str, Any], pd.DataFrame, str]:
    """
    Filter ACLED events within the matched admin polygons (GADM), per query.

    Returns (output_dict, counts_df, out_json_path).

    counts_df columns: query | n_polygons | n_events | n_with_population_best
    
    per_query entries:
      - n_polygons, polygons, events          (unchanged — pipeline compat)
      - event_records: list of dicts, one per matched event:
            {event_date, lat, lon, fatalities, population_best}
        population_best is None when the field is missing/null in the source.
    """

    if summary_df is None or summary_df.empty:
        raise ValueError("summary_df is empty (no matched locations).")
    if matched_poly is None or matched_poly.empty:
        raise ValueError("matched_poly is empty (no matched polygons).")

    # --- time window ---
    anchor_dt         = pd.to_datetime(start_date)
    lookback_start_dt = anchor_dt - pd.DateOffset(
        years=past_look_years, months=past_look_months, weeks=past_look_weeks
    )

    # --- pick join key ---
    if polygon_join_key is None:
        for k in ["UID", "GID_5", "GID_4", "GID_3", "GID_2", "GID_1", "GID_0"]:
            if k in summary_df.columns and k in matched_poly.columns:
                polygon_join_key = k
                break
    if polygon_join_key is None:
        raise ValueError(
            "Could not find a common join key between summary_df and matched_poly (UID or GID_*)."
        )

    # --- CRS ---
    polys = matched_poly.copy()
    if polys.crs is None:
        polys = polys.set_crs(epsg=4326)
    elif polys.crs.to_epsg() != 4326:
        polys = polys.to_crs(epsg=4326)

    # --- load ACLED ---
    with open(acled_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        events = data.get("events") or list(data.values())
        if events and isinstance(events[0], list):
            events = [e for sublist in events for e in sublist]
    elif isinstance(data, list):
        events = data
    else:
        raise ValueError("Unexpected ACLED JSON structure.")

    # ── DEBUG 1: what does a raw event actually look like? ──────────────────
    if debug and events:
        sample = events[0]
        print("\n[DEBUG] First raw event keys:", list(sample.keys()))
        print(f"[DEBUG] '{population_best_field}' value in first event:",
              repr(sample.get(population_best_field)))
        print(f"[DEBUG] '{fatalities_field}' value in first event:",
              repr(sample.get(fatalities_field)))
        print(f"[DEBUG] Total events in file: {len(events)}")
    # ────────────────────────────────────────────────────────────────────────

    _EMPTY_COUNTS = pd.DataFrame(
        columns=["query", "n_polygons", "n_events", "n_with_population_best"]
    )

    if not events:
        out = {"meta": {"iso3": iso3, "start_date": str(lookback_start_dt.date()),
                        "end_date": str(anchor_dt.date()), "n_events_considered": 0,
                        "note": "No events found in input ACLED file."}, "per_query": {}}
        Path(out_json_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        return out, _EMPTY_COUNTS, out_json_path

    # --- filter by time window ---
    filtered_events = []
    for e in events:
        raw_date = e.get(date_field)
        if not raw_date or raw_date == "Nui":
            continue
        dt = pd.to_datetime(raw_date, errors="coerce")
        if pd.isna(dt) or dt > anchor_dt or dt < lookback_start_dt:
            continue
        filtered_events.append(e)

    # ── DEBUG 2: how many survived the time filter? ──────────────────────────
    if debug:
        print(f"\n[DEBUG] Time window: {lookback_start_dt.date()} → {anchor_dt.date()}")
        print(f"[DEBUG] Events after time filter: {len(filtered_events)}")
        if filtered_events:
            # check population_best coverage in the time-filtered set
            n_pop = sum(
                1 for e in filtered_events
                if e.get(population_best_field) not in (None, "Nui", "", "null")
            )
            print(f"[DEBUG] Of those, events with non-null '{population_best_field}': {n_pop}")
            # show a few raw values so you can see the actual content
            pop_samples = [
                e.get(population_best_field) for e in filtered_events[:10]
            ]
            print(f"[DEBUG] First 10 '{population_best_field}' raw values: {pop_samples}")
    # ────────────────────────────────────────────────────────────────────────

    # --- build points GeoDataFrame ---
    ev_rows = []
    for idx, e in enumerate(filtered_events):
        ev_lat = e.get(lat_field)
        ev_lon = e.get(lon_field)

        if ev_lat is None or ev_lon is None or ev_lat == "Nui" or ev_lon == "Nui":
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

        # fatalities
        raw_fat = e.get(fatalities_field)
        try:
            ev_fatalities = float(raw_fat) if raw_fat not in (None, "Nui", "", "null") else 0.0
        except (TypeError, ValueError):
            ev_fatalities = 0.0

        # population_best — keep None honestly if missing
        raw_pop = e.get(population_best_field)
        try:
            ev_pop = float(raw_pop) if raw_pop not in (None, "Nui", "", "null") else None
        except (TypeError, ValueError):
            ev_pop = None

        ev_rows.append({
            "_event_idx": idx,
            "event_date":      e.get(date_field),
            "lat":             ev_lat,
            "lon":             ev_lon,
            "fatalities":      ev_fatalities,
            "population_best": ev_pop,
            "_event_obj":      e,
        })

    # ── DEBUG 3: coord + pop coverage after coord parsing ───────────────────
    if debug:
        print(f"\n[DEBUG] Events with usable coords: {len(ev_rows)}")
        n_pop_rows = sum(1 for r in ev_rows if r["population_best"] is not None)
        print(f"[DEBUG] Of those, with non-null population_best: {n_pop_rows}")
        if ev_rows:
            print(f"[DEBUG] Sample ev_rows[0]: event_date={ev_rows[0]['event_date']}, "
                  f"lat={ev_rows[0]['lat']}, lon={ev_rows[0]['lon']}, "
                  f"fatalities={ev_rows[0]['fatalities']}, "
                  f"population_best={ev_rows[0]['population_best']}")
    # ────────────────────────────────────────────────────────────────────────

    if not ev_rows:
        out = {"meta": {"iso3": iso3, "start_date": str(lookback_start_dt.date()),
                        "end_date": str(anchor_dt.date()),
                        "n_events_in_time_window": len(filtered_events),
                        "n_events_with_coords": 0,
                        "note": "No events had usable coordinates in the time window."},
               "per_query": {}}
        Path(out_json_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        return out, _EMPTY_COUNTS, out_json_path

    ev_df = pd.DataFrame(ev_rows)
    ev_gdf = gpd.GeoDataFrame(
        ev_df,
        geometry=gpd.points_from_xy(ev_df["lon"], ev_df["lat"]),
        crs="EPSG:4326",
    )

    # --- spatial join ---
    joined = gpd.sjoin(
        ev_gdf,
        polys[[polygon_join_key, "geometry"]],
        how="inner",
        predicate=predicate,
    )

    # ── DEBUG 4: spatial join result ─────────────────────────────────────────
    if debug:
        print(f"\n[DEBUG] Events inside matched polygons (after sjoin): {len(joined)}")
        if not joined.empty:
            n_pop_joined = joined["population_best"].notna().sum()
            print(f"[DEBUG] Of joined events, with non-null population_best: {n_pop_joined}")
            print(f"[DEBUG] joined population_best sample (first 10):\n",
                  joined["population_best"].head(10).tolist())
    # ────────────────────────────────────────────────────────────────────────

    # polygon id -> list of row dicts (already parsed, no need to re-read _event_obj)
    poly_to_records: Dict[Any, List[dict]] = {}
    for _, r in joined.iterrows():
        pid = r[polygon_join_key]
        poly_to_records.setdefault(pid, []).append({
            "event_date":      r["event_date"],
            "lat":             r["lat"],
            "lon":             r["lon"],
            "fatalities":      r["fatalities"],
            "population_best": r["population_best"] if pd.notna(r["population_best"]) else None,
        })

    # also keep _event_obj for the raw "events" list (pipeline compat)
    poly_to_events: Dict[Any, List[dict]] = {}
    for _, r in joined.iterrows():
        pid = r[polygon_join_key]
        poly_to_events.setdefault(pid, []).append(r["_event_obj"])

    def _event_key(e: dict):
        for k in ["event_id_cnty", "event_id_no_cnty", "data_id", "id"]:
            if k in e:
                return (k, e[k])
        return ("_fallback", json.dumps(e, sort_keys=True, ensure_ascii=False))

    per_query: Dict[str, Any] = {}
    counts = []

    for query, grp in summary_df.groupby("query"):
        poly_ids = grp[polygon_join_key].dropna().unique().tolist()

        # collect & de-dup raw event objects (pipeline compat)
        raw_events: Dict[tuple, dict] = {}
        for pid in poly_ids:
            for e in poly_to_events.get(pid, []):
                raw_events[_event_key(e)] = e

        # collect per-event records (de-dup by same key)
        seen_keys: set = set()
        event_records = []
        for pid in poly_ids:
            for rec in poly_to_records.get(pid, []):
                # build a dedup key from date+lat+lon (records don't carry event id)
                rkey = (rec["event_date"], rec["lat"], rec["lon"])
                if rkey not in seen_keys:
                    seen_keys.add(rkey)
                    event_records.append(rec)

        # sort by date
        event_records.sort(key=lambda x: x["event_date"] or "")

        n_with_pop = sum(1 for r in event_records if r["population_best"] is not None)

        per_query[query] = {
            # unchanged keys
            "n_polygons":    len(poly_ids),
            "polygons":      poly_ids,
            "events":        list(raw_events.values()),
            # new: one record per matched event with the fields you care about
            "event_records": event_records,   # [{event_date, lat, lon, fatalities, population_best}]
        }
        counts.append({
            "query":                query,
            "n_polygons":           len(poly_ids),
            "n_events":             len(event_records),
            "n_with_population_best": n_with_pop,
        })

    out = {
        "meta": {
            "iso3": iso3,
            "start_date":                  str(lookback_start_dt.date()),
            "end_date":                    str(anchor_dt.date()),
            "predicate":                   predicate,
            "polygon_join_key":            polygon_join_key,
            "n_events_total_in_file":      len(events),
            "n_events_in_time_window":     len(filtered_events),
            "n_events_with_coords":        len(ev_rows),
            "n_events_inside_any_polygon": int(len(joined)),
        },
        "per_query": per_query,
    }

    Path(out_json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    counts_df = (
        pd.DataFrame(counts)
        .sort_values("n_events", ascending=False)
        .reset_index(drop=True)
    )

    return out, counts_df, out_json_path


def plot_acled_events_map(
    acled_out: dict,
    matched_poly: gpd.GeoDataFrame,
    gpkg_path: str,
    iso3: str,
    title: str = "",
    min_marker_size: float = 10,
    max_marker_size: float = 300,
) -> plt.Figure:

    # ── 1. flatten all event_records across queries ──────────────────────────
    all_records = []
    for query, qdata in acled_out.get("per_query", {}).items():
        for rec in qdata.get("event_records", []):
            all_records.append(rec)

    if not all_records:
        raise ValueError("No event_records found in acled_out — nothing to plot.")

    rec_df = pd.DataFrame(all_records).drop_duplicates(subset=["event_date", "lat", "lon"])
    rec_df["fatalities"]      = pd.to_numeric(rec_df["fatalities"],      errors="coerce").fillna(0)
    rec_df["population_best"] = pd.to_numeric(rec_df["population_best"], errors="coerce")

    print(f"[MAP] Total unique events to plot: {len(rec_df)}")
    print(f"[MAP] Events with population_best: {rec_df['population_best'].notna().sum()}")
    print(f"[MAP] Fatalities range: {rec_df['fatalities'].min()} – {rec_df['fatalities'].max()}")
    print(f"[MAP] population_best range: {rec_df['population_best'].min()} – {rec_df['population_best'].max()}")

    # ── 2. load + dissolve admin-2 polygons ──────────────────────────────────
    gdf         = gpd.read_file(gpkg_path, layer="gadm_410")
    country_gdf = gdf[gdf["GID_0"] == iso3]
    if country_gdf.empty:
        raise ValueError(f"No GADM data for {iso3}")
    admin2_gdf = country_gdf.dissolve(by="GID_2", as_index=False)
    if admin2_gdf.crs is None:
        admin2_gdf = admin2_gdf.set_crs(epsg=4326)
    elif admin2_gdf.crs.to_epsg() != 4326:
        admin2_gdf = admin2_gdf.to_crs(epsg=4326)

    # ── 3. sum population_best per admin-2 polygon ───────────────────────────
    ev_gdf = gpd.GeoDataFrame(
        rec_df,
        geometry=gpd.points_from_xy(rec_df["lon"], rec_df["lat"]),
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(
        admin2_gdf[["GID_2", "geometry"]],
        ev_gdf[["population_best", "geometry"]],
        how="left",
        predicate="contains",
    )
    pop_by_admin2 = (
        joined.groupby("GID_2")["population_best"]
        .sum(min_count=1)
        .rename("pop_best_sum")
    )
    admin2_gdf = admin2_gdf.join(pop_by_admin2, on="GID_2")

    print(f"[MAP] Admin-2 regions with pop_best_sum > 0: "
          f"{(admin2_gdf['pop_best_sum'] > 0).sum()} / {len(admin2_gdf)}")

    # ── 4. CRS alignment for matched_poly ────────────────────────────────────
    polys = matched_poly.copy()
    if polys.crs is None:
        polys = polys.set_crs(epsg=4326)
    elif polys.crs.to_epsg() != 4326:
        polys = polys.to_crs(epsg=4326)

    # ── 5. marker size scaled to fatalities ──────────────────────────────────
    fat = rec_df["fatalities"].values
    fat_min, fat_max = fat.min(), fat.max()
    if fat_max > fat_min:
        sizes = min_marker_size + (fat - fat_min) / (fat_max - fat_min) * (max_marker_size - min_marker_size)
    else:
        sizes = np.full(len(fat), (min_marker_size + max_marker_size) / 2)

    # ── 6. choropleth colormap ────────────────────────────────────────────────
    pop_vals = admin2_gdf["pop_best_sum"].dropna()
    if pop_vals.empty or pop_vals.max() == 0:
        print("[MAP] WARNING: all population_best values are null/zero — choropleth will be flat.")
        norm_choro = mcolors.Normalize(vmin=0, vmax=1)
    else:
        norm_choro = mcolors.LogNorm(
            vmin=max(1, pop_vals.min()),
            vmax=pop_vals.max(),
        )
    cmap_choro = cm.get_cmap("YlOrRd")

    # ── 7. plot ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 12))

    admin2_gdf.plot(
        column="pop_best_sum",
        cmap=cmap_choro,
        norm=norm_choro,
        ax=ax,
        linewidth=0.3,
        edgecolor="gray",
        missing_kwds={"color": "whitesmoke", "label": "No data"},
        legend=False,
    )

    polys.boundary.plot(ax=ax, linewidth=1.8, edgecolor="steelblue", zorder=3)

    ax.scatter(
        rec_df["lon"],
        rec_df["lat"],
        s=sizes,
        color="steelblue",
        edgecolors="black",
        linewidths=0.4,
        alpha=0.85,
        zorder=5,
    )

    # ── colorbar with explicit readable ticks ────────────────────────────────
    sm_choro = cm.ScalarMappable(cmap=cmap_choro, norm=norm_choro)
    sm_choro.set_array([])
    cbar_choro = fig.colorbar(sm_choro, ax=ax, shrink=0.5, pad=0.01, aspect=30)
    cbar_choro.set_label("Population Exposed (sum per Admin-2)", fontsize=11)

    if not pop_vals.empty and pop_vals.max() > 0:
        vmin_cb   = max(1, pop_vals.min())
        vmax_cb   = pop_vals.max()
        log_ticks = np.logspace(np.log10(vmin_cb), np.log10(vmax_cb), num=6)
        log_ticks = np.unique([round(t, -int(np.floor(np.log10(t)))) for t in log_ticks])
        cbar_choro.set_ticks(log_ticks)
        cbar_choro.set_ticklabels([f"{int(t):,}" for t in log_ticks])
        cbar_choro.ax.tick_params(labelsize=10)

    # ── dot size legend ───────────────────────────────────────────────────────
    if fat_max > fat_min:
        legend_vals  = np.linspace(fat_min, fat_max, 4).astype(int)
        legend_sizes = min_marker_size + (legend_vals - fat_min) / (fat_max - fat_min) * (max_marker_size - min_marker_size)
        legend_handles = [
            mlines.Line2D([], [], marker="o", color="w", markerfacecolor="steelblue",
                          markeredgecolor="black",
                          markersize=np.sqrt(s),
                          label=f"{v} fatalities")
            for v, s in zip(legend_vals, legend_sizes)
        ]
        ax.legend(handles=legend_handles, title="Dot size = fatalities",
                  loc="lower left", fontsize=9, framealpha=0.8)

    ax.set_title(
        f"{title}\nFatalities (dots) & Population Exposed by Admin-2 colormap",
        fontsize=14, fontweight="bold",
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal")
    ax.grid(True, linewidth=0.3, alpha=0.5)
    plt.tight_layout()

    return fig

def plot_acled_timeseries_2weeks(
    acled_out: dict,
    end_date: str,
    title: str = "",
) -> plt.Figure:
    """
    Time series of ACLED events in the last 2 weeks before end_date.
    - X axis: event date
    - Y axis: population_best (population exposed)
    - One color per admin-2 region (query)
    - Arrow pointing to the peak event per admin-2
    - Total sum of all per-admin-2 peaks printed on the plot

    Parameters
    ----------
    acled_out : dict returned by filter_acled_events_by_admin
    end_date  : str "YYYY-MM-DD" — the anchor date (2-week window ends here)
    title     : str — plot title
    """

    end_dt   = pd.to_datetime(end_date)
    start_dt = end_dt - pd.Timedelta(weeks=2)

    # ── 1. collect records per query, filtered to 2-week window ──────────────
    query_records: Dict[str, pd.DataFrame] = {}

    for query, qdata in acled_out.get("per_query", {}).items():
        rows = []
        for rec in qdata.get("event_records", []):
            dt = pd.to_datetime(rec.get("event_date"), errors="coerce")
            if pd.isna(dt):
                continue
            if dt < start_dt or dt > end_dt:
                continue
            pop = rec.get("population_best")
            try:
                pop = float(pop) if pop not in (None, "Nui", "", "null") else None
            except (TypeError, ValueError):
                pop = None
            if pop is None:
                continue
            rows.append({"date": dt, "population_best": pop})

        if rows:
            df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
            query_records[query] = df

    if not query_records:
        raise ValueError(
            f"No events with population_best found in the 2-week window "
            f"{start_dt.date()} → {end_dt.date()}"
        )

    print(f"[TIMESERIES] Window: {start_dt.date()} → {end_dt.date()}")
    print(f"[TIMESERIES] Admin-2 regions with data: {len(query_records)}")

    # ── 2. color palette — one color per query ────────────────────────────────
    n_queries  = len(query_records)
    cmap_lines = cm.get_cmap("tab20", n_queries)
    colors     = {q: cmap_lines(i) for i, q in enumerate(query_records)}

    # ── 3. plot ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 7))

    peak_per_admin: Dict[str, float] = {}

    for query, df in query_records.items():
        color = colors[query]

        # peak first — needed for scatter split
        peak_idx = df["population_best"].idxmax()
        peak_row = df.loc[peak_idx]
        peak_val = peak_row["population_best"]
        peak_dt  = peak_row["date"]
        peak_per_admin[query] = peak_val

        # regular events — circles
        non_peak = df.drop(index=peak_idx)
        ax.scatter(
            non_peak["date"], non_peak["population_best"],
            color=color, s=45, marker="o", zorder=4, alpha=0.85,
        )

        # peak event — square, larger
        ax.scatter(
            [peak_dt], [peak_val],
            color=color, s=160, marker="s", zorder=5,
            edgecolors="black", linewidths=0.8, alpha=1.0,
        )

        # arrow only if there are multiple events
        if len(df) > 1:
            ax.annotate(
                "",
                xy=(peak_dt, peak_val),
                xytext=(peak_dt, peak_val * 0.6),
                arrowprops=dict(
                    arrowstyle="->",
                    color=color,
                    lw=1.8,
                ),
                zorder=5,
            )

        # small label at the peak
        ax.annotate(
            query,
            xy=(peak_dt, peak_val),
            xytext=(4, 6),
            textcoords="offset points",
            fontsize=9,
            color=color,
            fontweight="bold",
            zorder=6,
        )

    # ── 4. total of peaks ─────────────────────────────────────────────────────
    total_peak_pop = sum(peak_per_admin.values())
    print(f"\n[TIMESERIES] Peak population exposed per admin-2:")
    for q, v in sorted(peak_per_admin.items(), key=lambda x: -x[1]):
        print(f"  {q:40s}: {v:>12,.0f}")
    print(f"  {'TOTAL (sum of peaks)':40s}: {total_peak_pop:>12,.0f}")

    # ── 5. legend ─────────────────────────────────────────────────────────────
    legend_handles = [
        mlines.Line2D([], [], color=colors[q], marker="s", markersize=9,
                      markeredgecolor="black", linewidth=0,
                      label=f"{q}  (peak: {peak_per_admin[q]:,.0f})")
        for q in query_records
    ]
    ax.legend(
        handles=legend_handles,
        title="Admin-2 region",
        fontsize=12,
        title_fontsize=13,
        framealpha=0.85,
        ncol=max(1, n_queries // 15),
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        borderaxespad=0,
    )

    # ── 6. axes formatting ────────────────────────────────────────────────────
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=35, ha="right", fontsize=13)

    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    ax.tick_params(axis="y", labelsize=13)
    ax.set_xlabel("Date", fontsize=15)
    ax.set_ylabel("Population Exposed", fontsize=15)
    ax.set_title(
        f"{title}\nTotal Population Exposed (sum of admin-2 peaks): {total_peak_pop:,.0f}",
        fontsize=16, fontweight="bold",
    )
    ax.grid(True, linewidth=0.3, alpha=0.5)
    
    fig.tight_layout()
    fig.subplots_adjust(right=0.75)
    
    return fig, total_peak_pop