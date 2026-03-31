import re
import unicodedata
from collections import defaultdict
from typing import List, Tuple, Optional
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point
from matplotlib import cm
import matplotlib.colors as mcolors
import os
import json

gpkg_path  = '/eos/jeodpp/home/users/mihadar/data/Geospacial/gadm_410.gpkg'

def _norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[’'`]", "", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def gadm_match_locations(
    gpkg_path: str,
    locations: List[str],
    *,
    country: str,
    layer: str = "gadm_410",
    country_field: str = "COUNTRY",
    include_varnames: bool = True,
    out_json: str = "/eos/jeodpp/home/users/mihadar/data/from_emm/location_matches.json",
) -> Tuple[pd.DataFrame, gpd.GeoDataFrame]:
    """
    Match `locations` against GADM names (NAME_1..NAME_5; optionally VARNAME_1..VARNAME_5)
    within `country`. Saves found matches to JSON and returns:
      - summary_df (no geometry)
      - matched_polygons_gdf (unique polygons with geometry for plotting)

    Prints only for not-found locations: "<location> could not be found"
    """

    # 1) Load only country polygons (fast)
    safe_country = country.replace("'", "''")
    try:
        gadm = gpd.read_file(gpkg_path, layer=layer, where=f"{country_field} = '{safe_country}'")
    except TypeError:
        gadm = gpd.read_file(gpkg_path, layer=layer)
        gadm = gadm[gadm[country_field] == country]

    # Ensure WGS84
    if gadm.crs is None:
        gadm = gadm.set_crs(epsg=4326)
    elif gadm.crs.to_epsg() != 4326:
        gadm = gadm.to_crs(epsg=4326)

    # 2) Build lookup: normalized name -> list of (row_index, matched_field)
    name_fields = [f"NAME_{i}" for i in range(1, 6) if f"NAME_{i}" in gadm.columns]
    var_fields = [f"VARNAME_{i}" for i in range(1, 6) if f"VARNAME_{i}" in gadm.columns] if include_varnames else []
    lookup_fields = name_fields + var_fields

    lookup = defaultdict(list)
    # iterate only over relevant columns to keep it light
    for idx, row in gadm[lookup_fields].iterrows():
        for col in lookup_fields:
            val = row[col]
            if pd.isna(val) or not str(val).strip():
                continue
            # VARNAME columns sometimes contain multiple names separated by | or commas
            parts = re.split(r"[|,;/]", str(val))
            for p in parts:
                key = _norm(p)
                if key:
                    lookup[key].append((idx, col))

    # 3) Match input locations
    keep_cols = [c for c in [
        "UID",
        "GID_0", "NAME_0",
        "GID_1", "NAME_1", "TYPE_1", "ENGTYPE_1",
        "GID_2", "NAME_2", "TYPE_2", "ENGTYPE_2",
        "GID_3", "NAME_3", "TYPE_3", "ENGTYPE_3",
        "GID_4", "NAME_4", "TYPE_4", "ENGTYPE_4",
        "GID_5", "NAME_5", "TYPE_5", "ENGTYPE_5",
        "REGION", "VARREGION", "COUNTRY", "CONTINENT", "SUBCONT"
    ] if c in gadm.columns]

    records = []
    matched_indices = set()

    for loc in locations:
        key = _norm(loc)
        hits = lookup.get(key, [])
        if not hits:
            print(f"{loc} could not be found")
            continue

        # If multiple rows match, keep them all
        for idx, matched_col in hits:
            matched_indices.add(idx)
            rec = {"query": loc, "matched_field": matched_col}

            # helpful: level number if it matched NAME_2 etc
            m = re.search(r"_(\d)$", matched_col)
            rec["matched_level"] = int(m.group(1)) if m else None

            # add parent chain / attributes
            for c in keep_cols:
                rec[c] = gadm.at[idx, c] if c in gadm.columns else None

            records.append(rec)

    # 4) Output artifacts
    summary_df = pd.DataFrame(records)

    # Save only FOUND rows
    summary_df.to_json(out_json, orient="records", force_ascii=False, indent=2)
    print('Found locations saved in', out_json)

    # Return matched polygons for plotting
    matched_polys = gadm.loc[sorted(matched_indices)].copy() if matched_indices else gadm.iloc[0:0].copy()

    return summary_df, matched_polys


# plot
def plot_gadm_matches_onefig(
    gpkg_path: str,
    matched_polys: gpd.GeoDataFrame,
    *,
    country: str = "Sudan",
    layer: str = "gadm_410",
    country_field: str = "COUNTRY",
    points_df: pd.DataFrame | None = None,   # optional: df with latitude/longitude cols
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    title: str = "",
    base_alpha: float = 0.08,
    match_alpha: float = 0.18,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot full country boundary + highlight matched polygons in blue, in ONE figure.
    Optionally overlay point coordinates.

    matched_polys: output GeoDataFrame from gadm_match_locations() (polygons to highlight)
    points_df: optional dataframe containing lat/lon columns to plot points
    """

    # --- load full country map ---
    safe_country = country.replace("'", "''")
    try:
        country_gdf = gpd.read_file(gpkg_path, layer=layer, where=f"{country_field} = '{safe_country}'")
    except TypeError:
        country_gdf = gpd.read_file(gpkg_path, layer=layer)
        country_gdf = country_gdf[country_gdf[country_field] == country]

    # Ensure WGS84 everywhere
    if country_gdf.crs is None:
        country_gdf = country_gdf.set_crs(epsg=4326)
    elif country_gdf.crs.to_epsg() != 4326:
        country_gdf = country_gdf.to_crs(epsg=4326)

    if matched_polys is not None and not matched_polys.empty:
        if matched_polys.crs is None:
            matched_polys = matched_polys.set_crs(epsg=4326)
        elif matched_polys.crs.to_epsg() != 4326:
            matched_polys = matched_polys.to_crs(epsg=4326)

    # --- plot ---
    fig, ax = plt.subplots(figsize=(10, 10))

    # Base country polygons (light fill + thin boundary)
    country_gdf.plot(ax=ax, alpha=base_alpha, linewidth=0.4, edgecolor="gray")
    country_gdf.boundary.plot(ax=ax, linewidth=0.6, edgecolor="gray")

    # Highlight matched polygons (blue fill + blue boundary)
    if matched_polys is not None and not matched_polys.empty:
        matched_polys.plot(ax=ax, alpha=match_alpha, edgecolor="blue", linewidth=1.2)
        matched_polys.boundary.plot(ax=ax, edgecolor="blue", linewidth=1.4)

    # Optional: plot coordinate points (black dots)
    if points_df is not None and len(points_df) > 0 and lat_col in points_df.columns and lon_col in points_df.columns:
        pts = points_df.dropna(subset=[lat_col, lon_col]).copy()
        if len(pts) > 0:
            gpts = gpd.GeoDataFrame(
                pts,
                geometry=gpd.points_from_xy(pts[lon_col], pts[lat_col]),
                crs="EPSG:4326",
            )
            gpts.plot(ax=ax, markersize=20, alpha=0.9)  # default color; you asked blue only for regions

    ax.set_title(title or f"{country} — matched GADM regions")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal")
    # ax.grid(True)
    plt.tight_layout()

    return fig, ax



#### acled and idmc mapping 
def get_coordinates_from_acled_idmc(
    iso3,
    start_date: str,
    past_look_years: int = 0,
    past_look_months: int = 0,
    past_look_weeks: int = 0,
):
    """
    Extracts coordinates [lat, lon] from ACLED (time-filtered)
    and IDMC (unfiltered) data for the given ISO3 country code.

    Returns:
        tuple: (coordinates_acled, coordinates_idmc)
    """

    data_dir = '/eos/jeodpp/home/users/mihadar/data/'
    acled_path = os.path.join(data_dir, f'conflict context data/{iso3}/{iso3}_ACLED.json')
    idmc_path = os.path.join(data_dir, f'conflict context data/{iso3}/{iso3}_IDMC.json')

    coordinates_acled = []
    coordinates_idmc = []

    # -----------------
    # Time window setup
    # -----------------
    anchor_dt = pd.to_datetime(start_date)

    lookback_start_dt = anchor_dt - pd.DateOffset(
        years=past_look_years,
        months=past_look_months,
        weeks=past_look_weeks
    )

    # -------- ACLED --------
    if os.path.exists(acled_path):
        with open(acled_path, 'r', encoding='utf-8') as f:
            acled_data = json.load(f)

        events = acled_data.get("events", [])  # this matches your JSON structure

        for event in events:
            # ✅ correct key in your JSON
            event_date = event.get("event_date")
            if not event_date or event_date == "Nui":
                continue

            try:
                event_dt = pd.to_datetime(event_date, errors="coerce")
            except Exception:
                continue

            if pd.isna(event_dt):
                continue

            # Apply time filter
            if event_dt > anchor_dt:
                continue
            if event_dt < lookback_start_dt:
                continue

            # ✅ correct coordinate fields in your JSON
            lat = event.get("latitude")
            lon = event.get("longitude")

            if lat in (None, "Nui") or lon in (None, "Nui"):
                continue

            try:
                coordinates_acled.append([float(lat), float(lon)])
            except (TypeError, ValueError):
                continue
    else:
        print(f"ACLED file not found at {acled_path}")

    # -------- IDMC --------
    # (kept unchanged — no time filtering unless you want it)
    if os.path.exists(idmc_path):
        with open(idmc_path, 'r', encoding='utf-8') as f:
            idmc_data = json.load(f)

        for entry in idmc_data:
            if entry.get("iso3") != iso3:
                continue

            lat = entry.get("latitude")
            lon = entry.get("longitude")

            if lat is not None and lon is not None:
                try:
                    coordinates_idmc.append([float(lat), float(lon)])
                except (TypeError, ValueError):
                    continue

    else:
        print(f"IDMC file not found at {idmc_path}")

    return coordinates_acled, coordinates_idmc



# plot map with acled and idmc
def plot_admin2_with_coordinates(gpkg_path, iso3_code, coordinates, base_color, title="", matched_polys=None):
    """
    Plots two figures at ADMIN-2 level:
    1. Admin-2 map with gradient-colored points (based on order)
    2. Choropleth map of Admin-2 regions, colored by number of points inside

    - matched_polys (GeoDataFrame or None): polygons of interest; boundaries plotted in green on both figs
    """

    # Load GeoDataFrame
    gdf = gpd.read_file(gpkg_path, layer="gadm_410")

    # Filter to desired country
    country_gdf = gdf[gdf["GID_0"] == iso3_code]
    if country_gdf.empty:
        raise ValueError(f"No data found for country code: {iso3_code}")

    if "GID_2" not in country_gdf.columns:
        raise ValueError("GID_2 column not found. Your layer may not include Admin-2.")

    # Dissolve to Admin-2 (unique polygons per GID_2)
    admin2_gdf = country_gdf.dissolve(by="GID_2", as_index=False)

    # Convert coordinates into Points (lon, lat order!)
    point_geometries = []
    for lat, lon in coordinates:
        if lat is not None and lon is not None:
            point_geometries.append(Point(lon, lat))

    # Points are lat/lon by construction
    points_gdf = gpd.GeoDataFrame(geometry=point_geometries, crs="EPSG:4326")

    # Ensure points CRS matches admin2 CRS
    if admin2_gdf.crs is None:
        # If polygons CRS missing, assume WGS84 to match points (or set appropriately for your dataset)
        admin2_gdf = admin2_gdf.set_crs("EPSG:4326")
    if points_gdf.crs != admin2_gdf.crs:
        points_gdf = points_gdf.to_crs(admin2_gdf.crs)

    # Filter points within the country boundary (using admin2 union)
    points_within = points_gdf[points_gdf.geometry.within(admin2_gdf.unary_union)]

    # Ensure matched_polys is in same CRS if provided
    if matched_polys is not None and not matched_polys.empty:
        if matched_polys.crs is None:
            matched_polys = matched_polys.set_crs(admin2_gdf.crs)
        elif matched_polys.crs != admin2_gdf.crs:
            matched_polys = matched_polys.to_crs(admin2_gdf.crs)

    # --------------------
    # FIGURE 1: Points with Gradient Color (Admin-2 boundaries)
    # --------------------
    fig1, ax1 = plt.subplots(figsize=(10, 10))
    admin2_gdf.boundary.plot(ax=ax1, linewidth=0.4, edgecolor="gray")
    admin2_gdf.plot(ax=ax1, alpha=0.05)

    # Overlay matched polygons boundaries in green
    if matched_polys is not None and not matched_polys.empty:
        matched_polys.boundary.plot(ax=ax1, linewidth=1.6, edgecolor="green")

    n_points = len(points_within)
    if n_points > 0:
        cmap_pts = cm.get_cmap(base_color, n_points)
        for i, point in enumerate(points_within.geometry):
            ax1.plot(point.x, point.y, marker="o", markersize=2, color=cmap_pts(i / n_points), alpha=0.9)

    ax1.set_title(f"{title} - Point Density (Gradient) [Admin-2]")
    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")
    ax1.set_aspect("equal")
    ax1.grid(True)
    plt.tight_layout()

    # --------------------
    # FIGURE 2: Choropleth by Admin-2 Counts
    # --------------------
    # Use points as right dataframe; predicate contains on polygons is OK
    joined = gpd.sjoin(admin2_gdf, points_within, predicate="contains", how="left")

    # Count points per Admin-2
    count_series = joined.groupby("GID_2").size()
    admin2_gdf["point_count"] = admin2_gdf["GID_2"].map(count_series).fillna(0)

    max_count = admin2_gdf["point_count"].max()

    # Nice bins
    if max_count == 0:
        bins = [0, 1, 2, 3, 4, 5]
    else:
        magnitude = 10 ** (len(str(int(max_count))) - 1)
        nice_max = int(np.ceil(max_count / magnitude) * magnitude)
        bins = np.linspace(0, nice_max, 6)

    cmap_poly = cm.get_cmap(base_color, 5)
    norm = mcolors.BoundaryNorm(boundaries=bins, ncolors=5)

    fig2, ax2 = plt.subplots(figsize=(10, 10))
    admin2_gdf.plot(
        column="point_count",
        cmap=cmap_poly,
        linewidth=0.4,
        ax=ax2,
        edgecolor="gray",
        norm=norm,
        legend=True,
        legend_kwds={
            "label": "Number of Events",
            "orientation": "vertical",
            "shrink": 0.6,
            "boundaries": bins,
            "ticks": bins.astype(int),
        },
    )

    # Overlay matched polygons boundaries in green
    if matched_polys is not None and not matched_polys.empty:
        matched_polys.boundary.plot(ax=ax2, linewidth=1.6, edgecolor="green")

    ax2.set_title(f"{title} - Event Density by Admin-2")
    ax2.set_xlabel("Longitude")
    ax2.set_ylabel("Latitude")
    ax2.set_aspect("equal")
    plt.tight_layout()

    return fig1, fig2

