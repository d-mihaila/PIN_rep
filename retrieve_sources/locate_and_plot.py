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
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter



gpkg_path  = '/eos/jeodpp/home/users/mihadar/data/Geospacial/gadm_410.gpkg'

def _norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[’'`]", "", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def extract_locations(info_or_text):
    """
    Accepts either:
      - dict with key 'Locations'
      - string containing "'Locations': ..."
    Returns list[str]
    """
    if info_or_text is None:
        return []

    # Case 1: dict output from extract_disaster_info
    if isinstance(info_or_text, dict):
        loc_val = info_or_text.get("Locations")
        if not loc_val:
            return []
        if isinstance(loc_val, list):
            return [str(x).strip() for x in loc_val if str(x).strip()]
        return [s.strip() for s in str(loc_val).split(",") if s.strip()]

    # Case 2: string ("Processed content: {...}")
    text = str(info_or_text)
    m = re.search(r"'Locations'\s*:\s*(.*?)(?:\n|\})", text, flags=re.DOTALL)
    if not m:
        return []

    loc_blob = m.group(1).strip().rstrip().rstrip(",")
    return [x.strip() for x in loc_blob.split(",") if x.strip()]
    
def gadm_match_locations(
    gpkg_path: str,
    locations: List[str],
    *,
    country: str,
    layer: str = "gadm_410",
    country_field: str = "COUNTRY",
    include_varnames: bool = True,
    json_path: str,
) -> Tuple[pd.DataFrame, gpd.GeoDataFrame]:
    """
    Match `locations` against GADM names (NAME_1..NAME_5; optionally VARNAME_1..VARNAME_5)
    within `country`, BUT only SAVE/RETURN the corresponding ADMIN-2 region.

    Saves found matches to JSON and returns:
      - summary_df (admin-2 info only; no geometry)
      - matched_polygons_gdf (unique ADMIN-2 polygons with geometry for plotting)

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
    for idx, row in gadm[lookup_fields].iterrows():
        for col in lookup_fields:
            val = row[col]
            if pd.isna(val) or not str(val).strip():
                continue
            parts = re.split(r"[|,;/]", str(val))
            for p in parts:
                key = _norm(p)
                if key:
                    lookup[key].append((idx, col))

    # --- NEW: prepare an ADMIN-2 dissolved layer to pull attributes + geometry from ---
    if "GID_2" not in gadm.columns:
        raise ValueError("GID_2 column not found; cannot collapse matches to admin-2.")

    admin2_keep_cols = [c for c in [
        "GID_0", "NAME_0",
        "GID_1", "NAME_1", "TYPE_1", "ENGTYPE_1",
        "GID_2", "NAME_2", "TYPE_2", "ENGTYPE_2",
        "REGION", "VARREGION", "COUNTRY", "CONTINENT", "SUBCONT",
    ] if c in gadm.columns]

    admin2 = gadm[admin2_keep_cols + ["geometry"]].dropna(subset=["GID_2"]).dissolve(
        by="GID_2", as_index=False, aggfunc="first"
    )
    admin2 = admin2.set_index("GID_2", drop=False)

    # 3) Match input locations (same searching), but SAVE only admin-2 info
    records = []
    matched_gid2 = set()

    for loc in locations:
        key = _norm(loc)
        hits = lookup.get(key, [])
        if not hits:
            print(f"{loc} could not be found")
            continue

        for idx, matched_col in hits:
            gid2 = gadm.at[idx, "GID_2"]
            if pd.isna(gid2) or gid2 not in admin2.index:
                print(f"{loc} could not be found")
                continue

            matched_gid2.add(gid2)

            rec = {"query": loc, "matched_field": matched_col}

            # keep info about what level the TEXT matched at (optional, but useful)
            m = re.search(r"_(\d)$", matched_col)
            rec["matched_level"] = int(m.group(1)) if m else None

            # --- NEW: store only admin-2 columns (no NAME_3..NAME_5 etc.) ---
            for c in admin2_keep_cols:
                rec[c] = admin2.at[gid2, c] if c in admin2.columns else None

            records.append(rec)

    # 4) Output artifacts
    summary_df = pd.DataFrame(records)

    out_json = os.path.join(json_path, "matched_locations.json")
    summary_df.to_json(out_json, orient="records", force_ascii=False, indent=2)
    print("Found locations saved in", out_json)

    # --- NEW: return ONLY matched admin-2 polygons (not all admin-2, not admin3-5 parts) ---
    matched_polys = admin2.loc[sorted(matched_gid2)].copy() if matched_gid2 else admin2.iloc[0:0].copy()

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
    Plot ADMIN-2 borders in gray + highlight matched ADMIN-2 polygons in blue, in ONE figure.
    Optionally overlay point coordinates.
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

    # ---------- NEW: collapse base map to ADMIN-2 ----------
    if "GID_2" not in country_gdf.columns:
        raise ValueError("GID_2 column not found; cannot plot admin-2 borders.")

    admin2_base = country_gdf.dropna(subset=["GID_2"]).dissolve(
        by="GID_2", as_index=False, aggfunc="first"
    )
    # ------------------------------------------------------

    # --- plot ---
    fig, ax = plt.subplots(figsize=(10, 10))

    # Base ADMIN-2 borders only (gray)
    admin2_base.plot(ax=ax, alpha=base_alpha, linewidth=0.4, edgecolor="gray")
    admin2_base.boundary.plot(ax=ax, linewidth=0.6, edgecolor="gray")

    # Highlight matched ADMIN-2 polygons (blue)
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
            gpts.plot(ax=ax, markersize=20, alpha=0.9)

    ax.set_title(title or f"{country} — matched GADM admin-2 regions")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal")
    plt.tight_layout()

    return fig, ax


## plot acled
def plot_admin2_fatalities_heatmap(
    iso3_code,
    acled_rows,
    matched_polys=None,
    gadm_layer="gadm_410",
    base_color="Reds",
    title=""
):
    """
    Creates:
      FIG 1: Admin-2 choropleth by COUNT of events inside each polygon
             + green borders overlay for matched_polys
      FIG 2: Admin-2 choropleth by SUM of fatalities inside each polygon
             + green borders overlay for matched_polys

    Returns
    -------
    (fig1, fig2, admin2_gdf)
    """

    # --- Load GADM and build Admin-2 for country ---
    gdf = gpd.read_file(gpkg_path, layer=gadm_layer)

    country_gdf = gdf[gdf["GID_0"] == iso3_code].copy()
    if country_gdf.empty:
        raise ValueError(f"No data found for ISO3 code: {iso3_code}")
    if "GID_2" not in country_gdf.columns:
        raise ValueError("GID_2 column not found in GADM layer.")

    admin2_gdf = country_gdf.dissolve(by="GID_2", as_index=False)

    # --- Build points GeoDataFrame with fatalities ---
    points = []
    fats = []
    for r in acled_rows:
        lat = r.get("lat")
        lon = r.get("lon")
        fat = r.get("fatalities", 0)

        if lat is None or lon is None:
            continue

        try:
            lat = float(lat)
            lon = float(lon)
            fat = float(fat) if fat is not None else 0.0
        except (TypeError, ValueError):
            continue

        points.append(Point(lon, lat))  # lon, lat
        fats.append(fat)

    points_gdf = gpd.GeoDataFrame({"fatalities": fats}, geometry=points, crs="EPSG:4326")

    # If admin2 is not WGS84, convert points to match admin CRS
    if admin2_gdf.crs is None:
        admin2_gdf = admin2_gdf.set_crs("EPSG:4326")
    points_gdf = points_gdf.to_crs(admin2_gdf.crs)

    # Keep only points inside country outline
    points_within = points_gdf[points_gdf.geometry.within(admin2_gdf.unary_union)].copy()

    # --------------------
    # FIGURE 1: Choropleth by COUNT events per Admin-2
    # --------------------
    joined_ct = gpd.sjoin(admin2_gdf, points_within, predicate="contains", how="left")

    # count events per polygon (each joined point = 1)
    evt_count = joined_ct.groupby("GID_2")["fatalities"].count()
    admin2_gdf["event_count"] = admin2_gdf["GID_2"].map(evt_count).fillna(0).astype(int)

    max_ct = int(admin2_gdf["event_count"].max())

    # bins (5 classes)
    if max_ct <= 0:
        bins_ct = np.array([0, 1, 2, 3, 4, 5], dtype=float)
    else:
        nice_max = float(np.ceil(max_ct / 5) * 5)  # round up to multiple of 5
        bins_ct = np.linspace(0, nice_max, 6)

    cmap_ct = cm.get_cmap("Oranges", 5)
    norm_ct = mcolors.BoundaryNorm(boundaries=bins_ct, ncolors=5)

    fig1, ax1 = plt.subplots(figsize=(10, 10))
    admin2_gdf.plot(
        column="event_count",
        cmap=cmap_ct,
        linewidth=0.35,
        ax=ax1,
        edgecolor="gray",
        norm=norm_ct,
        legend=True,
        legend_kwds={
            "label": "Number of ACLED events",
            "orientation": "vertical",
            "shrink": 0.6,
            "boundaries": bins_ct,
            "ticks": bins_ct.astype(int)
        }
    )

    # overlay matched polys in green (optional)
    if matched_polys is not None and len(matched_polys) > 0:
        mp = matched_polys.copy()
        if mp.crs is None:
            mp = mp.set_crs(admin2_gdf.crs)
        mp = mp.to_crs(admin2_gdf.crs)
        mp.boundary.plot(ax=ax1, linewidth=1.6, edgecolor="green")

    ax1.set_title(f"{title} - ACLED events per Admin-2 (choropleth)")
    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")
    ax1.set_aspect("equal")
    ax1.grid(True)
    plt.tight_layout()

    # --------------------
    # FIGURE 2: Choropleth by SUM fatalities per Admin-2  (UNCHANGED)
    # --------------------
    joined = gpd.sjoin(admin2_gdf, points_within, predicate="contains", how="left")

    # sum fatalities per polygon
    fat_sum = joined.groupby("GID_2")["fatalities"].sum(min_count=1)
    admin2_gdf["fatalities_sum"] = admin2_gdf["GID_2"].map(fat_sum).fillna(0)

    max_fat = float(admin2_gdf["fatalities_sum"].max())

    # nice bins (5 classes)
    if max_fat <= 0:
        bins = np.array([0, 1, 2, 3, 4, 5], dtype=float)
    else:
        magnitude = 10 ** (len(str(int(max_fat))) - 1)
        nice_max = float(np.ceil(max_fat / magnitude) * magnitude)
        bins = np.linspace(0, nice_max, 6)

    cmap = cm.get_cmap(base_color, 5)
    norm = mcolors.BoundaryNorm(boundaries=bins, ncolors=5)

    fig2, ax2 = plt.subplots(figsize=(10, 10))
    admin2_gdf.plot(
        column="fatalities_sum",
        cmap=cmap,
        linewidth=0.35,
        ax=ax2,
        edgecolor="black",
        norm=norm,
        legend=True,
        legend_kwds={
            "label": "Sum of fatalities",
            "orientation": "vertical",
            "shrink": 0.6,
            "boundaries": bins,
            "ticks": bins.astype(int)
        }
    )

    # overlay matched polygons in GREEN borders
    if matched_polys is not None and len(matched_polys) > 0:
        mp = matched_polys.copy()
        if mp.crs is None:
            mp = mp.set_crs(admin2_gdf.crs)
        mp = mp.to_crs(admin2_gdf.crs)
        mp.boundary.plot(ax=ax2, linewidth=1.6, edgecolor="green")

    ax2.set_title(f"{title} - Fatalities by Admin-2 (choropleth)")
    ax2.set_xlabel("Longitude")
    ax2.set_ylabel("Latitude")
    ax2.set_aspect("equal")
    plt.tight_layout()

    return fig1, fig2, admin2_gdf




## plot idmc
def plot_admin2_idmc_figure_heatmap(
    iso3_code,
    idmc_rows,
    matched_polys=None,
    gadm_layer="gadm_410",
    base_color="Reds",
    title=""
):
    """
    Creates ONLY:
      FIG: Admin-2 choropleth by LINEAR SUM of IDMC 'figure' per polygon
           + green borders overlay for matched_polys

    Returns
    -------
    (fig, ax, admin2_gdf)
    """

    # --- Load GADM and build Admin-2 for country ---
    gdf = gpd.read_file(gpkg_path, layer=gadm_layer)

    country_gdf = gdf[gdf["GID_0"] == iso3_code].copy()
    if country_gdf.empty:
        raise ValueError(f"No data found for ISO3 code: {iso3_code}")
    if "GID_2" not in country_gdf.columns:
        raise ValueError("GID_2 column not found in GADM layer.")

    admin2_gdf = country_gdf.dissolve(by="GID_2", as_index=False)

    # --- Build points GeoDataFrame with figures ---
    points = []
    figs = []

    for r in idmc_rows:
        lat = r.get("lat")
        lon = r.get("lon")
        fig_val = r.get("figure", 0)

        if lat is None or lon is None:
            continue

        try:
            lat = float(lat)
            lon = float(lon)
            fig_val = float(fig_val) if fig_val is not None else 0.0
        except (TypeError, ValueError):
            continue

        points.append(Point(lon, lat))
        figs.append(fig_val)

    points_gdf = gpd.GeoDataFrame(
        {"figure": figs},
        geometry=points,
        crs="EPSG:4326"
    )

    # CRS harmonization
    if admin2_gdf.crs is None:
        admin2_gdf = admin2_gdf.set_crs("EPSG:4326")
    points_gdf = points_gdf.to_crs(admin2_gdf.crs)

    # Keep only points inside country outline
    points_within = points_gdf[points_gdf.geometry.within(admin2_gdf.unary_union)].copy()

    # --------------------
    # Choropleth by SUM(figure) per Admin-2 (LINEAR)
    # --------------------
    joined = gpd.sjoin(admin2_gdf, points_within, predicate="contains", how="left")

    fig_sum = joined.groupby("GID_2")["figure"].sum(min_count=1)
    admin2_gdf["figure_sum"] = admin2_gdf["GID_2"].map(fig_sum).fillna(0)

    max_fig = float(admin2_gdf["figure_sum"].max())

    # Build nice linear bins (5 classes)
    if max_fig <= 0:
        bins = np.array([0, 1, 2, 3, 4, 5], dtype=float)
    else:
        magnitude = 10 ** (len(str(int(max_fig))) - 1)
        nice_max = float(np.ceil(max_fig / magnitude) * magnitude)
        bins = np.linspace(0, nice_max, 6)

    cmap = cm.get_cmap(base_color, 5)
    norm = mcolors.BoundaryNorm(boundaries=bins, ncolors=5)

    fig, ax = plt.subplots(figsize=(10, 10))
    plot_obj = admin2_gdf.plot(
        column="figure_sum",
        cmap=cmap,
        linewidth=0.35,
        ax=ax,
        edgecolor="black",
        norm=norm,
        legend=True,
        legend_kwds={
            "label": "Displaced people (IDMC)",
            "orientation": "vertical",
            "shrink": 0.6,
            "boundaries": bins,
            "ticks": bins
        }
    )

    # Overlay matched polygons (green)
    if matched_polys is not None and len(matched_polys) > 0:
        mp = matched_polys.copy()
        if mp.crs is None:
            mp = mp.set_crs(admin2_gdf.crs)
        mp = mp.to_crs(admin2_gdf.crs)
        mp.boundary.plot(ax=ax, linewidth=1.6, edgecolor="green")

    # ---- Format colorbar labels as full numbers ----
    cbar_ax = plot_obj.get_figure().axes[-1]
    cbar_ax.yaxis.set_major_formatter(
        FuncFormatter(lambda x, _: f"{int(round(x)):,}")
    )
    cbar_ax.set_ylabel("Displaced people", rotation=90)

    # ---- Cosmetics ----
    ax.set_title(f"{title} - IDMC displacement by Admin-2")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal")
    plt.tight_layout()

    return fig, ax, admin2_gdf

## plot iom