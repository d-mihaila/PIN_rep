import rasterio
from rasterio.windows import from_bounds
import numpy as np
import matplotlib.pyplot as plt
from pyproj import Transformer
import os
import json
from rasterio.plot import show
from scipy.spatial.distance import cdist
from matplotlib.patches import Circle
from matplotlib.colors import LogNorm
import csv
import geopandas as gpd
# from geolocation import * # for the extract location thingy 
import rasterio
from rasterio.windows import from_bounds
from rasterio.windows import Window
from rasterio.features import geometry_mask
from matplotlib.ticker import FuncFormatter
import pandas as pd
from pathlib import Path
import re


gpkg_path  = '/eos/jeodpp/home/users/mihadar/data/Geospacial/gadm_410.gpkg'
pop_2020 = '/eos/jeodpp/home/users/mihadar/data/GHSL population /GHS_POP_GLOBE_2020/GHS_POP_E2020_GLOBE_R2023A_54009_100_V1_0.tif'
pop_2025 = '/eos/jeodpp/home/users/mihadar/data/GHSL population /GHS_POP_GLOBE_2025/GHS_POP_E2025_GLOBE_R2023A_54009_100_V1_0.tif'


## ghsl
def extract_admin_pop(summary_df, matched_poly, json_path):
    """
    Computes population within matched ADMIN-2 polygons only, saves JSON,
    and plots Sudan admin-2 borders (gray) + matched admin-2 regions choropleth (blue edges).
    """

    if pop_2025 is None:
        raise ValueError("Set pop_2025 before calling this function.")

    if summary_df.empty or matched_poly.empty:
        raise ValueError("No matched data available.")

    # --- ensure we are working with admin-2 join key only ---
    join_key = "GID_2"
    if join_key not in summary_df.columns:
        raise ValueError("summary_df must contain GID_2 for admin-2 population joins.")
    if join_key not in matched_poly.columns:
        raise ValueError("matched_poly must contain GID_2 for admin-2 population joins.")

    # --- FIX: avoid 'GID_2 is both index and column' ambiguity ---
    # Make sure both frames have a simple index (not GID_2 as an index level)
    summary_df = summary_df.reset_index(drop=True)
    matched_poly = matched_poly.reset_index(drop=True)

    # ---- Load raster ----
    with rasterio.open(pop_2025) as src:
        raster_crs = src.crs

        polys = matched_poly.copy()

        if polys.crs is None:
            polys = polys.set_crs(epsg=4326)

        if raster_crs is not None and polys.crs != raster_crs:
            polys = polys.to_crs(raster_crs)

        pop_vals = []
        for geom in polys.geometry:
            pop_vals.append(_zonal_sum_population(src, geom))

    polys["population_within_poly"] = pop_vals

    # ---- Join back to summary_df (ADMIN-2 only) ----
    pop_join = polys[[join_key, "population_within_poly"]].copy()
    pop_join = pop_join.drop_duplicates(subset=[join_key]).reset_index(drop=True)

    results_df = summary_df.merge(pop_join, on=join_key, how="left")

    # --- NEW: ensure ONE row per admin-2 (no duplicate populations from repeated hits) ---
    def _uniq_join(series):
        vals = [str(v) for v in series.dropna().unique()]
        return "; ".join(vals)

    # group to admin-2 uniqueness; keep first for static fields, combine provenance fields
    agg = {c: "first" for c in results_df.columns if c not in ["query", "matched_field", "matched_level"]}
    if "query" in results_df.columns:
        agg["query"] = _uniq_join
    if "matched_field" in results_df.columns:
        agg["matched_field"] = _uniq_join
    if "matched_level" in results_df.columns:
        agg["matched_level"] = _uniq_join

    results_df = results_df.groupby(join_key, as_index=False).agg(agg)
    
    # ---- Save JSON ----
    out_json_path = os.path.join(json_path, "population_ghsl.json")
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(
            results_df.to_dict(orient="records"),
            f,
            ensure_ascii=False,
            indent=2
        )

    # ---- Build plotting GeoDataFrame (matched admin-2 only) ----
    results_polys = matched_poly.merge(pop_join, on=join_key, how="left")

    # ---- Load Sudan outline for base layer (ADMIN-2 borders only) ----
    safe_country = "Sudan"
    try:
        sudan_outline = gpd.read_file(
            gpkg_path,
            layer="gadm_410",
            where=f"COUNTRY = '{safe_country}'"
        )
    except TypeError:
        sudan_outline = gpd.read_file(gpkg_path, layer="gadm_410")
        sudan_outline = sudan_outline[sudan_outline["COUNTRY"] == safe_country]

    # CRS harmonization
    if sudan_outline.crs != results_polys.crs:
        sudan_outline = sudan_outline.to_crs(results_polys.crs)

    # --- collapse base outline to admin-2 for plotting ---
    if "GID_2" in sudan_outline.columns:
        sudan_outline = sudan_outline.dropna(subset=["GID_2"]).dissolve(
            by="GID_2", as_index=False, aggfunc="first"
        )

    # =========================
    # --------- PLOT ----------
    # =========================

    fig, ax = plt.subplots(figsize=(11, 11))

    # Base Sudan admin-2 borders (gray thin)
    sudan_outline.plot(
        ax=ax,
        alpha=0.07,
        edgecolor="gray",
        linewidth=0.4
    )
    sudan_outline.boundary.plot(
        ax=ax,
        color="gray",
        linewidth=0.6
    )

    # Matched admin-2 regions (colored by population)
    plot_obj = results_polys.plot(
        column="population_within_poly",
        ax=ax,
        legend=True,
        edgecolor="blue",
        linewidth=1.2,
        alpha=0.75,
    )

    # ---- Fix colorbar formatting (disable scientific notation) ----
    cbar = plot_obj.get_figure().axes[-1]
    cbar.yaxis.set_major_formatter(
        FuncFormatter(lambda x, _: f"{int(x):,}")
    )
    cbar.set_ylabel("Population", rotation=90)

    # ---- Cosmetics ----
    ax.set_title("Sudan — Population within matched admin-2 regions", fontsize=14)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    return results_df, results_polys, fig, ax


## helper function 
def _pick_join_key(summary_df: pd.DataFrame, matched_poly: gpd.GeoDataFrame) -> str:
    """Pick a sensible join key between summary_df and matched_poly."""
    for k in ["UID", "GID_5", "GID_4", "GID_3", "GID_2", "GID_1", "GID_0"]:
        if k in summary_df.columns and k in matched_poly.columns:
            return k
    raise ValueError("No common key found to join summary_df with matched_poly (expected UID or GID_* columns).")


def _zonal_sum_population(src: rasterio.io.DatasetReader, geom) -> int:
    """
    Sum raster values inside a single polygon geometry.
    Reads only the polygon bounding window for speed.
    """
    if geom is None or geom.is_empty:
        return 0

    # ensure we have a geometry mapping
    geom_mapping = geom.__geo_interface__

    # window covering polygon bounds
    minx, miny, maxx, maxy = geom.bounds
    window = from_bounds(minx, miny, maxx, maxy, transform=src.transform)

    # read data for window
    data = src.read(1, window=window, masked=False)

    # handle nodata
    nodata = src.nodata
    if nodata is not None:
        data = np.where(data == nodata, np.nan, data)

    # build mask for pixels inside geometry (True inside)
    win_transform = src.window_transform(window)
    inside = ~geometry_mask(
        [geom_mapping],
        out_shape=data.shape,
        transform=win_transform,
        invert=False,   # geometry_mask returns True where OUTSIDE; we invert after (~)
        all_touched=False,
    )

    vals = data[inside]
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return 0

    # Population rasters are typically per-cell counts; sum and cast to int
    return int(np.round(np.sum(vals)))


### worldpop
zip_dir = Path("/eos/jeodpp/home/users/mihadar/data/WorldPop")

# global age sex structures
global_age_sex_path = zip_dir / 'global_agesex_structures_2025_CN_1km_R2025A_UA_v1'

# plot vulnerable groups stats
def parse_age_sex(filename: str):
    m = re.search(r"global_([mft])_(\d+)_", filename)
    if not m:
        raise ValueError(f"Cannot parse {filename}")
    sex = m.group(1)
    age = int(m.group(2))
    return sex, age


def _clip_window_to_raster(src, window: Window) -> Window | None:
    full = Window(0, 0, src.width, src.height)
    try:
        w = window.intersection(full)
    except Exception:
        return None
    if w.width <= 0 or w.height <= 0:
        return None
    return w


def _sum_in_poly_window(src, window: Window, inside_mask, geom_mapping) -> int:
    w = _clip_window_to_raster(src, window)
    if w is None:
        return 0

    data = src.read(1, window=w, masked=False)

    nodata = src.nodata
    if nodata is not None:
        data = np.where(data == nodata, np.nan, data)

    # Recompute mask if raster/window rounding differs
    if inside_mask is None or data.shape != inside_mask.shape:
        win_transform = src.window_transform(w)
        inside = ~geometry_mask(
            [geom_mapping],
            out_shape=data.shape,
            transform=win_transform,
            invert=False,
            all_touched=False,
        )
    else:
        inside = inside_mask

    vals = data[inside]
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return 0
    return int(np.round(np.sum(vals)))


def extract_admin_age_sex_groups(
    summary_df: pd.DataFrame,
    matched_poly: gpd.GeoDataFrame,
    worldpop_tif_dir: str,
    json_path: str,
    join_key: str = "GID_2",
    json_name: str = "population_age_sex_groups.json",
):
    """
    Per polygon (join_key), compute:
      - children_0_5_total   (both sexes, ages 0..5)
      - youth_6_15_total     (both sexes, ages 6..15)
      - women_16_59_female   (female only, ages 16..59)
      - elderly_60p_total    (both sexes, ages 60+)
      - vulnerable_people    (children + youth + women + elderly)
      - total_population     (both sexes, all ages)

    Saves JSON and returns results_df.
    """

    if summary_df is None or summary_df.empty:
        raise ValueError("summary_df is empty")
    if matched_poly is None or matched_poly.empty:
        raise ValueError("matched_poly is empty")

    if join_key not in summary_df.columns:
        raise ValueError(f"summary_df must contain {join_key} for joins.")
    if join_key not in matched_poly.columns:
        raise ValueError(f"matched_poly must contain {join_key} for joins.")

    summary_df = summary_df.reset_index(drop=True)
    polys = matched_poly.reset_index(drop=True).copy()

    tif_dir = os.path.abspath(worldpop_tif_dir)
    if not os.path.isdir(tif_dir):
        raise ValueError(f"worldpop_tif_dir is not a directory: {tif_dir}")

    tif_files = sorted(
        os.path.join(tif_dir, f)
        for f in os.listdir(tif_dir)
        if f.lower().endswith(".tif") and f.startswith("global_")
    )
    if not tif_files:
        raise ValueError(f"No WorldPop .tif files found in: {tif_dir}")

    # Align polygons CRS to rasters, and precompute stable windows/masks on a template raster
    with rasterio.open(tif_files[0]) as template:
        raster_crs = template.crs

        if polys.crs is None:
            polys = polys.set_crs(epsg=4326)
        if raster_crs is not None and polys.crs != raster_crs:
            polys = polys.to_crs(raster_crs)

        poly_specs = []
        for _, row in polys.iterrows():
            geom = row.geometry
            gid = row[join_key]

            if geom is None or geom.is_empty:
                poly_specs.append((gid, None, None, None))
                continue

            geom_mapping = geom.__geo_interface__

            minx, miny, maxx, maxy = geom.bounds
            w = from_bounds(minx, miny, maxx, maxy, transform=template.transform)
            w = w.round_offsets().round_lengths()

            win_transform = template.window_transform(w)
            inside = ~geometry_mask(
                [geom_mapping],
                out_shape=(int(w.height), int(w.width)),
                transform=win_transform,
                invert=False,
                all_touched=False,
            )

            poly_specs.append((gid, w, inside, geom_mapping))

    # Accumulate by region
    acc = {}
    def _ensure(gid):
        if gid not in acc:
            acc[gid] = {
                "children_0_5_total": 0,
                "youth_6_15_total": 0,
                "women_16_59_female": 0,
                "elderly_60p_total": 0,
                "total_population": 0,
            }

    for tif in tif_files:
        fname = os.path.basename(tif)
        sex_code, age = parse_age_sex(fname)
        if sex_code not in ("m", "f"):
            continue

        with rasterio.open(tif) as src:
            for gid, window, inside_mask, geom_mapping in poly_specs:
                _ensure(gid)
                if window is None:
                    continue

                s = _sum_in_poly_window(src, window, inside_mask, geom_mapping)

                acc[gid]["total_population"] += s

                if 0 <= age <= 5:
                    acc[gid]["children_0_5_total"] += s
                elif 6 <= age <= 15:
                    acc[gid]["youth_6_15_total"] += s
                elif age >= 60:
                    acc[gid]["elderly_60p_total"] += s

                if sex_code == "f" and 16 <= age <= 59:
                    acc[gid]["women_16_59_female"] += s

    groups_df = pd.DataFrame([{join_key: gid, **vals} for gid, vals in acc.items()])

    # Add vulnerable_people
    for col in ["children_0_5_total", "youth_6_15_total", "women_16_59_female", "elderly_60p_total"]:
        if col not in groups_df.columns:
            groups_df[col] = 0

    groups_df["vulnerable_people"] = (
        groups_df["children_0_5_total"].fillna(0)
        + groups_df["youth_6_15_total"].fillna(0)
        + groups_df["women_16_59_female"].fillna(0)
        + groups_df["elderly_60p_total"].fillna(0)
    ).astype(int)

    # Join back to summary_df
    results_df = summary_df.merge(groups_df, on=join_key, how="left")

    # Ensure one row per region (same uniqueness logic)
    def _uniq_join(series):
        vals = [str(v) for v in series.dropna().unique()]
        return "; ".join(vals)

    agg = {c: "first" for c in results_df.columns if c not in ["query", "matched_field", "matched_level"]}
    if "query" in results_df.columns:
        agg["query"] = _uniq_join
    if "matched_field" in results_df.columns:
        agg["matched_field"] = _uniq_join
    if "matched_level" in results_df.columns:
        agg["matched_level"] = _uniq_join

    results_df = results_df.groupby(join_key, as_index=False).agg(agg)

    # Save JSON only
    os.makedirs(json_path, exist_ok=True)
    out_json_path = os.path.join(json_path, json_name)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(results_df.to_dict(orient="records"), f, ensure_ascii=False, indent=2)

    return results_df

def plot_admin_population_table_for_metric(
    results_df: pd.DataFrame,
    metric_col: str,
    max_gid_level: int = 3,
    figsize=(12, 0.6),
    title: str | None = None,
):
    if results_df is None or results_df.empty:
        raise ValueError("results_df is empty")
    if metric_col not in results_df.columns:
        raise ValueError(f"'{metric_col}' column not found in results_df")

    keep_cols = []
    for i in range(max_gid_level + 1):
        gid = f"GID_{i}"
        name = f"NAME_{i}"
        if gid in results_df.columns:
            keep_cols.append(gid)
        if name in results_df.columns:
            keep_cols.append(name)
    keep_cols.append(metric_col)

    table_df = results_df[keep_cols].copy()

    nrows = len(table_df)
    fig_height = max(2, nrows * figsize[1])

    fig, ax = plt.subplots(figsize=(figsize[0], fig_height))
    ax.axis("off")

    display_df = table_df.copy()
    display_df[metric_col] = display_df[metric_col].apply(
        lambda x: f"{int(x):,}" if pd.notna(x) else ""
    )

    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        loc="center",
        cellLoc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")

    if title is None:
        title = f"{metric_col} per matched administrative region"

    ax.set_title(title, fontsize=13, pad=20)
    plt.tight_layout()
    plt.show()

    return fig, ax, table_df


## plot summary table

def plot_population_summary(
    ghsl_df: pd.DataFrame,
    worldpop_df: pd.DataFrame,
    figsize=(14, 0.6),
    join_keys=("GID_1", "GID_2"),
    ghsl_total_col: str = "population_within_poly",
    worldpop_total_col: str = "total_population",
    worldpop_vuln_col: str = "vulnerable_people",
    col_labels=(
        "Tot pop GHSL",
        "Tot pop Worldpop",
        "Vulnerable ppl",
        "% Vulnerable ppl",
    ),
    title: str = "Population summary per matched administrative region",
    percent_decimals: int = 1,
    how: str = "outer",
):
    if ghsl_df is None or ghsl_df.empty:
        raise ValueError("ghsl_df is empty")
    if worldpop_df is None or worldpop_df.empty:
        raise ValueError("worldpop_df is empty")

    # --- validate join keys ---
    missing_ghsl = [k for k in join_keys if k not in ghsl_df.columns]
    missing_wp = [k for k in join_keys if k not in worldpop_df.columns]
    if missing_ghsl:
        raise ValueError(f"Missing join keys in ghsl_df: {missing_ghsl}")
    if missing_wp:
        raise ValueError(f"Missing join keys in worldpop_df: {missing_wp}")

    # --- validate metric cols ---
    if ghsl_total_col not in ghsl_df.columns:
        raise ValueError(f"Missing GHSL total column '{ghsl_total_col}' in ghsl_df")
    for c in (worldpop_total_col, worldpop_vuln_col):
        if c not in worldpop_df.columns:
            raise ValueError(f"Missing WorldPop column '{c}' in worldpop_df")

    # --- keep only needed columns ---
    ghsl_keep = list(join_keys) + (["NAME_2"] if "NAME_2" in ghsl_df.columns else []) + [ghsl_total_col]
    wp_keep   = list(join_keys) + (["NAME_2"] if "NAME_2" in worldpop_df.columns else []) + [worldpop_total_col, worldpop_vuln_col]

    ghsl_sub = ghsl_df[ghsl_keep].copy()
    wp_sub   = worldpop_df[wp_keep].copy()

    merged = pd.merge(
        ghsl_sub,
        wp_sub,
        on=list(join_keys),
        how=how,
        suffixes=("_ghsl", "_wp"),
    )

    # --- resolve NAME_2 ---
    if "NAME_2_ghsl" in merged.columns or "NAME_2_wp" in merged.columns:
        merged["NAME_2"] = None
        if "NAME_2_ghsl" in merged.columns:
            merged["NAME_2"] = merged["NAME_2_ghsl"]
        if "NAME_2_wp" in merged.columns:
            merged["NAME_2"] = merged["NAME_2"].fillna(merged["NAME_2_wp"])
        merged.drop(columns=[c for c in ("NAME_2_ghsl", "NAME_2_wp") if c in merged.columns], inplace=True)

    if "NAME_2" not in merged.columns:
        raise ValueError("NAME_2 not found in either dataframe; cannot build the requested table.")

    # --- compute percent vulnerable ---
    total_wp = pd.to_numeric(merged[worldpop_total_col], errors="coerce")
    vuln_wp  = pd.to_numeric(merged[worldpop_vuln_col], errors="coerce")
    merged["_pct_vulnerable_worldpop"] = (vuln_wp / total_wp) * 100

    # --- final numeric table df ---
    ghsl_label, wp_total_label, wp_vuln_label, wp_pct_label = col_labels

    table_df = merged[["NAME_2", ghsl_total_col, worldpop_total_col, worldpop_vuln_col, "_pct_vulnerable_worldpop"]].copy()
    table_df = table_df.rename(
        columns={
            "NAME_2": "Admin 2 name",
            ghsl_total_col: ghsl_label,
            worldpop_total_col: wp_total_label,
            worldpop_vuln_col: wp_vuln_label,
            "_pct_vulnerable_worldpop": wp_pct_label,
        }
    )

    # --- display strings for plotting ---
    display_df = table_df.copy()

    def fmt_int(x):
        if pd.isna(x):
            return ""
        try:
            return f"{int(round(float(x))):,}"
        except Exception:
            return str(x)

    def fmt_pct(x):
        if pd.isna(x):
            return ""
        try:
            return f"{float(x):.{percent_decimals}f}%"
        except Exception:
            return str(x)

    display_df[ghsl_label]     = pd.to_numeric(display_df[ghsl_label], errors="coerce").apply(fmt_int)
    display_df[wp_total_label] = pd.to_numeric(display_df[wp_total_label], errors="coerce").apply(fmt_int)
    display_df[wp_vuln_label]  = pd.to_numeric(display_df[wp_vuln_label], errors="coerce").apply(fmt_int)
    display_df[wp_pct_label]   = pd.to_numeric(display_df[wp_pct_label], errors="coerce").apply(fmt_pct)

    # ==========================
    # Header order fix you asked
    # ==========================
    group_header = ["", "GHSL", "", "WorldPop", ""]              # top row
    second_header = list(display_df.columns)                     # second row (your original header)
    display_with_group = pd.concat(
        [pd.DataFrame([second_header], columns=display_df.columns), display_df],
        ignore_index=True,
    )

    # --- plot ---
    nrows = len(display_with_group)
    fig_height = max(2, nrows * figsize[1])
    fig, ax = plt.subplots(figsize=(figsize[0], fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=display_with_group.values,
        colLabels=group_header,  # <-- group header is row 0
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)

    # Bold both header rows: row 0 (colLabels) and row 1 (second_header)
    for (row, col), cell in table.get_celld().items():
        if row in (0, 1):
            cell.set_text_props(weight="bold")

    # Fake spanning for WorldPop across cols 2-4 on header row 0
    if (0, 2) in table.get_celld() and (0, 3) in table.get_celld() and (0, 4) in table.get_celld():
        table[(0, 2)].visible_edges = "LTB"  # no right
        table[(0, 3)].visible_edges = "TB"   # no left/right
        table[(0, 4)].visible_edges = "RTB"  # no left

    ax.set_title(title, fontsize=13, pad=20)
    plt.tight_layout()
    plt.show()

    # IMPORTANT: return a 3-tuple, so your unpacking works
    return fig, ax, table_df

def population_fetch(summary_df, matched_polys, event_folder):
    ghsl_results_df, results_polys, fig, ax = extract_admin_pop(
    summary_df=summary_df,
    matched_poly=matched_polys,
    json_path=event_folder,
    )
    
    worldpop_results_df = extract_admin_age_sex_groups(
    summary_df=summary_df,
    matched_poly=matched_polys,
    worldpop_tif_dir=str(global_age_sex_path),
    json_path=event_folder,
    join_key="GID_2",
)

    fig, ax, summary_df = plot_population_summary(
        ghsl_df=ghsl_results_df,
        worldpop_df=worldpop_results_df,
    )
    
    return worldpop_results_df