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
from geolocation import * # for the extract location thingy 
import rasterio
from rasterio.windows import from_bounds
from rasterio.features import geometry_mask
from matplotlib.ticker import FuncFormatter


gpkg_path  = '/eos/jeodpp/home/users/mihadar/data/Geospacial/gadm_410.gpkg'
pop_2020 = '/eos/jeodpp/home/users/mihadar/data/GHSL population /GHS_POP_GLOBE_2020/GHS_POP_E2020_GLOBE_R2023A_54009_100_V1_0.tif'
pop_2025 = '/eos/jeodpp/home/users/mihadar/data/GHSL population /GHS_POP_GLOBE_2025/GHS_POP_E2025_GLOBE_R2023A_54009_100_V1_0.tif'

def extract_population_counts(
    longitudes,
    latitudes,
    toponyms,
    radius_km=10, 
    tif_path = pop_2025,
):
    """
    Extract population and area statistics for multiple locations.
    """

    lons = np.array(longitudes)
    lats = np.array(latitudes)

    if len(lons) != len(lats):
        raise ValueError("Longitude/latitude length mismatch")

    if len(toponyms) != len(lons):
        raise ValueError("Toponyms must match number of locations")

    km_to_deg = radius_km / 111.0

    # Bounding box (+ buffer)
    min_lon = np.min(lons) - km_to_deg
    max_lon = np.max(lons) + km_to_deg
    min_lat = np.min(lats) - km_to_deg
    max_lat = np.max(lats) + km_to_deg

    with rasterio.open(tif_path) as src:
        transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)

        left, bottom = transformer.transform(min_lon, min_lat)
        right, top = transformer.transform(max_lon, max_lat)

        window = from_bounds(left, bottom, right, top, src.transform)
        data = src.read(1, window=window)

        nodata = src.nodata if src.nodata is not None else -200
        data = np.where(data == nodata, np.nan, data)

        rows, cols = data.shape

        # Cell area (km²)
        res_x = abs(src.transform.a)
        res_y = abs(src.transform.e)
        cell_area_km2 = (res_x * res_y) / 1e6

        # Coordinate grid
        xs = np.linspace(min_lon, max_lon, cols)
        ys = np.linspace(min_lat, max_lat, rows)
        lon_grid, lat_grid = np.meshgrid(xs, ys)

        radius_deg = radius_km / 111.0
        inside_any_radius = np.zeros(data.shape, dtype=bool)

        per_location_stats = []

        for name, lon, lat in zip(toponyms, lons, lats):
            dist = np.sqrt((lon_grid - lon)**2 + (lat_grid - lat)**2)
            mask = dist <= radius_deg

            inside_any_radius |= mask
            values = data[mask]
            values = values[~np.isnan(values)]

            if len(values) > 0:
                per_location_stats.append({
                    "name": name,
                    "total_population": int(np.sum(values)),
                    "mean_per_cell": float(np.mean(values)),
                    "cells": int(len(values)),
                    "area_km2": len(values) * cell_area_km2
                })
            else:
                per_location_stats.append({
                    "name": name,
                    "total_population": 0,
                    "mean_per_cell": 0.0,
                    "cells": 0,
                    "area_km2": 0.0
                })

        outside_mask = ~inside_any_radius
        outside_values = data[outside_mask]
        outside_values = outside_values[~np.isnan(outside_values)]

        # Area stats
        total_cells = np.count_nonzero(~np.isnan(data))
        inside_cells = np.count_nonzero(inside_any_radius & ~np.isnan(data))
        outside_cells = np.count_nonzero(outside_mask & ~np.isnan(data))

        total_area_km2 = total_cells * cell_area_km2
        inside_area_km2 = inside_cells * cell_area_km2
        outside_area_km2 = outside_cells * cell_area_km2

        total_population = int(np.nansum(data))
        outside_population = int(np.nansum(outside_values))

        outside_mean = float(np.mean(outside_values)) if len(outside_values) > 0 else 0.0

    # =========================
    # TABLE 1 — Population
    # =========================
    print("\n📊 Population statistics "
          f"(radius = {radius_km} km)\n")

    header = (
        f"{'Location':<15}"
        f"{'Total pop':>15}"
        f"{'Mean/cell':>15}"
        f"{'Area (km²)':>15}"
    )
    print(header)
    print("-" * len(header))

    for s in per_location_stats:
        print(
            f"{s['name']:<15}"
            f"{s['total_population']:>15,}"
            f"{s['mean_per_cell']:>15,.2f}"
            f"{s['area_km2']:>15,.2f}"
        )

    print(
        f"\nOutside radii population: {outside_population:,}"
        f"\nOutside radii mean/cell:  {outside_mean:,.2f}"
        f"\nTotal population:        {total_population:,}\n"
    )

    # =========================
    # TABLE 2 — Area coverage
    # =========================
    print("🗺️  Area coverage\n")

    area_header = (
        f"{'Region':<20}"
        f"{'Area (km²)':>15}"
        f"{'Percentage':>15}"
    )
    print(area_header)
    print("-" * len(area_header))

    print(
        f"{'Inside radii':<20}"
        f"{inside_area_km2:>15,.2f}"
        f"{inside_area_km2 / total_area_km2 * 100:>14.1f}%"
    )
    print(
        f"{'Outside radii':<20}"
        f"{outside_area_km2:>15,.2f}"
        f"{outside_area_km2 / total_area_km2 * 100:>14.1f}%"
    )
    print(
        f"{'Total':<20}"
        f"{total_area_km2:>15,.2f}"
        f"{'100.0%':>15}"
    )
    print()

    # =========================
    # Results dict
    # =========================
    results = {
        "per_location": per_location_stats,
        "outside": {
            "population": outside_population,
            "mean_per_cell": outside_mean,
            "area_km2": outside_area_km2,
            "cells": outside_cells
        },
        "area": {
            "total_km2": total_area_km2,
            "inside_km2": inside_area_km2,
            "outside_km2": outside_area_km2,
            "inside_pct": inside_area_km2 / total_area_km2 * 100,
            "outside_pct": outside_area_km2 / total_area_km2 * 100,
            "cell_area_km2": cell_area_km2
        },
        "total_population": total_population,
        "bbox": (min_lon, min_lat, max_lon, max_lat),
        "data": data
    }

    return results


def plot_population_results(
    results,
    longitudes,
    latitudes
):
    """
    Visualize population raster using the original dual-panel plot
    (linear + log scale), using results from extract_population_counts().
    """

    data_masked = results["data"]
    min_lon, min_lat, max_lon, max_lat = results["bbox"]
    total_population = results["total_population"]

    lons = np.array(longitudes)
    lats = np.array(latitudes)

    print("\n🎨 Creating visualization...")

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # =========================
    # Linear scale plot
    # =========================
    ax1 = axes[0]
    im1 = ax1.imshow(
        data_masked,
        cmap="YlOrRd",
        interpolation="nearest",
        extent=[min_lon, max_lon, min_lat, max_lat],
        aspect="auto"
    )

    ax1.set_title(
        f"Population Density - Linear Scale\nTotal: {total_population:,.0f} people",
        fontsize=12,
        fontweight="bold"
    )
    ax1.set_xlabel("Longitude", fontsize=11)
    ax1.set_ylabel("Latitude", fontsize=11)
    ax1.grid(True, alpha=0.3, linestyle="--")

    # Plot input locations
    ax1.scatter(
        lons,
        lats,
        c="blue",
        s=150,
        marker="*",
        edgecolors="white",
        linewidths=2,
        zorder=5,
        label="Input locations"
    )

    # Labels
    for i, (lon, lat) in enumerate(zip(lons, lats), 1):
        ax1.annotate(
            f"P{i}",
            (lon, lat),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7)
        )

    ax1.legend(loc="upper right")
    plt.colorbar(
        im1,
        ax=ax1,
        label="Population per 100m cell",
        fraction=0.046
    )

    # =========================
    # Log scale plot
    # =========================
    ax2 = axes[1]
    data_positive = np.where(data_masked > 0, data_masked, np.nan)

    if np.nansum(data_positive) > 0:
        vmin = max(np.nanmin(data_positive), 0.1)
        vmax = np.nanmax(data_positive)

        im2 = ax2.imshow(
            data_positive,
            cmap="YlOrRd",
            norm=LogNorm(vmin=vmin, vmax=vmax),
            interpolation="nearest",
            extent=[min_lon, max_lon, min_lat, max_lat],
            aspect="auto"
        )

        ax2.set_title(
            "Population Density - Log Scale\n(Better for seeing rural + urban)",
            fontsize=12,
            fontweight="bold"
        )
        ax2.set_xlabel("Longitude", fontsize=11)
        ax2.set_ylabel("Latitude", fontsize=11)
        ax2.grid(True, alpha=0.3, linestyle="--")

        ax2.scatter(
            lons,
            lats,
            c="blue",
            s=150,
            marker="*",
            edgecolors="white",
            linewidths=2,
            zorder=5,
            label="Input locations"
        )

        for i, (lon, lat) in enumerate(zip(lons, lats), 1):
            ax2.annotate(
                f"P{i}",
                (lon, lat),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7)
            )

        ax2.legend(loc="upper right")
        plt.colorbar(
            im2,
            ax=ax2,
            label="Population per 100m cell (log)",
            fraction=0.046
        )

    else:
        ax2.text(
            0.5,
            0.5,
            "No population data in this region\n(might be ocean/uninhabited)",
            ha="center",
            va="center",
            transform=ax2.transAxes,
            fontsize=14
        )
        ax2.set_title("Population Density - Log Scale (No Data)")

    plt.tight_layout()
    plt.show()
    
    
### put into csv file

def population_in_csv(toponyms, results, database_path):
    """
    Write GHSL population statistics into the CSV database.

    Data sources:
      - ghsl_toponym_<name> : per-location (within radius)
      - ghsl_region         : outside radii + total region

    CSV columns:
      ['Data source', 'indicator', 'value']
    """

    file_exists = os.path.exists(database_path)
    rows_to_write = []

    # =========================
    # Per-toponym (within radius)
    # =========================
    for stat in results["per_location"]:
        name = stat["name"]
        source = f"ghsl_toponym_{name}"

        rows_to_write.extend([
            [source, "total population", f"{stat['total_population']:,}"],
            [source, "mean population per cell", f"{stat['mean_per_cell']:,.2f}"],
            [source, "area (km2)", f"{stat['area_km2']:,.2f}"],
        ])

    # =========================
    # Regional (outside + total)
    # =========================
    region_source = "ghsl_region"

    area = results["area"]
    outside = results["outside"]

    rows_to_write.extend([
        # Population
        [region_source, "population outside radii", f"{outside['population']:,}"],
        [region_source, "mean population per cell outside radii", f"{outside['mean_per_cell']:,.2f}"],
        [region_source, "total population (region)", f"{results['total_population']:,}"],

        # Area absolute
        [region_source, "total area (km2)", f"{area['total_km2']:,.2f}"],
        [region_source, "area inside radii (km2)", f"{area['inside_km2']:,.2f}"],
        [region_source, "area outside radii (km2)", f"{area['outside_km2']:,.2f}"],

        # Area percentages
        [region_source, "area inside radii (%)", f"{area['inside_pct']:.1f}%"],
        [region_source, "area outside radii (%)", f"{area['outside_pct']:.1f}%"],
    ])

    # =========================
    # Write to CSV
    # =========================
    try:
        with open(database_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            if not file_exists:
                writer.writerow(['Data source', 'indicator', 'value'])

            writer.writerows(rows_to_write)

        print(f"✅ Successfully written {len(rows_to_write)} GHSL population rows to:")
        print(f"   {database_path}")

    except Exception as e:
        print(f"❌ Error writing GHSL population data to CSV: {e}")

        
        
        
## for the admin regions
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

POP_TIF_PATH = pop_2025

def extract_admin_pop(summary_df, matched_poly, out_json_path):
    """
    Computes population within matched polygons, saves JSON,
    and plots full Sudan outline + matched regions choropleth.
    """

    if POP_TIF_PATH is None:
        raise ValueError("Set POP_TIF_PATH before calling this function.")

    if summary_df.empty or matched_poly.empty:
        raise ValueError("No matched data available.")

    # ---- Load raster ----
    with rasterio.open(POP_TIF_PATH) as src:
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

    # ---- Join back to summary_df ----
    join_key = _pick_join_key(summary_df, matched_poly)

    pop_join = matched_poly[[join_key]].copy()
    pop_join["population_within_poly"] = polys["population_within_poly"].values

    results_df = summary_df.merge(pop_join, on=join_key, how="left")

    # ---- Save JSON ----
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(
            results_df.to_dict(orient="records"),
            f,
            ensure_ascii=False,
            indent=2
        )

    # ---- Build plotting GeoDataFrame ----
    results_polys = matched_poly.merge(
        pop_join.drop_duplicates(subset=[join_key]),
        on=join_key,
        how="left"
    )

    # ---- Load Sudan outline for base layer ----
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

    # =========================
    # --------- PLOT ----------
    # =========================

    fig, ax = plt.subplots(figsize=(11, 11))

    # Base Sudan map (full outline)
    sudan_outline.plot(
        ax=ax,
        alpha=0.07,
        edgecolor="black",
        linewidth=0.6
    )

    sudan_outline.boundary.plot(
        ax=ax,
        color="black",
        linewidth=0.8
    )

    # Matched admin regions (colored by population)
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
    ax.set_title("Sudan — Population within matched administrative regions", fontsize=14)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    return results_df, results_polys, fig, ax

## 

def plot_admin_population_table(results_df, max_gid_level=3, figsize=(12, 0.6)):
    """
    Print and plot a table containing columns up to GID_<max_gid_level>
    plus population_within_poly.

    Example:
      max_gid_level=3 keeps: GID_0..GID_3, NAME_0..NAME_3 (if present)
    """

    if results_df.empty:
        raise ValueError("results_df is empty")

    # ---- build column list ----
    keep_cols = []

    for i in range(max_gid_level + 1):
        gid = f"GID_{i}"
        name = f"NAME_{i}"
        if gid in results_df.columns:
            keep_cols.append(gid)
        if name in results_df.columns:
            keep_cols.append(name)

    if "population_within_poly" not in results_df.columns:
        raise ValueError("population_within_poly column not found")

    keep_cols.append("population_within_poly")

    table_df = results_df[keep_cols].copy()

    # ---- Print nicely ----
    with pd.option_context(
        "display.max_columns", None,
        "display.width", None,
        "display.max_colwidth", None
    ):
        print("\n===== ADMIN POPULATION TABLE =====\n")
        # print(table_df)

    # ---- Plot as figure table ----

    # Dynamic figure height based on rows
    nrows = len(table_df)
    fig_height = max(2, nrows * figsize[1])

    fig, ax = plt.subplots(figsize=(figsize[0], fig_height))
    ax.axis("off")

    # Format population nicely with commas
    display_df = table_df.copy()
    display_df["population_within_poly"] = display_df["population_within_poly"].apply(
        lambda x: f"{int(x):,}" if pd.notna(x) else ""
    )

    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        loc="center",
        cellLoc="center"
    )

    # Styling
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)

    # Bold header row
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")

    ax.set_title("Population per matched administrative region", fontsize=13, pad=20)

    plt.tight_layout()
    
    plt.show()

    return fig, ax, table_df