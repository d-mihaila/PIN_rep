import json
# from geolocation import *
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import os
import csv
import re
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from pyproj import Transformer
from pathlib import Path
from collections import defaultdict
from rasterio.transform import xy
from matplotlib import cm
from matplotlib.colors import Normalize


## here we read from worldpop

zip_dir = Path("/eos/jeodpp/home/users/mihadar/data/WorldPop")

# global age sex structures
global_age_sex_path = zip_dir / 'global_agesex_structures_2025_CN_1km_R2025A_UA_v1'

# sex disaggregated and migration data
spatial_data_path = zip_dir / 'SexDisaggregated_Migration/SpatialData'
Centroids = spatial_data_path / 'Centroids.shp'
Flowlines_Internal = spatial_data_path / 'Flowlines_Internal.shp'
Flowlines_International = spatial_data_path / 'Flowlines_International.shp'

internal_migration_path = zip_dir / 'SexDisaggregated_Migration/MigrationEstimates/Metadata_MigrEst_internal_v4.txt'
international_migration_path = zip_dir / 'SexDisaggregated_Migration/MigrationEstimates/Metadata_MigrEst_international_v7.txt'


### global age sex structures

def parse_age_sex(filename):
    """
    Extract sex and age from WorldPop age-sex filename.
    Returns (sex, age) where age is int.
    """
    # global_m_25_2025_CN_1km_R2025A_UA_v1.tif
    m = re.search(r'global_([mft])_(\d+)_', filename)
    if not m:
        raise ValueError(f"Cannot parse {filename}")
    sex = m.group(1)
    age = int(m.group(2))
    return sex, age


def extract_age_sex_population(
    longitudes,
    latitudes,
    toponyms,
    radius_km
):
    """
    Extract age–sex population within a radius for multiple locations.

    Returns
    -------
    pandas.DataFrame
        Columns: toponym, age, male, female, total
    """

    lons = np.array(longitudes)
    lats = np.array(latitudes)

    if len(lons) != len(lats):
        raise ValueError("Longitude/latitude length mismatch")

    if len(toponyms) != len(lons):
        raise ValueError("Toponyms must match number of locations")

    km_to_deg = radius_km / 111.0

    # Bounding box
    min_lon = np.min(lons) - km_to_deg
    max_lon = np.max(lons) + km_to_deg
    min_lat = np.min(lats) - km_to_deg
    max_lat = np.max(lats) + km_to_deg

    tif_dir = Path(global_age_sex_path)
    tif_files = sorted(tif_dir.glob("global_[mf]_*.tif"))

    # results[toponym][age][sex]
    results = defaultdict(lambda: defaultdict(lambda: {"male": 0, "female": 0}))

    # ---- open first raster to build grid & masks ----
    with rasterio.open(tif_files[0]) as src:
        transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)

        left, bottom = transformer.transform(min_lon, min_lat)
        right, top = transformer.transform(max_lon, max_lat)

        window = from_bounds(left, bottom, right, top, src.transform)

        with rasterio.open(tif_files[0]) as src:
            data0 = src.read(1, window=window)

        rows, cols = data0.shape

        row_inds, col_inds = np.meshgrid(
            np.arange(rows),
            np.arange(cols),
            indexing="ij"
        )

        xs, ys = xy(
            src.transform,
            row_inds + window.row_off,
            col_inds + window.col_off,
            offset="center"
        )
        lon_grid = np.array(xs).reshape(rows, cols)
        lat_grid = np.array(ys).reshape(rows, cols)


        radius_deg = radius_km / 111.0

        masks = {}
        for name, lon, lat in zip(toponyms, lons, lats):
            dist = np.sqrt((lon_grid - lon)**2 + (lat_grid - lat)**2)
            masks[name] = dist <= radius_deg

    # ---- loop through age/sex rasters ----
    for tif in tif_files:
        sex_code, age = parse_age_sex(tif.name)
        sex = "male" if sex_code == "m" else "female"

        with rasterio.open(tif) as src:
            data = src.read(1, window=window)
            nodata = src.nodata if src.nodata is not None else -200
            data = np.where(data == nodata, np.nan, data)

            for name in toponyms:
                values = data[masks[name]]
                values = values[~np.isnan(values)]
                results[name][age][sex] += int(np.sum(values))

    # ---- build tidy table ----
    rows = []
    for name in results:
        for age in sorted(results[name]):
            male = results[name][age]["male"]
            female = results[name][age]["female"]
            rows.append({
                "toponym": name,
                "age": age,
                "male": male,
                "female": female,
                "total": male + female
            })

    return pd.DataFrame(rows)

# save to csv
def worldpop_age_sex_to_csv(df, database_path):
    """
    Write WorldPop age–sex population data to CSV.

    Parameters
    ----------
    df : pandas.DataFrame
        Output of extract_age_sex_population
        Columns: toponym, age, male, female, total
    database_path : str or Path
        Output CSV path
    """

    file_exists = os.path.exists(database_path)
    rows_to_write = []

    # -------------------------
    # Global age ranges row
    # -------------------------
    ages = sorted(df["age"].unique().tolist())

    rows_to_write.append([
        "worldpop_age_split",
        "age ranges",
        str(ages)
    ])

    # -------------------------
    # Per-toponym rows
    # -------------------------
    for toponym in df["toponym"].unique():
        dfl = df[df["toponym"] == toponym].sort_values("age")

        female_list = dfl["female"].astype(int).tolist()
        male_list = dfl["male"].astype(int).tolist()
        total_list = dfl["total"].astype(int).tolist()

        source = f"worldpop_age_sex_{toponym}"

        rows_to_write.extend([
            [source, "population female", str(female_list)],
            [source, "population male", str(male_list)],
            [source, "population total", str(total_list)],
        ])

    # -------------------------
    # Write CSV
    # -------------------------
    try:
        with open(database_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            if not file_exists:
                writer.writerow(['Data source', 'indicator', 'value'])

            writer.writerows(rows_to_write)

        print(f"✅ Successfully written {len(rows_to_write)} WorldPop age–sex rows to:")
        print(f"   {database_path}")

    except Exception as e:
        print(f"❌ Error writing WorldPop age–sex data to CSV: {e}")


## plot it
def age_gradient_colors(ages, cmap_name):
    """
    Map ages to a color gradient.
    
    Parameters
    ----------
    ages : array-like of ints
        The ages to map
    cmap_name : str
        Name of matplotlib colormap (e.g., "Blues" or "Reds")
    
    Returns
    -------
    np.ndarray of RGBA colors
    """
    ages = np.array(ages)
    norm = (ages - ages.min()) / (ages.max() - ages.min())
    cmap = cm.get_cmap(cmap_name)
    return cmap(norm)


def plot_map_and_population_pie(
    longitudes,
    latitudes,
    toponyms,
    df,
    location_to_plot=None
):
    """
    Single figure:
    - Map with all locations
    - Pie chart for one selected location
    """

    if location_to_plot is None:
        location_to_plot = toponyms[0]

    # -----------------------------
    # Prepare totals for map
    # -----------------------------
    totals = (
        df.groupby("toponym")["total"]
        .sum()
        .reindex(toponyms)
    )

    # -----------------------------
    # Prepare data for pie
    # -----------------------------
    dfl = df[df["toponym"] == location_to_plot].sort_values("age")

    ages = dfl["age"].values
    male = dfl["male"].values
    female = dfl["female"].values

    total_pop = male.sum() + female.sum()

    values = np.concatenate([male, female])
    labels = (
        [f"M {a}" for a in ages] +
        [f"F {a}" for a in ages]
    )

    colors = np.concatenate([
        age_gradient_colors(ages, "Blues"),
        age_gradient_colors(ages, "Reds")
    ])

    # -----------------------------
    # Figure
    # -----------------------------
    fig = plt.figure(figsize=(14, 7))

    # ---- Map ----
    ax_map = fig.add_axes([0.05, 0.15, 0.4, 0.7])

    ax_map.scatter(
        longitudes,
        latitudes,
        s=totals / totals.max() * 1200 + 100,
        alpha=0.7
    )

    for lon, lat, name in zip(longitudes, latitudes, toponyms):
        ax_map.text(lon, lat, f" {name}", fontsize=10, va="center")

    ax_map.set_xlabel("Longitude")
    ax_map.set_ylabel("Latitude")
    ax_map.set_title("Locations sized by total population")

    # ---- Pie ----
    ax_pie = fig.add_axes([0.55, 0.2, 0.35, 0.6])

    ax_pie.pie(
        values,
        colors=colors,
        startangle=90,
        counterclock=False,
        wedgeprops=dict(edgecolor="white", linewidth=0.3)
    )

    ax_pie.set_title(
        f"Age–sex population structure\n{location_to_plot}",
        fontsize=12
    )

    ax_pie.text(
        0, -1.3,
        f"Total population: {int(total_pop):,}",
        ha="center",
        fontsize=11,
        fontweight="bold"
    )

    plt.show()

def plot_map_with_pies(longitudes, latitudes, toponyms, df):
    """
    Plot map with locations and mini pie charts at each location showing
    age-sex population structure (blue=male, red=female) with gradient by age.
    Includes a single colorbar for age.
    """

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot base points (optional: invisible, for scale reference)
    totals = df.groupby("toponym")["total"].sum().reindex(toponyms)
    ax.scatter(longitudes, latitudes, s=0)  # just for axis limits

    # Compute global age range for consistent color scaling
    all_ages = df["age"].values
    age_min, age_max = all_ages.min(), all_ages.max()
    norm = Normalize(vmin=age_min, vmax=age_max)
    cmap_m = cm.Blues
    cmap_f = cm.Reds

    def age_color(age, sex):
        """Return RGBA color for given age and sex."""
        if sex == "male":
            return cmap_m(norm(age))
        else:
            return cmap_f(norm(age))

    # Plot pie charts
    for lon, lat, name in zip(longitudes, latitudes, toponyms):
        dfl = df[df["toponym"] == name].sort_values("age")
        ages = dfl["age"].values
        male = dfl["male"].values
        female = dfl["female"].values

        values = np.concatenate([male, female])
        colors = [age_color(a, "male") for a in ages] + [age_color(a, "female") for a in ages]

        # Create inset axes in data coordinates
        size = max(totals.max()/20, 50)  # adjust pie size
        # Transform lon/lat to figure fraction for inset_axes
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        axins = inset_axes(ax, width=0.1, height=0.1, loc='center',
                           bbox_to_anchor=(lon, lat),
                           bbox_transform=ax.transData,
                           borderpad=0)
        axins.pie(values, colors=colors, startangle=90, counterclock=False,
                  wedgeprops=dict(edgecolor="white", linewidth=0.3))
        axins.set_aspect('equal')
        axins.set_xticks([])
        axins.set_yticks([])

        # Add total population below pie
        total_pop = male.sum() + female.sum()
        ax.text(lon, lat - 0.5, f"{total_pop:,}", ha="center", fontsize=9)

    # Axes and labels
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Population age-sex structure per location")

    # Colorbar for age
    from matplotlib.colors import ListedColormap
    import matplotlib.colorbar as cbar

    # Create a dummy mappable for colorbar
    sm = cm.ScalarMappable(cmap=cmap_m, norm=norm)
    sm.set_array([])

    cbar_m = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02, label="Male age (years)")
    smf = cm.ScalarMappable(cmap=cmap_f, norm=norm)
    smf.set_array([])
    cbar_f = fig.colorbar(smf, ax=ax, fraction=0.03, pad=0.06, label="Female age (years)")

    plt.show()