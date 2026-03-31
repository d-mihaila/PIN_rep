import numpy as np
import matplotlib.pyplot as plt
import math
import json
import pandas as pd
from collections import defaultdict
from typing import Tuple
import geopandas as gpd
from shapely.geometry import Point


def fatalities_per_a2(
    acled_json_path: str,
    pop_path: str,
    gpkg_path: str,
    gpkg_layer: str | None = None,
    use_nearest_if_missing: bool = True,
    nearest_max_km: float = 25.0,
    top_n: int = 15,
):
    """
    Spatially assigns ACLED events (lat/lon points) to GADM Admin2 polygons (GID_2),
    aggregates fatalities per Admin2, merges with WorldPop by GID_2, and returns/prints:

      Top N Admin2 by fatalities (post-pop merge)

    Also adds:
      - vulnerable_people (from WorldPop)
      - pct_fatalities = fatalities / total_population * 100
      - PIN rule (percent units):
          if pct_fatalities > 5:            PIN += total_population
          elif 0 < pct_fatalities <= 5:     PIN += vulnerable_people
    """

    # -------------------
    # Load ACLED point rows
    # -------------------
    rows = extract_lat_lon_fatalities_from_acled(acled_json_path)
    if not rows:
        print("No lat/lon rows extracted from ACLED.")
        return pd.DataFrame(
            columns=["GID_2","NAME_1","NAME_2","fatalities","total_population","vulnerable_people","pct_fatalities","pin_contrib"]
        )

    acled_df = pd.DataFrame(rows)
    acled_df["fatalities"] = pd.to_numeric(acled_df["fatalities"], errors="coerce").fillna(0.0)

    gdf_pts = gpd.GeoDataFrame(
        acled_df,
        geometry=[Point(xy) for xy in zip(acled_df["lon"], acled_df["lat"])],
        crs="EPSG:4326",
    )

    # -------------------
    # Load GADM polygons (Admin2)
    # -------------------
    gadm = gpd.read_file(gpkg_path) if gpkg_layer is None else gpd.read_file(gpkg_path, layer=gpkg_layer)

    if "GID_2" not in gadm.columns:
        raise ValueError(f"GADM layer must contain 'GID_2'. Columns: {list(gadm.columns)[:40]}")

    for namecol in ("NAME_1", "NAME_2"):
        if namecol not in gadm.columns:
            # still works (names just absent)
            pass

    if gadm.crs is None:
        gadm = gadm.set_crs("EPSG:4326")
    if gadm.crs != gdf_pts.crs:
        gadm = gadm.to_crs(gdf_pts.crs)

    keep_cols = [c for c in ["GID_2","NAME_1","NAME_2"] if c in gadm.columns]
    gadm_small = gadm[keep_cols + ["geometry"]].copy()

    # -------------------
    # Spatial join: within
    # -------------------
    joined = gpd.sjoin(gdf_pts, gadm_small, how="left", predicate="within")

    # -------------------
    # Optional nearest fallback for unmatched points
    # -------------------
    if use_nearest_if_missing and joined["GID_2"].isna().any():
        pts_miss = joined[joined["GID_2"].isna()].copy()
        if not pts_miss.empty:
            try:
                pts_miss_m = pts_miss.to_crs(epsg=3857)
                gadm_m = gadm_small.to_crs(epsg=3857)

                nearest = gpd.sjoin_nearest(
                    pts_miss_m,
                    gadm_m,
                    how="left",
                    distance_col="dist_m",
                )

                max_m = float(nearest_max_km) * 1000.0
                nearest_ok = nearest[nearest["dist_m"] <= max_m].copy()

                for col in ["GID_2","NAME_1","NAME_2"]:
                    if col in nearest_ok.columns:
                        joined.loc[nearest_ok.index, col] = nearest_ok[col].values
            except Exception:
                # keep silent as requested (minimal printing)
                pass

    joined_ok = joined[joined["GID_2"].notna()].copy()
    if joined_ok.empty:
        print("No points could be assigned to Admin2 polygons.")
        return pd.DataFrame(
            columns=["GID_2","NAME_1","NAME_2","fatalities","total_population","vulnerable_people","pct_fatalities","pin_contrib"]
        )

    # -------------------
    # Aggregate fatalities per Admin2 (from joined points)
    # -------------------
    group_cols = ["GID_2"] + [c for c in ["NAME_1","NAME_2"] if c in joined_ok.columns]

    fat_admin2 = (
        joined_ok.groupby(group_cols, as_index=False)
                 .agg(fatalities=("fatalities", "sum"))
    )
    fat_admin2["fatalities"] = pd.to_numeric(fat_admin2["fatalities"], errors="coerce").fillna(0.0)

    # -------------------
    # Load WorldPop JSON and merge by GID_2
    # -------------------
    with open(pop_path, "r", encoding="utf-8") as f:
        pop_obj = json.load(f)

    if isinstance(pop_obj, dict) and "data" in pop_obj and isinstance(pop_obj["data"], list):
        pop_rows = pop_obj["data"]
    elif isinstance(pop_obj, list):
        pop_rows = pop_obj
    else:
        raise ValueError("Unrecognized population JSON structure (expected list or {'data':[...]}).")

    pop_df = pd.DataFrame(pop_rows).copy()

    for c in ("GID_2", "total_population", "vulnerable_people"):
        if c not in pop_df.columns:
            raise ValueError(f"Population JSON must contain '{c}' (from your extract_admin_age_sex_groups output).")

    pop_df["GID_2_norm"] = pop_df["GID_2"].astype(str).str.strip()
    pop_df["total_population"] = pd.to_numeric(pop_df["total_population"], errors="coerce")
    pop_df["vulnerable_people"] = pd.to_numeric(pop_df["vulnerable_people"], errors="coerce")

    keep_pop_cols = ["GID_2_norm", "total_population", "vulnerable_people"]
    # If pop also has names, we can prefer them
    for c in ("NAME_1", "NAME_2"):
        if c in pop_df.columns:
            keep_pop_cols.append(c)

    pop_small = pop_df[keep_pop_cols].copy()

    fat_admin2["GID_2_norm"] = fat_admin2["GID_2"].astype(str).str.strip()
    merged = fat_admin2.merge(pop_small, on="GID_2_norm", how="left", suffixes=("", "_pop"))

    # Keep only rows with valid population
    merged = merged[merged["total_population"].notna() & (merged["total_population"] > 0)].copy()
    if merged.empty:
        print("No Admin2 overlaps found between ACLED (spatially assigned) and WorldPop by GID_2.")
        return pd.DataFrame(
            columns=["GID_2","NAME_1","NAME_2","fatalities","total_population","vulnerable_people","pct_fatalities","pin_contrib"]
        )

    # Prefer pop names if present
    if "NAME_1_pop" in merged.columns and merged["NAME_1_pop"].notna().any():
        merged["NAME_1"] = merged["NAME_1_pop"]
    if "NAME_2_pop" in merged.columns and merged["NAME_2_pop"].notna().any():
        merged["NAME_2"] = merged["NAME_2_pop"]

    # -------------------
    # Compute pct + PIN contribution (UPDATED RULE)
    # -------------------
    merged["pct_fatalities"] = (merged["fatalities"] / merged["total_population"]) * 100.0  # percent units

    def _pin_contrib(row):
        pct = float(row["pct_fatalities"])  # e.g. 0.1 means 0.1%
        if pct <= 0.0:
            return 0.0
        elif pct < 1.0:
            # take pct% of vulnerable_people
            pct = pct * 100
            return (pct / 100) * float(row["vulnerable_people"])
        elif pct < 5:
            return float(row["vulnerable_people"])
        else:
            # for 5% and above, add whole population
            return float(row["total_population"])

    merged["pin_contrib"] = merged.apply(_pin_contrib, axis=1)
    PIN = float(merged["pin_contrib"].sum())

    # -------------------
    # Final output: Top N by fatalities (post-pop merge)
    # -------------------
    out_cols = ["GID_2", "NAME_1", "NAME_2", "fatalities", "total_population", "vulnerable_people", "pct_fatalities", "pin_contrib"]
    out = merged[out_cols].copy()
    out = out.sort_values("fatalities", ascending=False).head(top_n).reset_index(drop=True)

    print(f"\nTop {top_n} Admin2 by fatalities (post-pop merge):")
    print(out.to_string(index=False))

    print("\n[PIN summary]")
    print(f"PIN (rule-based) = {int(round(PIN)):,}")
    
        # -------------------
    # Totals (affected Admin2 only)
    # -------------------
    # "Affected" = Admin2 with fatalities > 0
    affected = merged[merged["fatalities"] > 0].copy()

    total_population_affected = float(affected["total_population"].sum())
    total_vulnerable_affected = float(affected["vulnerable_people"].sum())
    PIN_total = float(merged["pin_contrib"].sum())  # already rule-based

    print("\n[Totals - affected Admin2 only]")
    print(f"Total population (affected): {int(round(total_population_affected)):,}")
    print(f"Vulnerable population (affected): {int(round(total_vulnerable_affected)):,}")
    print(f"People in Need (PIN): {int(round(PIN_total)):,}")

    # -------------------
    # Bar plot
    # -------------------
    import matplotlib.pyplot as plt

    labels = [
        "Total Population\n(affected)",
        "Vulnerable Population\n(affected)",
        "People in Need (PIN)"
    ]
    values = [
        total_population_affected,
        total_vulnerable_affected,
        PIN_total
    ]

    plt.figure(figsize=(10, 5))
    bars = plt.bar(labels, values)

    plt.ylabel("People")
    plt.title("Affected Population & PIN Summary")

    # Add formatted labels above bars
    ax = plt.gca()
    ymax = max(values) if values else 0
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.01 * ymax,
            f"{int(round(val)):,}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    plt.show()

    return out


### with DCRM 
def extract_admin2_with_index(
    summary_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    From the summary_df returned by gadm_match_locations,
    return a DataFrame containing ONLY:
        - GID_2
        - NAME_2
        - admin1_index

    The admin1_index is derived from NAME_1 via an internal dictionary.
    """

    gpkg_path  = '/eos/jeodpp/home/users/mihadar/data/Geospacial/gadm_410.gpkg'

    admin1_dict = {
        "North Darfur": 9.9,
        "South Darfur": 9.0,
        "North Kurdufan": 8.1,
        "West Kurdufan": 7.8,
        "South Kurdufan": 7.3,
        "Khartoum": 6.8,
        "White Nile": 5.4,
        "West Darfur": 5.3,
        "East Darfur": 5.2,
        "Al Jazirah": 5.1,
        "Central Darfur": 5.0,
        "Al Qadarif": 3.8,
        "River Nile": 3.7,
        "Blue Nile": 3.4,
        "Sennar": 3.3,
        "Northern": 2.7,
        "Red Sea": 1.4,
        "Kassala": 1.3
    }

    required_cols = {"GID_2", "NAME_2", "NAME_1"}
    missing = required_cols - set(summary_df.columns)
    if missing:
        raise ValueError(f"Missing required columns in summary_df: {missing}")

    df = summary_df.copy()

    # Map Admin-1 → index
    df["admin1_index"] = df["NAME_1"].map(admin1_dict)

    # Keep only requested columns
    result = df[["GID_2", "NAME_2", "admin1_index"]].drop_duplicates()

    return result.reset_index(drop=True)


def compute_pin_with_table(
    summary_df_with_index: pd.DataFrame,
    worldpop_results_df: pd.DataFrame,
    *,
    gid_col: str = "GID_2",
    name_col: str = "NAME_2",
    index_col: str = "admin1_index",
    total_col: str = "total_population",
    vuln_col: str = "vulnerable_people",
):
    # --- Ensure wide table displays fully ---
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 2000)
    pd.set_option("display.expand_frame_repr", False)

    # --- Keep only required columns ---
    s = summary_df_with_index[[gid_col, name_col, index_col]].copy()
    w = worldpop_results_df[[gid_col, total_col, vuln_col]].copy()

    # --- Force ONE row per GID_2 ---
    s = s.groupby(gid_col, as_index=False).agg({
        name_col: "first",
        index_col: "first"
    })

    w[total_col] = pd.to_numeric(w[total_col], errors="coerce").fillna(0)
    w[vuln_col] = pd.to_numeric(w[vuln_col], errors="coerce").fillna(0)

    w = w.groupby(gid_col, as_index=False).agg({
        total_col: "sum",
        vuln_col: "sum"
    })

    df = s.merge(w, on=gid_col, how="left").fillna(0)

    used_population = []
    used_type = []
    pin = 0

    for _, row in df.iterrows():
        idx = row[index_col]

        if idx > 8:
            val = int(row[total_col])
            used_population.append(val)
            used_type.append("total_population")
            pin += val

        elif 5 < idx < 8:
            val = int(row[vuln_col])
            used_population.append(val)
            used_type.append("vulnerable_people")
            pin += val

        else:
            print(f"{row[name_col]} below conflict risk index")
            used_population.append(0)
            used_type.append("excluded")

    df["used_type"] = used_type
    df["used_population"] = used_population

    debug_table = df[
        [gid_col, name_col, index_col, total_col, vuln_col, "used_type", "used_population"]
    ]

    pin_formatted = f"{pin:,}"

    # Print clean full table
    print("\n===== PIN DEBUG TABLE =====\n")
    print(debug_table.to_string(index=False))
    print("\nEstimated PIN:", pin_formatted)

    return debug_table, pin_formatted