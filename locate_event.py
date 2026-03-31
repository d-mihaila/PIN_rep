import os
import json
import csv
import pandas as pd
import reverse_geocoder as rg
from geopy.distance import geodesic
from geolocation import * # for the extract location thingy 
from pathlib import Path
from typing import Dict, Any, Tuple, List
import geopandas as gpd
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.lines as mlines
import matplotlib.ticker as mticker
import matplotlib.dates as mdates


# eventually take in the jupyter notebook for testing_displacement.ipynb please 
    # and double check if the other files form IDMC can also be useful. 

    
iso3 = 'SDN'
country = 'Sudan'
data_dir = '/eos/jeodpp/home/users/mihadar/data/'
out_path = os.path.join(data_dir, f'for report/{iso3}/{iso3}_idmc_filtered.json')
data_dir = '/eos/jeodpp/home/users/mihadar/data/'
path_to_idmc_organised = os.path.join(data_dir, f'for report/{iso3}/{iso3}_IDMC.json')

# fetch the information for the country, region and date. 
# What countries might be relevant -- look at locations

def relevant_countries(locations, country_iso2, radius=1000):
    """
    Given a list of locations and a primary ISO2 code, returns a list 
    of unique ISO2 codes within the radius by probing points 
    along the circular boundary.
    """
    # Start with the primary country (normalized to uppercase)
    relevant_iso2_list = {country_iso2.upper()}
    
    # We sample the boundary every 30 degrees (12 points around the circle)
    # This provides a more 'circular' check than just N, S, E, W
    bearings = range(0, 360, 30) 

    for loc in locations:
        lat, lon = loc['latitude'], loc['longitude']
        center_point = (lat, lon)
        
        # 1. Generate the 'probe' coordinates
        # We start with the center point itself
        coords_to_check = [center_point]
        
        # 2. Add points along the circumference of the radius
        for bearing in bearings:
            # Calculate the point on the edge of the circle
            destination = geodesic(kilometers=radius).destination(center_point, bearing)
            coords_to_check.append((destination.latitude, destination.longitude))
        
        # 3. Batch search for all points (center + 12 circumference points)
        # reverse_geocoder uses an offline KD-tree, so searching 13 points is nearly instant
        results = rg.search(coords_to_check)
        
        for res in results:
            # Extract the 2-letter country code
            found_cc = res.get('cc', '').upper()
            
            if found_cc:
                relevant_iso2_list.add(found_cc)

    return list(relevant_iso2_list)



# countries in x temporal window
def fetch_displacements(event_start_date, iso3_input, past_look_years=0, past_look_months=0, past_look_weeks=0):
    """
    Reads from the raw IDMC list and filters by a single ISO3 or a list of ISO3s.
    Returns a dictionary of matched events within the temporal window.
    """
    # 1. Setup the temporal window (looking backwards)
    anchor_dt = pd.to_datetime(event_start_date)
    lookback_start_dt = anchor_dt - pd.DateOffset(
        years=past_look_years, 
        months=past_look_months, 
        weeks=past_look_weeks
    )
    
    # 2. Handle ISO3 input: ensure it's a list even if a single string is passed
    if isinstance(iso3_input, str):
        iso3_list = [iso3_input]
    else:
        iso3_list = iso3_input

    file_path = '/eos/jeodpp/home/users/mihadar/data/IDMC/idmc_displacements_raw.json'
    matched_events = {}

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        # 3. Iterate through the LIST of events
        for entry in raw_data:
            # CHECK: Is this event's country in our target list?
            if entry.get("iso3") in iso3_list:
                disp_start = entry.get("displacement_start_date")
                disp_end = entry.get("displacement_end_date")

                # Helper to check if a date falls within the lookback window
                def is_in_range(date_str):
                    if not date_str or date_str == 'x':
                        return False
                    try:
                        dt = pd.to_datetime(date_str)
                        return lookback_start_dt <= dt <= anchor_dt
                    except:
                        return False

                # 4. Filter Logic: If either start or end is in range, add to dictionary
                if is_in_range(disp_start) or is_in_range(disp_end):
                    start_label = disp_start if disp_start else "x"
                    end_label = disp_end if disp_end else "x"
                    
                    # Include the ISO3 in the title so you know which country it belongs to
                    event_iso = entry.get("iso3")
                    title = f"[{event_iso}] {start_label} - {end_label}"
                    
                    # Ensure unique keys
                    if title in matched_events:
                        title = f"{title} (ID: {entry.get('id')})"
                    
                    matched_events[title] = entry

        # After the loop is finished:
        output_path = '/eos/jeodpp/home/users/mihadar/data/IDMC/matched_events.json'

        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Write the file (using 'w' to overwrite)
        with open(output_path, 'w', encoding='utf-8') as out_f:
            json.dump(matched_events, out_f, indent=4, ensure_ascii=False)

        print(f"SUCCESS: Matched events saved to {output_path}")

        return matched_events

    except Exception as e:
        print(f"An ERROR occurred: {e}")
        return {}
    
def clean_matched_events():
    file_path = '/eos/jeodpp/home/users/mihadar/data/IDMC/matched_events.json'
    
    # Define the exact order and list of indicators you want to keep
    target_indicators = [
        "latitude", "longitude", "figure", "displacement_type", 
        "category", "subcategory", "type", "subtype", 
        "event_start_date", "event_end_date", "locations_name", 
        "standard_popup_text", "standard_info_text", "sources", 
        "source_url", "role"
    ]

    try:
        # 1. Load the existing matched events
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        cleaned_data = {}

        # 2. Process each event
        for event_key, event_details in data.items():
            # Create a new dictionary for the event with only target indicators in order
            cleaned_event = {indicator: event_details.get(indicator) for indicator in target_indicators}
            cleaned_data[event_key] = cleaned_event

        # 3. Save the cleaned data back to the file (overwrite)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(cleaned_data, f, indent=4, ensure_ascii=False)

        print(f"SUCCESS: Cleaned data saved to {file_path}")
        print(f"Processed {len(cleaned_data)} events.")

    except FileNotFoundError:
        print(f"ERROR: The file {file_path} was not found.")
    except Exception as e:
        print(f"An ERROR occurred: {e}")

# filtering the gathered events by location more restrictedlt... 
def filter_matched_events(latitudes, longitudes, radius=100):
    """
    Filters the matched_events.json file. If an event is further than 
    'radius' km from ALL of the provided lat/long pairs, it is deleted.
    """
    file_path = '/eos/jeodpp/home/users/mihadar/data/IDMC/matched_events.json'
    
    # 1. Prepare the investigation points as a list of tuples
    investigation_points = list(zip(latitudes, longitudes))
    
    try:
        # 2. Load the current matched events
        if not os.path.exists(file_path):
            print(f"Error: {file_path} does not exist.")
            return
            
        with open(file_path, 'r', encoding='utf-8') as f:
            matched_events = json.load(f)

        # 3. Identify which keys to keep
        keys_to_keep = {}
        
        for event_key, event_data in matched_events.items():
            event_lat = event_data.get('latitude')
            event_lon = event_data.get('longitude')
            
            if event_lat is None or event_lon is None:
                continue # Skip events with no coordinates
            
            event_coord = (event_lat, event_lon)
            is_nearby = False
            
            # Check the event against every investigation point
            for point in investigation_points:
                distance = geodesic(event_coord, point).kilometers
                if distance <= radius:
                    is_nearby = True
                    break # We found one match, no need to check other points for this event
            
            if is_nearby:
                keys_to_keep[event_key] = event_data

        # 4. Overwrite the file with the filtered results
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(keys_to_keep, f, indent=4, ensure_ascii=False)

        print(f"Filtering complete. Radius: {radius}km.")
        print(f"Events remaining: {len(keys_to_keep)} (Deleted {len(matched_events) - len(keys_to_keep)})")

    except Exception as e:
        print(f"An error occurred during filtering: {e}")
        
    
# preprocessing the file    

def idmc_organisation(path_to_original):
    """
    Groups IDMC displacement events by ISO3 code and sorts them 
    chronologically (newest to oldest) within each country.
    """

    try:
        with open(path_to_original, 'r') as f:
            raw_data = json.load(f)

        grouped = {}
        for entry in raw_data:
            # Determine ISO3 code
            iso3 = entry.get("iso3")
            if not iso3:
                country_name = entry.get("country", "").lower()
                iso3 = COUNTRY_TO_ISO3.get(country_name, "UNKNOWN")

            if iso3 not in grouped:
                grouped[iso3] = []
            grouped[iso3].append(entry)

        # Sort the outer dictionary keys (ISO3 codes) alphabetically
        sorted_iso3_keys = sorted(grouped.keys())

        # Build the final structure and sort events within each country
        organised_data = {}
        for code in sorted_iso3_keys:
            # Sort events from newest to oldest by displacement_start_date
            events = sorted(
                grouped[code], 
                key=lambda x: x.get("displacement_start_date", ""), 
                reverse=True
            )
            organised_data[code] = events

        # Write to the new file
        with open('/eos/jeodpp/home/users/mihadar/data/IDMC/idmc_organised.json', 'w') as out_f:
            json.dump(organised_data, out_f, indent=4)

        return "Success: File 'idmc_organised.json' has been created."

    except Exception as e:
        return f"Error: {str(e)}"


    
def write_idmc_to_csv(database_path):
    idmc_filtered = '/eos/jeodpp/home/users/mihadar/data/IDMC/matched_events.json'
    
    # Check if the source JSON exists
    if not os.path.exists(idmc_filtered):
        print(f"Error: {idmc_filtered} not found.")
        return

    # Check if database file exists for header logic
    file_exists = os.path.exists(database_path)
    
    try:
        with open(idmc_filtered, 'r', encoding='utf-8') as f:
            matched_events = json.load(f)

        rows_to_write = []

        for event_key, data in matched_events.items():
            # 1. Construct Data Source string: iso3_subtype_locations_name
            # Extract iso3 from the key (e.g., "[PAK]") or from data if available
            iso3 = event_key.split(']')[0].replace('[', '') if ']' in event_key else "UNKNOWN"
            subtype = data.get('subtype')
            loc_name = data.get('locations_name')
            
            # Filter out nulls and join with underscores
            source_parts = [iso3]
            if subtype: source_parts.append(subtype)
            if loc_name: source_parts.append(loc_name)
            data_source = "_".join(source_parts) + 'IDMC'

            # 2. Map Indicators to values
            # figure -> 'number displaced'
            rows_to_write.append([data_source, 'number displaced', data.get('figure')])
            
            # latitude, longitude -> 'location' with [lat, long]
            lat = data.get('latitude')
            lon = data.get('longitude')
            rows_to_write.append([data_source, 'location', f"[{lat}, {lon}]"])
            
            # event_start_date
            rows_to_write.append([data_source, 'event_start_date', data.get('event_start_date')])
            
            # event_end_date
            rows_to_write.append([data_source, 'event_end_date', data.get('event_end_date')])
            
            # source_url
            rows_to_write.append([data_source, 'source_url', data.get('source_url')])

        # 3. Write to CSV
        with open(database_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            if not file_exists:
                writer.writerow(['Data source', 'indicator', 'value'])
            
            writer.writerows(rows_to_write)

        print(f"Successfully written {len(rows_to_write)} IDMC rows to: {database_path}")

    except Exception as e:
        print(f"An error occurred while writing to CSV: {e}")

 


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
    predicate: str = "within",  # use "intersects" for border-touching points
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Filter IDMC displacement events for ONE country (iso3) by:
      1) time window (backwards from event_start_date)
      2) event point inside ANY matched_polys geometry

    Saves JSON with only matched events and returns:
      (matched_events_dict, counts_df)

    counts_df summarizes #events per polygon (if UID/GID available).
    """

    if matched_polys is None or matched_polys.empty:
        raise ValueError("matched_polys is empty — no admin regions to filter within.")

    # ---- time window ----
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

    # ---- load raw IDMC list ----
    with open(idmc_raw_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    if not isinstance(raw_data, list):
        raise ValueError("Expected idmc_displacements_raw.json to be a LIST of events.")

    # ---- filter by iso3 + time window + coords ----
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

        candidates.append({
            "_entry": entry,
            "lat": lat,
            "lon": lon,
        })

    # If nothing passes time+coords, save empty result
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

    # ---- points gdf ----
    pts_df = pd.DataFrame(candidates)
    pts_gdf = gpd.GeoDataFrame(
        pts_df,
        geometry=gpd.points_from_xy(pts_df["lon"], pts_df["lat"]),
        crs="EPSG:4326"
    )

    # ---- polygons CRS ----
    polys = matched_polys.copy()
    if polys.crs is None:
        polys = polys.set_crs(epsg=4326)
    elif polys.crs.to_epsg() != 4326:
        polys = polys.to_crs(epsg=4326)

    # Choose a polygon identifier for counts/traceability
    poly_id_col = None
    for c in ["UID", "GID_5", "GID_4", "GID_3", "GID_2", "GID_1"]:
        if c in polys.columns:
            poly_id_col = c
            break

    polys_for_join = polys[[poly_id_col, "geometry"]].copy() if poly_id_col else polys[["geometry"]].copy()

    # ---- spatial join: keep points inside ANY polygon ----
    joined = gpd.sjoin(pts_gdf, polys_for_join, how="inner", predicate=predicate)

    # collect matched entries (preserve original dict)
    matched_entries = []
    for _, r in joined.iterrows():
        e = r["_entry"]
        # add traceability fields (optional)
        if poly_id_col:
            e = dict(e)  # copy so we don't mutate original
            e["_matched_polygon_id"] = r.get(poly_id_col)
        matched_entries.append(e)

    # de-duplicate (by ID if present, else by stable JSON)
    def _event_key(e: dict):
        for k in ["id", "event_id", "record_id"]:
            if k in e:
                return (k, e[k])
        return ("_fallback", json.dumps(e, sort_keys=True, ensure_ascii=False))

    uniq = {}
    for e in matched_entries:
        uniq[_event_key(e)] = e
    matched_entries = list(uniq.values())

    
    print(f"[DEBUG] len(matched_entries): {len(matched_entries)}")
    if len(matched_entries) > 0:
        print(f"[DEBUG] first entry keys: {list(matched_entries[0].keys())}")
        print(f"[DEBUG] sample _matched_polygon_id values: "
              f"{[e.get('_matched_polygon_id') for e in matched_entries[:5]]}")
    else:
        print("[DEBUG] matched_entries is EMPTY — no events matched any polygon")

    counts_df = (
        pd.DataFrame(matched_entries)
        .dropna(subset=["_matched_polygon_id"]))
    
    # counts per polygon (if id col exists)
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
            "n_events_with_coords": len(candidates),  # same as above after filtering
            "n_events_inside_regions": len(matched_entries),
            "polygon_id_col": poly_id_col,
        },
        "events": matched_entries
    }

    Path(out_json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    return out, counts_df

def plot_idmc_events_map(
    idmc_out,
    matched_poly: gpd.GeoDataFrame,
    gpkg_path: str,
    iso3: str,
    title: str = "",
    min_marker_size: float = 10,
    max_marker_size: float = 300,
) -> plt.Figure:

    # ── 1. flatten — handle all dict formats ─────────────────────────────────
    all_records = []

    # format A: {"meta":..., "events": [...]}
    if "events" in idmc_out:
        for event in idmc_out["events"]:
            if not isinstance(event, dict):
                continue
            lat  = event.get("latitude")
            lon  = event.get("longitude")
            fig  = event.get("figure")
            date = event.get("displacement_date") or event.get("event_start_date")
            if lat is None or lon is None:
                continue
            try:
                lat = float(lat)
                lon = float(lon)
            except (TypeError, ValueError):
                continue
            try:
                fig = float(fig) if fig not in (None, "x", "", "null") else None
            except (TypeError, ValueError):
                fig = None
            all_records.append({"event_date": date, "lat": lat, "lon": lon, "figure": fig})

    # format B: {"per_query": {query: {"event_records": [...]}}}
    elif "per_query" in idmc_out:
        for query, qdata in idmc_out["per_query"].items():
            if not isinstance(qdata, dict):
                continue
            for rec in qdata.get("event_records", []):
                all_records.append(rec)

    # format C: raw {title_string: event_dict, ...}
    else:
        for key, event in idmc_out.items():
            if not isinstance(event, dict):
                continue
            lat  = event.get("latitude")
            lon  = event.get("longitude")
            fig  = event.get("figure")
            date = event.get("displacement_date") or event.get("event_start_date")
            if lat is None or lon is None:
                continue
            try:
                lat = float(lat)
                lon = float(lon)
            except (TypeError, ValueError):
                continue
            try:
                fig = float(fig) if fig not in (None, "x", "", "null") else None
            except (TypeError, ValueError):
                fig = None
            all_records.append({"event_date": date, "lat": lat, "lon": lon, "figure": fig})

    if not all_records:
        raise ValueError("No event_records found in idmc_out — nothing to plot.")

    rec_df = pd.DataFrame(all_records).drop_duplicates(subset=["event_date", "lat", "lon"])
    rec_df["figure"] = pd.to_numeric(rec_df["figure"], errors="coerce")

    print(f"[MAP] Total unique IDMC events to plot: {len(rec_df)}")
    print(f"[MAP] Events with figure:               {rec_df['figure'].notna().sum()}")
    print(f"[MAP] Figure range: {rec_df['figure'].min():,.0f} – {rec_df['figure'].max():,.0f}")

    # ── 2. load + dissolve admin-2 ────────────────────────────────────────────
    gdf         = gpd.read_file(gpkg_path, layer="gadm_410")
    country_gdf = gdf[gdf["GID_0"] == iso3]
    if country_gdf.empty:
        raise ValueError(f"No GADM data for {iso3}")
    admin2_gdf = country_gdf.dissolve(by="GID_2", as_index=False)
    if admin2_gdf.crs is None:
        admin2_gdf = admin2_gdf.set_crs(epsg=4326)
    elif admin2_gdf.crs.to_epsg() != 4326:
        admin2_gdf = admin2_gdf.to_crs(epsg=4326)

    # ── 3. sum figure per admin-2 ─────────────────────────────────────────────
    ev_gdf = gpd.GeoDataFrame(
        rec_df,
        geometry=gpd.points_from_xy(rec_df["lon"], rec_df["lat"]),
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(
        admin2_gdf[["GID_2", "geometry"]],
        ev_gdf[["figure", "geometry"]],
        how="left",
        predicate="contains",
    )
    fig_by_admin2 = (
        joined.groupby("GID_2")["figure"]
        .sum(min_count=1)
        .rename("figure_sum")
    )
    admin2_gdf = admin2_gdf.join(fig_by_admin2, on="GID_2")

    print(f"[MAP] Admin-2 regions with figure_sum > 0: "
          f"{(admin2_gdf['figure_sum'] > 0).sum()} / {len(admin2_gdf)}")

    # ── 4. CRS alignment for matched_poly ─────────────────────────────────────
    polys = matched_poly.copy()
    if polys.crs is None:
        polys = polys.set_crs(epsg=4326)
    elif polys.crs.to_epsg() != 4326:
        polys = polys.to_crs(epsg=4326)

    # ── 5. marker size scaled to figure ──────────────────────────────────────
    fig_vals   = rec_df["figure"].fillna(0).values
    fmin, fmax = fig_vals.min(), fig_vals.max()
    if fmax > fmin:
        sizes = min_marker_size + (fig_vals - fmin) / (fmax - fmin) * (max_marker_size - min_marker_size)
    else:
        sizes = np.full(len(fig_vals), (min_marker_size + max_marker_size) / 2)

    # ── 6. choropleth colormap ────────────────────────────────────────────────
    choro_vals = admin2_gdf["figure_sum"].dropna()
    if choro_vals.empty or choro_vals.max() == 0:
        print("[MAP] WARNING: all figure values are null/zero — choropleth will be flat.")
        norm_choro = mcolors.Normalize(vmin=0, vmax=1)
    else:
        norm_choro = mcolors.LogNorm(
            vmin=max(1, choro_vals.min()),
            vmax=choro_vals.max(),
        )
    cmap_choro = cm.get_cmap("YlGnBu")

    # ── 7. plot ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 12))

    admin2_gdf.plot(
        column="figure_sum",
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
        color="crimson",
        edgecolors="black",
        linewidths=0.4,
        alpha=0.85,
        zorder=5,
    )

    # ── colorbar with explicit readable ticks ────────────────────────────────
    sm_choro = cm.ScalarMappable(cmap=cmap_choro, norm=norm_choro)
    sm_choro.set_array([])
    cbar = fig.colorbar(sm_choro, ax=ax, shrink=0.5, pad=0.01, aspect=30)
    cbar.set_label("Displaced People (sum per Admin-2)", fontsize=11)

    if not choro_vals.empty and choro_vals.max() > 0:
        vmin_cb   = max(1, choro_vals.min())
        vmax_cb   = choro_vals.max()
        log_ticks = np.logspace(np.log10(vmin_cb), np.log10(vmax_cb), num=6)
        log_ticks = np.unique([round(t, -int(np.floor(np.log10(t)))) for t in log_ticks])
        cbar.set_ticks(log_ticks)
        cbar.set_ticklabels([f"{int(t):,}" for t in log_ticks])
        cbar.ax.tick_params(labelsize=10)

    # ── dot size legend ───────────────────────────────────────────────────────
    if fmax > fmin:
        legend_vals  = np.linspace(fmin, fmax, 4).astype(int)
        legend_sizes = min_marker_size + (legend_vals - fmin) / (fmax - fmin) * (max_marker_size - min_marker_size)
        legend_handles = [
            mlines.Line2D([], [], marker="o", color="w", markerfacecolor="crimson",
                          markeredgecolor="black", markersize=np.sqrt(s),
                          label=f"{v:,.0f} displaced")
            for v, s in zip(legend_vals, legend_sizes)
        ]
        ax.legend(handles=legend_handles, title="Dot size = displaced people",
                  loc="lower left", fontsize=9, framealpha=0.8)

    ax.set_title(
        f"{title}\nDisplaced People (dots) & Displacement by Admin-2 colormap",
        fontsize=14, fontweight="bold",
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal")
    ax.grid(True, linewidth=0.3, alpha=0.5)
    plt.tight_layout()

    return fig

#### 2 weeks time series: 
def plot_idmc_timeseries_2weeks(
    idmc_out: dict,
    end_date: str,
    title: str = "",
) -> Tuple[plt.Figure, float]:
    """
    Time series of IDMC displacement events in the last 2 weeks before end_date.
    - X axis: event date
    - Y axis: figure (displaced people)
    - One color per admin-2 region (query)
    - Square marker at peak event per admin-2
    - Transparent circles for all other events
    - Polynomial line of best fit per admin-2
    - Legend outside the figure on the right
    - Total sum of ALL figures across all events printed in title

    Parameters
    ----------
    idmc_out : dict returned by fetch_idmc_displacements_by_admin
    end_date : str "YYYY-MM-DD" — the anchor date (2-week window ends here)
    title    : str — plot title

    Returns
    -------
    fig             : matplotlib Figure
    total_figure    : float — sum of ALL figure values in the window
    """

    end_dt   = pd.to_datetime(end_date)
    start_dt = end_dt - pd.Timedelta(weeks=2)

    # ── 1. flatten events from idmc_out — handle all formats ─────────────────
    # We need both a per-query grouping (for colors) AND the raw records.
    # idmc_out["events"] is a flat list; each event has a UID/polygon_id
    # so we group by that to get per-admin coloring.

    raw_events = []

    if "events" in idmc_out:
        raw_events = idmc_out["events"]
    elif "per_query" in idmc_out:
        for qdata in idmc_out["per_query"].values():
            if isinstance(qdata, dict):
                raw_events.extend(qdata.get("event_records", []))
    else:
        # format C: raw {title: event_dict}
        for key, event in idmc_out.items():
            if isinstance(event, dict):
                raw_events.append(event)

    # ── 2. filter to 2-week window and build per-admin groups ─────────────────
    # use locations_name or polygon_id as the grouping label
    query_records: Dict[str, pd.DataFrame] = {}
    admin_rows: Dict[str, list] = {}

    for event in raw_events:
        # date — try displacement_date first, then event_start_date
        date_str = event.get("displacement_date") or event.get("event_start_date")
        if not date_str:
            continue
        dt = pd.to_datetime(date_str, errors="coerce")
        if pd.isna(dt) or dt < start_dt or dt > end_dt:
            continue

        fig = event.get("figure")
        try:
            fig = float(fig) if fig not in (None, "x", "", "null") else None
        except (TypeError, ValueError):
            fig = None
        if fig is None:
            continue

        # label for this admin — prefer locations_name, fall back to polygon_id
        label = (
            event.get("locations_name")
            or str(event.get("polygon_id", "unknown"))
        )

        admin_rows.setdefault(label, []).append({"date": dt, "figure": fig})

    for label, rows in admin_rows.items():
        df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
        query_records[label] = df

    if not query_records:
        raise ValueError(
            f"No IDMC events with figure found in the 2-week window "
            f"{start_dt.date()} → {end_dt.date()}"
        )

    print(f"[TIMESERIES] Window: {start_dt.date()} → {end_dt.date()}")
    print(f"[TIMESERIES] Admin regions with data: {len(query_records)}")

    # ── 3. color palette ──────────────────────────────────────────────────────
    n_queries  = len(query_records)
    cmap_lines = cm.get_cmap("tab20", n_queries)
    colors     = {q: cmap_lines(i) for i, q in enumerate(query_records)}

    # ── 4. plot ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(16, 8))

    peak_per_admin: Dict[str, float] = {}

    for query, df in query_records.items():
        color = colors[query]

        # peak first — needed for scatter split and arrow
        peak_idx = df["figure"].idxmax()
        peak_row = df.loc[peak_idx]
        peak_val = peak_row["figure"]
        peak_dt  = peak_row["date"]
        peak_per_admin[query] = peak_val

        # regular events — transparent circles
        non_peak = df.drop(index=peak_idx)
        ax.scatter(
            non_peak["date"], non_peak["figure"],
            color=color, s=45, marker="o", zorder=4, alpha=0.25,
        )

        # peak event — opaque square
        ax.scatter(
            [peak_dt], [peak_val],
            color=color, s=160, marker="s", zorder=5,
            edgecolors="black", linewidths=0.8, alpha=1.0,
        )

        # arrow to peak (only if >1 event)
        if len(df) > 1:
            ax.annotate(
                "",
                xy=(peak_dt, peak_val),
                xytext=(peak_dt, peak_val * 0.6),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.8),
                zorder=5,
            )

        # polynomial line of best fit
        date_nums = (df["date"] - df["date"].min()).dt.days.values
        fig_vals  = df["figure"].values

        if len(date_nums) >= 2:
            deg      = 2 if len(date_nums) >= 3 else 1
            coeffs   = np.polyfit(date_nums, fig_vals, deg=deg)
            x_smooth     = np.linspace(date_nums.min(), date_nums.max(), 200)
            y_smooth     = np.polyval(coeffs, x_smooth)
            dates_smooth = df["date"].min() + pd.to_timedelta(x_smooth, unit="D")
            ax.plot(
                dates_smooth, y_smooth,
                color=color, linewidth=2.0, linestyle="-",
                alpha=0.6, zorder=3,
            )

    # ── 5. total — SUM of ALL figures (not just peaks) ────────────────────────
    total_figure = sum(
        df["figure"].sum() for df in query_records.values()
    )
    print(f"\n[TIMESERIES] Total displaced people (sum of all figures): {total_figure:,.0f}")
    print(f"\n[TIMESERIES] Peak figure per admin:")
    for q, v in sorted(peak_per_admin.items(), key=lambda x: -x[1]):
        print(f"  {q:40s}: {v:>12,.0f}")

    # ── 6. legend outside on the right ───────────────────────────────────────
    legend_handles = [
        mlines.Line2D([], [], color=colors[q], marker="s", markersize=9,
                      markeredgecolor="black", linewidth=0,
                      label=f"{q}  (peak: {peak_per_admin[q]:,.0f})")
        for q in query_records
    ]
    ax.legend(
        handles=legend_handles,
        title="Admin region",
        fontsize=12,
        title_fontsize=13,
        framealpha=0.85,
        ncol=max(1, n_queries // 15),
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        borderaxespad=0,
    )

    # ── 7. axes formatting ────────────────────────────────────────────────────
    ax.set_xlim(start_dt - pd.Timedelta(days=0.5), end_dt + pd.Timedelta(days=0.5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=35, ha="right", fontsize=13)

    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    ax.tick_params(axis="y", labelsize=13)
    ax.set_xlabel("Date", fontsize=15)
    ax.set_ylabel("Displaced People (figure)", fontsize=15)
    ax.set_title(
        f"{title}\nTotal Displaced People (sum of all figures): {total_figure:,.0f}",
        fontsize=16, fontweight="bold",
    )
    ax.grid(True, linewidth=0.3, alpha=0.5)

    fig.tight_layout()
    fig.subplots_adjust(right=0.75)

    return fig, total_figure