import os
import json
import csv
import re
import logging
import traceback
from datetime import datetime
from geoparser import Geoparser
import sqlite3, os
import pandas as pd
from worldpoppy import wp_raster, bbox_from_location
import numpy as np
import xarray as xr
import requests
import numpy as np
import rioxarray as rxr

# replace if not found
db_path = os.path.expanduser("/storage/mihadar/data/geoparser/geonames.db")


'''
NOTE: locations here mean ALL locations mentioned in the articles. 
Some of them are not useful infomration about WHERE the x disaster happened. 
A function for filtering should be added. 
'''


# class GetLocation():
#     # in the end it should all be happening here
#     def __init__(self):
#         pass

#     def run:
#         pass




def troubleshoot_geodata(db_path):
    with sqlite3.connect(db_path, isolation_level=None, uri=True) as conn:
        tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)
    print(tables)


    with sqlite3.connect(db_path) as conn:
        for tbl in ["names", "locations"]:
            print(f"\n=== {tbl.upper()} SCHEMA ===")
            schema = conn.execute(
                "SELECT sql FROM sqlite_master WHERE name=?", (tbl,)
            ).fetchone()
            print(schema[0] if schema else f"No schema for {tbl}")

            print(f"\n=== FIRST 10 ROWS from {tbl} ===")
            df = pd.read_sql(f"SELECT * FROM {tbl} LIMIT 10;", conn)
            display(df)



#### TO DO HERE PLEASE SAVE LAT AND LONG! 
def get_location(languages, content, disaster, year, debug=True, data_dir="/eos/jeodpp/home/users/mihadar/data/EMM"):
    """
    Extracts geographic locations from a list of news articles using Geoparser.
    Processes all articles at once (content list), saves structured results to JSON.

    Args:
        languages (list[str]): List of language codes (e.g., ["en", "fr", "de"]).
        content (list[str]): Corresponding list of news article texts.
        disaster (str): Disaster name (for file naming).
        year (int or str): Year (for file naming).
        debug (bool): Print and log detailed debug information.
        data_dir (str): Directory to store the output JSON file.

    Returns:
        str: Path to the saved JSON file.
    """
    # Check and prepare geodata
    # troubleshoot_geodata(db_path)

    geoparser = Geoparser(gazetteer="geonames")
    geoparser.gazetteer.db_path = db_path

    # Parse ALL texts at once
    docs = geoparser.parse(content)
    results = []

    # Each doc corresponds to one text string from content
    for idx, doc in enumerate(docs):
        entry = {
            "article_index": idx,
            "language": languages[idx] if idx < len(languages) else None,
            "locations": []
        }

        longitudes = []
        latitudes = []
        toponyms = []
        for toponym, location in zip(doc.toponyms, doc.locations):
            if location:
                entry["locations"].append({
                    "toponym": toponym.text,
                    "resolved_name": location.get("name"),
                    "country_name": location.get("country_name"),
                    "feature_type": location.get("feature_type"),
                    "latitude": location.get("latitude"),
                    "longitude": location.get("longitude"),
                    "score": getattr(toponym, "score", None)
                })
                latitudes.append(location.get("latitude"))
                longitudes.append(location.get("longitude"))
                toponyms.append(toponym)
            else:
                entry["locations"].append({
                    "toponym": toponym.text,
                    "resolved_name": None
                })

        results.append(entry)

        if debug:
            print(f"\n[DEBUG] Document #{idx} ({entry['language']})")
            print(f"Text sample: {doc.text[:100]}...")
            for loc in entry["locations"]:
                print(f"  - {loc['toponym']} → {loc.get('resolved_name')} ({loc.get('country_name')})")

    # Save structured JSON file
    os.makedirs(data_dir, exist_ok=True)
    safe_disaster = disaster.lower().replace(" ", "_")
    out_path = os.path.join(data_dir, f"{safe_disaster}_{year}_locations.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    if debug:
        print(f"\n Saved {len(results)} parsed documents to {out_path}")

    return latitudes, latitudes, toponyms, out_path

def get_location_coordinates(locations, debug=True):
    """
    Resolves a list of location names into latitude and longitude coordinates
    using the same Geoparser + GeoNames gazetteer pipeline.

    Args:
        locations (list[str]): List of location name strings (e.g. ["Paris", "Berlin"]).
        debug (bool): Print debug information.

    Returns:
        tuple:
            latitudes (list[float or None])
            longitudes (list[float or None])
    """

    # Initialize geoparser
    geoparser = Geoparser(gazetteer="geonames")
    geoparser.gazetteer.db_path = db_path

    # Treat each location name as its own "document"
    docs = geoparser.parse(locations)

    latitudes = []
    longitudes = []

    for idx, doc in enumerate(docs):

        lat = None
        lon = None

        # Usually one toponym per short input string
        for toponym, location in zip(doc.toponyms, doc.locations):
            if location:
                lat = location.get("latitude")
                lon = location.get("longitude")

                if debug:
                    print(
                        f"[DEBUG] {toponym.text} → "
                        f"{location.get('name')} "
                        f"({location.get('country_name')}) "
                        f"lat={lat}, lon={lon}"
                    )
            else:
                if debug:
                    print(f"[DEBUG] {toponym.text} → NOT RESOLVED")

        latitudes.append(lat)
        longitudes.append(lon)

    return latitudes, longitudes


def get_lang_cont(json_path: str):
    """
    Extracts languages and content from a JSON file.
    Replaces apostrophes in the content with spaces.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        docs = json.load(f)

    languages = [doc.get("language", "") for doc in docs]
    content = [
        doc.get("page_content", "").replace("'", " ") for doc in docs
    ]

    return languages, content


def clean_locations(locations):
    '''Here we make sure that the locations are actually within the range of the disasters 
    - news can also mention reporting countries, or helping ones - which we should not take into account 
    when calculating the local PIN number.'''
    # we take the country or the locations and test if the geolocation computed is more than x km away from that
    pass
    
    

## according to the location (longitude and latitude really), get the population information

def region_info(location, radius_km=1, year=2020, iso3=None,
                data_dir="/eos/jeodpp/home/users/mihadar/data/geopop"):
    """
    Extract population information from a locally stored WorldPop TIFF file.

    Parameters
    ----------
    location : list or tuple
        [longitude, latitude]
    radius_km : float
        Radius of area around the point (km)
    year : int
        Year of WorldPop data
    iso3 : str
        ISO3 country code (mandatory for offline mode)
    data_dir : str
        Directory where TIFF files are stored

    Returns
    -------
    dict with:
        - pixel_population
        - total_population_radius_km
        - resolution_m
        - metadata
    """

    if iso3 is None:
        raise ValueError("You must provide iso3='ITA' (or another ISO code).")

    iso3 = iso3.upper()
    lon, lat = location

    # -------------------------------
    # 1. Load offline TIFF
    # -------------------------------
    filepath = os.path.join(data_dir, f"{iso3}_ppp_{year}.tif")
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"WorldPop file not found:\n{filepath}\n"
            f"Download it first using download_worldpop_country()."
        )

    raster = rxr.open_rasterio(filepath).squeeze()

    # Raster CRS is EPSG:4326 for WorldPop (lat/lon)
    # Ensure it's recognized
    if raster.rio.crs is None:
        raster = raster.rio.write_crs("EPSG:4326")

    # -------------------------------
    # 2. Create bounding box around point
    # -------------------------------
    bbox = bbox_from_location((lon, lat), width_km=radius_km)
    min_lon, min_lat, max_lon, max_lat = bbox

    # Crop raster to bounding box
    cropped = raster.rio.clip_box(minx=min_lon, miny=min_lat,
                                  maxx=max_lon, maxy=max_lat)

    # -------------------------------
    # 3. Estimate raster resolution (m)
    # -------------------------------
    try:
        dx = float(abs(raster.x[1] - raster.x[0]))
        dy = float(abs(raster.y[1] - raster.y[0]))
        meters_per_deg_lat = 111320
        meters_per_deg_lon = 111320 * np.cos(np.radians(lat))
        res_x_m = dx * meters_per_deg_lon
        res_y_m = dy * meters_per_deg_lat
        resolution_m = (res_x_m + res_y_m) / 2
    except:
        resolution_m = None

    # -------------------------------
    # 4. Pixel value at (lon, lat)
    # -------------------------------
    pixel_value = raster.sel(x=lon, y=lat, method="nearest").values.item()

    # -------------------------------
    # 5. Total population in the radius box
    # -------------------------------
    total_pop = float(np.nansum(cropped.values))

    return {
        "pixel_population": pixel_value,
        "total_population_radius_km": total_pop,
        "resolution_m": resolution_m,
        "metadata": {
            "iso3": iso3,
            "location_lon_lat": location,
            "radius_km": radius_km,
            "year": year,
            "source_tif": filepath,
            "note": (
                "Population extracted from offline WorldPop raster via rioxarray. "
                "Pixel is nearest raster cell. Total population is sum inside "
                "the bounding box approximating the radius."
            )
        }
    }
    
    
# to put in the actual database
db_path = os.path.expanduser("/storage/mihadar/data/geoparser/geonames.db")

def extract_unique_locations(file_path):
    """
    Extract unique locations from the JSON file.
    Removes duplicates based on toponym, latitude, and longitude.
    """
    # Read the JSON file
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Use a set to track unique locations (toponym, lat, lon)
    unique_locations = {}
    
    # Iterate through all articles
    for article in data:
        # Get locations from each article
        locations = article.get('locations', [])
        
        # Process each location
        for loc in locations:
            toponym = loc.get('toponym')
            latitude = loc.get('latitude')
            longitude = loc.get('longitude')
            
            # Create a unique key (toponym, lat, lon)
            key = (toponym, latitude, longitude)
            
            # Only add if not already present
            if key not in unique_locations:
                unique_locations[key] = {
                    'toponym': toponym,
                    'latitude': latitude,
                    'longitude': longitude
                }
    
    # Print the unique locations
    print(f"Found {len(unique_locations)} unique locations:\n")
    print(f"{'Toponym':<30} {'Latitude':<15} {'Longitude':<15}")
    print("-" * 60)
    
    for loc in unique_locations.values():
        print(f"{loc['toponym']:<30} {loc['latitude']:<15} {loc['longitude']:<15}")
    
    return list(unique_locations.values())
 
    
def write_locations_to_database(unique_locs, database_path):
    """
    Write unique locations to the database CSV file.
    
    Parameters:
    - unique_locs: List of unique location dictionaries
    - disaster: Disaster type (e.g., 'flood')
    - country: Country name (e.g., 'pakistan')
    - month: Month number or name
    - year: Year (e.g., 2022)
    """

    # Create the CSV file with headers
    with open(database_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Data source', 'indicator', 'value'])
        
        # Write each unique location
        for loc in unique_locs:
            toponym = loc['toponym']
            latitude = loc['latitude']
            longitude = loc['longitude']
            
            # Format value as [Latitude, Longitude]
            value = f"[{latitude}, {longitude}]"
            
            # Write row: 'geoparser', toponym, [lat, lon]
            writer.writerow(['geoparser', toponym, value])
    
    print(f"\nDatabase created at: {database_path}")
    print(f"Written {len(unique_locs)} locations to database")


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
    
    
COUNTRY_TO_ISO3 = {
    "afghanistan": "AFG",
    "albania": "ALB",
    "algeria": "DZA",
    "andorra": "AND",
    "angola": "AGO",
    "antigua and barbuda": "ATG",
    "argentina": "ARG",
    "armenia": "ARM",
    "australia": "AUS",
    "austria": "AUT",
    "azerbaijan": "AZE",
    "bahamas": "BHS",
    "bahrain": "BHR",
    "bangladesh": "BGD",
    "barbados": "BRB",
    "belarus": "BLR",
    "belgium": "BEL",
    "belize": "BLZ",
    "benin": "BEN",
    "bhutan": "BTN",
    "bolivia": "BOL",
    "bosnia and herzegovina": "BIH",
    "botswana": "BWA",
    "brazil": "BRA",
    "brunei": "BRN",
    "bulgaria": "BGR",
    "burkina faso": "BFA",
    "burundi": "BDI",
    "cabo verde": "CPV",
    "cambodia": "KHM",
    "cameroon": "CMR",
    "canada": "CAN",
    "central african republic": "CAF",
    "chad": "TCD",
    "chile": "CHL",
    "china": "CHN",
    "colombia": "COL",
    "comoros": "COM",
    "congo": "COG",
    "costa rica": "CRI",
    "côte d’ivoire": "CIV",
    "croatia": "HRV",
    "cuba": "CUB",
    "cyprus": "CYP",
    "czechia": "CZE",
    "denmark": "DNK",
    "djibouti": "DJI",
    "dominica": "DMA",
    "dominican republic": "DOM",
    "ecuador": "ECU",
    "egypt": "EGY",
    "el salvador": "SLV",
    "equatorial guinea": "GNQ",
    "eritrea": "ERI",
    "estonia": "EST",
    "eswatini": "SWZ",
    "ethiopia": "ETH",
    "fiji": "FJI",
    "finland": "FIN",
    "france": "FRA",
    "gabon": "GAB",
    "gambia": "GMB",
    "georgia": "GEO",
    "germany": "DEU",
    "ghana": "GHA",
    "greece": "GRC",
    "grenada": "GRD",
    "guatemala": "GTM",
    "guinea": "GIN",
    "guinea-bissau": "GNB",
    "guyana": "GUY",
    "haiti": "HTI",
    "honduras": "HND",
    "hungary": "HUN",
    "iceland": "ISL",
    "india": "IND",
    "indonesia": "IDN",
    "iran": "IRN",
    "iraq": "IRQ",
    "ireland": "IRL",
    "israel": "ISR",
    "italy": "ITA",
    "jamaica": "JAM",
    "japan": "JPN",
    "jordan": "JOR",
    "kazakhstan": "KAZ",
    "kenya": "KEN",
    "kiribati": "KIR",
    "kuwait": "KWT",
    "kyrgyzstan": "KGZ",
    "laos": "LAO",
    "latvia": "LVA",
    "lebanon": "LBN",
    "lesotho": "LSO",
    "liberia": "LBR",
    "libya": "LBY",
    "liechtenstein": "LIE",
    "lithuania": "LTU",
    "luxembourg": "LUX",
    "madagascar": "MDG",
    "malawi": "MWI",
    "malaysia": "MYS",
    "maldives": "MDV",
    "mali": "MLI",
    "malta": "MLT",
    "marshall islands": "MHL",
    "mauritania": "MRT",
    "mauritius": "MUS",
    "mexico": "MEX",
    "micronesia": "FSM",
    "moldova": "MDA",
    "monaco": "MCO",
    "mongolia": "MNG",
    "montenegro": "MNE",
    "morocco": "MAR",
    "mozambique": "MOZ",
    "myanmar": "MMR",
    "namibia": "NAM",
    "nauru": "NRU",
    "nepal": "NPL",
    "netherlands": "NLD",
    "new zealand": "NZL",
    "nicaragua": "NIC",
    "niger": "NER",
    "nigeria": "NGA",
    "north korea": "PRK",
    "north macedonia": "MKD",
    "norway": "NOR",
    "oman": "OMN",
    "pakistan": "PAK",
    "palau": "PLW",
    "panama": "PAN",
    "papua new guinea": "PNG",
    "paraguay": "PRY",
    "peru": "PER",
    "philippines": "PHL",
    "poland": "POL",
    "portugal": "PRT",
    "qatar": "QAT",
    "romania": "ROU",
    "russia": "RUS",
    "rwanda": "RWA",
    "saint kitts and nevis": "KNA",
    "saint lucia": "LCA",
    "saint vincent and the grenadines": "VCT",
    "samoa": "WSM",
    "san marino": "SMR",
    "são tomé and príncipe": "STP",
    "saudi arabia": "SAU",
    "senegal": "SEN",
    "serbia": "SRB",
    "seychelles": "SYC",
    "sierra leone": "SLE",
    "singapore": "SGP",
    "slovakia": "SVK",
    "slovenia": "SVN",
    "solomon islands": "SLB",
    "somalia": "SOM",
    "south africa": "ZAF",
    "south korea": "KOR",
    "south sudan": "SSD",
    "spain": "ESP",
    "sri lanka": "LKA",
    "sudan": "SDN",
    "suriname": "SUR",
    "sweden": "SWE",
    "switzerland": "CHE",
    "syria": "SYR",
    "tajikistan": "TJK",
    "tanzania": "TZA",
    "thailand": "THA",
    "timor-leste": "TLS",
    "togo": "TGO",
    "tonga": "TON",
    "trinidad and tobago": "TTO",
    "tunisia": "TUN",
    "turkey": "TUR",
    "turkmenistan": "TKM",
    "tuvalu": "TUV",
    "uganda": "UGA",
    "ukraine": "UKR",
    "united arab emirates": "ARE",
    "united kingdom": "GBR",
    "united states": "USA",
    "uruguay": "URY",
    "uzbekistan": "UZB",
    "vanuatu": "VUT",
    "venezuela": "VEN",
    "vietnam": "VNM",
    "yemen": "YEM",
    "zambia": "ZMB",
    "zimbabwe": "ZWE"
}