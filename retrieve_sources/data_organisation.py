import json
import os
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import math
from geoparser import Geoparser
import geopandas as gpd
import matplotlib.colors as mcolors
import numpy as np
import matplotlib.cm as cm
from matplotlib import colormaps
from datetime import datetime
import matplotlib.ticker as mticker


base_folder = f'/eos/jeodpp/home/users/mihadar/data/'
country_folder =  os.path.join(base_folder, '/per_country')
event_folder = os.path.join(base_folder, '/case_studies')


def ensure_country_event_data(country: str, iso3: str) -> str:
    """
    Ensures /.../case_study/{country}/all_country_data exists and contains
    iso3_ACLED.json, iso3_IDMC.json.
    If not, extracts them.

    Returns:
        country_folder (str): Path to the all_country_data folder.
    """
    expected_files = {
        "ACLED":     os.path.join(country_folder, f"{iso3}_ACLED.json"),
        "IDMC":      os.path.join(country_folder, f"{iso3}_IDMC.json"),
    }

    # If folder exists and all files exist, do nothing
    if os.path.isdir(country_folder) and all(os.path.isfile(p) for p in expected_files.values()):
        return country_folder

    # Otherwise, (re)create folder and extract
    os.makedirs(country_folder, exist_ok=True)

    data_dir = "/eos/jeodpp/home/users/mihadar/data/"
    acled_path = os.path.join(data_dir, "ACLED", "ACLED_events_by_country.json")
    idmc_path = os.path.join(data_dir, "IDMC", "idmc_organised.json")

    new_file = extract_country_events(acled_path, iso3, "ACLED", output_folder=country_folder)
    print(f"Saved filtered ACLED file: {new_file}")

    new_file = extract_country_events(idmc_path, iso3, "IDMC", output_folder=country_folder)
    print(f"Saved filtered IDMC file: {new_file}")

    return country_folder


def extract_country_events(file_path, iso3, source_title, output_folder):
    """
    Extract events for a specific country from a JSON file and save to output_folder.

    Returns:
    - output_file_path (str): Path to the saved JSON file.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Load the original JSON
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Extract events for the specific country
    country_events = data.get(iso3)

    # clean data
    clean_events = replace_nan_with_none(country_events)

    if clean_events is None:
        raise ValueError(f"No data found for country code '{iso3}' in {file_path}")

    # Define output file path
    output_file_path = os.path.join(output_folder, f"{iso3}_{source_title}.json")

    # Save only the events (drop country layer)
    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump(clean_events, f, ensure_ascii=False, indent=4)

    return output_file_path


def replace_nan_with_none(obj):
    if isinstance(obj, dict):
        return {k: replace_nan_with_none(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [replace_nan_with_none(v) for v in obj]
    elif isinstance(obj, float) and math.isnan(obj):
        return None
    else:
        return obj