import json
import os
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
from datetime import timedelta
from collections import defaultdict
from PIN.locate_event import *


# EMM
# Locations + Worldpop + INFORM Risk

# ACLED
def plot_acled_population_filtered(iso3, start_date, end_date):
    """
    Reads ACLED JSON, filters by date range, returns filtered events
    and plots rolling monthly sum of population_best over time.
    Parameters:
    - iso3: country ISO3 code
    - start_date: string "YYYY-MM-DD"
    - end_date: string "YYYY-MM-DD"
    Returns:
    - dates: list of datetime objects for each filtered event
    - population_best: list of floats for each filtered event
    - coordinates: list of [lat, lon] pairs for each filtered event
    - fig: matplotlib figure with the time series
    """

    data_dir = '/eos/jeodpp/home/users/mihadar/data/'
    acled_path = os.path.join(data_dir, f'for report/{iso3}/{iso3}_acled.json')
    with open(acled_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    if isinstance(raw_data, dict):
        events = list(raw_data.values())
    elif isinstance(raw_data, list):
        events = raw_data
    else:
        raise ValueError("Unexpected JSON structure")

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt   = datetime.strptime(end_date,   "%Y-%m-%d")

    records         = []
    dates           = []
    population_best = []
    coordinates     = []

    for event in events:
        date_str = event.get("event_date")
        pop      = event.get("population_best")

        if date_str and pop is not None:
            try:
                date_obj = datetime.strptime(date_str[:10], "%Y-%m-%d")
                pop = float(pop)
                if start_dt <= date_obj <= end_dt:

                    lat = event.get("latitude")
                    lon = event.get("longitude")
                    if lat in (None, "Nui") or lon in (None, "Nui"):
                        continue
                    try:
                        lat_f = float(lat)
                        lon_f = float(lon)
                    except (TypeError, ValueError):
                        continue

                    records.append((date_obj, pop))
                    dates.append(date_obj)
                    population_best.append(pop)
                    coordinates.append([lat_f, lon_f])

            except (ValueError, TypeError):
                continue

    if not records:
        raise ValueError(f"No records found for {iso3} between {start_date} and {end_date}")

    records.sort(key=lambda x: x[0])

    # Plot
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(dates, population_best, color='red', linewidth=2, alpha=0.9)
    ax.fill_between(dates, population_best, alpha=0.15, color='red')
    ax.set_xlim(start_dt, end_dt)
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))
    ax.set_title(f"{iso3} — Population Affected (30-day rolling window)\n{start_date} to {end_date}", fontsize=16, fontweight='bold')
    ax.set_xlabel("Date", fontsize=13)
    ax.set_ylabel("Population Best (sum)", fontsize=13)
    ax.grid(True)
    fig.tight_layout()

    return dates, population_best, coordinates, fig

# IDMC 
def plot_idmc_filtered(iso3, start_date, end_date):
    """
    Reads IDMC JSON, filters by date range, plots rolling 30-day sum of 'figure',
    and returns dates, figures, coordinates and the plot.
    Parameters:
    - iso3: country ISO3 code
    - start_date: string "YYYY-MM-DD"
    - end_date: string "YYYY-MM-DD"
    Returns:
    - dates: list of datetime objects for each filtered event
    - figures: list of floats for each filtered event
    - coordinates: list of [lat, lon] pairs for each filtered event
    - fig: matplotlib figure with the time series
    """
    data_dir = '/eos/jeodpp/home/users/mihadar/data/'
    idmc_path = os.path.join(data_dir, f'for report/{iso3}/{iso3}_IDMC.json')
    with open(idmc_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    if isinstance(raw_data, dict):
        events = list(raw_data.values())
    elif isinstance(raw_data, list):
        events = raw_data
    else:
        raise ValueError("Unexpected JSON structure")

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt   = datetime.strptime(end_date,   "%Y-%m-%d")

    records     = []
    dates       = []
    figures     = []
    coordinates = []

    for event in events:
        date_str = event.get("displacement_date")
        figure   = event.get("figure")

        if date_str and figure is not None:
            try:
                date_obj = datetime.strptime(date_str[:10], "%Y-%m-%d")
                figure = float(figure)
                if start_dt <= date_obj <= end_dt:

                    lat = event.get("latitude")
                    lon = event.get("longitude")
                    if lat is None or lon is None:
                        continue
                    try:
                        lat_f = float(lat)
                        lon_f = float(lon)
                    except (TypeError, ValueError):
                        continue

                    records.append((date_obj, figure))
                    dates.append(date_obj)
                    figures.append(figure)
                    coordinates.append([lat_f, lon_f])

            except (ValueError, TypeError):
                continue

    if not records:
        raise ValueError(f"No IDMC records found for {iso3} between {start_date} and {end_date}")

    records.sort(key=lambda x: x[0])

    # Plot
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(dates, figures, color='darkblue', linewidth=2, alpha=0.9)
    ax.fill_between(dates, figures, alpha=0.15, color='lightblue')
    ax.set_xlim(start_dt, end_dt)
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))
    ax.set_title(f"{iso3} — IDMC Displacements (30-day rolling window)\n{start_date} to {end_date}", fontsize=16, fontweight='bold')
    ax.set_xlabel("Date", fontsize=13)
    ax.set_ylabel("Displaced Figures (sum)", fontsize=13)
    ax.grid(True)
    fig.tight_layout()

    return dates, figures, coordinates, fig


# INFORM Severity
def load_severity(iso3):
    """
    Loads and processes INFORM Severity data for a given ISO3 country code.

    Returns
    -------
    event_data : dict
        Keys are event IDs (e.g. "001", "002", ...) plus two special keys:
          - "SUM"       : sum across ALL events
          - "SUM_EXCL"  : sum excluding the first event (e.g. "001")
        Each value is a dict:
          {
            "crisis_type": str,          # only present for individual events, "" for sums
            "People exposed":    [(date, float), ...],
            "People affected":   [(date, float), ...],
            "People displaced":  [(date, float), ...],
            "People in Need":    [(date, float), ...],
          }
    first_event_id : str
        The ID of the first (excluded) event in SUM_EXCL.
    """
    categories = ["People exposed", "People affected", "People displaced", "People in Need"]

    data_dir    = '/eos/jeodpp/home/users/mihadar/data/'
    inform_path = os.path.join(data_dir, f'for report/{iso3}/{iso3}_ISeverity.json')

    with open(inform_path, 'r', encoding='utf-8') as f:
        country_data = json.load(f)

    if not isinstance(country_data, dict) or not country_data:
        raise ValueError(f"No valid data for ISO3 code: {iso3}")

    first_event_id = list(country_data.keys())[0]

    # ── Per-event extraction ───────────────────────────────────────────────────
    event_data = {}
    for event_id, inner_data in country_data.items():
        entry = {"crisis_type": (inner_data.get("type_of_crisis") or "").strip()}
        for cat in categories:
            points = []
            for record in inner_data.get(cat, []):
                fig_val  = record.get("Figure")
                date_str = record.get("Date")
                if fig_val is not None and date_str:
                    try:
                        points.append((datetime.strptime(date_str[:10], "%Y-%m-%d"), float(fig_val)))
                    except (ValueError, TypeError):
                        continue
            entry[cat] = sorted(points, key=lambda x: x[0])
        event_data[event_id] = entry

    # ── Helper: sum a subset of events per category ───────────────────────────
    def _sum_events(ids):
        summed = {}
        for cat in categories:
            by_date = defaultdict(float)
            for eid in ids:
                for date_obj, val in event_data[eid][cat]:
                    by_date[date_obj] += val
            summed[cat] = sorted(by_date.items())
        summed["crisis_type"] = ""
        return summed

    all_ids  = list(event_data.keys())
    excl_ids = [eid for eid in all_ids if eid != first_event_id]

    event_data["SUM"]      = _sum_events(all_ids)
    event_data["SUM_EXCL"] = _sum_events(excl_ids)

    return event_data, first_event_id

def plot_severity(iso3, event_data, first_event_id):
    """
    Plots INFORM Severity data from event_data returned by load_severity().

    Produces 3 figures:
      fig1 — one subplot per individual event
      fig2 — combined SUM (all events): all 4 categories + displaced & in need with peak annotations
      fig3 — combined SUM_EXCL (excluding first event): same layout as fig2

    Parameters
    ----------
    iso3           : str
    event_data     : dict  (as returned by load_severity)
    first_event_id : str   (as returned by load_severity)

    Returns
    -------
    fig1, fig2, fig3 : matplotlib figures
    """
    from scipy.signal import find_peaks
    import numpy as np

    plt.ioff()

    categories = {
        "People exposed":   "orange",
        "People affected":  "green",
        "People displaced": "blue",
        "People in Need":   "red",
    }
    priority_categories = {
        "People displaced": "blue",
        "People in Need":   "red",
    }

    individual_ids = [k for k in event_data if k not in ("SUM", "SUM_EXCL")]

    # ── Shared x-axis range across all individual events ──────────────────────
    all_dates = [
        date_obj
        for eid in individual_ids
        for cat in categories
        for date_obj, _ in event_data[eid][cat]
    ]
    global_min = min(all_dates)
    global_max = max(all_dates)

    # ══════════════════════════════════════════════════════════════════════════
    # fig1 — individual events
    # ══════════════════════════════════════════════════════════════════════════
    n_events = len(individual_ids)
    fig1, axes = plt.subplots(n_events, 1, figsize=(14, 6 * n_events), sharex=True)
    if n_events == 1:
        axes = [axes]

    for ax, eid in zip(axes, individual_ids):
        entry = event_data[eid]
        for cat, color in categories.items():
            if entry[cat]:
                dates, figures = zip(*entry[cat])
                ax.plot(dates, figures, label=cat, color=color, linewidth=2, alpha=0.8)
        ax.set_xlim(global_min, global_max)
        ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))
        ax.set_title(f"{iso3} — {eid}  {entry['crisis_type']}".strip(), fontsize=18)
        ax.set_ylabel("Number of People", fontsize=13)
        ax.legend(fontsize=14)
        ax.grid(True)

    axes[-1].set_xlabel("Date", fontsize=15)
    fig1.suptitle(f"Population Situation — {iso3} — Individual Events",
                  fontsize=22, fontweight='bold', y=1.01)
    fig1.tight_layout()

    # ══════════════════════════════════════════════════════════════════════════
    # Helper: build fig2 / fig3 from a SUM key
    # ══════════════════════════════════════════════════════════════════════════
    def _plot_sum(sum_key, title_suffix=""):
        sum_entry = event_data[sum_key]

        # x-axis range for this sum
        sum_dates = [d for cat in categories for d, _ in sum_entry[cat]]
        smin, smax = min(sum_dates), max(sum_dates)

        fig, (ax_all, ax_pin) = plt.subplots(2, 1, figsize=(14, 14), sharex=True)

        # ── Top: all 4 categories ─────────────────────────────────────────────
        for cat, color in categories.items():
            if sum_entry[cat]:
                dates, figures = zip(*sum_entry[cat])
                ax_all.plot(dates, figures, label=cat, color=color, linewidth=2.5, alpha=0.9)

        ax_all.set_xlim(smin, smax)
        ax_all.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))
        ax_all.set_title(f"{iso3} — All Events Combined {title_suffix}".strip(),
                         fontsize=18, fontweight='bold')
        ax_all.set_ylabel("Number of People", fontsize=13)
        ax_all.legend(fontsize=14)
        ax_all.grid(True)

        # ── Bottom: displaced + in need with peak annotations ─────────────────
        for cat, color in priority_categories.items():
            if sum_entry[cat]:
                dates, figures = zip(*sum_entry[cat])
                ax_pin.plot(dates, figures, label=cat, color=color, linewidth=2.5, alpha=0.9)

        # Peak annotations on People in Need
        cat = "People in Need"
        if sum_entry[cat]:
            dates, figures = zip(*sum_entry[cat])
            figures_arr  = np.array(figures)
            peak_indices, _ = find_peaks(figures_arr, prominence=figures_arr.max() * 0.1)

            for idx in peak_indices:
                peak_date = dates[idx]
                peak_val  = figures_arr[idx]

                # Find which individual event contributed most on this date
                contributions = sorted(
                    [
                        (dict(event_data[eid][cat]).get(peak_date, 0), eid)
                        for eid in individual_ids
                    ],
                    reverse=True,
                )
                source_event = contributions[0][1]
                label = f"{peak_date.strftime('%Y-%m-%d')}\n{source_event}"

                ax_pin.axvline(x=peak_date, color='orange', linestyle=':', linewidth=1.2, alpha=0.8)
                ax_pin.annotate(
                    label,
                    xy=(peak_date, peak_val),
                    xytext=(peak_date, figures_arr.max() * 1.05),
                    textcoords='data',
                    fontsize=11,
                    color='black',
                    ha='center',
                    fontweight='bold',
                    arrowprops=dict(arrowstyle='-', color='orange', linestyle='dotted', alpha=0.6),
                )

            ax_pin.set_ylim(top=figures_arr.max() * 1.25)

        ax_pin.set_xlim(smin, smax)
        ax_pin.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))
        ax_pin.set_title(f"{iso3} — People Displaced & In Need {title_suffix}".strip(),
                         fontsize=18, fontweight='bold')
        ax_pin.set_ylabel("Number of People", fontsize=13)
        ax_pin.set_xlabel("Date", fontsize=15)
        ax_pin.legend(fontsize=14)
        ax_pin.grid(True)

        fig.suptitle(f"Population Situation — {iso3} {title_suffix}".strip(),
                     fontsize=22, fontweight='bold', y=1.01)
        fig.tight_layout()
        return fig

    fig2 = _plot_sum("SUM")
    fig3 = _plot_sum("SUM_EXCL", title_suffix=f"(excl. {first_event_id})")

    return fig1, fig2, fig3

def plot_severity_with_acled_idmc(iso3, event_data, first_event_id,
                                   acled_rolling_dates, acled_rolling_values,
                                   idmc_rolling_dates,  idmc_rolling_values,
                                   start_date, end_date):
    """
    Plots INFORM Severity SUM and SUM_EXCL combined figures, overlaid with
    ACLED and IDMC rolling 30-day values on the same axes.

    Parameters
    ----------
    iso3                : str
    event_data          : dict  (as returned by load_severity)
    first_event_id      : str   (as returned by load_severity)
    acled_rolling_dates : list of datetime
    acled_rolling_values: list of float
    idmc_rolling_dates  : list of datetime
    idmc_rolling_values : list of float
    start_date          : str "YYYY-MM-DD"
    end_date            : str "YYYY-MM-DD"

    Returns
    -------
    fig2, fig3 : matplotlib figures
    """
    from scipy.signal import find_peaks
    import numpy as np

    plt.ioff()

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt   = datetime.strptime(end_date,   "%Y-%m-%d")

    individual_ids = [k for k in event_data if k not in ("SUM", "SUM_EXCL")]

    categories = {
        "People exposed":   "orange",
        "People affected":  "green",
        "People displaced": "blue",
        "People in Need":   "red",
    }
    priority_categories = {
        "People displaced": "blue",
        "People in Need":   "red",
    }

    def _plot_sum(sum_key, title_suffix=""):
        sum_entry = event_data[sum_key]

        fig, (ax_all, ax_pin) = plt.subplots(2, 1, figsize=(14, 14), sharex=True)

        # ── Top: all 4 INFORM categories ──────────────────────────────────────
        for cat, color in categories.items():
            if sum_entry[cat]:
                dates, figures = zip(*sum_entry[cat])
                ax_all.plot(dates, figures, label=cat, color=color, linewidth=2.5, alpha=0.9)

        # ACLED + IDMC overlaid on top subplot
        ax_all.plot(acled_rolling_dates, acled_rolling_values,
                    label='ACLED population affected (30-day rolling)',
                    color='red', linewidth=2, alpha=0.7, linestyle='--')
        ax_all.plot(idmc_rolling_dates, idmc_rolling_values,
                    label='IDMC displacements (30-day rolling)',
                    color='blue', linewidth=2, alpha=0.7, linestyle='--')

        ax_all.set_xlim(start_dt, end_dt)
        ax_all.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))
        ax_all.set_title(f"{iso3} — All Events Combined {title_suffix}".strip(),
                         fontsize=18, fontweight='bold')
        ax_all.set_ylabel("Number of People", fontsize=13)
        ax_all.legend(fontsize=12)
        ax_all.grid(True)

        # ── Bottom: displaced + in need + ACLED + IDMC ────────────────────────
        for cat, color in priority_categories.items():
            if sum_entry[cat]:
                dates, figures = zip(*sum_entry[cat])
                ax_pin.plot(dates, figures, label=cat, color=color, linewidth=2.5, alpha=0.9)

        ax_pin.plot(acled_rolling_dates, acled_rolling_values,
                    label='ACLED population affected (30-day rolling)',
                    color='red', linewidth=2, alpha=0.7, linestyle='--')
        ax_pin.plot(idmc_rolling_dates, idmc_rolling_values,
                    label='IDMC displacements (30-day rolling)',
                    color='blue', linewidth=2, alpha=0.7, linestyle='--')

        # Peak annotations on People in Need
        cat = "People in Need"
        if sum_entry[cat]:
            dates, figures = zip(*sum_entry[cat])
            figures_arr = np.array(figures)
            peak_indices, _ = find_peaks(figures_arr, prominence=figures_arr.max() * 0.1)

            for idx in peak_indices:
                peak_date = dates[idx]
                peak_val  = figures_arr[idx]

                contributions = sorted(
                    [
                        (dict(event_data[eid][cat]).get(peak_date, 0), eid)
                        for eid in individual_ids
                    ],
                    reverse=True,
                )
                source_event = contributions[0][1]
                label = f"{peak_date.strftime('%Y-%m-%d')}\n{source_event}"

                ax_pin.axvline(x=peak_date, color='orange', linestyle=':', linewidth=1.2, alpha=0.8)
                ax_pin.annotate(
                    label,
                    xy=(peak_date, peak_val),
                    xytext=(peak_date, figures_arr.max() * 1.05),
                    textcoords='data',
                    fontsize=11,
                    color='black',
                    ha='center',
                    fontweight='bold',
                    arrowprops=dict(arrowstyle='-', color='orange', linestyle='dotted', alpha=0.6),
                )

            ax_pin.set_ylim(top=figures_arr.max() * 1.25)

        ax_pin.set_xlim(start_dt, end_dt)
        ax_pin.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))
        ax_pin.set_title(f"{iso3} — People Displaced & In Need {title_suffix}".strip(),
                         fontsize=18, fontweight='bold')
        ax_pin.set_ylabel("Number of People", fontsize=13)
        ax_pin.set_xlabel("Date", fontsize=15)
        ax_pin.legend(fontsize=12)
        ax_pin.grid(True)

        fig.suptitle(f"Population Situation — {iso3} {title_suffix}".strip(),
                     fontsize=22, fontweight='bold', y=1.01)
        fig.tight_layout()
        return fig

    fig2 = _plot_sum("SUM")
    fig3 = _plot_sum("SUM_EXCL", title_suffix=f"(excl. {first_event_id})")

    return fig2, fig3


## helpers
def replace_nan_with_none(obj):
    if isinstance(obj, dict):
        return {k: replace_nan_with_none(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [replace_nan_with_none(v) for v in obj]
    elif isinstance(obj, float) and math.isnan(obj):
        return None
    else:
        return obj 
    

def extract_country_events(file_path, iso3, source_title):
    """
    Extract events for a specific country from a JSON file and save to a new file inside a
    folder named '{conflict_context}_{iso3}'.
    
    Parameters:
    - file_path (str): Path to the original JSON file.
    - iso3 (str): ISO3 country code to extract.
    - source_title (str): Title to include in the output file name.
    - conflict_context (str): Prefix for the folder, e.g., conflict type or context.
    - output_folder (str): Base folder where the country-specific folder will be created.
    
    Returns:
    - output_file_path (str): Path to the saved JSON file.
    """
    # Create the folder {conflict_context}_{iso3} inside output_folder
    country_folder = os.path.join('/eos/jeodpp/home/users/mihadar/data/for report', f"{iso3}")
    os.makedirs(country_folder, exist_ok=True)
    
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
    output_file_path = os.path.join(country_folder, f"{iso3}_{source_title}.json")
    
    # Save only the events (drop country layer)
    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump(clean_events, f, ensure_ascii=False, indent=4)
    
    return output_file_path


## mapping


