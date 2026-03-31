import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from pathlib import Path
import pandas as pd
import matplotlib.ticker as mticker   


def filter_country_json(source_path: str, iso3: str) -> None:
    # Load the original JSON
    with open(source_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Check if the ISO3 exists
    if iso3 not in data:
        raise ValueError(f"ISO3 code '{iso3}' not found in the JSON root.")

    # Extract the country-specific dictionary
    country_data = data[iso3]

    # Build output path in the same directory
    directory = os.path.dirname(source_path)
    output_filename = f"{iso3}_filtered_events.json"
    output_path = os.path.join(directory, output_filename)

    # Save only that dictionary
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(country_data, f, indent=4, ensure_ascii=False)

    print(f"Filtered file saved to: {output_path}")
    
    return output_path


# Use a non-interactive backend (safe for scripts/servers)
matplotlib.use("Agg")


def plot_severity_timeseries(filtered_json_path: str) -> str:
    """
    Reads a country-filtered JSON file containing multiple events (root keys),
    and for each event creates ONE plot:
      - title = event["type_of_crisis"] (fallback: event_id)
      - x-axis = Date
      - y-axis = Figure
      - lines (legend) = People exposed / affected / displaced / in need

    Outputs:
      - Saves individual PNGs into <same_dir>/plots/
      - Creates a zip file <same_dir>/event_plots.zip containing the PNGs

    Returns:
      Path to the created zip file.
    """

    wanted_keys = {
        "People exposed": "People exposed",
        "People affected": "People affected",
        "People displaced": "People displaced",
        "People in need": "People in Need",
    }

    def parse_date(s: Any) -> Optional[datetime]:
        if s is None:
            return None
        s = str(s).strip()
        for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(s, fmt)
            except ValueError:
                pass
        dt = pd.to_datetime(s, errors="coerce")
        if pd.isna(dt):
            return None
        return dt.to_pydatetime()

    def parse_number(x: Any) -> Optional[float]:
        if x is None:
            return None
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip()
        if s.lower() in {"x", "na", "n/a", ""}:
            return None
        s = s.replace(",", "")
        s = re.sub(r"[^0-9.\-]", "", s)
        if not s or s in {"-", "."}:
            return None
        try:
            return float(s)
        except ValueError:
            return None

    def thousands_formatter(x, pos):
        return f"{int(x):,}"
        
    # Load JSON
    with open(filtered_json_path, "r", encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)

    for event_id, event in data.items():
        if not isinstance(event, dict):
            continue

        title = event.get("type_of_crisis") or event_id
        series: Dict[str, List[Tuple[datetime, float]]] = {}

        for label, key in wanted_keys.items():
            entries = event.get(key, [])
            if not isinstance(entries, list):
                continue

            rows: List[Tuple[datetime, float]] = []
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                d = parse_date(entry.get("Date"))
                v = parse_number(entry.get("Figure"))
                if d is None or v is None:
                    continue
                rows.append((d, v))

            if rows:
                rows.sort(key=lambda t: t[0])
                series[label] = rows

        if not series:
            continue

        # Plot
        fig, ax = plt.subplots(figsize=(9, 5))

        for label, rows in series.items():
            xs = [d for d, _ in rows]
            ys = [v for _, v in rows]
            ax.plot(xs, ys, marker="o", linewidth=1.5, label=label)

        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.set_ylabel("Figure")
        ax.legend()
        ax.tick_params(axis="x", rotation=45)

        # Apply thousands formatting
        ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))

        plt.tight_layout()
        # plt.show()
        
### IOM
def plot_displaced_timeseries(
    json_path: Union[str, Path],
    *,
    start_year: int = 2021,               # <<< filter start year
    figsize: Tuple[int, int] = (10, 5),
    line_kwargs: Optional[dict] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Plot the number of displaced people (y‑axis) against reporting dates (x‑axis)
    **starting from the given year** (default = 2021).

    Parameters
    ----------
    json_path : str | pathlib.Path
        Path to the JSON file that contains the top‑level ``idps`` block.
    start_year : int, default 2021
        Only records with a reporting date on or after 1‑Jan‑``start_year`` are plotted.
    figsize : tuple[int, int], default (10, 5)
        Figure size in inches.
    line_kwargs : dict | None, default None
        Extra arguments passed to ``ax.plot`` (colour, linestyle, marker …).
    ax : matplotlib.axes.Axes | None, default None
        Provide an existing Axes (e.g. for sub‑plots) or let the function create one.

    Returns
    -------
    matplotlib.figure.Figure
        The Figure object – Jupyter will display it automatically.
    """

    # -------------------------------------------------
    # 3️⃣ Load JSON and locate the `idps` block
    # -------------------------------------------------
    json_path = Path(json_path)
    if not json_path.is_file():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    idps_block = data.get("idps")
    if idps_block is None:
        raise KeyError("'idps' key not found in the JSON root object")

    # -------------------------------------------------
    # 4️⃣ Normalise payload (dict OR list per year)
    # -------------------------------------------------
    records: List[Tuple[datetime, float]] = []   # (date, displaced)

    def _parse_date(s: str) -> datetime:
        """Try a few common formats; fall back to pandas."""
        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%Y-%m-%dT%H:%M:%S%z"):
            try:
                return datetime.strptime(s, fmt)
            except ValueError:
                continue
        dt = pd.to_datetime(s, errors="coerce")
        if pd.isna(dt):
            raise ValueError(f"Unable to parse date string: {s!r}")
        return dt.to_pydatetime()

    for year_key, payload in idps_block.items():
        # make the payload iterable (list of dicts)
        if isinstance(payload, dict):
            payloads = [payload]
        elif isinstance(payload, list):
            payloads = payload
        else:
            raise TypeError(
                f"Unexpected type for year {year_key!r}: {type(payload)} – "
                "expected dict or list of dicts"
            )

        for rec in payloads:
            if not isinstance(rec, dict):
                continue

            raw_date = rec.get("reporting_date")
            raw_idps = rec.get("idps")          # displaced count

            if raw_date is None or raw_idps is None:
                continue

            # ---- date ----
            try:
                dt = _parse_date(str(raw_date).strip())
            except Exception as e:
                raise ValueError(f"Year {year_key}: {e}")

            # ---- displaced number (int/float or string with commas) ----
            try:
                displaced = float(str(raw_idps).replace(",", "").strip())
            except ValueError:
                continue

            records.append((dt, displaced))

    if not records:
        raise ValueError("No valid (date, displaced) records found in the JSON file.")

    # -------------------------------------------------
    # 5️⃣ Build DataFrame and apply the year filter
    # -------------------------------------------------
    df = pd.DataFrame(records, columns=["date", "displaced"])
    df.sort_values("date", inplace=True)

    # Keep only rows from start_year onward
    cutoff = datetime(start_year, 1, 1)
    df = df[df["date"] >= cutoff]

    if df.empty:
        raise ValueError(f"No data points on or after {cutoff.date()} – check the start_year.")

    # -------------------------------------------------
    # 6️⃣ Plot
    # -------------------------------------------------
    line_kwargs = line_kwargs or {}

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    ax.plot(df["date"], df["displaced"],
            marker="o", linewidth=1.5, **line_kwargs)

    ax.set_title(f"People displaced (≥ {start_year})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Displaced (people)")

    ax.tick_params(axis="x", rotation=45)

    # y‑axis in thousands
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))

    plt.tight_layout()
    # No explicit fig.show() – Jupyter will render the figure automatically.
    return fig

### ACLED
def load_json_flexible(path):
    """
    Loads either:
      - a standard JSON file containing a list of dicts
      - or a JSON Lines file (one JSON object per line)
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8-sig")  # utf-8-sig handles possible BOM
    text_stripped = text.strip()

    # Case A: standard JSON (likely a list)
    if text_stripped.startswith("["):
        return json.loads(text_stripped)

    # Case B: JSON lines
    events = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        events.append(json.loads(line))
    return events


def _clean_population(val: Any) -> Optional[float]:
    """
    Turn whatever is stored in the JSON into a real number.

    * Removes commas (e.g. "12,345")
    * Strips surrounding whitespace
    * Returns None for empty / null / non‑numeric entries
    """
    if val is None:
        return None

    # most ACLED dumps store the number as a string, sometimes with commas
    if isinstance(val, str):
        val = val.replace(",", "").strip()
        if val == "":
            return None

    # At this point `val` is either a clean string of digits or already a number
    try:
        return int(val)          # fast path for integer‑like strings
    except (ValueError, TypeError):
        try:
            return float(val)    # fallback for decimals
        except (ValueError, TypeError):
            return None          # anything else is treated as missing


# --------------------------------------------------------------
# 2️⃣  Extraction function – returns clean Python lists
# --------------------------------------------------------------
def extract_eventdate_population(
    events: List[dict],
    date_key: str = "event_date",
    pop_key: str = "population_best",
) -> Tuple[List[pd.Timestamp], List[Optional[float]]]:
    """
    Parameters
    ----------
    events : list[dict]
        Raw JSON records.
    date_key, pop_key : str
        Keys that hold the date string and the population string/number.

    Returns
    -------
    dates : list[pd.Timestamp]      # NaT for unparsable dates
    pops  : list[float|None]        # numeric population or None
    """
    dates: List[pd.Timestamp] = []
    pops:  List[Optional[float]] = []

    for ev in events:
        # ---- date -------------------------------------------------
        raw_date = ev.get(date_key, None)
        # pd.to_datetime will give NaT for None / bad strings
        dates.append(pd.to_datetime(raw_date, errors="coerce"))

        # ---- population -------------------------------------------
        raw_pop = ev.get(pop_key, None)
        pops.append(_clean_population(raw_pop))

    return dates, pops


# --------------------------------------------------------------
# 3️⃣  Build the DataFrame and aggregate per month (sum)
# --------------------------------------------------------------
def aggregate_monthly_sum(
    dates: List[pd.Timestamp],
    pops:  List[Optional[float]],
    offset: str = "M",          # "M" = month‑end, "MS" = month‑start
) -> pd.Series:
    """
    Returns a Series indexed by month (Timestamp) whose values are the
    **sum** of all population_best that fall in that month.
    """
    df = pd.DataFrame(
        {"event_date": dates, "population_best": pops}
    )

    # Keep only rows where BOTH the date and the population are present
    df = df.dropna(subset=["event_date", "population_best"])

    # Ensure numeric type (float is fine – pandas will sum correctly)
    df["population_best"] = pd.to_numeric(df["population_best"], errors="coerce")

    # Set the datetime as the index – required for .resample()
    df = df.set_index("event_date").sort_index()

    # Resample to the chosen monthly offset and sum
    monthly = df["population_best"].resample(offset).sum()

    # Drop months that ended up empty (all NaNs)
    monthly = monthly.dropna()

    return monthly


# --------------------------------------------------------------
# 4️⃣  Pretty‑print a short table (first N months + total)
# --------------------------------------------------------------
def print_monthly_table(series: pd.Series, head: int = 12) -> None:
    """
    Shows the first `head` rows and the grand total.
    """
    df = series.reset_index()
    df.columns = ["month", "population_total"]
    print("\n=== First {} months ===".format(head))
    print(df.head(head).to_string(index=False))
    print("\n=== Grand total for the whole period ===")
    print(f"{series.sum():,.0f}")


# --------------------------------------------------------------
# 5️⃣  Plotting helper
# --------------------------------------------------------------
def plot_monthly_series(
    series: pd.Series,
    ylabel: str = "monthly total population_best",
    title: str = "Population exposed to violence – monthly totals",
    figsize: Tuple[int, int] = (12, 6),
) -> None:
    """
    Simple line‑plot with markers; y‑axis formatted with thousands separators.
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(series.index, series.values, marker="o", linestyle="-", color="#0066CC")

    ax.set_xlabel("Month")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Add commas to large numbers (1 234 567)
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))

    ax.grid(which="major", axis="y", linestyle="--", alpha=0.5)

    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()


### from EMM 
def plot_weekly_results(results):
    """
    results: dict returned by run_past_week_windows():
      {
        "period_start": "YYYY-MM-DD",
        "period_end": "YYYY-MM-DD",
        "weekly": [
            {"num_relevant_docs": int, "topic_counts": {topic:int, ...}},
            ...
        ]
      }

    Produces 2 figures:
      1) relevant articles per week
      2) topic counts per week (all topics on same axes)
    """
    if not results or "weekly" not in results or not results["weekly"]:
        raise ValueError("results['weekly'] is empty; nothing to plot.")

    period_start = pd.to_datetime(results["period_start"])
    period_end   = pd.to_datetime(results["period_end"])
    weekly = results["weekly"]

    # Reconstruct week starts/ends and midpoints from period_start + k weeks
    week_starts = [period_start + pd.Timedelta(days=3 * k) for k in range(len(weekly))]
    week_mids   = [ws + pd.Timedelta(hours=36) for ws in week_starts]  # midpoint of 72h window

    # Build dataframe
    df = pd.DataFrame(weekly).copy()
    df["week_mid"] = week_mids

    # Ensure columns exist
    if "num_relevant_docs" not in df.columns:
        df["num_relevant_docs"] = 0

    # Expand topic counts into columns
    def _get_topic_val(row, topic):
        tc = row if isinstance(row, dict) else {}
        return int(tc.get(topic, 0))

    if "topic_counts" not in df.columns:
        df["topic_counts"] = [{} for _ in range(len(df))]
        
    topic_map = {
        "fatalities_injuries": "Fatalities, Violence",
        "infrastructure": "Infrastructure Impact",
        "displacement": "Displacement",
        "humanitarian_needs": "Food, water, WASH inaccessibility",
        "vulnerable_groups": "Vulnerable groups",
    }

    for key, label in topic_map.items():
        df[label] = df["topic_counts"].apply(lambda tc: int((tc or {}).get(key, 0)))


    df = df.sort_values("week_mid")

    # ----- Build month ticks: place tick at mid-month (15th) for each month in range -----
    plot_start = df["week_mid"].min() - pd.Timedelta(days=3)
    plot_end   = df["week_mid"].max() + pd.Timedelta(days=3)

    month_starts = pd.date_range(
        plot_start.normalize().replace(day=1),
        plot_end.normalize().replace(day=1),
        freq="MS",
    )
    month_mids = month_starts + pd.Timedelta(days=14)  # ~15th of month
    month_labels = [d.strftime("%B") for d in month_mids]

    def _apply_month_axis(ax):
        ax.set_xticks(month_mids)
        ax.set_xticklabels(month_labels, rotation=0)
        ax.set_xlim(plot_start, plot_end)   # ← was period_start, period_end
        ax.grid(True, axis="y", alpha=0.3)

    # =========================
    # Figure 1: total per week
    # =========================
    plt.figure(figsize=(11, 4))
    plt.plot(df["week_mid"], df["num_relevant_docs"], marker="o")
    avg_docs = df["num_relevant_docs"].mean()
    plt.axhline(avg_docs, color="red", linestyle=":", linewidth=1.5,
                label=f"Average ({avg_docs:.1f} articles)")
    vals = df["num_relevant_docs"].values
    dates = df["week_mid"].values
    for i in range(1, len(vals) - 1):
        if vals[i] > avg_docs and vals[i] > vals[i-1] and vals[i] > vals[i+1]:
            peak_date = pd.Timestamp(dates[i])
            plt.annotate(
                peak_date.strftime("%d %b"),
                xy=(peak_date, vals[i]),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center", fontsize=8, color="steelblue"
            )
    plt.legend()
    plt.xlabel("Month")
    plt.ylabel("Number of relevant articles (per window)")
    plt.title("Relevant article counts per window")
    _apply_month_axis(plt.gca())
    plt.tight_layout()
    plt.show()

    # ======================================
    # Figure 2: topic densities per week
    # ======================================
    plt.figure(figsize=(11, 4))
    for label in topic_map.values():
        plt.plot(df["week_mid"], df[label], marker="o", label=label)
    plt.xlabel("Month")
    plt.ylabel("Number of articles mentioning topic (per window)")
    plt.title("Number of topic mentions (binary per article, summed per window)")
    _apply_month_axis(plt.gca())
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return df["num_relevant_docs"], avg_docs