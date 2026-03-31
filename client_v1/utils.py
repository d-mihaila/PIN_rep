import pandas as pd
import numpy as np
import time
import pycountry
from client_v1.formatting_utils import fixed_width_wrap, format_docs, format_doc_minimal
from client_v1.config_clients import client, client1
import re
import json
from openai import OpenAI
import geopandas as gpd
from shapely.geometry import Polygon
from shapely.ops import unary_union
import osmnx as osm
import ast
import logging
from typing import Any, Dict, Optional

# world = gpd.read_file('./data/ne_110m_admin_0_countries.shp')


model = "llama-3.3-70b-instruct"

def geocode_emdat(location):
    def process_geocoding(location_to_geocode):
        try:
            return osm.geocode_to_gdf(location_to_geocode)["geometry"].iloc[0]
        except Exception:
            return None

    geocoded_location = process_geocoding(location)

    if geocoded_location is None:
        print(f"Error geocoding location '{location}'. Trying to correct with GPT-4.")
        response = client1.chat.completions.create(
            model="gpt-4o",
            stream=False,
            messages=[{"role": "user", "content": f"Correct spelling or grammar or substitute with most commonly used location name by Google Maps, give me only the answer in the form 'Country, Location' filled with the corrected Country and Location: '{location}'"}]
        )
        corrected_location = response.choices[0].message.content.strip()
        geocoded_location = process_geocoding(corrected_location)

    return geocoded_location


def get_country_boundary(country_name):
    # Filter the world GeoDataFrame for the country
    country = world[world['NAME'] == country_name]
    if not country.empty:
        # Return the country's geometry
        return country.geometry.iloc[0]
    else:
        # Return None if country not found
        return None

def get_geometries(row):
    country = row['Country']
    locations = row['Locations']
    
    # Return NaN if locations is NaN
    if pd.isna(locations):
        return None
    
    # Get the country's boundary
    country_boundary = get_country_boundary(country)
    
    # If no country boundary is found, return None
    if country_boundary is None:
        return None
    
    locations_list = locations.split(', ')
    
    # Get polygons for each location, ignoring None results
    polygons = [geocode_emdat(f"{country}, {location}") for location in locations_list]
    polygons = [polygon for polygon in polygons if polygon is not None]
    
    # Filter polygons to remove those outside the country boundary
    valid_polygons = [polygon for polygon in polygons if polygon.within(country_boundary)]
    
    # If there are no valid polygons, return None
    if not valid_polygons:
        return None
    
    # Combine them into a single geometry using unary_union
    combined_geometry = unary_union(valid_polygons)
    
    return combined_geometry


def fact_check(client1, disaster, month, year, location, page_content):
    """
    Checks if the document content is relevant to the specified disaster event using a language model.

    :param disaster: Name of the disaster event
    :param month: Month of the disaster event.
    :param year: Year of the disaster event.
    :param location: Specific location affected by the disaster.
    :param page_content: The content of the document.
    :return: "Yes" if relevant, otherwise "No".
    """
    # Construct the prompt for the LLM
    prompt = (
        f"Is the following document referring to the {disaster} disaster "
        f"that occurred in {location} during {month} {year}? "
        f"Please answer only with 'Yes' or 'No' without adding anything else.\n\n"
        f"Document Content: {page_content}"
    )

    # Call the language model with the prompt
    completion = client1.chat.completions.create(
        model=model,  # Replace with the appropriate model for your use case
        messages=[
            {"role": "system", "content": "You are an expert in disaster event analysis."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    #print(completion.choices[0].message.content.strip())
    return completion.choices[0].message.content.strip()


def extract_triplets(nested_list):
    def traverse_structure(structure):
        triplets = []
        for item in structure:
            if isinstance(item, list):
                if len(item) == 3 and (item[1] == 'causes' or item[1] == 'prevents'):
                    # It's a valid triplet
                    triplets.append(item)
                else:
                    # Recursively traverse deeper if it’s a nested list
                    triplets.extend(traverse_structure(item))
        return triplets

    # Start the recursive extraction
    return traverse_structure(nested_list)

def needs_correction(relationships):
    # Check for excessive nesting or malformed entries
    return any(isinstance(item, list) and len(item) != 3 for item in relationships)

def extract_unique_nodes(relationships):
    # Extract triplets and initialize a set for unique nodes
    triplets = extract_triplets(relationships)
    unique_nodes = set()

    for triplet in triplets:
        try:
            # Ensure elements are hashable types like strings
            node1, node2 = triplet[0], triplet[2]
            unique_nodes.add(node1)
            unique_nodes.add(node2)
        except TypeError as e:
            print(f"Skipping malformed nodes in row {index}: {triplet}. Error: {e}")

    return list(unique_nodes)

def balance_brackets(s):
    s = s.replace('}', ']')
    s = s.rstrip(",]")
    s = re.sub(r'([a-zA-Z]), \[', r'\1"], [', s)
    s = s.split("using shorter")[0].strip().rstrip('.')  # Use lowercase "using shorter"
    open_count = s.count('[')
    close_count = s.count(']')

    if open_count > close_count:
        s += ']' * (open_count - close_count)
    elif close_count > open_count:
        s = '[' * (close_count - open_count) + s

    return s

def clean_structure(s):
    s = s.split("however")[0]  # Use lowercase "however"
    s = re.sub(r'\]\s+and\s+\[', '], [', s)
    s = re.sub(r'\]\n\n\[', '], [', s)
    s = s.rsplit(']', 1)[0] + ']'
    return s

def remove_duplicate_keywords(relationships):
    cleaned_relationships = []
    for relation in relationships:
        cleaned_relation = []
        previous_word = None
        for word in relation:
            if word != previous_word:
                cleaned_relation.append(word)
            previous_word = word
        cleaned_relationships.append(cleaned_relation)
    return cleaned_relationships

def extract_relationships_from_string(s):
    lines = s.strip().split('\n')
    relationships = []

    for line in lines:
        line = re.sub(r'^[-\d.]+\s*', '', line.strip())
        if not line:
            continue

        if 'causes' in line:
            parts = line.split('causes')
            if len(parts) == 2:
                cause, effect = parts
                relationships.append([cause.strip(), 'causes', effect.strip()])
        elif 'prevents' in line:
            parts = line.split('prevents')
            if len(parts) == 2:
                prevention, effect = parts
                relationships.append([prevention.strip(), 'prevents', effect.strip()])

    relationships = [[elem.strip('", ') if isinstance(elem, str) else elem for elem in relation] for relation in relationships]
    return relationships

def extract_list_from_string(s):
    s = s.lower()  # Convert to lowercase
    first_bracket_index = s.find('[')
    if first_bracket_index != -1:
        s = s[first_bracket_index:]

    s = balance_brackets(s)
    s = clean_structure(s)

    try:
        relationships = json.loads(s)
        relationships = [[elem.strip('", ') if isinstance(elem, str) else elem for elem in relation] for relation in relationships]
    except json.JSONDecodeError:
        relationships = extract_relationships_from_string(s)

    return remove_duplicate_keywords(relationships)

def transform_triplets(relationships):
    def clean_element(element):
        if isinstance(element, list):
            element = ' '.join(element)
        return re.sub(r'[^a-zA-Z\s-]', '', element).strip()

    transformed_list = [
        (clean_element(source), clean_element(relation), clean_element(target))
        for triplet in relationships if len(triplet) == 3
        for source, relation, target in [triplet]
    ]

    # Filter to only include triplets with 'causes' or 'prevents' as the relation
    filtered_list = [
        triplet for triplet in transformed_list
        if triplet[1] in {"causes", "prevents"}
    ]

    # Remove triplets with any element longer than 50 characters
    filtered_list = [
        triplet for triplet in filtered_list
        if all(len(element) <= 50 for element in triplet)
    ]
    
    return filtered_list

def process_graph(s):
    try:
        if isinstance(s, str):
            relationships = extract_list_from_string(s)
            return transform_triplets(relationships)
        else:
            return None
    except Exception as e:
        return None 


def iso3_to_iso2(iso3_code):
    # Iterate through countries in pycountry and find a match for the ISO3 code
    country = pycountry.countries.get(alpha_3=iso3_code)
    if country:
        return country.alpha_2
    else:
        return None  # Return None if no match is found

def generate_date_ranges(start_dt, num_weeks=4):
    start_dt = pd.to_datetime(start_dt)
    date_ranges = []
    for i in range(num_weeks):
        start = start_dt + pd.Timedelta(weeks=i)
        end = start + pd.Timedelta(weeks=1)
        date_ranges.append((start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')))
    return date_ranges



def process_documents(client1, docs, iso2, country, disaster, month, year, location, format_fn=format_doc_minimal, sleep_interval=1, **kwargs):
    """
    Filters and formats documents using specified criteria and formatting functions.

    :param docs: List of documents to process.
    :param iso2: ISO 2-letter country code for filtering.
    :param country: Country name to check in the title.
    :param disaster: Name of the disaster event.
    :param location: Specific location affected by the disaster.
    :param format_fn: Function to format documents.
    :param sleep_interval: Time to sleep between fact_check calls to manage rate limits.
    :param kwargs: Additional arguments for the formatting function.
    :return: Tuple of formatted string of filtered documents and count of relevant documents.
    """
    
    # Initial filtering based on country code and title
    filtered_docs = [
        entry for entry in docs
        if entry['metadata']['source']['country'] == iso2 or
           country in entry['metadata']['title']
    ]

    relevant_docs = []
    
    # Further filter using the fact_check function
    for entry in filtered_docs:
        if fact_check(client1, disaster, month, year, location, entry['page_content']) == "Yes":
            
            relevant_docs.append(entry)
        time.sleep(sleep_interval)  # Add a sleep interval to avoid hitting rate limits
    
    num_relevant_docs = len(relevant_docs)
    print("Num filtered docs after factcheck = ", num_relevant_docs)

    # Format the relevant documents
    # formatted_docs = format_docs(relevant_docs, doc_fn=format_fn, **kwargs)
    
    return relevant_docs, num_relevant_docs



def extract_factsheet_sections(txt: str) -> dict:
    """
    Returns a dict with the former 'column headers' as keys.
    Keys are lowercase to match your old downstream usage.
    """
    column_names = [
        "Key information",
        "Severity",
        "Key drivers",
        "Main impacts, exposure, and vulnerability",
        "Likelihood of multi-hazard risks",
        "Best practices for managing this risk",
        "Recommendations and supportive measures for recovery"
    ]

    # Prepare output (lowercase keys like before)
    out = {col.lower(): "" for col in column_names}

    # Match any of the headers
    pattern = '|'.join([re.escape(col.lower()) for col in column_names])
    matches = [(m.start(), m.end(), m.group()) for m in re.finditer(pattern, txt.lower())]

    for i, (start, end, col_name) in enumerate(matches):
        content_start = end
        content_end = matches[i + 1][0] if i + 1 < len(matches) else len(txt)
        content = txt[content_start:content_end].strip()
        content = re.sub(r'\n\s*-\s*', '; ', content).replace('\n', ' ')
        out[col_name] = content

    return out


def clean_text(text):
    if text is None:
        return ""
    if isinstance(text, float) and pd.isna(text):
        return ""
    text = str(text)
    cleaned_text = re.sub(r'^[^\w]+|[^\w]+$', '', text)
    return cleaned_text.strip()

def process_storyline_dict(sections: dict):
    """
    Cleans each section.
    If 'unknown' appears >= 5 times overall, returns None (same behavior as your old code).
    """
    keys = [
        "key information",
        "severity",
        "key drivers",
        "main impacts, exposure, and vulnerability",
        "likelihood of multi-hazard risks",
        "best practices for managing this risk",
        "recommendations and supportive measures for recovery",
    ]

    cleaned = dict(sections)
    for k in keys:
        cleaned[k] = clean_text(cleaned.get(k, ""))

    combined_text = "\n".join([f"{k}: {cleaned.get(k,'')}" for k in keys])
    unknown_count = combined_text.lower().count("unknown")

    return cleaned if unknown_count < 5 else None

    
def custom_sum(x, y):
    if x is None and y is None:
        return np.nan
    elif x is None:
        return y
    elif y is None:
        return x
    else:
        return x + y

log = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Helper: turn “12 million”, “over 80,000”, etc. into a proper number.
# ----------------------------------------------------------------------
_NUMBER_RE = re.compile(
    r"""
    (?P<sign>[-+]?)\s*
    (?P<int>\d{1,3}(?:,\d{3})*|\d+)
    (?:\.(?P<frac>\d+))?
    \s*(?P<suffix>million|billion|thousand)?
    """,
    re.IGNORECASE | re.VERBOSE,
)

_MAG = {
    None: 1,
    "thousand": 1_000,
    "million": 1_000_000,
    "billion": 1_000_000_000,
}


def _human_to_number(txt: str) -> Optional[int]:
    """
    Convert a human‑readable number (e.g. “12 million”, “80,000”, “over 5”) to an int.
    Returns None if the pattern does not match – the caller can then decide to keep the
    original string or replace it with None.
    """
    m = _NUMBER_RE.fullmatch(txt.strip())
    if not m:
        return None

    sign = -1 if m.group("sign") == "-" else 1
    integer = int(m.group("int").replace(",", ""))
    frac = m.group("frac")
    value = integer
    if frac:
        # 8.9 million → 8_900_000
        value = integer + float(f"0.{frac}")

    magnitude = _MAG[m.group("suffix").lower() if m.group("suffix") else None]
    return int(sign * value * magnitude)


# ----------------------------------------------------------------------
# Helper: safely parse a raw {...} block.  Any field that cannot be parsed
# becomes None (null in JSON).  Only the *numeric* fields are converted to
# numbers – everything else stays as a string (or None).
# ----------------------------------------------------------------------
_NUMERIC_FIELDS = {
    "People affected",
    "People displaced",
    "People in need",
    "People assisted",
    "Fatalities",
    "Economic losses",
}

def _safe_parse_block(raw_block: str) -> Optional[Dict[str, Any]]:
    """
    Return a dict where every key that could be extracted is present.
    If a value cannot be turned into a valid JSON literal we insert None.
    """
    # Strip outer braces if they exist – we will add them back later
    block = raw_block.strip()
    if block.startswith("{"):
        block = block[1:]
    if block.endswith("}"):
        block = block[:-1]

    # Split on commas that are *outside* of brackets/quotes.
    # A simple regex works for the very flat structures we expect.
    parts = re.split(r',\s*(?=[^"]*"(?:[^"]*"[^"]*")*$)', block)
    result: Dict[str, Any] = {}

    for part in parts:
        if not part.strip():
            continue
        # Expected form:  "key": value
        kv_match = re.match(r'\s*["\']?(?P<key>[^"\':]+)["\']?\s*:\s*(?P<val>.+)', part)
        if not kv_match:
            # totally malformed – we cannot even get a key, skip it
            log.debug("Unable to split key/value from part: %s", part)
            continue

        key = kv_match.group("key").strip()
        raw_val = kv_match.group("val").strip().rstrip(",")

        # ------------------------------------------------------------------
        # 1️⃣  Try a *strict* JSON decode of the value (covers numbers,
        #     booleans, null, quoted strings, arrays, objects)
        # ------------------------------------------------------------------
        try:
            val = json.loads(raw_val)
        except json.JSONDecodeError:
            # ------------------------------------------------------------------
            # 2️⃣  If strict JSON fails, see if it is a human‑readable number
            # ------------------------------------------------------------------
            num = _human_to_number(raw_val)
            if num is not None:
                val = num
            else:
                # ------------------------------------------------------------------
                # 3️⃣  Anything else we coerce to a plain string (so that
                #     stray quotes do not break the parser)
                # ------------------------------------------------------------------
                # strip surrounding quotes if they exist
                if (raw_val.startswith('"') and raw_val.endswith('"')) or (
                    raw_val.startswith("'") and raw_val.endswith("'")
                ):
                    val = raw_val[1:-1]
                else:
                    val = raw_val

        # ------------------------------------------------------------------
        # 4️⃣  For the *numeric* fields we only keep numbers; if we ended up
        #     with a non‑numeric value we replace it with None.
        # ------------------------------------------------------------------
        if key in _NUMERIC_FIELDS:
            if isinstance(val, (int, float)):
                result[key] = val
            else:
                # try one last conversion attempt (e.g. "12 million" that
                # survived as a string)
                conv = _human_to_number(str(val))
                result[key] = conv if conv is not None else None
        else:
            # All other fields (Locations, etc.) are stored as‑is.
            result[key] = val

    # If we did not manage to extract any key/value pair, signal failure
    return result if result else None


# ----------------------------------------------------------------------
# Main function – only the parsing part has been changed.
# ----------------------------------------------------------------------
def extract_disaster_info(
    disaster: str,
    month: str,
    year: str,
    country: str,
    formatted_docs: str,
    client1,
):
    """
    Extracts specific disaster information from formatted documents using a language model
    and ensures the JSON format is correct.  Any malformed field is replaced with ``None``.
    """
    json_template = {
        "People affected": None,
        "People displaced": None,
        "People in need": None,
        "People assisted": None,
        "Fatalities": None,
        "Economic losses": None,
        "Locations": [],
    }

    # --------------------------------------------------------------
    # Build the prompt
    # --------------------------------------------------------------
    prompt = (
        f"You are an expert in conflict analysis. Based on the content related to the {disaster} conflict "
        f"that occurred in {country} during {month} {year}, please fill in the following JSON template. "
        f"For 'Locations', list all mentioned cities or provinces only within {country}, ignoring any outside of {country}, and separate them by commas. "
        f"For 'People affected','People displaced', 'People in need', 'People assisted', 'Fatalities', and 'Economic losses', return only the total amount according to the Document Content; do not include any additional words or text. "
        f"Use 'None' for any field where the information is not available.\n\n"
        f"Document Content: {formatted_docs}\n\n"
        f"JSON Template:\n{json_template}"
    )

    # --------------------------------------------------------------
    # Call the language model
    # --------------------------------------------------------------
    completion = client1.chat.completions.create(
        model=model,  # <- make sure `model` is defined in the surrounding scope
        messages=[
            {"role": "system", "content": "You are an expert in disaster event analysis."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    response_content = completion.choices[0].message.content.strip()

    # --------------------------------------------------------------
    # 1️⃣  Try a direct JSON decode (the happy path)
    # --------------------------------------------------------------
    try:
        extracted_data = json.loads(response_content)
        return _postprocess(extracted_data)
    except json.JSONDecodeError:
        pass

    # --------------------------------------------------------------
    # 2️⃣  Pull the first {...} block from the LLM output
    # --------------------------------------------------------------
    block = _extract_first_braced_object(response_content)
    if not block:
        log.warning("No JSON object found in model response.")
        log.debug("Raw response: %s", response_content)
        return None

    # --------------------------------------------------------------
    # 3️⃣  Try a quick sanitisation + JSON decode (covers simple cases)
    # --------------------------------------------------------------
    sanitized = _sanitize_to_json(block)
    try:
        extracted_data = json.loads(sanitized)
        return _postprocess(extracted_data)
    except json.JSONDecodeError as e:
        log.debug("Sanitised block still not valid JSON: %s", e)
        log.debug("Sanitized block: %s", sanitized)

    # --------------------------------------------------------------
    # 4️⃣  *** NEW *** – robust per‑field parsing, inserting null on error
    # --------------------------------------------------------------
    robust = _safe_parse_block(block)
    if robust is not None:
        log.info("Recovered data with per‑field fallback – malformed fields set to null.")
        return _postprocess(robust)

    # --------------------------------------------------------------
    # 5️⃣  LAST RESORT – Python literal eval (keeps None, single quotes, etc.)
    # --------------------------------------------------------------
    try:
        py_obj = ast.literal_eval(block)
        if isinstance(py_obj, dict):
            return _postprocess(py_obj)
    except Exception as e2:
        log.debug("ast.literal_eval also failed: %s", e2)

    # --------------------------------------------------------------
    # 6️⃣  All attempts failed – give up for this week
    # --------------------------------------------------------------
    log.error("Failed to extract any usable JSON for %s %s %s – returning None.", disaster, month, year)
    return None

# def extract_disaster_info(disaster, month, year, country, formatted_docs, client1):
#     """
#     Extracts specific disaster information from formatted documents using a language model
#     and ensures the JSON format is correct.

#     :param disaster: Name of the disaster event.
#     :param month: Month of the disaster event.
#     :param year: Year of the disaster event.
#     :param country: The country where the disaster occurred.
#     :param formatted_docs: The formatted content of the documents.
#     :return: A dictionary with extracted information or None if not available.
#     """
#     json_template = {
#         "People affected": None,
#         'People displaced': None,
#         'People in need': None,
#         'People assisted': None,
#         "Fatalities": None,
#         "Economic losses": None,
#         "Locations": []
#     }

#     # Construct the prompt for the LLM
#     prompt = (
#         f"You are an expert in conflict analysis. Based on the content related to the {disaster} conflict "
#         f"that occurred in {country} during {month} {year}, please fill in the following JSON template. "
#         f"For 'Locations', list all mentioned cities or provinces only within {country}, ignoring any outside of {country}, and separate them by commas. "
#         f"For 'People affected','People displaced', 'People in need', 'People assisted', 'Fatalities', and 'Economic losses', return only the total amount according to the Document Content; do not include any additional words or text. "
#         f"Use 'None' for any field where the information is not available.\n\n"
#         f"Document Content: {formatted_docs}\n\n"
#         f"JSON Template:\n{json_template}"
#     )

#     # Call the language model with the prompt
#     completion = client1.chat.completions.create(
#         model=model,  # Replace with your model
#         messages=[
#             {"role": "system", "content": "You are an expert in disaster event analysis."},
#             {"role": "user", "content": prompt}
#         ],
#         temperature=0
#     )

#     # Extract the response content
#     response_content = completion.choices[0].message.content.strip()

#     # 1) Try direct JSON parse
#     try:
#         extracted_data = json.loads(response_content)
#         return _postprocess(extracted_data)
#     except json.JSONDecodeError:
#         pass

#     # 2) Extract {...} block and try parse
#     block = _extract_first_braced_object(response_content)
#     if not block:
#         print("No JSON object found in model response.")
#         print("Raw response:", response_content)
#         return None

#     # 3) Sanitize and parse
#     sanitized = _sanitize_to_json(block)
#     try:
#         extracted_data = json.loads(sanitized)
#         return _postprocess(extracted_data)
#     except json.JSONDecodeError as e:
#         print("Still failed to decode JSON. Error:", e)
#         print("Sanitized block:", sanitized)

#         # 4) LAST resort: try Python-literal parsing (handles None, single quotes)
#         try:
#             py_obj = ast.literal_eval(block)
#             if isinstance(py_obj, dict):
#                 return _postprocess(py_obj)
#         except Exception as e2:
#             print("ast.literal_eval also failed:", e2)

#         return None

def gpt_graph(prompt):
    completion = client1.chat.completions.create(
        model=model,  # Replace with the appropriate model for your use case
        messages=[
        {"role": "system", "content": "You are a disaster manager expert in risk dynamics."},
        {
            "role": "user",
            "content": prompt,
        }
    ]
    )

    # Extract the content from the response
    message_content = completion.choices[0].message.content
    return message_content



def parse_factsheet(response_content):
    """
    Parses the structured text response to extract disaster information.

    :param response_content: The structured text response from the language model.
    :return: A dictionary with extracted information.
    """

    def extract_value(label, text):
        # Improved pattern: Look for the label followed by any content until the next label or end of text
        pattern = rf"{label}:\s*([^\n]*?)(?=\n[A-Za-z\s]+:|$)"
        match = re.search(pattern, text, re.IGNORECASE)
        return match.group(1).strip() if match else None

    
    factsheet = {
        "People affected": extract_value("People affected", response_content),
        "Fatalities": extract_value("Fatalities", response_content),
        "Economic losses": extract_value("Economic losses", response_content),
        "Locations": extract_value("Locations", response_content),
    }

    return factsheet


def save_and_log_skipped(events, row):
    """Save processed events and log skipped rows."""
    if events:
        emdat2 = pd.concat(events)
        emdat2.to_csv(os.path.join(folder_path, f"emm_filtered_docs_{emdat2.iloc[-1]['DisNo.'].replace('-', '')}.csv"), index=False)
        print("Data saved up to disaster num. =", emdat2.iloc[-1]["DisNo."], "File saved at:", os.path.join(folder_path, f"emm_filtered_docs_{emdat2.iloc[-1]['DisNo.'].replace('-', '')}.csv"))
    # with open(skipped_rows_file, "a") as f:
    #     f.write(f"{row['DisNo.']}\n")

def safe_get(d, key, default=None):
    """Handle both dict-like and attribute-like docs."""
    if isinstance(d, dict):
        return d.get(key, default)
    return getattr(d, key, default)


## for json info from the extract disaster info: 
def _extract_first_braced_object(text: str) -> str | None:
    # Extract first {...} block (non-greedy)
    m = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    return m.group(0) if m else None

def _sanitize_to_json(text: str) -> str:
    """
    Best-effort conversion of 'almost-json' or 'python-dict-like' into valid JSON.
    Handles:
      - None -> null
      - single quotes -> double quotes (careful)
      - unquoted keys -> quoted keys (for expected keys)
      - Locations: Khartoum, X, Y -> "Locations": "Khartoum, X, Y" (then post-process)
    """
    expected_keys = ["People affected", "Fatalities", "Economic losses", "Locations"]

    # Normalize None/Null variants
    text = re.sub(r"\bNone\b", "null", text)
    text = re.sub(r"\bNULL\b", "null", text, flags=re.IGNORECASE)

    # Quote expected keys if they appear unquoted: People affected: -> "People affected":
    for k in expected_keys:
        text = re.sub(rf"(?<!\")\b{re.escape(k)}\b(?!\")\s*:", f"\"{k}\":", text)

    # Convert single quotes around keys/strings to double quotes (common LLM output)
    # This is imperfect but usually helps for dict-like responses.
    text = re.sub(r"'", '"', text)

    # If Locations value is unquoted and contains commas, wrap it in quotes:
    # "Locations": Khartoum, South Khartoum -> "Locations": "Khartoum, South Khartoum"
    text = re.sub(
        r'("Locations"\s*:\s*)([^"\[\{][^}\n]*)',
        lambda m: m.group(1) + json.dumps(m.group(2).strip().rstrip(",")),
        text
    )

    return text

def _postprocess(extracted: dict) -> dict:
    # Convert "Locations" comma-string -> list
    loc = extracted.get("Locations", None)
    if isinstance(loc, str):
        extracted["Locations"] = [x.strip() for x in loc.split(",") if x.strip()]
    elif loc is None:
        extracted["Locations"] = []

    # Ensure numeric fields are int/float or None
    for k in ["People affected", "Fatalities", "Economic losses"]:
        v = extracted.get(k)
        if isinstance(v, str):
            v_clean = v.replace(",", "").strip()
            if v_clean.lower() in ("null", "none", ""):
                extracted[k] = None
            else:
                # try int then float
                try:
                    extracted[k] = int(v_clean)
                except ValueError:
                    try:
                        extracted[k] = float(v_clean)
                    except ValueError:
                        extracted[k] = None

    return extracted



### for the topics countrs per week 'historical emm'
TOPIC_DEFS_5 = {
    "fatalities_injuries": (
        "Any mention of people killed or deaths/fatalities/death toll (e.g., killed, dead, bodies, mass graves)."
        "Any violent events OR injuries. Includes wounded/injured, clashes, attacks, shelling, airstrikes, shooting, "
        "armed conflict, raids/looting if described with violence."
    ),
    "infrastructure": (
        "Any infrastructure/building damage (homes destroyed,"
        "burned, rubble, damaged roads/bridges/power/water facilities)."
    ),
    "displacement": (
        "Displacement (internal displacement, refugees, fleeing, evacuations, homeless)."
    ),
    "humanitarian_needs": (
        "Food/water/healthcare access or breakdown. Includes hunger/famine/food shortages or prices, lack of safe water "
        "or sanitation (WASH), hospitals/clinics disrupted, lack of medicines, outbreaks tied to service collapse."
    ),
    "vulnerable_groups": (
        "Impacts on vulnerable groups and protection/education. Includes women/girls/children, GBV/sexual violence/"
        "exploitation/trafficking, unaccompanied children, and schools closed/education disrupted."
    ),
}

def _extract_json_object(text: str) -> Dict[str, Any]:
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if m:
        return json.loads(m.group(1))
    m = re.search(r"(\{.*\})", text, flags=re.DOTALL)
    if m:
        return json.loads(m.group(1))
    raise ValueError("Could not parse JSON from model output")

def count_topics(content: str, client1, topic_defs: Dict[str, str] = TOPIC_DEFS_5) -> Dict[str, int]:
    """
    Returns {topic: 0/1} for ONE article. Binary presence only (no extra weight for repeats).
    """
    if not content or not content.strip():
        return {t: 0 for t in topic_defs.keys()}

    content_snip = content.strip()
    if len(content_snip) > 8000:
        content_snip = content_snip[:8000] + "\n[TRUNCATED]"

    topic_list = [{"topic": k, "definition": v} for k, v in topic_defs.items()]

    prompt = f"""
You are extracting weekly indicators from conflict-related news.

Task:
For the single article below, decide for each topic whether it is present (1) or absent (0).
Rules:
- Binary per topic per article: if mentioned at least once => 1, else 0.
- Use broad matching: synonyms, paraphrases, nearby phrasing all count.
- If a number is given (e.g., "20 killed", "dozens wounded") that counts.
- Do NOT require the exact keyword.

Return ONLY valid JSON:
{{
  "topics": {{
    "<topic_name>": 0 or 1,
    ...
  }}
}}

Topics:
{json.dumps(topic_list, ensure_ascii=False)}

Article:
\"\"\"{content_snip}\"\"\"
""".strip()

    resp = client1.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": "You output strictly valid JSON only."},
            {"role": "user", "content": prompt},
        ],
    )

    raw = resp.choices[0].message.content
    obj = _extract_json_object(raw)
    topics_obj = obj.get("topics", {})

    out = {}
    for t in topic_defs.keys():
        val = topics_obj.get(t, 0)
        out[t] = 1 if str(val).strip() in {"1", "true", "True"} else 0
    return out

def update_week_topic_counts(week_topic_counts: dict, topic_hits: dict):
    """
    week_topic_counts: dict(topic -> int)
    topic_hits: dict(topic -> 0/1) for a single article
    """
    for topic, hit in topic_hits.items():
        week_topic_counts[topic] += int(hit)