from __future__ import annotations
import sys
from pathlib import Path

# Resolve the absolute path of the folder that contains client_v1
PROJECT_ROOT = Path(__file__).resolve().parent   # ← folder where script.ipynb lives
# If you are running a notebook, __file__ does not exist, so use cwd()
PROJECT_ROOT = Path.cwd()                        # works in Jupyter too

# Insert the path at the front of sys.path (higher priority than site‑packages)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Now the imports work
from client_v1.settings import EmmRetrieversSettings
from openai import OpenAI
from datetime import datetime
from client_v1.settings import *
from client_v1.formatting_utils import format_docs
from client_v1.config_clients import client, client1
from client_v1.client import EmmAugmentedRetriever, EmmRetrieverV1
import pandas as pd
# from client_v1.utils import *
from client_v1.jrc_openai import JRCChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import json
import httpx
from copy import deepcopy
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from datetime import date
from typing import List, Dict, Optional, Literal
from pydantic import BaseModel, Field, validator


####### helper functions ##########
# ──────────────────────────────────────────────────────────────────────
#  utils.py
# ──────────────────────────────────────────────────────────────────────
import re
import json
import ast
from typing import Dict, List
import pandas as pd


def clean_text(text: str | None) -> str:
    """Trim whitespace and strip surrounding non‑alphanumerics."""
    if not text:
        return ""
    txt = str(text).strip()
    return re.sub(r"^[^\w]+|[^\w]+$", "", txt).strip()


def extract_factsheet_sections(txt: str) -> Dict[str, str]:
    """
    Returns a dict where keys are the *lower‑cased* section titles (exactly the
    field names of `FactsheetSections`).  The function is deliberately simple:
    it looks for each header, grabs everything up to the next header, and
    normalises line‑breaks.
    """
    column_names = [
        "Key information",
        "Severity",
        "Key drivers",
        "Main impacts, exposure, and vulnerability",
        "Likelihood of multi‑hazard risks",
        "Best practices for managing this risk",
        "Recommendations and supportive measures for recovery",
    ]

    # Build a regex that captures any header, case‑insensitively
    header_pat = re.compile(
        "|".join([re.escape(h) for h in column_names]),
        flags=re.IGNORECASE,
    )

    # Find all header positions
    matches = [(m.start(), m.end(), m.group().lower()) for m in header_pat.finditer(txt)]
    if not matches:
        # If nothing matched we still return an empty dict – the pydantic model
        # will turn missing values into “unknown”.
        return {h.lower().replace(" ", "_"): "" for h in column_names}

    out: Dict[str, str] = {}
    for i, (start, end, header) in enumerate(matches):
        # content runs until the next header (or end‑of‑string)
        next_start = matches[i + 1][0] if i + 1 < len(matches) else len(txt)
        raw = txt[end:next_start].strip()
        # turn bullet lists into a semi‑colon separated string
        cleaned = re.sub(r"\n\s*-\s*", "; ", raw).replace("\n", " ")
        out[header.replace(" ", "_")] = cleaned.strip()

    # Fill missing keys with empty string (pydantic will later replace with “unknown”)
    for col in column_names:
        key = col.lower().replace(" ", "_")
        out.setdefault(key, "")

    return out


def _extract_first_braced_object(text: str) -> str | None:
    """Return the first {...} block (including nested braces)."""
    stack = []
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if not stack:
                start = i
            stack.append("{")
        elif ch == "}":
            if stack:
                stack.pop()
                if not stack and start is not None:
                    return text[start : i + 1]
    return None


def _sanitize_to_json(braced: str) -> str:
    """
    Very light‑weight sanitiser:
      * replace single quotes with double quotes
      * add quotes around bare keys (e.g. `people: 10` → `"people": 10`)
      * turn Python's `None` into JSON `null`
    """
    # 1️⃣ single‑quote → double‑quote
    s = braced.replace("'", '"')

    # 2️⃣ ensure keys are quoted
    s = re.sub(r'(?<=\{|,)\s*([A-Za-z0-9_]+)\s*:', r' "\1":', s)

    # 3️⃣ Python None → JSON null
    s = s.replace("None", "null")

    return s



log = logging.getLogger(__name__)


def _postprocess_disaster_info(raw: Dict[str, Any]) -> DisasterInfo:
    """
    Normalise the raw dict that comes from the LLM into the strict
    `DisasterInfo` model.  Handles:
      * numbers that arrive as strings ("12,000") → int
      * economic losses that include a currency symbol → float
      * location strings that are comma‑separated → List[str]
    """
    def _to_int(val):
        if val is None:
            return None
        # strip commas, spaces, possible “people” word
        s = re.sub(r"[^\d]", "", str(val))
        return int(s) if s else None

    def _to_float(val):
        if val is None:
            return None
        s = re.sub(r"[^\d\.]", "", str(val))
        return float(s) if s else None

    locations = raw.get("Locations") or raw.get("locations") or []
    if isinstance(locations, str):
        locations = [c.strip() for c in locations.split(",") if c.strip()]

    return DisasterInfo(
        people_affected=_to_int(raw.get("People affected")),
        fatalities=_to_int(raw.get("Fatalities")),
        economic_losses=_to_float(raw.get("Economic losses")),
        locations=locations,
    )


def _call_llm_for_disaster_info(
    disaster: str,
    month: str,
    year: int,
    country: str,
    formatted_docs: str,
    client1,
    model: str,
) -> DisasterInfo | None:
    """
    Sends the prompt to the LLM and returns a validated `DisasterInfo`.
    All parsing (JSON → fallback → ast.literal_eval) lives here.
    """
    json_template = {
        "People affected": None,
        "Fatalities": None,
        "Economic losses": None,
        "Locations": []
    }

    prompt = (
        f"You are an expert in disaster event analysis. Based on the content related to the "
        f"{disaster} disaster that occurred in {country} during {month} {year}, please fill in the "
        f"following JSON template. For 'Locations', list all mentioned cities or provinces only "
        f"within {country}, ignoring any outside of {country}, and separate them by commas. "
        f"For 'People affected', 'Fatalities', and 'Economic losses', return only the total amount "
        f"according to the Document Content; do not include any additional words or text. "
        f"Use 'None' for any field where the information is not available.\n\n"
        f"Document Content:\n{formatted_docs}\n\n"
        f"JSON Template:\n{json.dumps(json_template, indent=2)}"
    )

    completion = client1.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert in disaster event analysis."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    raw = completion.choices[0].message.content.strip()

    # 1️⃣ Try straight JSON
    try:
        return _postprocess_disaster_info(json.loads(raw))
    except json.JSONDecodeError:
        pass

    # 2️⃣ Look for the first {...} block
    block = _extract_first_braced_object(raw)
    if not block:
        log.warning("No JSON block found in LLM response for disaster info.")
        return None

    # 3️⃣ Sanitise and try again
    sanitized = _sanitize_to_json(block)
    try:
        return _postprocess_disaster_info(json.loads(sanitized))
    except json.JSONDecodeError:
        pass

    # 4️⃣ Fallback to Python literal eval (handles None, single quotes)
    try:
        py_obj = ast.literal_eval(block)
        if isinstance(py_obj, dict):
            return _postprocess_disaster_info(py_obj)
    except Exception as exc:  # pragma: no cover – rarely hit
        log.error("Failed to parse disaster info from LLM: %s", exc)

    return None

###### CLASSES #############

class FactsheetSections(BaseModel):
    """All sections that come from the “story” LLM."""
    key_information: str = ""
    severity: str = ""
    key_drivers: str = ""
    main_impacts_exposure_and_vulnerability: str = ""
    likelihood_of_multi_hazard_risks: str = ""
    best_practices_for_managing_this_risk: str = ""
    recommendations_and_supportive_measures_for_recovery: str = ""

    @validator("*", pre=True)
    def empty_to_unknown(cls, v):
        """If a section is empty, normalise it to the string 'unknown'."""
        txt = str(v).strip()
        return txt if txt else "unknown"


class DisasterInfo(BaseModel):
    """Result of `extract_disaster_info`."""
    people_affected: Optional[int] = None
    fatalities: Optional[int] = None
    economic_losses: Optional[float] = None  # USD millions (or whatever you decide)
    locations: List[str] = Field(default_factory=list)


class WeekResult(BaseModel):
    """Everything we want to store for a single week."""
    week_start_dt: date
    week_end_dt: date
    num_relevant_docs: int
    topic_counts: Dict[str, int]                     # e.g. {"flood": 3, "heat": 0}
    factsheet_sections: FactsheetSections
    disaster_info: DisasterInfo


class MasterFactsheet(BaseModel):
    """Top‑level container written to disk."""
    disaster: str
    country: str
    weeks: Dict[date, WeekResult] 

    
    

def build_week_payload(
    week_start: date,
    week_end: date,
    formatted_docs: str,
    story: str,
    num_relevant_docs: int,
    topic_counts: Dict[str, int],
    disaster: str,
    month: str,
    year: int,
    country: str,
    client1,
    model: str,
) -> WeekResult:
    """
    One‑stop function that:
      1️⃣ extracts the factsheet sections from `story`;
      2️⃣ extracts the disaster‑info JSON from the LLM;
      3️⃣ bundles everything together in a `WeekResult`.
    """
    # ------------------------------------------------------------------ #
    # 1️⃣ Factsheet sections (story → structured dict)
    # ------------------------------------------------------------------ #
    raw_sections = extract_factsheet_sections(story)
    factsheet = FactsheetSections(**raw_sections)

    # ------------------------------------------------------------------ #
    # 2️⃣ Disaster info (LLM → validated model)
    # ------------------------------------------------------------------ #
    disaster_info = _call_llm_for_disaster_info(
        disaster=disaster,
        month=month,
        year=year,
        country=country,
        formatted_docs=formatted_docs,
        client1=client1,
        model=model,
    )
    if disaster_info is None:
        disaster_info = DisasterInfo()   # all fields become None / []

    # ------------------------------------------------------------------ #
    # 3️⃣ Assemble final week payload
    # ------------------------------------------------------------------ #
    return WeekResult(
        week_start_dt=week_start,
        week_end_dt=week_end,
        num_relevant_docs=num_relevant_docs,
        topic_counts=topic_counts,
        factsheet_sections=factsheet,
        disaster_info=disaster_info,
    )



##### setting ######
model = "gpt-4o"

def run_retrieval(event_details, folder_path):
    # unpack info 
    disaster = event_details["disaster"]
    country = event_details["country"]
    iso2 = event_details["iso2"]
    iso3 = event_details["iso3"]
    month = event_details["month"]
    year = event_details["year"]
    location = event_details["location"]
    start_dt = event_details["start_dt"]
    end_dt = event_details["end_dt"]
    
    # SAVE the filtered documents
    os.makedirs(folder_path, exist_ok=True)
    
    EXAMPLE_QUESTION = (
        f"What are the latest developments on the {disaster} disaster occurred in {country} "
        f"on {month} {year} that affected {location}?"
    )

    print(f'we are interested in the {disaster} happening in {country} and retrieve news from EMM in the timewindow {start_dt} - {end_dt}')

    
    response = client.post(
            "/r/rag-minimal/query",
            params={"cluster_name": settings.DEFAULT_CLUSTER, "index": f"mine_e_emb16-e1f7_prod4_{year}"}, # only change if something is happening v late december -- or 'mine_e_emb16-e1f7_prod4_live'
            json={
                "query": EXAMPLE_QUESTION,
                "lambda_mult": 0.9,
                "spec": {"search_k": 50, "fetch_k": 100},
                    "filter": {
                    "max_chunk_no": 1,
                    "min_chars": 100,
                    "start_dt": start_dt,
                    "end_dt": end_dt,
                    },
                },
                timeout=40.0
            )

    response.raise_for_status()  # Ensure the request was successful
    search_resp = response.json()
    documents = search_resp["documents"]

    # print(documents)

    print(' RETRIEVAL RESULTS: \n')
    print(f"total documents retrieved: {len(documents)}")
    
    # process, fact_check and format the documents
    formatted_docs, num_relevant_docs = process_documents(
        client1, documents, iso2, country, disaster, month, year, location, debug=True)
    
    print(f'number of relevant articles is: {num_relevant_docs}')
    # print(formatted_docs[0])
    if num_relevant_docs > 0:
        # SAVE the filtered documents
        articles = []
        for d in formatted_docs:
            articles.append({
                "title": d["metadata"].get("title"),
                "pubdate": d["metadata"].get("pubdate"),
                "source": d["metadata"].get("source"),
                "link": d["metadata"].get("link"),
                "guid": d["metadata"].get("guid"),
                "chunk_no": d["metadata"].get("chunk_no"),
                "page_content": d.get("page_content"),
                "score": d.get("score")
            })

        event_json = {
            "disaster_type": disaster,
            "country": country,
            "iso2": iso2,
            "location": location,
            "start_dt": str(start_dt),
            "query": EXAMPLE_QUESTION,
            "articles": articles,
            "n_documents": len(articles)
        }

        json_path = os.path.join(folder_path, f"{disaster}_{country}_{start_dt}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(event_json, f, ensure_ascii=False, indent=2)

        print(f"Saved formatted EMM articles → {json_path}")

        # now, FACTSHEET + graph 
        INSTRUCTIONS = (
            f"Complete the following factsheet on the {disaster} event in {country}: \n"
            " - Key information: [Quick summary with location and date.] \n"
            " - Severity: [Low, Medium, High] \n"
            " - Key drivers: [Main causes of the disaster.] \n"
            " - Main impacts, exposure, and vulnerability: [Economic damage, people affected, fatalities, effects on communities and infrastructure.] \n"
            "- Likelihood of multi-hazard risks: [Chances of subsequent related hazards.] \n"
            "- Best practices for managing this risk: [Effective prevention and mitigation strategies.] \n"
            "- Recommendations and supportive measures for recovery: [Guidelines for aid and rebuilding.]\n"
            " Important: Use only the information provided about the event. Do not add any assumptions or external data. "
            "If specific details are missing or uncertain, indicate them as 'unknown'."
        )

        llm_model = JRCChatOpenAI(
            model=model,
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_API_BASE_URL,
        )

        system_prompt = (
            "You are an assistant tasked with providing concise and factual updates on disaster events. "
            "Based on the context provided, answer the queries with clear and actionable information. "
            "If the information is uncertain or not found, recommend verifying with official channels. "
            "For incomplete or unknown answers, respond with 'unknown'."
            "\n\n"
            "{context}"
            "\n\n"
            "The original question was {question}"
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{instructions}"),
            ]
        )

        # Chain expects a dict with keys: question, instructions
        rag_chain = (
            {
                "context": lambda _: formatted_docs,     # inject docs you already computed
                "question": lambda x: x["question"],     # simple passthrough
                "instructions": lambda x: x["instructions"],
            }
            | prompt
            | llm_model
        )

        r = rag_chain.invoke({"question": EXAMPLE_QUESTION, "instructions": INSTRUCTIONS})
        story = fixed_width_wrap(r.content)

        graph_prompt = f"""Create a knowledge graph that captures the main causal relationships presented in the text. Follow these guidelines: 
        - Only use two relationship types: 'causes' or 'prevents' (so e.g. either 'A causes B' or 'A prevents B'), no other type of relations are allowed.
        - Minimize the number of nodes, use short node names with no more than two words if possible.
        - Focus on drivers and impacts, specifying the type of impact or damage if mentioned explicitly (e.g. 'blackout' or 'infrastructure damage' or 'crop devastation').
        - Do not use both a factor and its opposite (e.g., "early warning" and "lack of early warning") in the same graph. Represent the relationship with either "causes" or "prevents," not both. Avoid duplicating similar nodes.

        Example:
        prompt: In the past few days, several districts of Limpopo province, north-eastern South Africa experienced heavy rain and hailstorms, causing floods and severe weather-related incidents that resulted in casualties and damage. Local authorities warned population, and many were evacuated.
        graph: [["heavy rain", "causes", "flooding"], ["hailstorms", "causes", "flooding"], ["flooding", "causes", "damages"], ["flooding", "causes", "casualties"], ["early warning", "prevents", "casualties"]]

        prompt: {story}
        graph:"""

        graph = gpt_graph(graph_prompt)

        # 1) factsheet sections from the story (your old "columns")
        factsheet_sections = extract_factsheet_sections(story)  # NEW helper below

        factsheet_sections = process_storyline_dict(factsheet_sections)  # NEW helper below
        
        info = extract_disaster_info(disaster, month, year, country, formatted_docs, client1)
        # print('all of the info:', info)
        # print('can i index these things?', info[0])
        # print('trying to find the locations', info['Locations'])
        # print(story)
        
        # build factsheet payload (separate from event_json!)
        factsheet_json = {
            "disaster_type": disaster,
            "country": country,
            "iso2": iso2,
            "location": location,
            "start_dt": str(start_dt),
            "query": EXAMPLE_QUESTION,
            "nNews": num_relevant_docs,
            "story": story,
            "sections": factsheet_sections or {},
            "causal_graph": graph,
            "extracted_fields": info,
        }
        

        factsheet_path = os.path.join(folder_path, f"{disaster}_{country}_{start_dt}_factsheet.json")
        with open(factsheet_path, "w", encoding="utf-8") as f:
            json.dump(factsheet_json, f, ensure_ascii=False, indent=2)

        print(f"Saved factsheet JSON → {factsheet_path}")

    else:
        print("No news retrieved about disaster:", disaster)

        factsheet_json = {
            "disaster_type": disaster,
            "country": country,
            "iso2": iso2,
            "location": location,
            "start_dt": str(start_dt),
            "query": EXAMPLE_QUESTION,
            "nNews": 0,
            "status": "no_news",
            "story": None,
            "sections": {},
            "causal_graph": None,
            "extracted_fields": {},
        }

        factsheet_path = os.path.join(folder_path, f"{disaster}_{country}_{start_dt}_factsheet.json")
        with open(factsheet_path, "w", encoding="utf-8") as f:
            json.dump(factsheet_json, f, ensure_ascii=False, indent=2)

        print(f"Saved 'no news' factsheet JSON → {factsheet_path}")
        
    return info, factsheet_path, num_relevant_docs
    
                
                
def run_past_week_windows(event_details):
    """
    Slides 1-week windows from (start_dt - past_weeks_window weeks) up to start_dt.

    Example: start_dt=2023-04-15, past_weeks_window=6 =>
      2023-03-04..2023-03-11, 2023-03-11..2023-03-18, ... , 2023-04-08..2023-04-15

    Returns:
      {
        "period_start": <start of very first window>,
        "period_end": <end of last window>,
        "weekly": [ {week_start, week_end, total_retrieved, num_relevant_docs}, ... ]
      }
    """
    # read inputs (do NOT change original dict)
    start_dt = pd.to_datetime(event_details["start_dt"])

    # NEW: interpret the window count as weeks (rename in your configs if you want)
    past_weeks_window = int(event_details.get("past_weeks_window",
                                             event_details.get("past_months_window", 0)))

    # past_weeks_window = past_months * 4
    # build first window start
    first_start = start_dt - pd.DateOffset(weeks=past_weeks_window)

    weekly_results = []

    for k in range(past_weeks_window):
        win_start = first_start + pd.DateOffset(weeks=k)
        win_end   = win_start + pd.DateOffset(weeks=1)

        ed = deepcopy(event_details)
        ed["start_dt"] = win_start.strftime("%Y-%m-%d")
        ed["end_dt"]   = win_end.strftime("%Y-%m-%d")

        # --- your retrieval + processing logic (unchanged) ---
        disaster = ed["disaster"]
        country  = ed["country"]
        iso2     = ed["iso2"]
        year     = ed["year"]
        month    = ed["month"]
        location = ed["location"]
        start_s  = ed["start_dt"]
        end_s    = ed["end_dt"]

        EXAMPLE_QUESTION = (
            f"What are the latest developments on the {disaster} disaster occurred in {country} "
            f"on {month} {year} that affected {location}?"
        )

        print(f"\nWeek {k+1}/{past_weeks_window}: {start_s} -> {end_s}")

        response = client.post(
            "/r/rag-minimal/query",
            params={
                "cluster_name": settings.DEFAULT_CLUSTER,
                "index": f"mine_e_emb16-e1f7_prod4_{year}",
            },
            json={
                "query": EXAMPLE_QUESTION,
                "lambda_mult": 0.9,
                "spec": {"search_k": 30, "fetch_k": 80},
                "filter": {
                    "max_chunk_no": 1,
                    "min_chars": 100,
                    "start_dt": start_s,
                    "end_dt": end_s,
                },
            },
            timeout=40.0,
        )

        response.raise_for_status()
        documents = response.json().get("documents", [])
        total_docs = len(documents)

        week_topic_counts = defaultdict(int)

        formatted_docs, num_relevant_docs = process_documents(
            client1, documents, iso2, country, disaster, month, year, location, debug=True
        )
        
        if num_relevant_docs > 0:
            for d in formatted_docs:
                content = d.get("page_content", "")
                hits = count_topics(content, client1)   # {topic:0/1}
                update_week_topic_counts(week_topic_counts, hits)

        # store in weekly_results
        weekly_results.append({
            "num_relevant_docs": num_relevant_docs,
            "topic_counts": dict(week_topic_counts),  # sums of binaries across articles
        })
        
        # extract some impacts! and locations
        # now, FACTSHEET + graph 
        INSTRUCTIONS = (
            f"Complete the following factsheet on the {disaster} event in {country}: \n"
            " - Key information: [Quick summary with location and date.] \n"
            " - Severity: [Low, Medium, High] \n"
            " - Main impacts, exposure, and vulnerability: [Economic damage, people affected, fatalities, people displaced, effects or damage on communities and infrastructure.] \n"
            "- Likelihood of multi-hazard risks: [Chances of subsequent related hazards.] \n"
            " Important: Use only the information provided about the event. Do not add any assumptions or external data. "
            "If specific details are missing or uncertain, indicate them as 'unknown'."
        )

        llm_model = JRCChatOpenAI(
            model=model,
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_API_BASE_URL,
        )

        system_prompt = (
            "You are an assistant tasked with providing concise and factual updates on disaster events. "
            "Based on the context provided, answer the queries with clear and actionable information. "
            "If the information is uncertain or not found, recommend verifying with official channels. "
            "For incomplete or unknown answers, respond with 'unknown'."
            "\n\n"
            "{context}"
            "\n\n"
            "The original question was {question}"
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{instructions}"),
            ]
        )

        # Chain expects a dict with keys: question, instructions
        rag_chain = (
            {
                "context": lambda _: formatted_docs,     # inject docs you already computed
                "question": lambda x: x["question"],     # simple passthrough
                "instructions": lambda x: x["instructions"],
            }
            | prompt
            | llm_model
        )

        r = rag_chain.invoke({"question": EXAMPLE_QUESTION, "instructions": INSTRUCTIONS})
        story = fixed_width_wrap(r.content)
        
        factsheet_sections = extract_factsheet_sections(story)
        
        info = extract_disaster_info(disaster, month, year, country, formatted_docs, client1)
        print('all of the info:', info)
        
        factsheet_json = {
            'week_start_dt': first_start.strftime("%Y-%m-%d"),
            'week_end_dt': start_dt.strftime("%Y-%m-%d"),
            #number of relevant articles 
            #each category form the instructions
            #each category from the extract disaster info
            #each counting occurance of the topics --- from weekly_results
            
        }
        
        # print(f"  total retrieved: {total_docs}")
        # print(f"  relevant after processing: {num_relevant_docs}")
        
        rapid_escalation_track = {
        "period_start": first_start.strftime("%Y-%m-%d"),
        "period_end": start_dt.strftime("%Y-%m-%d"),
        "weekly": weekly_results,
    }
        
        factsheet_path = os.path.join(folder_path, f"{disaster}_{country}_{start_dt}_factsheet.json")
        with open(factsheet_path, "w", encoding="utf-8") as f:
            json.dump(factsheet_json, f, ensure_ascii=False, indent=2)

        print(f"Saved 'no news' factsheet JSON → {factsheet_path}")
        
    return factsheet_path, info, rapid_escalation_track


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
    week_starts = [period_start + pd.Timedelta(weeks=k) for k in range(len(weekly))]
    week_mids   = [ws + pd.Timedelta(days=3.5) for ws in week_starts]  # midpoint of week

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
        "fatalities": "Fatalities",
        "violence_injuries": "Violence acts",
        "displacement_infrastructure": "Displacement and Infrastructure",
        "humanitarian_services": "Food, water, WASH inaccessibility",
        "vulnerable_groups": "Vulnerable groups",
    }

    for key, label in topic_map.items():
        df[label] = df["topic_counts"].apply(lambda tc: int((tc or {}).get(key, 0)))


    df = df.sort_values("week_mid")

    # ----- Build month ticks: place tick at mid-month (15th) for each month in range -----
    month_starts = pd.date_range(
        period_start.normalize().replace(day=1),
        period_end.normalize().replace(day=1),
        freq="MS",
    )
    month_mids = month_starts + pd.Timedelta(days=14)  # ~15th of month
    month_labels = [d.strftime("%B") for d in month_mids]

    def _apply_month_axis(ax):
        ax.set_xticks(month_mids)
        ax.set_xticklabels(month_labels, rotation=0)
        ax.set_xlim(period_start, period_end)
        ax.grid(True, axis="y", alpha=0.3)

    # =========================
    # Figure 1: total per week
    # =========================
    plt.figure(figsize=(11, 4))
    plt.plot(df["week_mid"], df["num_relevant_docs"], marker="o")
    plt.xlabel("Month")
    plt.ylabel("Number of relevant articles (per week)")
    plt.title("Weekly relevant article counts")
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
    plt.ylabel("Number of articles mentioning topic (per week)")
    plt.title("Weekly topic densities (binary per article, summed per week)")
    _apply_month_axis(plt.gca())
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    
##########
log = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Helper – turn a datetime range into a list of week‑start dates
# ----------------------------------------------------------------------
def _split_into_weeks(start: datetime, end: datetime) -> List[date]:
    """Return every Monday (or the supplied start day) that falls inside [start, end]."""
    weeks = []
    cur = start
    while cur <= end:
        weeks.append(cur.date())
        cur += timedelta(days=7)
    return weeks


# ----------------------------------------------------------------------
# Your existing “process_documents”, “count_topics”, … stay unchanged.
# ----------------------------------------------------------------------


def run_weekly_pipeline_new(
    event_details: Dict[str, Any],
    client1,
    rag_chain,
    model: str,
) -> Dict[str, Any]:
    """
    Slides a 1‑week window from (start_dt - past_weeks_window) up to start_dt.
    For every window it:
      * fetches the documents,
      * runs your existing `process_documents`,
      * counts topics,
      * runs the RAG chain that produces the “story”,
      * builds a **WeekResult** (via `build_week_payload`).

    Returns a dict that matches the shape of `MasterFactsheet` (except the
    top‑level `weeks` mapping, which is built later).
    """
    # ------------------------------------------------------------------ #
    # 1️⃣  Interpret the window size (weeks)
    # ------------------------------------------------------------------ #
    start_dt = pd.to_datetime(event_details["start_dt"])
    past_weeks_window = int(
        event_details.get(
            "past_weeks_window",
            event_details.get("past_months_window", 0),
        )
    )
    first_start = start_dt - pd.DateOffset(weeks=past_weeks_window)

    weekly_results: List[WeekResult] = []

    for k in range(past_weeks_window):
        win_start = first_start + pd.DateOffset(weeks=k)
        win_end   = win_start + pd.DateOffset(weeks=1)

        # clone the dict so we don't mutate the original
        ed = deepcopy(event_details)
        ed["start_dt"] = win_start.strftime("%Y-%m-%d")
        ed["end_dt"]   = win_end.strftime("%Y-%m-%d")

        # ------------------------------------------------------------------
        # a) Build the question that the RAG chain will answer
        # ------------------------------------------------------------------
        disaster = ed["disaster"]
        country  = ed["country"]
        iso2     = ed["iso2"]
        year     = ed["year"]
        month    = ed["month"]
        location = ed["location"]
        start_s  = ed["start_dt"]
        end_s    = ed["end_dt"]

        EXAMPLE_QUESTION = (
            f"What are the latest developments on the {disaster} disaster occurred in "
            f"{country} on {month} {year} that affected {location}?"
        )
        log.info("Week %d/%d → %s – %s", k + 1, past_weeks_window, start_s, end_s)

        # ------------------------------------------------------------------
        # b) Retrieve raw docs from your elastic‑search endpoint
        # ------------------------------------------------------------------
        response = client1.post(
            "/r/rag-minimal/query",
            params={
                "cluster_name": settings.DEFAULT_CLUSTER,
                "index": f"mine_e_emb16-e1f7_prod4_{year}",
            },
            json={
                "query": EXAMPLE_QUESTION,
                "lambda_mult": 0.9,
                "spec": {"search_k": 30, "fetch_k": 80},
                "filter": {
                    "max_chunk_no": 1,
                    "min_chars": 100,
                    "start_dt": start_s,
                    "end_dt": end_s,
                },
            },
            timeout=40.0,
        )
        response.raise_for_status()
        documents = response.json().get("documents", [])
        total_docs = len(documents)

        # ------------------------------------------------------------------
        # c) Process / filter the retrieved docs (your existing function)
        # ------------------------------------------------------------------
        week_topic_counts = defaultdict(int)

        formatted_docs, num_relevant_docs = process_documents(
            client1,
            documents,
            iso2,
            country,
            disaster,
            month,
            year,
            location,
            debug=True,
        )

        # count topics (your existing helper)
        for d in formatted_docs:
            hits = count_topics(d.get("page_content", ""), client1)
            for t, v in hits.items():
                week_topic_counts[t] += v

        # ------------------------------------------------------------------
        # d) Run the RAG chain that gives you the “story” (the factsheet text)
        # ------------------------------------------------------------------
        rag_output = rag_chain.invoke(
            {"question": EXAMPLE_QUESTION, "instructions": INSTRUCTIONS}
        )
        story = rag_output.content   # already a string

        # ------------------------------------------------------------------
        # e) Build the WeekResult (single source of truth)
        # ------------------------------------------------------------------
        week_payload = build_week_payload(
            week_start=win_start.date(),
            week_end=win_end.date(),
            formatted_docs="\n".join([doc["page_content"] for doc in formatted_docs]),
            story=story,
            num_relevant_docs=num_relevant_docs,
            topic_counts=dict(week_topic_counts),
            disaster=disaster,
            month=month,
            year=year,
            country=country,
            client1=client1,
            model=model,
        )
        weekly_results.append(week_payload)

    # ------------------------------------------------------------------ #
    # 5️⃣  Return the container that the top‑level writer will consume
    # ------------------------------------------------------------------ #
    return {
        "period_start": first_start.date(),
        "period_end":   (first_start + pd.DateOffset(weeks=past_weeks_window)).date(),
        "weekly":       weekly_results,
    }
