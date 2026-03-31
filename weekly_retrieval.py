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
from client_v1.utils import *
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
import re
import json
import ast
from typing import Dict, List
import pandas as pd
import os


output_dir = '/eos/jeodpp/home/users/mihadar/PIN_project'

def run_past_week_windows(event_details):
    """
    Slides 1‑week windows from (start_dt - number_windows weeks) up to start_dt
    and writes *one* JSON file that contains a separate entry for every week.
    """
    # ----------------------------------------------------------------------
    # 1️⃣  Setup – unchanged
    # ----------------------------------------------------------------------
    start_dt = pd.to_datetime(event_details["start_dt"])
    number_windows = int(
        event_details.get("number_windows",
                          event_details.get("number_windows", 0))
    )
    first_start = start_dt - pd.DateOffset(days=3 * number_windows)

    weekly_results = []                # keep the summary you already return
    all_weeks_payload = []             # ← NEW: list that will become the master JSON

    extracted_info_list = []
    # ----------------------------------------------------------------------
    # 2️⃣  Loop over each weekly window (the body is *exactly* what you already have)
    # ----------------------------------------------------------------------
    for k in range(number_windows):
        win_start = first_start + pd.DateOffset(days=3 * k)
        win_end   = win_start + pd.DateOffset(days=3)

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
        
        output_dir = event_details.get("output_dir", os.getcwd())
        os.makedirs(output_dir, exist_ok=True)

        EXAMPLE_QUESTION = (
            f"What are the latest developments on the {disaster} disaster occurred in {country} "
            f"on {month} {year} that affected {location}?"
        )

        print(f"\nWeek {k+1}/{number_windows}: {start_s} -> {end_s}")

        response = client.post(
            "/r/rag-minimal/query",
            params={
                "cluster_name": settings.DEFAULT_CLUSTER,
                "index": "mine_e_emb16-e1f7_prod4_live" #f"mine_e_emb16-e1f7_prod4_{year}" #"mine_e_emb16-e1f7_prod4_live" # or the live one for pakistan / iran "mine_e_emb16-e1f7_prod4_live"
            },
            json={
                "query": EXAMPLE_QUESTION,
                "lambda_mult": 0.9,
                "spec": {"search_k": 50, "fetch_k": 100},
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
                
                
            # extract more info
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

            factsheet_sections = extract_factsheet_sections(story)  # NEW helper below

            factsheet_sections = process_storyline_dict(factsheet_sections)  # NEW helper below

            info = extract_disaster_info(disaster, month, year, country, formatted_docs, client1)
            print('all of the info:', info)

            # build factsheet payload (separate from event_json!)
            factsheet_json = {
                "start_dt": str(start_dt),
                "nNews": num_relevant_docs,
                "story": story,
                "sections": factsheet_sections or {},
                "extracted_fields": info,
            }
            week_title = win_start.strftime("%Y-%m-%d")
            week_entry = {"title": week_title, "factsheet": factsheet_json}
            all_weeks_payload.append(week_entry)
            
            # only save the info and the start_dt in a json -- append to it whenever info != null

            if info:                                   # skip empty dict / None
                info_entry = {
                    "title": week_title,               # same title as the factsheet
                    "start_dt": week_title,            # week‑start date (duplicate for clarity)
                    "extracted_fields": info,
                }
                extracted_info_list.append(info_entry)
                print(f"   📥  info stored for week {week_title}")

        else:
            # ---- no relevant docs → do NOT add an entry -----------------------
            print(f"🟡  Week {k+1}/{number_windows}: No news retrieved for {disaster}")

        # ------------------------------------------------------------------
        # 3.4  Store the lightweight weekly summary (kept for every week)
        # ------------------------------------------------------------------
        weekly_results.append(
            {
                "num_relevant_docs": num_relevant_docs,
                "topic_counts": dict(week_topic_counts),
            }
        )

        # --------------------------------------------------------------
        # 3.5  Debug print for this iteration
        # --------------------------------------------------------------
        print(f"✅  window {k+1}/{number_windows} processed – nNews = {num_relevant_docs}")

    # ------------------------------------------------------------------
    # 4️⃣  AFTER THE LOOP – write ONE combined JSON file
    # ------------------------------------------------------------------
    combined_name   = f"combined_factsheets_{start_dt}.json"
    info_name        = f"extracted_info_{start_dt}.json"
    dummy_name       = f"{disaster}_{country}_{start_dt}_factsheet.json"

    master_path = os.path.join(output_dir, combined_name)
    info_path   = os.path.join(output_dir, info_name)
    dummy_path  = os.path.join(output_dir, dummy_name)

    # ── 2️⃣  Write / prepend the **combined factsheets** file ---------------
    if os.path.exists(master_path):
        try:
            with open(master_path, "r", encoding="utf-8") as f:
                existing = json.load(f)

            if isinstance(existing, list):
                # prepend new weeks so the newest appear first (optional)
                all_weeks_payload = all_weeks_payload + existing
                print(
                    f"🔀  Loaded existing master file ({len(existing)} weeks) "
                    f"and will prepend the new {number_windows} weeks."
                )
            else:
                print("⚠️  Existing master file is not a list – it will be overwritten.")
        except Exception as e:
            print(f"⚠️  Could not read existing master file ({e}) – creating a fresh one.")
    else:
        print("📂  No existing master file – creating a new one.")

    with open(master_path, "w", encoding="utf-8") as f:
        json.dump(all_weeks_payload, f, ensure_ascii=False, indent=2)

    print(f"\n🚀  **All files written to ONE file:** {master_path}")
    print(f"🧮  Total window‑entries in file: {len(all_weeks_payload)}")

    # ── 3️⃣  Write / prepend the **extracted‑info** file --------------------
    if os.path.exists(info_path):
        try:
            with open(info_path, "r", encoding="utf-8") as f:
                existing_info = json.load(f)

            if isinstance(existing_info, list):
                extracted_info_list = extracted_info_list + existing_info
                print(
                    f"🔀  Loaded existing info file ({len(existing_info)} entries) "
                    f"and will prepend the new {len(extracted_info_list)} entries."
                )
            else:
                print("⚠️  Existing info file is not a list – it will be overwritten.")
        except Exception as e:
            print(f"⚠️  Could not read existing info file ({e}) – creating a fresh one.")
    else:
        print("📂  No existing extracted‑info file – creating a new one.")

    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(extracted_info_list, f, ensure_ascii=False, indent=2)

    print(f"\n📄  Extracted‑info JSON written to: {info_path}")
    print(f"🧮  Entries stored in that file: {len(extracted_info_list)}")


    
    return info_path, {
        "period_start": first_start.strftime("%Y-%m-%d"),
        "period_end": start_dt.strftime("%Y-%m-%d"),
        "weekly": weekly_results,
    }


## rapid escalation check to determine medium window
