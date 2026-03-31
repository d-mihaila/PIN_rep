import os
from pathlib import Path

# set working directory to project_root (one level up from folder2)
PROJECT_ROOT = Path.cwd().parent
os.chdir(PROJECT_ROOT)

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


def run_past_month_windows(event_details):
    """
    Slides 1-month windows from (start_dt - past_months_window months) up to start_dt.
    Example: start_dt=2023-04-15, past_months_window=6 =>
      2022-10-15..2022-11-15, 2022-11-15..2022-12-15, ... , 2023-03-15..2023-04-15

    Returns: list of dicts with counts per window.
    """
    # read inputs (do NOT change original dict)
    start_dt = pd.to_datetime(event_details["start_dt"])
    past_months_window = int(event_details["past_months_window"])

    # build first window start
    first_start = start_dt - pd.DateOffset(months=past_months_window)

    results = []

    for k in range(past_months_window):
        win_start = first_start + pd.DateOffset(months=k)
        win_end   = win_start + pd.DateOffset(months=1)

        ed = deepcopy(event_details)
        ed["start_dt"] = win_start.strftime("%Y-%m-%d")
        ed["end_dt"]   = win_end.strftime("%Y-%m-%d")

        # --- your retrieval + processing logic (same as run_retrieval, but no saving) ---
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

        print(f"\nWindow {k+1}/{past_months_window}: {start_s} -> {end_s}")

        response = client.post(
            "/r/rag-minimal/query",
            params={
                "cluster_name": settings.DEFAULT_CLUSTER,
                "index": f"mine_e_emb16-e1f7_prod4_{year}",
            },
            json={
                "query": EXAMPLE_QUESTION,
                "lambda_mult": 0.9,
                "spec": {"search_k": 20, "fetch_k": 50},
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

        formatted_docs, num_relevant_docs = process_documents(
            client1, documents, iso2, country, disaster, month, year, location, debug=True
        )
        
        
        results.append({
            "window_start": start_s,
            "window_end": end_s,
            "total_retrieved": total_docs,
            "num_relevant_docs": num_relevant_docs,
        })

        print(f"  total retrieved: {total_docs}")
        print(f"  relevant after processing: {num_relevant_docs}")

    return results

def plot_filtered_counts(counts_by_window):
    df = pd.DataFrame(counts_by_window).copy()
    if df.empty:
        raise ValueError("counts_by_window is empty; nothing to plot.")

    # Ensure datetimes and correct ordering
    df["window_start"] = pd.to_datetime(df["window_start"])
    df = df.sort_values("window_start")

    # Format x-axis labels (start date only)
    x_labels = df["window_start"].dt.strftime("%Y-%m-%d")

    plt.figure(figsize=(10, 4))
    plt.plot(df["window_start"], df["num_relevant_docs"], marker="o")

    plt.xticks(df["window_start"], x_labels, rotation=0)
    plt.xlabel("window start date")
    plt.ylabel("number of relevant articles found")
    plt.title("Rapid Escalation check - EMM occurrence of the conflict")

    plt.tight_layout()
    plt.show()
    
    
    
                
                
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

        # print(f"  total retrieved: {total_docs}")
        # print(f"  relevant after processing: {num_relevant_docs}")

    return {
        "period_start": first_start.strftime("%Y-%m-%d"),
        "period_end": start_dt.strftime("%Y-%m-%d"),
        "weekly": weekly_results,
    }


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