from __future__ import annotations
import json
import re
from collections import Counter
from typing import Any, Dict, List
import pandas as pd

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
from client_v1.utils import process_documents
from client_v1.config_clients import client, client1
from client_v1.client import EmmAugmentedRetriever, EmmRetrieverV1
import pandas as pd
from client_v1.utils import *
from client_v1.jrc_openai import JRCChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import httpx
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D
import textwrap

GLOBAL_QUERY = (
    "Summarise the conflict events that happened per region "
    "following the instructions you are given. Focus on patterns of violence, "
    "actors involved, civilian targeting, and severity dynamics. "
    "Do NOT use fatalities as the primary severity indicator."
)

INPUT_ACLED_JSON = "/eos/jeodpp/home/users/mihadar/data/case_studies/ACLED.json"
OUTPUT_JSON      = "/eos/jeodpp/home/users/mihadar/data/case_studies/ACLED_with_factsheets.json"

MODEL_NAME = "gpt-4o"


# ==========================================================
# SCHEMA (your shape, but we will NOT include most_severe_events)
# ==========================================================

FACTSHEET_SCHEMA = {
    "key_information": {
        "time_range": {"start": None, "end": None},
        "query": None,
        "n_events": None,
        "dominant_event_types": [],
        "dominant_sub_event_types": [],
    },
    "actors": {
        "primary_actors": [],
        "secondary_actors": [],
        "notable_actor_pairs": [],
    },
    "violence_profile": {
        "civilian_targeting_flagged_events": None,
        "severity_overview": None,
        "severity_breakdown": {"high": None, "medium": None, "low": None},
    },
    "patterns": {
        "notable_tactics_or_methods": [],
        "locations_mentioned": [],
        "temporal_pattern": None,
    },
    "notes_summary": None,
    "data_quality": {
        "attribution_uncertainty": None,
        "notes_limitations": None
    }
}

EVENT_FACT_FIELDS = [
    "event_date", "disorder_type",
    "event_type", "sub_event_type",
    "actor1", "assoc_actor_1",
    "actor2", "assoc_actor_2",
    "civilian_targeting", "notes"
]


# ==========================================================
# PASS 1 — DETERMINISTIC EXTRACTION + SEVERITY
# ==========================================================

SEVERITY_WEIGHTS = {
    ("event_type", "Battles"): 3,
    ("event_type", "Violence against civilians"): 4,
    ("event_type", "Explosions/Remote violence"): 4,
    ("event_type", "Riots"): 2,
    ("event_type", "Strategic developments"): 2,
    ("event_type", "Protests"): 1,

    ("sub_event_type", "Air/drone strike"): 3,
    ("sub_event_type", "Shelling/artillery/missile attack"): 3,
    ("sub_event_type", "Armed clash"): 2,
    ("sub_event_type", "Attack"): 2,
    ("sub_event_type", "Excessive force against protesters"): 2,
    ("sub_event_type", "Non-state actor overtakes territory"): 4,
}

NOTE_KEYWORDS = {
    r"\bairstrike|airstrikes\b": 3,
    r"\bdrone\b": 3,
    r"\bshelling|artillery|missile\b": 3,
    r"\bgrenade|ied\b": 3,
    r"\bshoot(?:ing|ings)?|gunfire\b": 2,
    r"\blive bullets?\b": 2,
    r"\b(teargas|tear gas)\b": 1,
    r"\bhostage|kidnap|abduct|detain(?:ed|ing)?\b": 2,
    r"\bhospital|clinic|school\b": 3
}

# numeric info patterns from notes (rough but useful)
RE_KILLED   = re.compile(r"\b(\d+)\s+(?:people|person|civilians?)\s+were\s+killed\b", re.IGNORECASE)
RE_INJURED  = re.compile(r"\b(\d+)\s+(?:people|person)\s+were\s+injured\b", re.IGNORECASE)
RE_WOUNDED  = re.compile(r"\b(\d+)\s+were\s+wounded\b", re.IGNORECASE)
RE_HOSTAGES = re.compile(r"\bhostages?\b", re.IGNORECASE)
RE_DETAIN   = re.compile(r"\bdetain(?:ed|ing)?\b", re.IGNORECASE)
RE_ABDUCT   = re.compile(r"\babduct(?:ed|ing)?|kidnap(?:ped|ping)?\b", re.IGNORECASE)


def civilian_targeting_flag(value):
    return isinstance(value, str) and ("civilian targeting" in value.lower())


def extract_event_facts(events):
    return [{k: e.get(k) for k in EVENT_FACT_FIELDS} for e in events]


def severity_score(event):
    s = 0
    et = event.get("event_type") or ""
    st = event.get("sub_event_type") or ""
    s += SEVERITY_WEIGHTS.get(("event_type", et), 0)
    s += SEVERITY_WEIGHTS.get(("sub_event_type", st), 0)

    if civilian_targeting_flag(event.get("civilian_targeting")):
        s += 2

    notes = event.get("notes") or ""
    for pat, w in NOTE_KEYWORDS.items():
        if re.search(pat, notes, flags=re.IGNORECASE):
            s += w
    return int(s)


def severity_band(score):
    if score >= 7:
        return "High"
    if score >= 4:
        return "Medium"
    return "Low"


def enrich_with_severity(event_facts):
    out = []
    for e in event_facts:
        sc = severity_score(e)
        e2 = dict(e)
        e2["severity_score"] = sc
        e2["severity"] = severity_band(sc)
        e2["civilian_targeting_flag"] = civilian_targeting_flag(e.get("civilian_targeting"))
        out.append(e2)
    return out


def compute_basic_stats(events):
    dates = [e.get("event_date") for e in events if e.get("event_date")]

    event_counts = Counter()
    sub_event_counts = Counter()
    actor1_counts = Counter()
    actor2_counts = Counter()
    pair_counts   = Counter()

    civ_count = 0

    for e in events:
        et = e.get("event_type") or "Unknown"
        st = e.get("sub_event_type") or "Unknown"
        a1 = e.get("actor1") or "Unknown"
        a2 = e.get("actor2") or "Unknown"

        event_counts[et] += 1
        sub_event_counts[st] += 1
        actor1_counts[a1] += 1
        actor2_counts[a2] += 1
        pair_counts[f"{a1} ↔ {a2}"] += 1

        if civilian_targeting_flag(e.get("civilian_targeting")):
            civ_count += 1

    return {
        "time_range": {"start": min(dates) if dates else None, "end": max(dates) if dates else None},
        "n_events": len(events),
        "event_counts": dict(event_counts),
        "sub_event_counts": dict(sub_event_counts),
        "actor1_counts": dict(actor1_counts),
        "actor2_counts": dict(actor2_counts),
        "pair_counts": dict(pair_counts),
        "civilian_targeting_flagged_events": civ_count
    }


def top_n(counter, n=5):
    return [k for k, _ in Counter(counter).most_common(n)]


def extract_note_metrics(events):
    """
    Pull useful numeric/indicator signals from notes (not fatalities field):
    - mentions of killed/injured/wounded counts
    - hostage/detention/abduction mentions
    """
    killed_sum = 0
    injured_sum = 0
    wounded_sum = 0
    hostages_mentions = 0
    detain_mentions = 0
    abduct_mentions = 0

    for e in events:
        notes = e.get("notes") or ""
        for m in RE_KILLED.finditer(notes):
            killed_sum += int(m.group(1))
        for m in RE_INJURED.finditer(notes):
            injured_sum += int(m.group(1))
        for m in RE_WOUNDED.finditer(notes):
            wounded_sum += int(m.group(1))

        hostages_mentions += 1 if RE_HOSTAGES.search(notes) else 0
        detain_mentions   += 1 if RE_DETAIN.search(notes) else 0
        abduct_mentions   += 1 if RE_ABDUCT.search(notes) else 0

    return {
        "notes_killed_mentions_sum": killed_sum,
        "notes_injured_mentions_sum": injured_sum + wounded_sum,
        "hostage_events_mentioned": hostages_mentions,
        "detention_events_mentioned": detain_mentions,
        "abduction_events_mentioned": abduct_mentions
    }


def severity_aggregate(event_facts_scored):
    """
    Provide a single region severity score + breakdown.
    """
    if not event_facts_scored:
        return {
            "severity_avg_score": 0.0,
            "severity_max_score": 0,
            "severity_breakdown": {"high": 0, "medium": 0, "low": 0}
        }

    scores = [e["severity_score"] for e in event_facts_scored]
    bands  = [e["severity"] for e in event_facts_scored]
    c = Counter(bands)

    return {
        "severity_avg_score": round(sum(scores) / len(scores), 2),
        "severity_max_score": max(scores),
        "severity_breakdown": {
            "high": c.get("High", 0),
            "medium": c.get("Medium", 0),
            "low": c.get("Low", 0)
        }
    }


# ==========================================================
# PASS 1 FACTSHEET (AGGREGATED ONLY)
# ==========================================================

def build_deterministic_factsheet(region_name, events):
    """
    Produces ONE aggregated factsheet per region.
    Does NOT store _event_facts and does NOT store most_severe_events.
    """
    basic = compute_basic_stats(events)

    facts = enrich_with_severity(extract_event_facts(events))
    sev = severity_aggregate(facts)
    note_metrics = extract_note_metrics(events)

    # severity-weighted actor ranking (simple and useful)
    actor1_weighted = Counter()
    actor2_weighted = Counter()
    for ef in facts:
        actor1_weighted[ef.get("actor1") or "Unknown"] += ef["severity_score"]
        actor2_weighted[ef.get("actor2") or "Unknown"] += ef["severity_score"]

    return {
        "key_information": {
            "time_range": basic["time_range"],
            "query": GLOBAL_QUERY,
            "n_events": basic["n_events"],
            "dominant_event_types": top_n(basic["event_counts"], 5),
            "dominant_sub_event_types": top_n(basic["sub_event_counts"], 5),
        },
        "actors": {
            "primary_actors": top_n(basic["actor1_counts"], 7),
            "secondary_actors": top_n(basic["actor2_counts"], 7),
            "notable_actor_pairs": top_n(basic["pair_counts"], 5),
            # extra useful rankings (counts + severity-weighted)
            "actor1_rank_by_count": top_n(basic["actor1_counts"], 10),
            "actor2_rank_by_count": top_n(basic["actor2_counts"], 10),
            "actor1_rank_by_severity": [k for k, _ in actor1_weighted.most_common(10)],
            "actor2_rank_by_severity": [k for k, _ in actor2_weighted.most_common(10)],
        },
        "violence_profile": {
            "civilian_targeting_flagged_events": basic["civilian_targeting_flagged_events"],
            "severity_overview": (
                f"Avg severity score={sev['severity_avg_score']}, max={sev['severity_max_score']}. \n"
                f"Band counts: High={sev['severity_breakdown']['high']}, "
                f"Medium={sev['severity_breakdown']['medium']}, "
                f"Low={sev['severity_breakdown']['low']}.  \n"
            ),
            "severity_breakdown": sev["severity_breakdown"],
            # extra numeric
            "severity_avg_score": sev["severity_avg_score"],
            "severity_max_score": sev["severity_max_score"],
            **note_metrics
        },
        "patterns": {
            # deterministic hints from keyword tallies
            "notable_tactics_or_methods": [
                k for k, _ in Counter(
                    k
                    for e in events
                    for k in [
                        "airstrike/drone" if re.search(r"\bairstrike|drone\b", (e.get("notes") or ""), re.I) else None,
                        "shelling/artillery" if re.search(r"\bshelling|artillery|missile\b", (e.get("notes") or ""), re.I) else None,
                        "small arms fire" if re.search(r"\bshoot|gunfire|live bullets?\b", (e.get("notes") or ""), re.I) else None,
                        "detention/hostages" if re.search(r"\bhostage|detain|abduct|kidnap\b", (e.get("notes") or ""), re.I) else None,
                        "attack on protected site" if re.search(r"\bhospital|clinic|school\b", (e.get("notes") or ""), re.I) else None,
                    ]
                    if k is not None
                ).most_common(5)
            ],
            "locations_mentioned": [],
            "temporal_pattern": None,
        },
        "notes_summary": None,  # filled by LLM (short 4–5 sentences)
        "data_quality": {
            "attribution_uncertainty": None,
            "notes_limitations": "Deterministic baseline uses ACLED coded fields + keyword/numeric extraction from notes.",
        }
    }, facts  # return facts ONLY for LLM input; we do NOT store them in output


# ==========================================================
# PASS 2 — LLM SUMMARY (TEXT ONLY, 4–5 SENTENCES)
# ==========================================================

def build_llm_prompt(region, event_facts_scored, aggregated_fs):
    """
    Ask for a short paragraph, not JSON.
    Provide a compact input: aggregated stats + a trimmed set of event facts.
    """
    # keep prompt bounded
    ef = event_facts_scored[:40]

    return f"""
{GLOBAL_QUERY}

REGION: {region}

You are given:
A) Aggregated stats (JSON)
B) Up to 40 event records with fields including type/subtype, actors, civilian targeting flag, severity score, and notes.

Task:
Write ONE short conflict-analysis summary paragraph of 4–5 sentences.
Include:
- key violence patterns and tactics (airstrikes/shelling/clashes etc if present),
- main actors and how they interact,
- civilian targeting implications,
- any quantitative signals from notes (killed/injured mentioned, hostage/detention mentioned),
- any attribution uncertainty if the notes indicate it.

Do NOT list events one-by-one. Do NOT output JSON. Output plain text only.

AGGREGATED_STATS_JSON:
{json.dumps(aggregated_fs, ensure_ascii=False)}

EVENT_FACTS_JSON:
{json.dumps(ef, ensure_ascii=False)}
""".strip()


def call_llm_text(prompt, client1):
    # Try Responses API
    if hasattr(client1, "responses") and hasattr(client1.responses, "create"):
        try:
            resp = client1.responses.create(
                model=MODEL_NAME,
                input=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            if hasattr(resp, "output_text") and resp.output_text:
                return resp.output_text.strip()
            return resp.output[0].content[0].text.strip()
        except Exception:
            pass

    # Fallback to Chat Completions
    if hasattr(client1, "chat") and hasattr(client1.chat, "completions") and hasattr(client1.chat.completions, "create"):
        resp = client1.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        return (resp.choices[0].message.content or "").strip()

    raise RuntimeError("client1 supports neither responses.create nor chat.completions.create")


## for visualising 
def save_region_summary_figure(region_name,
                               factsheet,
                               output_dir,
                               width=10,
                               height=6,
                               dpi=150):
    """
    Save a white-page figure with the region conflict summary text.

    Parameters
    ----------
    region_name : str
    factsheet : dict
        The aggregated factsheet (per region)
    output_dir : str or Path
        Folder where PNG files will be saved
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = factsheet.get("notes_summary", "")
    if not summary:
        summary = "No summary available."

    # Optional header metadata
    ki = factsheet.get("key_information", {})
    vp = factsheet.get("violence_profile", {})

    header = (
        f"Region: {region_name}\n"
        f"Time range: {ki.get('time_range', {}).get('start')} → {ki.get('time_range', {}).get('end')}\n"
        f"Events: {ki.get('n_events')} | "
        f"Civilian targeting flagged: {vp.get('civilian_targeting_flagged_events')}\n"
        f"Severity: {vp.get('severity_overview')}\n\n"
    )

    full_text = header + summary

    # Wrap text nicely for figure width
    wrapped_text = "\n".join(textwrap.wrap(full_text, width=110))

    # Create figure
    fig = plt.figure(figsize=(width, height), dpi=dpi)
    fig.patch.set_facecolor("white")

    # Remove axes
    plt.axis("off")

    # Draw text
    plt.text(
        0.01,
        0.99,
        wrapped_text,
        va="top",
        ha="left",
        fontsize=11,
        family="monospace",
        wrap=True
    )

    # Filename safe formatting
    safe_name = region_name.replace("/", "_").replace(" ", "_").replace('"', "")
    out_file = output_dir / f"{safe_name}_factsheet_summary.png"

    plt.savefig(out_file, bbox_inches="tight", facecolor="white")
    plt.close()

    return out_file

# 

# ==========================================================
# PIPELINE DRIVER
# ==========================================================

def run_acled_factsheet():
    with open(INPUT_ACLED_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    from client_v1.config_clients import client1

    FIG_DIR = "/eos/jeodpp/home/users/mihadar/data/case_studies/figures"
    # KG_DIR  = "/eos/jeodpp/home/users/mihadar/data/case_studies/figures/kgs"

    skipped_regions = []

    for region in data["per_query"]:
        events = data["per_query"][region]["events"]

        # ✅ HARD SKIP EMPTY REGIONS
        if not events or len(events) == 0:
            print(f"Skipping region (no events): {region}")
            skipped_regions.append(region)
            continue

        print(f"Processing region: {region} ({len(events)} events)")

        aggregated_fs, facts_for_llm = build_deterministic_factsheet(region, events)

        # LLM summary
        prompt = build_llm_prompt(region, facts_for_llm, aggregated_fs)
        aggregated_fs["notes_summary"] = call_llm_text(prompt, client1)

        # Store ONLY the single factsheet
        data["per_query"][region]["factsheet"] = aggregated_fs

        # 1) Save summary figure per region
        fig_path = save_region_summary_figure(
            region_name=region,
            factsheet=aggregated_fs,
            output_dir=FIG_DIR
        )
        print("Saved summary figure:", fig_path)

        # -----------------------------------------
        # OPTIONAL KG BLOCK (keep commented if unused)
        # -----------------------------------------

        # kg_name, kg_df = save_region_kg(
        #     region=region,
        #     factsheet=aggregated_fs,
        #     output_dir=KG_DIR
        # )
        #
        # if kg_name is not None:
        #     print("Saved KG figure:", f"{KG_DIR}/{kg_name}.png")
        #
        # data["per_query"][region]["factsheet"]["knowledge_graph_triplets"] = (
        #     kg_df.to_dict(orient="records")
        # )

    # Save final output
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print("DONE → Output written to:", OUTPUT_JSON)
    print(f"Skipped {len(skipped_regions)} regions with zero events")



if __name__ == "__main__":
    run_acled_factsheet()
    
    
    
## knowledge graph
# def _short(s, max_words=3):
#     if s is None:
#         return None
#     s = str(s).strip()
#     s = re.sub(r"\s+", " ", s)
#     return " ".join(s.split()[:max_words])

# def build_region_triplets(region, factsheet):
#     trips = []

#     ki = factsheet.get("key_information", {})
#     actors = factsheet.get("actors", {})
#     vp = factsheet.get("violence_profile", {})
#     patterns = factsheet.get("patterns", {})

#     # ---- region -> severity info
#     if "severity_avg_score" in vp:
#         trips.append({"source": _short(region), "edge": "has_score", "target": f"avg {vp['severity_avg_score']}"})
#     if "severity_max_score" in vp:
#         trips.append({"source": _short(region), "edge": "has_score", "target": f"max {vp['severity_max_score']}"})

#     sb = vp.get("severity_breakdown") or {}
#     if sb:
#         trips.append({"source": _short(region), "edge": "has_level",
#                       "target": f"H{sb.get('high',0)} M{sb.get('medium',0)} L{sb.get('low',0)}"})

#     # ---- region -> dominant types
#     for et in (ki.get("dominant_event_types") or [])[:3]:
#         trips.append({"source": _short(region), "edge": "dominates", "target": _short(et)})
#     for st in (ki.get("dominant_sub_event_types") or [])[:3]:
#         trips.append({"source": _short(region), "edge": "dominates", "target": _short(st)})

#     # ---- actor rankings (edges)
#     prim = actors.get("primary_actors") or []
#     sec  = actors.get("secondary_actors") or []
#     pairs = actors.get("notable_actor_pairs") or []

#     for i, a in enumerate(prim[:4], start=1):
#         trips.append({"source": _short(a), "edge": "rank", "target": f"primary {i}"})
#     for i, a in enumerate(sec[:4], start=1):
#         trips.append({"source": _short(a), "edge": "rank", "target": f"secondary {i}"})

#     # actor pairs: "A ↔ B"
#     for p in pairs[:5]:
#         if "↔" in p:
#             a, b = [x.strip() for x in p.split("↔", 1)]
#             trips.append({"source": _short(a), "edge": "clashes_with", "target": _short(b)})
#             trips.append({"source": _short(b), "edge": "clashes_with", "target": _short(a)})

#     # ---- civilian targeting proxy
#     civ_n = vp.get("civilian_targeting_flagged_events", 0)
#     if civ_n and civ_n > 0:
#         src = _short(prim[0]) if prim else _short(region)
#         trips.append({"source": src, "edge": "targets", "target": "civilians"})

#     # ---- tactics/methods
#     for tac in (patterns.get("notable_tactics_or_methods") or [])[:5]:
#         src = _short(prim[0]) if prim else _short(region)
#         trips.append({"source": src, "edge": "uses", "target": _short(tac)})

#     # ---- numeric note metrics if you kept them in violence_profile
#     for k in ["notes_killed_mentions_sum", "notes_injured_mentions_sum",
#               "hostage_events_mentioned", "detention_events_mentioned", "abduction_events_mentioned"]:
#         v = vp.get(k, 0)
#         if v:
#             trips.append({"source": _short(region), "edge": "mentions", "target": f"{k}:{v}"})

#     # de-duplicate
#     seen = set()
#     uniq = []
#     for t in trips:
#         key = (t["source"], t["edge"], t["target"])
#         if t["source"] and t["target"] and key not in seen:
#             seen.add(key)
#             uniq.append(t)

#     return uniq

# ## print the graph
# def plot_cgraph(kg_df, name_plot,
#                 out_folder, save=True):

#     # Create output directory if it doesn't exist
#     if save:
#         os.makedirs(out_folder, exist_ok=True)

#     # Create directed graph
#     G = nx.from_pandas_edgelist(
#         kg_df,
#         "source",
#         "target",
#         edge_attr=True,
#         create_using=nx.MultiDiGraph()
#     )

#     edge_colors_dict = {
#         "causes": "red",
#         "prevents": "green",

#         # New KG edges
#         "clashes_with": "orange",
#         "targets": "purple",
#         "uses": "blue",
#         "dominates": "gray",
#         "has_score": "black",
#         "has_level": "black",
#         "rank": "brown",
#         "mentions": "gray",
#         "summary": "gray",
#     }

#     default_edge_color = "gray"

#     edge_color_list = [
#         edge_colors_dict.get(G[u][v][key].get("edge"), default_edge_color)
#         for u, v, key in G.edges(keys=True)
#     ]

#     # Plot
#     plt.figure(figsize=(12, 12))

#     # Layout trick
#     central_node = 'central'
#     G.add_node(central_node)

#     pos = nx.spring_layout(G, k=1.5, iterations=100)

#     pos.pop(central_node)
#     G.remove_node(central_node)

#     # Draw nodes
#     nx.draw_networkx_nodes(
#         G, pos,
#         node_color='skyblue',
#         node_size=800,
#         alpha=0.8
#     )

#     # Draw edges
#     nx.draw_networkx_edges(
#         G, pos,
#         edge_color=edge_color_list,
#         arrows=True,
#         width=2
#     )

#     # Line breaks for long labels
#     labels_with_linebreaks = {
#         node: node.replace(" and ", " and\n")
#         for node in G.nodes()
#     }

#     nx.draw_networkx_labels(
#         G, pos,
#         labels=labels_with_linebreaks,
#         font_size=10,
#         font_weight='bold',
#         verticalalignment='center',
#         horizontalalignment='center'
#     )

#     # Legend
#     legend_elements = [
#         Line2D([0], [0], color=color, label=edge_type, lw=2)
#         for edge_type, color in edge_colors_dict.items()
#     ]
#     plt.legend(handles=legend_elements, loc='best')

#     plt.gca().set_aspect('equal', adjustable='box')
#     plt.margins(x=0.1, y=0.1)
#     plt.axis('off')
#     plt.tight_layout()

#     # ✅ SAVE HERE
#     if save:
#         file_path = os.path.join(out_folder, f"{name_plot}.png")
#         plt.savefig(file_path, dpi=300, bbox_inches="tight")
#         print(f"Saved graph to: {file_path}")

#     plt.show()
#     plt.close()


# def save_region_kg(region, factsheet, output_dir):
#     """
#     Builds deterministic KG triplets for region, saves KG PNG via plot_cgraph.
#     Returns: path to saved image (as printed by plot_cgraph), and the df.
#     """
#     os.makedirs(output_dir, exist_ok=True)

#     triplets = build_region_triplets(region, factsheet)
#     kg_df = pd.DataFrame(triplets)

#     if kg_df.empty:
#         return None, kg_df

#     # Your plot_cgraph expects columns: source, target, edge
#     name_plot = f"KG_{region}".replace("/", "_").replace(" ", "_").replace('"', "")
#     plot_cgraph(
#         kg_df,
#         name_plot=name_plot,
#         out_folder=output_dir, save=True
#     )
#     # plot_cgraph prints the saved path; we also return df for debugging
#     return name_plot, kg_df
    
