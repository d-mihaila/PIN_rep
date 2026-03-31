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
import os
import json
import httpx

##### setting ######
model = "gpt-4o"

# SAVE the filtered documents
folder_path = '/home/mihadar/data/from_emm'
os.makedirs(folder_path, exist_ok=True)
   
# event_info = {
#     "disaster": "armed conflict, war",
#     "country": "Cambodia",
#     "iso2": "KH",
#     "location": "Cambodia",
#     "month": "July",
#     "year": 2024
# }

# time_window = {
#     "months": 3,
#     "weeks": 0,
#     "days": 0
# }

# event_info["start_dt"] = pd.to_datetime("2022-06-22")

# event_info["end_dt"] = (
#     event_info["start_dt"]
#     + pd.DateOffset(
#         months=time_window["months"],
#         weeks=time_window["weeks"],
#         days=time_window["days"]
#     )
# )

# event_info["start_dt_str"] = event_info["start_dt"].strftime("%Y-%m-%d")
# event_info["end_dt_str"] = event_info["end_dt"].strftime("%Y-%m-%d")
    
# # unpacking event info
# start_dt = event_info["start_dt_str"]
# end_dt = event_info["end_dt_str"]
# disaster = event_info["disaster"]
# country = event_info["country"]
# iso2 = event_info["iso2"]
# location = event_info["location"]
# month = event_info["month"]
# year = event_info["year"]

disaster = "civil war, violence"
country = "Sudan"
iso2 = "SD"
iso3 = 'SDN'
month = "April"
year = 2025
location = "Sudan"
start_dt = pd.to_datetime("2025-04-15")
months_add, weeks_add, days_add = 0, 0, 3
end_dt = start_dt + pd.DateOffset(months=months_add, weeks=weeks_add, days = days_add)
start_dt = start_dt.strftime('%Y-%m-%d')
end_dt = end_dt.strftime('%Y-%m-%d')


EXAMPLE_QUESTION = (
    f"What are the latest developments on the {disaster} disaster occurred in {country} "
    f"on {month} {year} that affected {location}?"
)

print(f'we are interested in the {disaster} happening in {country} and retrieve news from EMM in the timewindow {start_dt} - {end_dt}')

def run_retrieval():
    response = client.post(
            "/r/rag-minimal/query",
            params={"cluster_name": settings.DEFAULT_CLUSTER, "index": f"mine_e_emb16-e1f7_prod4_{year}"}, # only change if something is happening v late december mine_e_emb16-e1f7_prod4_live  
            json={
                "query": EXAMPLE_QUESTION,
                "lambda_mult": 0.9,
                "spec": {"search_k": 10, "fetch_k": 50},
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
        
    return info, factsheet_path