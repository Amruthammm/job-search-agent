# ===============================
# Job Search Agent (UK) — LangChain create_agent + Gemini + Gradio
# ===============================
# Setup:
#   pip install -r requirements.txt
#   Free Adzuna keys (UK jobs): https://developer.adzuna.com/  -> set ADZUNA_APP_ID / ADZUNA_APP_KEY
#   Free Reed keys (UK jobs):   https://www.reed.co.uk/developers -> set REED_API_KEY
#   At least one of these key pairs is required — the tools return an error message if unset.
#
# This version uses LangChain's prebuilt create_agent() instead of a hand-built LangGraph
# StateGraph — it wires the same "LLM decides -> tool runs -> LLM reads result -> ..." loop
# for you, and its checkpointer keeps full conversation state (including Gemini's tool-call
# "thought signatures") automatically, so we no longer have to manage that by hand.

import os
import uuid

import requests
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver
import gradio as gr


# ===============================
# Gemini LLM
# ===============================
llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0,
    google_api_key="",   # never hardcode keys in the file
)


# ===============================
# Tools
# ===============================
@tool
def search_jobs_adzuna(keywords: str, location: str = "London", max_results: int = 5) -> str:
    """Search UK job listings on Adzuna. keywords: e.g. '.NET developer'. location: UK city or 'remote'."""
    app_id, app_key = "", ""
    if not (app_id and app_key):
        return "[Adzuna] Missing ADZUNA_APP_ID / ADZUNA_APP_KEY — set them to use this tool."

    r = requests.get(
        "https://api.adzuna.com/v1/api/jobs/gb/search/1",
        params={"app_id": app_id, "app_key": app_key, "what": keywords,
                "where": location, "results_per_page": max_results, "content-type": "application/json"},
        timeout=15,
    )
    r.raise_for_status()
    jobs = [{
        "title": j.get("title"),
        "company": j.get("company", {}).get("display_name"),
        "location": j.get("location", {}).get("display_name"),
        "salary": f"£{int(j['salary_min']):,}–£{int(j['salary_max']):,}" if j.get("salary_min") else "n/a",
        "url": j.get("redirect_url"),
    } for j in r.json().get("results", [])]
    return _fmt(jobs, "Adzuna")


@tool
def search_jobs_reed(keywords: str, location: str = "London", max_results: int = 5) -> str:
    """Search UK job listings on Reed.co.uk. keywords: e.g. 'C# developer'. location: UK town/city."""
    api_key = ""
    if not api_key:
        return "[Reed] Missing REED_API_KEY — set it to use this tool."

    r = requests.get(
        "https://www.reed.co.uk/api/1.0/search",
        params={"keywords": keywords, "locationName": location, "resultsToTake": max_results},
        auth=(api_key, ""),
        timeout=15,
    )
    r.raise_for_status()
    jobs = [{
        "title": j.get("jobTitle"),
        "company": j.get("employerName"),
        "location": j.get("locationName"),
        "salary": f"£{int(j['minimumSalary']):,}–£{int(j['maximumSalary']):,}" if j.get("minimumSalary") else "n/a",
        "url": j.get("jobUrl"),
    } for j in r.json().get("results", [])]
    return _fmt(jobs, "Reed")


@tool
def salary_to_monthly(annual_gbp: int) -> str:
    """Convert an annual UK salary (GBP) to a rough monthly gross figure."""
    return f"£{annual_gbp / 12:,.0f} per month (gross)"


def _fmt(jobs, source):
    if not jobs:
        return f"[{source}] No jobs found."
    lines = [f"[{source}] {len(jobs)} result(s):"]
    for i, j in enumerate(jobs, 1):
        lines.append(f"{i}. {j['title']} — {j['company']} — {j['location']} — {j['salary']}\n   {j['url']}")
    return "\n".join(lines)


# ===============================
# Agent (built with create_agent)
# ===============================
SYSTEM_PROMPT = (
    "You are a UK job search assistant. Use the search tools to find real listings, "
    "then summarise the best matches as a short markdown list with title, company, location, "
    "salary and link. Ask a clarifying question if the role or location is unclear."
)

# InMemorySaver = a checkpointer: it stores each conversation's full message state (keyed by
# thread_id), so we don't need to hand-build or hand-mutate the message list ourselves — that
# also means Gemini's tool-call thought signatures are preserved automatically across turns.
agent = create_agent(
    model=llm,
    tools=[search_jobs_adzuna, search_jobs_reed, salary_to_monthly],
    system_prompt=SYSTEM_PROMPT,
    checkpointer=InMemorySaver(),
)


# ===============================
# Gradio Chat UI
# ===============================
def chat(user_message, history, thread_id):
    # One thread_id per browser session (created below) — the checkpointer uses it to look up
    # that session's full message history, so we only ever need to send the newest message.
    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke({"messages": [{"role": "user", "content": user_message}]}, config=config)
    return result["messages"][-1].content


demo = gr.ChatInterface(
    fn=chat,
    title="UK Job Search Agent",
    description="Ask e.g. 'Find senior .NET jobs in London paying over £70k' or 'remote C# roles in the UK'.",
    additional_inputs=[gr.State(lambda: str(uuid.uuid4()))],  # fresh thread_id per session
)

if __name__ == "__main__":
    demo.launch()