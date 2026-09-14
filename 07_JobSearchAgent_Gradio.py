# ===============================
# Job Search Agent (UK) — LangGraph + Gemini + Gradio
# ===============================
# Setup:
#   pip install langchain langchain-google-genai langgraph gradio requests
#   Free Adzuna keys (UK jobs): https://developer.adzuna.com/  -> set ADZUNA_APP_ID / ADZUNA_APP_KEY
#   Free Reed keys (UK jobs):   https://www.reed.co.uk/developers -> set REED_API_KEY
#   At least one of these key pairs is required — the tools return an error message if unset.

import os
import requests
from typing import Literal

from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

from langgraph.graph import StateGraph, START, END, MessagesState
import gradio as gr


# ===============================
# Gemini 2.5 Flash LLM
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


tools = [search_jobs_adzuna, search_jobs_reed, salary_to_monthly]
tools_by_name = {t.name: t for t in tools}
llm_with_tools = llm.bind_tools(tools)


# ===============================
# Nodes
# ===============================
SYSTEM_PROMPT = (
    "You are a UK job search assistant. Use the search tools to find real listings, "
    "then summarise the best matches as a short markdown list with title, company, location, "
    "salary and link. Ask a clarifying question if the role or location is unclear."
)


def llm_call(state: MessagesState):
    response = llm_with_tools.invoke([SystemMessage(content=SYSTEM_PROMPT)] + state["messages"])
    return {"messages": [response]}


def tool_node(state: MessagesState):
    results = []
    for tool_call in state["messages"][-1].tool_calls:
        observation = tools_by_name[tool_call["name"]].invoke(tool_call["args"])
        results.append(ToolMessage(content=str(observation), tool_call_id=tool_call["id"]))
    return {"messages": results}


def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    return "tool_node" if state["messages"][-1].tool_calls else END


# ===============================
# Build Agent Graph
# ===============================
builder = StateGraph(MessagesState)

builder.add_node("llm_call", llm_call)
builder.add_node("tool_node", tool_node)

builder.add_edge(START, "llm_call")
builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
builder.add_edge("tool_node", "llm_call")

agent = builder.compile()


# ===============================
# Gradio Chat UI
# ===============================
def chat(user_message, history, lc_messages):
    # Keep the real LangChain messages (incl. tool calls + Gemini thought signatures) in gr.State.
    # Rebuilding them from Gradio's plain-text history would drop the signatures and Gemini 3 rejects that.
    # lc_messages is a mutable list stored in gr.State — mutate it in place (rather than reassign/return
    # it) so the update persists even on Gradio versions whose ChatInterface has no additional_outputs.
    lc_messages.append(HumanMessage(content=user_message))
    result = agent.invoke({"messages": lc_messages})
    lc_messages.clear()
    lc_messages.extend(result["messages"])
    return result["messages"][-1].content


demo = gr.ChatInterface(
    fn=chat,
    title="UK Job Search Agent",
    description="Ask e.g. 'Find senior .NET jobs in London paying over £70k' or 'remote C# roles in the UK'.",
    additional_inputs=[gr.State([])],
)

if __name__ == "__main__":
    demo.launch()