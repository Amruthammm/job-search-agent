# UK Job Search Agent

A small tool-calling AI agent that searches live UK job listings, built with
[LangGraph](https://github.com/langchain-ai/langgraph), Google Gemini, and a
[Gradio](https://gradio.app) chat UI.

It follows the classic agent loop: the LLM reads your message, decides
whether to call a tool (a job search API, or a salary converter), reads the
tool's result, and either calls another tool or replies — looping until it
has a final answer.

## Features

- Chat UI built with `gr.ChatInterface`
- Searches [Adzuna](https://developer.adzuna.com/) and
  [Reed.co.uk](https://www.reed.co.uk/developers) job listings (UK)
- A `salary_to_monthly` helper tool for quick salary conversions
- Built on Gemini 3 via `langchain-google-genai`, with full support for
  Gemini's "thought signature" requirement on multi-turn tool calls

## Requirements

- Python 3.10+ (required by current `langchain-google-genai`)
- A free [Google AI Studio](https://aistudio.google.com/) API key
- At least one of:
  - Adzuna `App ID` + `App Key` — sign up at
    [developer.adzuna.com/signup](https://developer.adzuna.com/signup)
  - Reed API key — sign up at
    [reed.co.uk/developers](https://www.reed.co.uk/developers)

## Setup

```bash
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Set your API keys as environment variables in the same terminal session:

```bash
export GOOGLE_API_KEY="your-gemini-key"
export ADZUNA_APP_ID="your-adzuna-app-id"
export ADZUNA_APP_KEY="your-adzuna-app-key"
export REED_API_KEY="your-reed-key"          # optional if you have Adzuna
```

(On Windows use `set VAR=value` in Command Prompt, or `$env:VAR="value"` in
PowerShell.)

## Run

```bash
python 07_JobSearchAgent_Gradio.py
```

Then open the local URL Gradio prints (usually `http://127.0.0.1:7860`) and
try:

- "Find senior .NET developer jobs in London"
- "Remote full-stack C# roles in the UK"
- "What is £80,000 a year per month?"

## How it works

- `search_jobs_adzuna` / `search_jobs_reed` — `@tool`-decorated functions
  that call the respective job search REST APIs and return a formatted list
  of results. If the required API key(s) aren't set, they return a plain
  error string instead of failing.
- `salary_to_monthly` — a small offline helper tool, no API key needed.
- A `StateGraph` (from LangGraph) wires together an `llm_call` node (asks
  Gemini what to do next) and a `tool_node` (executes whichever tool Gemini
  asked for), looping until Gemini replies without requesting a tool.
- The Gradio `chat()` function keeps the full LangChain message history
  (including Gemini's tool-call "thought signatures") in a `gr.State` object
  so multi-turn tool calls keep working correctly across messages.

## Notes

- Never commit your API keys. Use environment variables or a local `.env`
  file (add `.env` to `.gitignore`) instead of hardcoding them.
- Without any job-search API keys set, the tools still run — they'll just
  tell Gemini the required key is missing, and Gemini will fall back to
  suggesting manual searches on LinkedIn, Reed, Indeed, etc.
