# AI Ocean Assistant — Streamlit UI

A Streamlit front-end for the LangGraph Planner → Agent marine intelligence
pipeline (weather + ocean data, agentic recommendation).

## Files

- `marine_graph.py` — the original pipeline from `project.py`, with the
  interactive CLI loop removed. Exposes a compiled LangGraph `app`.
- `tools.py` — your existing `weather_agent` / `get_ocean_data` tools (unchanged).
- `streamlit_app.py` — the UI. Imports `app` from `marine_graph.py` and calls
  `app.invoke(...)` for each question.
- `requirements.txt` — updated to include `streamlit` and `python-dotenv`.
- `.env.example` — copy to `.env` and fill in your Mistral API key.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# edit .env and set MISTRAL_API_KEY=...
```

## Run

```bash
streamlit run streamlit_app.py
```

Then open the local URL Streamlit prints (usually http://localhost:8501).

## Using the app

1. In the sidebar, set your location — either type latitude/longitude
   directly, or switch to "Pick on map" and confirm the coordinates.
2. Type a natural-language question (weather, sea conditions, fishing safety,
   fishing-zone recommendation, etc.) and click **Ask**.
3. The app shows:
   - A recommendation card (safety status, summary, actionable advice)
   - The planner's execution plan (optional, toggle in sidebar)
   - Tabs with the raw weather and ocean forecast tables/charts
   - Optional raw JSON output for debugging
4. Previous queries in the session are kept in a collapsible history section.

## Notes / things you may want to adjust

- `marine_graph.py` is a straight extraction of your pipeline logic — no
  behavior was changed, only the bottom `while(True): input(...)` REPL loop
  was removed since Streamlit provides its own input loop.
- Each call to `app.invoke(...)` currently makes fresh LLM + tool calls; if
  you want to avoid recomputation, consider adding `st.cache_data` around a
  wrapper function keyed by `(question, latitude, longitude)`.
- The agent pipeline can raise `ValueError` if the underlying LLM doesn't
  return valid JSON (as in your original code) — the UI catches this and
  shows the error/traceback in an expander rather than crashing.
