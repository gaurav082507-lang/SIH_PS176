"""
AI Ocean / Marine Intelligence Assistant — Streamlit UI

Wraps the LangGraph pipeline defined in marine_graph.py (Planner -> Agent)
with a simple, friendly chat-style interface.

Run with:
    streamlit run streamlit_app.py
"""

import json
import traceback

import pandas as pd
import streamlit as st

from marine_graph import app as marine_app

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Ocean Assistant",
    page_icon="🌊",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stat-card {
        background: #f0f7fb;
        border: 1px solid #d5e8f2;
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 10px;
    }
    .status-safe { color: #12794f; font-weight: 700; }
    .status-caution { color: #b8860b; font-weight: 700; }
    .status-unsafe { color: #c0392b; font-weight: 700; }
    .status-insufficient { color: #6b6b6b; font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# SESSION STATE
# ============================================================

if "history" not in st.session_state:
    st.session_state.history = []  # list of {question, lat, lon, result}

# ============================================================
# SIDEBAR — LOCATION & INPUTS
# ============================================================

with st.sidebar:
    st.title("🌊 AI Ocean Assistant")
    st.caption("Agentic marine intelligence: weather + ocean data, planned and analyzed by AI agents.")

    st.subheader("📍 Location")

    loc_mode = st.radio(
        "Set location",
        ["Manual entry", "Pick on map"],
        horizontal=True,
    )

    default_lat, default_lon = 19.0760, 72.8777  # Mumbai coast, as a sane default

    if loc_mode == "Manual entry":
        latitude = st.number_input(
            "Latitude", min_value=-90.0, max_value=90.0,
            value=default_lat, format="%.6f",
        )
        longitude = st.number_input(
            "Longitude", min_value=-180.0, max_value=180.0,
            value=default_lon, format="%.6f",
        )
    else:
        st.caption("Click on the map to drop a pin, then confirm below.")
        map_df = pd.DataFrame(
            {"lat": [st.session_state.get("picked_lat", default_lat)],
             "lon": [st.session_state.get("picked_lon", default_lon)]}
        )
        st.map(map_df, zoom=4)
        latitude = st.number_input(
            "Latitude (confirm/adjust)", min_value=-90.0, max_value=90.0,
            value=st.session_state.get("picked_lat", default_lat), format="%.6f",
        )
        longitude = st.number_input(
            "Longitude (confirm/adjust)", min_value=-180.0, max_value=180.0,
            value=st.session_state.get("picked_lon", default_lon), format="%.6f",
        )
        st.session_state["picked_lat"] = latitude
        st.session_state["picked_lon"] = longitude

    st.divider()
    show_raw = st.checkbox("Show raw JSON output", value=False)
    show_plan = st.checkbox("Show planner output", value=True)

    st.divider()
    if st.button("🗑️ Clear history", use_container_width=True):
        st.session_state.history = []
        st.rerun()

# ============================================================
# MAIN — QUERY INPUT
# ============================================================

st.header("Ask the Ocean Assistant")
st.caption("e.g. \"Is it safe to go fishing tomorrow morning?\" or \"What are the sea conditions near me today?\"")

with st.form("query_form", clear_on_submit=False):
    question = st.text_area(
        "Your question",
        placeholder="Find the best fishing zone near me today...",
        height=90,
    )
    col1, col2 = st.columns([1, 5])
    with col1:
        submitted = st.form_submit_button("🚀 Ask", use_container_width=True)

# ============================================================
# RUN THE GRAPH
# ============================================================

def status_class(status: str) -> str:
    s = (status or "").strip().lower()
    if s == "safe":
        return "status-safe"
    if s == "caution":
        return "status-caution"
    if s == "unsafe":
        return "status-unsafe"
    return "status-insufficient"


def render_result(question, latitude, longitude, result):
    st.subheader("📋 Recommendation")

    rec = result.get("recommendation", {}) or {}
    status = rec.get("safety_status", "Insufficient Data")
    summary = rec.get("summary", "No summary available.")
    action = rec.get("recommendation", "No recommendation available.")

    st.markdown(
        f"""
        <div class="stat-card">
        <b>Location:</b> {latitude:.4f}, {longitude:.4f}<br>
        <b>Safety status:</b> <span class="{status_class(status)}">{status}</span><br>
        <b>Summary:</b> {summary}<br>
        <b>Recommendation:</b> {action}
        </div>
        """,
        unsafe_allow_html=True,
    )

    pipeline_status = result.get("status", "unknown")
    if pipeline_status == "success":
        st.success(f"Pipeline status: {pipeline_status}")
    elif pipeline_status == "partial":
        st.warning(f"Pipeline status: {pipeline_status} — some data may be missing.")
    else:
        st.error(f"Pipeline status: {pipeline_status}")

    if show_plan and result.get("plan"):
        with st.expander("🧭 Planner output"):
            st.text(result["plan"])

    tab1, tab2 = st.tabs(["🌤️ Weather data", "🌊 Ocean data"])

    with tab1:
        wd = result.get("weather_data", {}) or {}
        if not wd:
            st.info("No weather data returned.")
        else:
            hourly = wd.get("hourly")
            daily = wd.get("daily")
            if daily:
                st.markdown("**Daily forecast**")
                st.dataframe(pd.DataFrame(daily), use_container_width=True)
            if hourly:
                st.markdown("**Hourly forecast (first 24h)**")
                df = pd.DataFrame(hourly)
                st.dataframe(df.head(24), use_container_width=True)
                if "temperature" in df.columns and "datetime" in df.columns:
                    st.line_chart(df.head(48).set_index("datetime")[["temperature"]])
            if not hourly and not daily:
                st.json(wd)

    with tab2:
        od = result.get("ocean_data", {}) or {}
        if not od:
            st.info("No ocean data returned.")
        else:
            hourly = od.get("hourly")
            if hourly:
                df = pd.DataFrame(hourly)
                st.markdown("**Hourly marine forecast (first 24h)**")
                st.dataframe(df.head(24), use_container_width=True)
                cols = [c for c in ["wave_height", "sea_surface_temperature"] if c in df.columns]
                if cols and "datetime" in df.columns:
                    st.line_chart(df.head(48).set_index("datetime")[cols])
            else:
                st.json(od)

    if show_raw:
        with st.expander("🔍 Raw JSON result"):
            st.json(result)


if submitted:
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        initial_state = {
            "latitude": latitude,
            "longitude": longitude,
            "user_question": question,
            "weather_data": {},
            "ocean_data": {},
            "plan": "",
            "status": "in_progress",
            "recommendation": {},
        }

        with st.spinner("Planning and gathering marine data... this can take a moment."):
            try:
                result = marine_app.invoke(initial_state)
                st.session_state.history.insert(
                    0,
                    {
                        "question": question,
                        "latitude": latitude,
                        "longitude": longitude,
                        "result": result,
                        "error": None,
                    },
                )
            except Exception as e:
                st.session_state.history.insert(
                    0,
                    {
                        "question": question,
                        "latitude": latitude,
                        "longitude": longitude,
                        "result": None,
                        "error": f"{e}\n\n{traceback.format_exc()}",
                    },
                )

# ============================================================
# DISPLAY HISTORY (most recent first)
# ============================================================

if not st.session_state.history:
    st.info("Ask a question above to get started — e.g. weather, sea conditions, or fishing-zone safety.")
else:
    latest = st.session_state.history[0]
    st.divider()
    st.markdown(f"### 🗨️ \"{latest['question']}\"")

    if latest["error"]:
        st.error("The agent pipeline failed to return a valid result.")
        with st.expander("Error details"):
            st.code(latest["error"])
    else:
        render_result(latest["question"], latest["latitude"], latest["longitude"], latest["result"])

    if len(st.session_state.history) > 1:
        with st.expander(f"📜 Previous queries ({len(st.session_state.history) - 1})"):
            for item in st.session_state.history[1:]:
                st.markdown(f"**\"{item['question']}\"** — ({item['latitude']:.4f}, {item['longitude']:.4f})")
                if item["error"]:
                    st.error("Failed")
                else:
                    rec = (item["result"] or {}).get("recommendation", {}) or {}
                    st.write(f"Status: {rec.get('safety_status', 'N/A')} — {rec.get('summary', '')}")
                st.markdown("---")
