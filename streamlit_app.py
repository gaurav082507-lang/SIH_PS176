"""
AI Ocean / Marine Intelligence Assistant — Streamlit UI

Wraps the LangGraph pipeline defined in marine_graph.py (Planner -> Agent)
with a friendly interface, including a LIVE diagram of the pipeline nodes
(Planner, Agent, Weather Tool, Ocean Tool, Recommendation) lighting up as
they execute.

Run with:
    streamlit run streamlit_app.py
"""

import time
import traceback

import pandas as pd
import streamlit as st
from langchain_core.callbacks.base import BaseCallbackHandler

try:
    import folium
    from streamlit_folium import st_folium
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False

from marine_graph import planner_node, agent_node

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
    /* ============================================================
       OCEAN GRADIENT THEME
       ============================================================ */

    /* ---- App background ---- */
    .stApp {
        background: linear-gradient(160deg, #041c32 0%, #04293a 35%, #064663 65%, #0a7ea4 100%);
        background-attachment: fixed;
    }

    /* Make default text light against the dark gradient */
    .stApp, .stApp p, .stApp li, .stApp label, .stMarkdown, h1, h2, h3, h4, h5 {
        color: #eaf6fb;
    }

    /* ---- Sidebar ---- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(200deg, #011627 0%, #022b3a 55%, #033a52 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
    }
    section[data-testid="stSidebar"] * { color: #dcefff; }

    /* ---- Hero banner ---- */
    .hero-banner {
        background: linear-gradient(120deg, #00c6ae 0%, #0074b7 50%, #7b2ff7 100%);
        border-radius: 18px;
        padding: 28px 30px;
        margin-bottom: 18px;
        box-shadow: 0 8px 28px rgba(0, 116, 183, 0.35);
    }
    .hero-banner h1 {
        margin: 0 0 4px 0;
        font-size: 30px;
        color: #ffffff !important;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    .hero-banner p {
        margin: 0;
        color: rgba(255,255,255,0.92) !important;
        font-size: 14.5px;
    }

    /* ---- Sidebar title ---- */
    .sidebar-title {
        background: linear-gradient(120deg, #00e6c3, #4facfe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 24px;
        font-weight: 800;
        margin-bottom: 0px;
    }

    /* ---- Buttons (Ask, Clear history, etc.) ---- */
    .stButton button, .stFormSubmitButton button, button[kind="secondaryFormSubmit"] {
        background: linear-gradient(90deg, #00c6ae 0%, #0074b7 100%);
        color: #ffffff;
        border: none;
        border-radius: 10px;
        font-weight: 700;
        padding: 0.55em 1.4em;
        box-shadow: 0 4px 14px rgba(0, 116, 183, 0.4);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .stButton button:hover, .stFormSubmitButton button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 18px rgba(0, 198, 174, 0.5);
        color: #ffffff;
    }

    /* ---- Text areas / number inputs / radio ---- */
    .stTextArea textarea, .stNumberInput input {
        background: rgba(255,255,255,0.07) !important;
        color: #eaf6fb !important;
        border: 1px solid rgba(255,255,255,0.18) !important;
        border-radius: 8px !important;
    }
    .stTextArea textarea::placeholder { color: rgba(234,246,251,0.5) !important; }

    /* ---- Section headers with gradient underline ---- */
    .grad-subheader {
        font-size: 19px;
        font-weight: 800;
        margin: 6px 0 10px 0;
        display: inline-block;
        background: linear-gradient(90deg, #4facfe, #00e6c3);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        border-bottom: 2px solid rgba(79,172,254,0.35);
        padding-bottom: 4px;
    }

    /* ---- Recommendation / stat card (glassmorphism) ---- */
    .stat-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.96) 0%, rgba(235,248,255,0.96) 100%);
        border: 1px solid rgba(255,255,255,0.5);
        border-radius: 14px;
        padding: 16px 20px;
        margin-bottom: 10px;
        color: #1a2b33;
        box-shadow: 0 6px 20px rgba(0,0,0,0.25);
    }
    .stat-card b { color: #0d1b21; }
    .status-safe { color: #12794f; font-weight: 700; }
    .status-caution { color: #b8860b; font-weight: 700; }
    .status-unsafe { color: #c0392b; font-weight: 700; }
    .status-insufficient { color: #565f66; font-weight: 700; }

    /* ---- Pipeline diagram ---- */
    .pipeline-wrap {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 4px;
        padding: 18px 6px 8px 6px;
        overflow-x: auto;
    }
    .pipe-node {
        flex: 1;
        min-width: 130px;
        border-radius: 14px;
        padding: 12px 10px;
        text-align: center;
        font-size: 13px;
        font-weight: 700;
        border: 2px solid;
        transition: all 0.3s ease;
        position: relative;
        box-shadow: 0 4px 14px rgba(0,0,0,0.18);
    }
    .pipe-node .icon { font-size: 22px; display: block; margin-bottom: 4px; }
    .pipe-node .sub { font-weight: 400; font-size: 11px; margin-top: 2px; opacity: 0.9; }

    .pipe-arrow {
        font-size: 20px;
        background: linear-gradient(90deg, #4facfe, #00e6c3);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        flex-shrink: 0;
        padding: 0 2px;
        font-weight: 900;
    }

    .pipe-pending {
        background: linear-gradient(135deg, #3a4a55, #2a3944);
        border-color: rgba(255,255,255,0.18); color: #b8c6cf;
        box-shadow: none;
    }
    .pipe-running {
        background: linear-gradient(135deg, #4facfe 0%, #00c6ae 100%);
        border-color: #7fdcff; color: #ffffff;
        animation: pulse 1.1s infinite;
    }
    .pipe-done {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        border-color: #6bf7b0; color: #ffffff;
    }
    .pipe-skipped {
        background: linear-gradient(135deg, #f7971e 0%, #ffd200 100%);
        border-color: #ffe27a; color: #3a2a00; border-style: dashed;
    }
    .pipe-error {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        border-color: #ff9c8a; color: #ffffff;
    }

    @keyframes pulse {
        0%   { box-shadow: 0 0 0 0 rgba(79,172,254,0.55); }
        70%  { box-shadow: 0 0 0 10px rgba(79,172,254,0); }
        100% { box-shadow: 0 0 0 0 rgba(79,172,254,0); }
    }

    .tools-row {
        display: flex;
        gap: 8px;
        justify-content: center;
        margin-top: 8px;
    }
    .tool-chip {
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 11px;
        font-weight: 700;
        border: 1.5px solid;
        display: inline-block;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    }

    /* ---- Expanders / tabs blend into dark theme ---- */
    .streamlit-expanderHeader { color: #eaf6fb !important; }
    div[data-testid="stExpander"] {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# PIPELINE DIAGRAM
# ============================================================

STATUS_CLASS = {
    "pending": "pipe-pending",
    "running": "pipe-running",
    "done": "pipe-done",
    "skipped": "pipe-skipped",
    "error": "pipe-error",
}

STATUS_LABEL = {
    "pending": "Waiting",
    "running": "Running…",
    "done": "Done",
    "skipped": "Not needed",
    "error": "Failed",
}

TOOL_LABELS = {
    "weather_tool": ("☁️", "Weather Tool"),
    "ocean_tool": ("🌊", "Ocean Tool"),
    "tide_tool": ("🌙", "Tide Tool"),
}


def _tool_chip_html(key, status):
    icon, label = TOOL_LABELS[key]
    cls = STATUS_CLASS.get(status, "pipe-pending")
    return (
        f'<span class="tool-chip {cls}">{icon} {label} · '
        f'{STATUS_LABEL.get(status, status)}</span>'
    )


def render_pipeline_html(status: dict) -> str:
    """Build the HTML for the live node diagram from a status dict."""

    def node(key, icon, title, sub=""):
        cls = STATUS_CLASS.get(status.get(key, "pending"), "pipe-pending")
        label = STATUS_LABEL.get(status.get(key, "pending"), "")
        sub_html = f'<div class="sub">{sub}</div>' if sub else ""
        return (
            f'<div class="pipe-node {cls}">'
            f'<span class="icon">{icon}</span>{title}'
            f'<div class="sub">{label}</div>'
            f'{sub_html}'
            f'</div>'
        )

    arrow = '<div class="pipe-arrow">➜</div>'

    tools_html = (
        '<div class="tools-row">'
        + _tool_chip_html("weather_tool", status.get("weather_tool", "pending"))
        + _tool_chip_html("ocean_tool", status.get("ocean_tool", "pending"))
        + _tool_chip_html("tide_tool", status.get("tide_tool", "pending"))
        + "</div>"
    )

    html = (
        '<div class="pipeline-wrap">'
        + node("planner", "🧭", "Planner Agent", "Builds execution plan")
        + arrow
        + node("agent", "🤖", "Data &amp; Analysis Agent", "Calls tools, analyzes")
        + arrow
        + node("recommendation", "📋", "Recommendation", "Final structured answer")
        + "</div>"
        + tools_html
    )
    return html


class ToolTracker(BaseCallbackHandler):
    """
    LangChain callback handler that flips weather_tool / ocean_tool / tide_tool
    status to 'running' -> 'done' as the underlying agent actually calls them,
    and re-renders the pipeline placeholder live.
    """

    TOOL_KEY_MAP = {
        "weather_agent": "weather_tool",
        "get_ocean_data": "ocean_tool",
        "get_tide": "tide_tool",
    }

    def __init__(self, status: dict, placeholder):
        self.status = status
        self.placeholder = placeholder
        self._run_id_to_key = {}

    def _redraw(self):
        self.placeholder.markdown(render_pipeline_html(self.status), unsafe_allow_html=True)

    def on_tool_start(self, serialized, input_str, *, run_id=None, **kwargs):
        name = (serialized or {}).get("name", "")
        key = self.TOOL_KEY_MAP.get(name)
        if key:
            self._run_id_to_key[run_id] = key
            self.status[key] = "running"
            self._redraw()

    def on_tool_end(self, output, *, run_id=None, **kwargs):
        key = self._run_id_to_key.get(run_id)
        if key:
            self.status[key] = "done"
            self._redraw()

    def on_tool_error(self, error, *, run_id=None, **kwargs):
        key = self._run_id_to_key.get(run_id)
        if key:
            self.status[key] = "error"
            self._redraw()


# ============================================================
# SESSION STATE
# ============================================================

if "history" not in st.session_state:
    st.session_state.history = []  # list of {question, lat, lon, result, error, node_status}

# ============================================================
# SIDEBAR — LOCATION & INPUTS
# ============================================================

with st.sidebar:
    st.markdown('<div class="sidebar-title">🌊 AI Ocean Assistant</div>', unsafe_allow_html=True)
    st.caption("Agentic marine intelligence: weather + ocean data, planned and analyzed by AI agents.")

    st.markdown('<div class="grad-subheader">📍 Location</div>', unsafe_allow_html=True)

    loc_mode = st.radio(
        "Set location",
        ["Manual entry", "Pick on map"],
        horizontal=True,
    )

    default_lat, default_lon = 19.0760, 72.8777  # Mumbai coast, as a sane default

    if "picked_lat" not in st.session_state:
        st.session_state["picked_lat"] = default_lat
        st.session_state["picked_lon"] = default_lon

    # Detect a mode switch, or an external update (e.g. a map click handled
    # further down the page) so the relevant widget re-seeds from the
    # canonical picked_lat/picked_lon instead of showing a stale value.
    # (Streamlit ignores `value=` on reruns once a widget `key` exists, and
    # won't let us overwrite a widget's key *after* it has already run this
    # script pass — so any external update sets this flag and reruns, and
    # we consume it here, *before* the widgets below are instantiated.)
    mode_changed = st.session_state.get("prev_loc_mode") != loc_mode
    st.session_state["prev_loc_mode"] = loc_mode
    force_resync = st.session_state.pop("force_resync_location", False)

    if loc_mode == "Manual entry":
        if mode_changed or force_resync or "manual_lat_input" not in st.session_state:
            st.session_state["manual_lat_input"] = float(st.session_state["picked_lat"])
            st.session_state["manual_lon_input"] = float(st.session_state["picked_lon"])

        latitude = st.number_input(
            "Latitude", min_value=-90.0, max_value=90.0,
            format="%.6f", key="manual_lat_input",
        )
        longitude = st.number_input(
            "Longitude", min_value=-180.0, max_value=180.0,
            format="%.6f", key="manual_lon_input",
        )
        st.session_state["picked_lat"] = latitude
        st.session_state["picked_lon"] = longitude
    else:
        st.caption("👉 Click anywhere on the map (main area) to drop a pin.")

        if mode_changed or force_resync or "map_lat_input" not in st.session_state:
            st.session_state["map_lat_input"] = float(st.session_state["picked_lat"])
            st.session_state["map_lon_input"] = float(st.session_state["picked_lon"])

        latitude = st.number_input(
            "Latitude (from map click)", min_value=-90.0, max_value=90.0,
            format="%.6f", key="map_lat_input",
        )
        longitude = st.number_input(
            "Longitude (from map click)", min_value=-180.0, max_value=180.0,
            format="%.6f", key="map_lon_input",
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

st.markdown(
    """
    <div class="hero-banner">
        <h1>🌊 Ask the Ocean Assistant</h1>
        <p>e.g. "Is it safe to go fishing tomorrow morning?" or "What are the sea conditions near me today?"</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if loc_mode == "Pick on map":
    if not FOLIUM_AVAILABLE:
        st.warning(
            "Interactive map picking needs the `streamlit-folium` and `folium` "
            "packages. Add them to requirements.txt and reinstall "
            "(`pip install streamlit-folium folium`), or switch to 'Manual entry' "
            "in the sidebar."
        )
    else:
        st.markdown('<div class="grad-subheader">📍 Click on the map to set your location</div>', unsafe_allow_html=True)
        m = folium.Map(
            location=[st.session_state["picked_lat"], st.session_state["picked_lon"]],
            zoom_start=5,
        )
        folium.Marker(
            [st.session_state["picked_lat"], st.session_state["picked_lon"]],
            tooltip="Selected location",
        ).add_to(m)

        map_result = st_folium(
            m,
            height=380,
            width=None,  # stretch to container width
            returned_objects=["last_clicked"],
            key="location_picker_map",
        )

        if map_result and map_result.get("last_clicked"):
            clicked_lat = map_result["last_clicked"]["lat"]
            clicked_lon = map_result["last_clicked"]["lng"]
            if (
                round(clicked_lat, 6) != round(st.session_state["picked_lat"], 6)
                or round(clicked_lon, 6) != round(st.session_state["picked_lon"], 6)
            ):
                st.session_state["picked_lat"] = clicked_lat
                st.session_state["picked_lon"] = clicked_lon
                # Can't write the sidebar's widget keys (map_lat_input /
                # map_lon_input) directly here — that widget already ran
                # earlier in this script pass, and Streamlit disallows
                # mutating a widget's key after it has been instantiated
                # in the same run. Instead, flag a resync; the sidebar
                # block consumes this flag *before* creating the widgets
                # on the next rerun, so it picks up picked_lat/picked_lon.
                st.session_state["force_resync_location"] = True
                st.rerun()

        st.caption(
            f"📌 Selected: {st.session_state['picked_lat']:.6f}, "
            f"{st.session_state['picked_lon']:.6f} — fine-tune in the sidebar if needed."
        )

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
# HELPERS
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


def render_result(latitude, longitude, result):
    st.markdown('<div class="grad-subheader">📋 Recommendation</div>', unsafe_allow_html=True)

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

    tab1, tab2, tab3 = st.tabs(["🌤️ Weather data", "🌊 Ocean data", "🌙 Tide data"])

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

    with tab3:
        td = result.get("tide_data", {}) or {}
        if not td:
            st.info("No tide data returned (not needed for this query, or unavailable).")
        elif td.get("success") is False:
            st.warning(f"Tide lookup failed: {td.get('error', 'Unknown error')}")
        else:
            # get_tide's response can come through as the raw tool output
            # ({"success", "data": {"station", "tides": {...}}}) or already
            # unwrapped by the LLM — handle both shapes defensively.
            inner = td.get("data", td)
            station = inner.get("station") or {}
            tides = inner.get("tides") or inner

            if station:
                st.markdown(
                    f"**Nearest station:** {station.get('name', 'N/A')} "
                    f"({station.get('distance_km', '?')} km away)"
                )

            high_tide = tides.get("high_tide") or []
            low_tide = tides.get("low_tide") or []

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**⬆️ High tide**")
                if high_tide:
                    st.dataframe(pd.DataFrame(high_tide), use_container_width=True, hide_index=True)
                else:
                    st.caption("No high tide entries returned.")
            with col_b:
                st.markdown("**⬇️ Low tide**")
                if low_tide:
                    st.dataframe(pd.DataFrame(low_tide), use_container_width=True, hide_index=True)
                else:
                    st.caption("No low tide entries returned.")

            if not high_tide and not low_tide:
                st.json(td)

    if show_raw:
        with st.expander("🔍 Raw JSON result"):
            st.json(result)


# ============================================================
# RUN THE PIPELINE (with live node visualization)
# ============================================================

if submitted:
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        st.divider()
        st.markdown('<div class="grad-subheader">⚙️ Pipeline execution</div>', unsafe_allow_html=True)

        node_status = {
            "planner": "pending",
            "agent": "pending",
            "weather_tool": "pending",
            "ocean_tool": "pending",
            "tide_tool": "pending",
            "recommendation": "pending",
        }
        diagram_placeholder = st.empty()
        diagram_placeholder.markdown(render_pipeline_html(node_status), unsafe_allow_html=True)

        state = {
            "latitude": latitude,
            "longitude": longitude,
            "user_question": question,
            "weather_data": {},
            "ocean_data": {},
            "tide_data": {},
            "plan": "",
            "status": "in_progress",
            "recommendation": {},
        }

        result = None
        error = None

        try:
            # ---- Step 1: Planner ----
            node_status["planner"] = "running"
            diagram_placeholder.markdown(render_pipeline_html(node_status), unsafe_allow_html=True)

            plan_out = planner_node(state)
            state.update(plan_out)

            node_status["planner"] = "done"
            node_status["agent"] = "running"
            diagram_placeholder.markdown(render_pipeline_html(node_status), unsafe_allow_html=True)

            # ---- Step 2: Agent (tools tracked live via callback) ----
            tracker = ToolTracker(node_status, diagram_placeholder)
            agent_out = agent_node(state, callbacks=[tracker])
            state.update(agent_out)

            # Any tool never called by the agent -> mark as "not needed"
            for tool_key in ("weather_tool", "ocean_tool", "tide_tool"):
                if node_status[tool_key] == "pending":
                    node_status[tool_key] = "skipped"

            node_status["agent"] = "done"
            node_status["recommendation"] = "running"
            diagram_placeholder.markdown(render_pipeline_html(node_status), unsafe_allow_html=True)

            time.sleep(0.15)  # brief beat so "running" is visible before "done"
            node_status["recommendation"] = "done"
            diagram_placeholder.markdown(render_pipeline_html(node_status), unsafe_allow_html=True)

            result = state

        except Exception as e:
            error = f"{e}\n\n{traceback.format_exc()}"
            # mark whichever stage was in-flight as failed
            for key in ("planner", "agent", "recommendation"):
                if node_status[key] == "running":
                    node_status[key] = "error"
            diagram_placeholder.markdown(render_pipeline_html(node_status), unsafe_allow_html=True)

        st.session_state.history.insert(
            0,
            {
                "question": question,
                "latitude": latitude,
                "longitude": longitude,
                "result": result,
                "error": error,
                "node_status": dict(node_status),
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
        render_result(latest["latitude"], latest["longitude"], latest["result"])

    if len(st.session_state.history) > 1:
        with st.expander(f"📜 Previous queries ({len(st.session_state.history) - 1})"):
            for item in st.session_state.history[1:]:
                st.markdown(f"**\"{item['question']}\"** — ({item['latitude']:.4f}, {item['longitude']:.4f})")
                if item.get("node_status"):
                    st.markdown(render_pipeline_html(item["node_status"]), unsafe_allow_html=True)
                if item["error"]:
                    st.error("Failed")
                else:
                    rec = (item["result"] or {}).get("recommendation", {}) or {}
                    st.write(f"Status: {rec.get('safety_status', 'N/A')} — {rec.get('summary', '')}")
                st.markdown("---")
