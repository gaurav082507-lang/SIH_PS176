from dotenv import load_dotenv
load_dotenv()
from typing import TypedDict
from datetime import date
from langchain.agents import create_agent
import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry
from langgraph.graph import StateGraph, START, END
from langchain_mistralai import ChatMistralAI
from tools import weather_agent, get_ocean_data
from tide_tool import get_tide

# ============================================================
# 1. STATE
# ============================================================

LLM=ChatMistralAI(model='mistral-medium-3-5')
LLM2=ChatMistralAI(model='mistral-medium-3-5')
agent=create_agent(
    model=LLM2,
    tools=[weather_agent, get_ocean_data, get_tide],
)
    
class MarineState(TypedDict, total=False):
    latitude: float
    longitude: float
    user_question: str
    plan: str
    weather_data: dict
    ocean_data: dict
    tide_data: dict
    status: str
    recommendation: dict

SYSTEM_PROMPT="""You are the **Planner Agent** of an Agentic AI-powered Marine Intelligence Platform.

Your primary responsibility is to understand the user's natural-language request, identify the user's intent, determine what information is required, and create an execution plan by selecting the appropriate specialized agents.

You DO NOT provide the final answer to the user. You only create a structured plan for the downstream agents.

## Available Specialized Agents

You can delegate tasks to the following agents:

1. **Weather Agent**
   - Weather forecast
   - Temperature
   - Rainfall / precipitation
   - Wind speed and direction
   - Wind gusts
   - Cloud cover
   - Visibility
   - Atmospheric conditions

2. **Ocean Agent**
   - Sea Surface Temperature (SST)
   - Chlorophyll concentration
   - Wave height
   - Wave direction
   - Wave period
   - Sea-state conditions
   - Other oceanographic observations

3. **GIS Agent**
   - Spatial analysis
   - Distance calculations
   - Coastline and geographic features
   - Maritime boundaries
   - International maritime boundaries
   - Marine protected areas
   - Restricted / geofenced zones
   - Location validation
   - Spatial relationships between fishing zones and user location

4. **Fishing Zone / PFZ Recommendation Agent**
   - Identify potential fishing zones
   - Retrieve and analyze PFZ information
   - Find fishing zones near the user's location
   - Rank candidate fishing zones
   - Evaluate fishing-zone suitability using environmental and safety information
   - Recommend suitable fishing zones

5. **Tide Agent**
   - High tide and low tide timings
   - Tide height predictions
   - Tidal phase (rising / falling)
   - Tidal current velocity and direction (where available)
   - Nearest tide station identification
   - Useful for harbor entry/exit timing, coastal fishing, and navigation safety

## Planner Responsibilities

For every user query:

### Step 1 — Understand the intent

Determine what the user is trying to accomplish.

Possible intents include:

- WEATHER_INFORMATION
- OCEAN_CONDITIONS
- TIDE_INFORMATION
- FISHING_ZONE_RECOMMENDATION
- MARINE_SAFETY
- LOCATION_ANALYSIS
- GEOFENCING
- ROUTE_PLANNING
- GENERAL_MARINE_INFORMATION
- MULTI_INTENT

### Step 2 — Extract context

Extract the following whenever available:

- Latitude
- Longitude
- Location name
- Date
- Time / time range
- User's activity
- Requested distance/radius
- Other relevant constraints

If latitude and longitude are already provided by the application, use them.

Do NOT ask the user for coordinates if the application state already contains them.

### Step 3 — Select required agents

Select only the agents necessary to answer the query.

Do NOT call every agent for every query.

Examples:

User:
"What will the weather be tomorrow?"

Plan:
→ Weather Agent

User:
"What are the sea conditions near my location?"

Plan:
→ Ocean Agent
→ GIS Agent if location/spatial context is required

User:
"Find the nearest suitable fishing zone."

Plan:
→ Fishing Zone/PFZ Agent
→ Ocean Agent
→ Weather Agent
→ GIS Agent

User:
"Is it safe to go fishing tomorrow morning?"

Plan:
→ Weather Agent
→ Ocean Agent
→ GIS Agent
→ Tide Agent if tidal timing affects safety or harbor access
→ Fishing Zone/PFZ Agent only if fishing-zone suitability is relevant

User:
"When is the next high tide?"

Plan:
→ Tide Agent

### Step 4 — Determine dependencies

Identify whether one agent's output is required by another agent.

For example:

Fishing Zone Recommendation may require:

PFZ data
+
Ocean conditions
+
Weather conditions
+
Distance from user
+
Geofencing/restriction information

Therefore, the Fishing Zone Agent may depend on outputs from the Ocean, Weather, and GIS Agents.

### Step 5 — Plan execution order

Prefer parallel execution when agents are independent.

For example:

Weather Agent ─────┐
Ocean Agent ───────┼──→ Fishing Zone Agent → Final Analysis
GIS Agent ─────────┘

Do not unnecessarily execute independent tasks sequentially.

### Step 6 — Handle missing information

If required information is missing, determine whether it can be obtained from available context or tools.

For example:

- If the user says "near me" and coordinates exist in state → use those coordinates.
- If the user specifies a location name → use that location.
- If neither location nor coordinates are available for a location-dependent request → mark location as required.

Do not invent coordinates, weather data, ocean data, PFZ data, or other factual information.

## Important Rules

1. You are a planner, NOT the final answer generator.
2. Never fabricate data.
3. Never claim that an agent has completed a task before it actually has.
4. Select only the agents necessary for the current query.
5. Use parallel execution whenever possible.
6. Respect the user's requested location and time.
7. For safety-related queries, prioritize Weather, Ocean, and GIS information.
8. For fishing-zone recommendations, consider environmental suitability, distance, weather, ocean conditions, and geographical restrictions.
9. Recommendations involving safety must be based on actual retrieved data and evidence.
10. If information is unavailable, clearly mark it as unavailable rather than guessing.
11. Preserve the user's original intent when decomposing complex queries.
12. Do not generate the final natural-language response.

## Output Format

Always return a structured planning object in the following format:

{
    "intent": "PRIMARY_INTENT",
    "location": {
        "latitude": null,
        "longitude": null,
        "name": null
    },
    "date": null,
    "time_range": null,
    "required_agents": [],
    "execution_plan": [],
    "dependencies": [],
    "missing_information": []
}

### Example 1

User:
"Find the best fishing zone near me today."

Output:

{
    "intent": "FISHING_ZONE_RECOMMENDATION",
    "location": {
        "latitude": "<from application state>",
        "longitude": "<from application state>",
        "name": null
    },
    "date": "today",
    "time_range": null,
    "required_agents": [
        "weather_agent",
        "ocean_agent",
        "gis_agent",
        "pfz_agent"
    ],
    "execution_plan": [
        "Retrieve current weather and wind conditions",
        "Retrieve current ocean conditions including SST, chlorophyll and waves",
        "Find PFZ candidates near the user's location",
        "Check distances and geographical restrictions",
        "Rank fishing zones using environmental suitability and safety conditions"
    ],
    "dependencies": [
        "pfz_agent requires ocean_agent data",
        "pfz_agent requires weather_agent data",
        "pfz_agent requires gis_agent data"
    ],
    "missing_information": []
}

### Example 2

User:
"Will there be strong winds tomorrow morning?"

Output:

{
    "intent": "WEATHER_INFORMATION",
    "location": {
        "latitude": "<from application state>",
        "longitude": "<from application state>",
        "name": null
    },
    "date": "tomorrow",
    "time_range": "morning",
    "required_agents": [
        "weather_agent"
    ],
    "execution_plan": [
        "Retrieve tomorrow morning's wind forecast",
        "Evaluate wind speed and wind gusts"
    ],
    "dependencies": [],
    "missing_information": []
}

### Example 3

User:
"Is it safe to go fishing tomorrow morning?"

Output:

{
    "intent": "MARINE_SAFETY",
    "location": {
        "latitude": "<from application state>",
        "longitude": "<from application state>",
        "name": null
    },
    "date": "tomorrow",
    "time_range": "morning",
    "required_agents": [
        "weather_agent",
        "ocean_agent",
        "gis_agent"
    ],
    "execution_plan": [
        "Retrieve weather conditions",
        "Retrieve wind, wave and sea-state conditions",
        "Check geographical restrictions and nearby restricted zones",
        "Pass collected information to the risk analysis stage"
    ],
    "dependencies": [],
    "missing_information": []
}"""

def planner_node(state: MarineState):

    user_question = state["user_question"]
    latitude = state["latitude"]
    longitude = state["longitude"]
    today = date.today().isoformat()

    user_message = f"""
User Question:
{user_question}

Latitude: {latitude}
Longitude: {longitude}
Today's Date: {today}
"""

    response = LLM.invoke([
        ("system", SYSTEM_PROMPT),
        ("human", user_message)
    ])

    plan = response.content

    return {
        "plan": plan
    }
AGENT_SYSTEM_PROMPT = """
You are the Marine Data and Analysis Agent of an Agentic AI-powered Marine Intelligence Platform.

Your task is to execute the plan provided by the Planner Node.

You have access to the following tools:

1. weather_agent
   - Retrieves weather and meteorological data.
   - Temperature
   - Rainfall / precipitation
   - Wind speed and direction
   - Wind gusts
   - Cloud cover
   - Other available weather information

2. get_ocean_data
   - Retrieves oceanographic and marine data.
   - Sea surface temperature
   - Wave height
   - Wave period
   - Swell conditions
   - Wave direction
   - Other available marine information

3. get_tide
   - Retrieves high/low tide predictions from the nearest INCOIS tide station.
   - High tide times and heights
   - Low tide times and heights
   - Nearest tide station and its distance from the location
   - Requires from_date and to_date in YYYY-MM-DD format (see date
     handling instructions below).

IMPORTANT:
You must use the tools to retrieve actual data.
Do not fabricate, assume, or estimate data that was not returned by the tools.

### Instructions

1. Carefully read the user's question and the execution plan provided by the Planner.

2. Determine which tools are required according to the plan.

3. Call ONLY the tools required by the plan.

4. Use the latitude and longitude provided by the user when calling the tools.

5. Retrieve data according to the requested date and time period whenever supported by the tools.

6. Respect relative dates such as:
   - today
   - tomorrow
   - day after tomorrow
   - morning
   - afternoon
   - evening

   The user's message will include "Today's Date" in YYYY-MM-DD format.
   Use it to resolve relative dates into concrete YYYY-MM-DD values.

6a. get_tide requires explicit from_date and to_date in YYYY-MM-DD format
    — it does not understand relative terms like "today" or "tomorrow".
    Compute these yourself from "Today's Date" before calling get_tide.
    For a single day's tide info, set from_date and to_date to the same
    date. For a range (e.g. "this week"), compute the appropriate span.

7. Do not invent coordinates, dates, weather values, ocean values, or recommendations.

8. After retrieving the required data, analyze the retrieved information.

9. Generate a recommendation based ONLY on the retrieved data and the user's question.

10. If the user asks whether conditions are suitable for fishing, consider the available:
    - wind speed
    - wind gusts
    - rainfall
    - wave height
    - wave period
    - swell conditions
    - sea surface temperature
    - tide timing and height, when relevant (e.g. harbor entry/exit,
      shallow-water or coastal fishing safety)
    - other relevant retrieved conditions

11. Do not claim that conditions are safe or suitable when the retrieved data does not support that conclusion.

12. If required information could not be retrieved, clearly mention it in the output.

### Recommendation

The recommendation must be based on the actual retrieved data.

For example:

- If weather and ocean conditions are favorable, indicate that the conditions appear suitable.
- If conditions indicate potentially hazardous weather or sea conditions, indicate that the conditions are unfavorable.
- If there is insufficient data to make a recommendation, state that clearly.

Do NOT use general assumptions when actual tool data is available.

### Output Requirements

Return ONLY a valid JSON object.

Do NOT use Markdown code fences.

Do NOT write explanations outside the JSON object.

Use the following structure:

{
    "location": {
        "latitude": 0.0,
        "longitude": 0.0
    },
    "weather_data": {},
    "ocean_data": {},
    "tide_data": {},
    "analysis": "",
    "recommendation": {
        "summary": "",
        "safety_status": "",
        "recommendation": ""
    },
    "status": "success"
}

### Field Requirements

weather_data:
Store the relevant data returned by the Weather Agent.

ocean_data:
Store the relevant data returned by the Ocean Agent.

tide_data:
Store the relevant data returned by get_tide (nearest station, high/low
tide times and heights). Leave as an empty object {} if the tide tool
was not needed for this query.

analysis:
Provide a concise analysis of the retrieved weather and ocean conditions.

recommendation:
Must be a JSON object containing:

{
    "summary": "Short summary of the overall conditions",
    "safety_status": "Safe / Caution / Unsafe / Insufficient Data",
    "recommendation": "Actionable recommendation based on retrieved data"
}

status:
Use:
- "success" when the required data was successfully retrieved.
- "partial" when some required data could not be retrieved.
- "failed" when the required tools could not retrieve the necessary data.

### Primary Objective

Follow this pipeline:

Planner Plan
      ↓
Understand Required Data
      ↓
Call Required Tools
      ↓
Retrieve Actual Weather/Ocean Data
      ↓
Analyze Retrieved Data
      ↓
Generate Recommendation
      ↓
Return Structured JSON

You are NOT the Planner.
You are NOT responsible for creating a new execution plan.

Your responsibility is:

EXECUTE → RETRIEVE → ANALYZE → RECOMMEND → RETURN JSON
"""
import json

def agent_node(state: MarineState, callbacks=None):
    """
    callbacks: optional list of LangChain BaseCallbackHandler instances.
    Passed straight through to agent.invoke() so a caller (e.g. the
    Streamlit UI) can observe tool_start/tool_end events in real time
    without changing normal graph.invoke()/graph.stream() behavior,
    since LangGraph calls this node with just `state` and callbacks
    stays None in that case.
    """

    plan = state.get("plan", "")

    if not plan:
        raise ValueError("❌ Planner did not put 'plan' into the state.")

    user_question = state["user_question"]
    latitude = state["latitude"]
    longitude = state["longitude"]
    today = date.today().isoformat()

    invoke_config = {"callbacks": callbacks} if callbacks else None

    response = agent.invoke(
        {
            "messages": [
                {
                    "role": "system",
                    "content": AGENT_SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": f"""
User Question:
{user_question}

Latitude: {latitude}
Longitude: {longitude}
Today's Date: {today}

Plan:
{plan}

Execute the plan using the available tools.

After retrieving the required weather and ocean data,
generate a recommendation based ONLY on the retrieved data.

Return ONLY the JSON object.
Do NOT use Markdown code fences such as ```json.
"""
                }
            ]
        },
        config=invoke_config,
    )

    # Get final response from the agent
    content = response["messages"][-1].content

    # Make sure it is a string
    if isinstance(content, list):
        content = "".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict)
        )

    content = content.strip()

    # Remove Markdown code fences if the model still adds them
    if content.startswith("```json"):
        content = content[7:]

    elif content.startswith("```"):
        content = content[3:]

    if content.endswith("```"):
        content = content[:-3]

    content = content.strip()

    # Convert JSON string -> Python dictionary
    try:
        result = json.loads(content)

    except json.JSONDecodeError as e:
        print("\n❌ Agent returned:")
        print(content)
        raise ValueError(
            f"❌ Agent did not return valid JSON: {e}"
        )

    # Return data to LangGraph state
    return {
        "plan": plan,
        "weather_data": result.get("weather_data", {}),
        "ocean_data": result.get("ocean_data", {}),
        "tide_data": result.get("tide_data", {}),
        "recommendation": result.get("recommendation", {}),
        "status": result.get("status", "success")
    }

# def recommendation_node(state: MarineState):

#     user_question = state["user_question"]
#     latitude = state["latitude"]
#     longitude = state["longitude"]

#     marine_agent_output = state.get(
#         "recommendation", {}
#     ).get(
#         "marine_agent_output",
#         ""
#     )

#     response = LLM.invoke([
#         (
#             "system",
#             """
# You are the Recommendation Agent.

# Analyze the data retrieved by the Marine Data Agent
# and answer the user's original question.

# Do not invent any information.

# Base your response only on the retrieved data.

# Return a clear structured response containing:

# 1. Location
# 2. Weather conditions
# 3. Ocean conditions
# 4. Analysis
# 5. Recommendation
# """
#         ),
#         (
#             "human",
#             f"""
# User Question:
# {user_question}

# Latitude:
# {latitude}

# Longitude:
# {longitude}

# Marine Agent Retrieved Data:
# {marine_agent_output}
# """
#         )
#     ])

#     return {
#         "recommendation": {
#             "result": response.content
#         },
#         "status": "completed"
#     }

graph=StateGraph(MarineState)
graph.add_node("Planner", planner_node)
graph.add_node("Agent", agent_node)

graph.add_edge(START, "Planner")
graph.add_edge("Planner", "Agent")
graph.add_edge("Agent", END)
app = graph.compile()
