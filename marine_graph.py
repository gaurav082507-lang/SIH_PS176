from dotenv import load_dotenv
load_dotenv()

import json
from typing import TypedDict

from langchain.agents import create_agent
from langchain_mistralai import ChatMistralAI
from langgraph.graph import StateGraph, START, END

from tools import weather_agent, get_ocean_data


# ============================================================
# 1. LLMs
# ============================================================

LLM = ChatMistralAI(
    model="mistral-medium-3-5"
)

LLM2 = ChatMistralAI(
    model="mistral-medium-3-5"
)


# ============================================================
# 2. MARINE DATA AGENT
# ============================================================

agent = create_agent(
    model=LLM2,
    tools=[
        weather_agent,
        get_ocean_data
    ],
)


# ============================================================
# 3. STATE
# ============================================================

class MarineState(TypedDict, total=False):

    latitude: float
    longitude: float

    user_question: str

    plan: str

    weather_data: dict
    ocean_data: dict

    recommendation: dict

    status: str


# ============================================================
# 4. PLANNER SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are the Planner Agent of an Agentic AI-powered Marine Intelligence Platform.

Your responsibility is to understand the user's natural-language request,
identify the required information, and create an execution plan for the
Marine Data Agent.

You DO NOT provide the final answer to the user.

You ONLY create a structured execution plan.

============================================================
AVAILABLE AGENTS / TOOLS
============================================================

Currently available specialized tools:

1. Weather Agent

The Weather Agent can retrieve:

- Weather forecast
- Temperature
- Rainfall / precipitation
- Wind speed
- Wind direction
- Wind gusts
- Cloud cover
- Visibility
- Atmospheric conditions
- Other available meteorological information


2. Ocean Agent

The Ocean Agent can retrieve:

- Sea Surface Temperature (SST)
- Wave height
- Wave direction
- Wave period
- Swell conditions
- Sea-state conditions
- Other available oceanographic information


IMPORTANT:

GIS Agent and PFZ Agent are planned components of the platform,
but they are NOT currently available as executable tools.

Therefore:

DO NOT assign GIS Agent or PFZ Agent in the current execution plan.

============================================================
STEP 1 — UNDERSTAND USER INTENT
============================================================

Determine the primary intent.

Possible intents:

- WEATHER_INFORMATION
- OCEAN_CONDITIONS
- MARINE_SAFETY
- FISHING_CONDITIONS
- GENERAL_MARINE_INFORMATION
- MULTI_INTENT

============================================================
STEP 2 — EXTRACT CONTEXT
============================================================

Extract whenever available:

- Latitude
- Longitude
- Location name
- Date
- Time
- Time range
- User activity
- Requested distance/radius
- Other constraints

Latitude and longitude are provided by the application state.

DO NOT ask the user for coordinates if coordinates already exist
in the application state.

============================================================
STEP 3 — SELECT REQUIRED AGENTS
============================================================

Select ONLY the tools required for the user's request.

Examples:

User:
"What will the weather be tomorrow?"

Required:
- weather_agent


User:
"What are the sea conditions near my location?"

Required:
- get_ocean_data


User:
"Is it safe to go fishing tomorrow morning?"

Required:
- weather_agent
- get_ocean_data


User:
"Will strong winds affect fishing tomorrow?"

Required:
- weather_agent


User:
"What is the sea temperature?"

Required:
- get_ocean_data


User:
"Give me complete marine conditions for tomorrow."

Required:
- weather_agent
- get_ocean_data

============================================================
STEP 4 — DEPENDENCIES
============================================================

Identify whether one tool's output depends on another.

Weather and ocean retrieval are generally independent.

Therefore, when both are required, they may be retrieved independently.

The Marine Data Agent will combine the retrieved information
during analysis.

============================================================
STEP 5 — EXECUTION PLAN
============================================================

Create a clear sequence of tasks.

For example:

1. Retrieve required weather conditions.
2. Retrieve required ocean conditions.
3. Analyze the retrieved information.
4. Generate a recommendation based only on retrieved data.

============================================================
IMPORTANT RULES
============================================================

1. You are a planner, NOT the final answer generator.

2. Never fabricate weather data.

3. Never fabricate ocean data.

4. Never invent coordinates.

5. Never invent dates.

6. Respect the user's requested date and time.

7. Select only the tools actually required.

8. Do not select GIS Agent or PFZ Agent because they are not
   currently executable tools.

9. For marine safety queries, prioritize both weather and ocean
   information whenever relevant.

10. For fishing-condition queries, consider weather and ocean
    conditions.

11. Do not claim that conditions are safe or unsafe yourself.

12. The downstream Marine Data Agent will perform the final analysis.

============================================================
OUTPUT FORMAT
============================================================

Return ONLY the following structured planning object:

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

Do NOT return Markdown.

Do NOT return explanations outside the object.
"""


# ============================================================
# 5. PLANNER NODE
# ============================================================

def planner_node(state: MarineState):

    user_question = state["user_question"]

    latitude = state.get("latitude")
    longitude = state.get("longitude")

    user_message = f"""
User Question:
{user_question}

Latitude:
{latitude}

Longitude:
{longitude}

Create the execution plan according to the planner instructions.
"""

    response = LLM.invoke(
        [
            ("system", SYSTEM_PROMPT),
            ("human", user_message)
        ]
    )

    plan = response.content

    if isinstance(plan, list):
        plan = "".join(
            item.get("text", "")
            for item in plan
            if isinstance(item, dict)
        )

    return {
        "plan": plan.strip()
    }


# ============================================================
# 6. MARINE DATA AGENT SYSTEM PROMPT
# ============================================================

AGENT_SYSTEM_PROMPT = """
You are the Marine Data and Analysis Agent of an Agentic AI-powered
Marine Intelligence Platform.

Your responsibility is:

EXECUTE
    ↓
RETRIEVE
    ↓
ANALYZE
    ↓
RECOMMEND
    ↓
RETURN STRUCTURED JSON


============================================================
AVAILABLE TOOLS
============================================================

1. weather_agent

Retrieves meteorological information including:

- Temperature
- Rainfall / precipitation
- Wind speed
- Wind direction
- Wind gusts
- Cloud cover
- Visibility
- Other available weather information


2. get_ocean_data

Retrieves oceanographic information including:

- Sea Surface Temperature
- Wave height
- Wave direction
- Wave period
- Swell conditions
- Sea-state conditions
- Other available marine information


============================================================
IMPORTANT
============================================================

You MUST use the tools to retrieve actual data.

Never fabricate:

- Weather values
- Ocean values
- Coordinates
- Dates
- Times
- Forecasts
- Recommendations


============================================================
EXECUTION RULES
============================================================

1. Carefully read the user's question.

2. Carefully read the Planner's execution plan.

3. Call ONLY the tools required by the plan.

4. Use the latitude and longitude supplied by the application.

5. Respect requested dates.

6. Respect requested times such as:

   - morning
   - afternoon
   - evening
   - night

7. Respect relative dates such as:

   - today
   - tomorrow
   - day after tomorrow

8. If weather information is required, call weather_agent.

9. If ocean information is required, call get_ocean_data.

10. If both are required, call both tools.

11. After retrieving the data, analyze ONLY the actual returned
    information.

12. Do not use general assumptions when actual data is available.

13. If required information is unavailable, clearly state that
    insufficient data is available.


============================================================
MARINE SAFETY
============================================================

For safety-related questions, consider the retrieved:

Weather:
- Wind speed
- Wind gusts
- Rainfall
- Visibility
- Temperature
- Other relevant conditions

Ocean:
- Wave height
- Wave period
- Wave direction
- Swell
- Sea surface temperature
- Sea-state conditions


IMPORTANT:

Do not claim "Safe" merely because conditions appear normal.

The safety_status must reflect ONLY what can reasonably be supported
by the retrieved data.

If the available information is insufficient:

"safety_status": "Insufficient Data"


============================================================
FISHING CONDITIONS
============================================================

When the user asks about fishing conditions, analyze the available:

- Wind
- Wind gusts
- Rainfall
- Visibility
- Wave height
- Wave period
- Swell
- Sea-state
- Sea surface temperature

Do not invent fishing-zone/PFZ information.

PFZ functionality will be added later when the PFZ tool becomes
available.


============================================================
OUTPUT
============================================================

Return ONLY a valid JSON object.

Do NOT use Markdown.

Do NOT use ```json.

Do NOT write anything outside the JSON object.

Use exactly this structure:

{
    "location": {
        "latitude": 0.0,
        "longitude": 0.0
    },
    "weather_data": {},
    "ocean_data": {},
    "analysis": "",
    "recommendation": {
        "summary": "",
        "safety_status": "",
        "recommendation": ""
    },
    "status": "success"
}


============================================================
STATUS
============================================================

Use:

"success"

when all required data was successfully retrieved.

"partial"

when some required data could not be retrieved.

"failed"

when the required tools could not retrieve the necessary data.


============================================================
PRIMARY OBJECTIVE
============================================================

Planner Plan
     ↓
Understand Required Data
     ↓
Call Required Tools
     ↓
Retrieve Actual Data
     ↓
Analyze Data
     ↓
Generate Recommendation
     ↓
Return JSON

You are NOT the Planner.

You are NOT responsible for creating a new execution plan.

Your responsibility is:

EXECUTE → RETRIEVE → ANALYZE → RECOMMEND → RETURN JSON
"""


# ============================================================
# 7. MARINE DATA AGENT NODE
# ============================================================

def agent_node(state: MarineState, callbacks=None):

    plan = state.get("plan", "")

    if not plan:
        raise ValueError(
            "Planner did not put 'plan' into the state."
        )

    user_question = state["user_question"]

    latitude = state.get("latitude")
    longitude = state.get("longitude")

    invoke_config = {}

    if callbacks:
        invoke_config["callbacks"] = callbacks

    user_message = f"""
User Question:
{user_question}

Latitude:
{latitude}

Longitude:
{longitude}

Planner Execution Plan:
{plan}

Execute the plan using the available tools.

Retrieve the actual required data.

Then analyze the retrieved data and generate the recommendation.

Return ONLY the JSON object.
"""

    response = agent.invoke(
        {
            "messages": [
                {
                    "role": "system",
                    "content": AGENT_SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": user_message
                }
            ]
        },
        config=invoke_config
    )

    # ========================================================
    # GET FINAL AGENT RESPONSE
    # ========================================================

    content = response["messages"][-1].content

    # Mistral/LangChain may sometimes return structured content
    if isinstance(content, list):

        text_parts = []

        for item in content:

            if isinstance(item, dict):

                text = item.get("text")

                if text:
                    text_parts.append(text)

        content = "".join(text_parts)

    content = str(content).strip()


    # ========================================================
    # REMOVE MARKDOWN CODE FENCES
    # ========================================================

    if content.startswith("```json"):

        content = content[len("```json"):]

    elif content.startswith("```"):

        content = content[len("```"):]

    if content.endswith("```"):

        content = content[:-3]

    content = content.strip()


    # ========================================================
    # PARSE JSON
    # ========================================================

    try:

        result = json.loads(content)

    except json.JSONDecodeError as e:

        print("\n========================================")
        print("AGENT RETURNED INVALID JSON")
        print("========================================")
        print(content)
        print("========================================\n")

        raise ValueError(
            f"Agent did not return valid JSON: {e}"
        )


    # ========================================================
    # RETURN DATA TO LANGGRAPH STATE
    # ========================================================

    return {

        "plan": plan,

        "weather_data":
            result.get("weather_data", {}),

        "ocean_data":
            result.get("ocean_data", {}),

        "recommendation":
            result.get("recommendation", {}),

        "status":
            result.get("status", "success")
    }


# ============================================================
# 8. LANGGRAPH WORKFLOW
# ============================================================

graph = StateGraph(MarineState)


# Add nodes
graph.add_node(
    "Planner",
    planner_node
)

graph.add_node(
    "Agent",
    agent_node
)


# ============================================================
# GRAPH EDGES
# ============================================================

graph.add_edge(
    START,
    "Planner"
)

graph.add_edge(
    "Planner",
    "Agent"
)

graph.add_edge(
    "Agent",
    END
)


# ============================================================
# COMPILE GRAPH
# ============================================================

app = graph.compile()
