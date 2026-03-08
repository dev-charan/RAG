import os
import requests
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

load_dotenv()

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant",
)

# Tools
@tool
def calculator(expression: str) -> str:
    """Use this tool for any math calculation. Input should be a math expression like '25 * 48'"""
    return str(eval(expression))

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city. Input should be a city name like 'Mumbai'"""
    geo = requests.get(f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1").json()
    if not geo.get("results"):
        return f"City '{city}' not found."
    lat = geo["results"][0]["latitude"]
    lon = geo["results"][0]["longitude"]
    weather = requests.get(f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true").json()
    current = weather["current_weather"]
    return f"{city}: {current['temperature']}°C, wind speed {current['windspeed']} km/h"

tools = [calculator, get_weather]

# LLM with tools bound
llm_with_tools = llm.bind_tools(tools)

# State
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Nodes
def call_llm(state: State):
    result = llm_with_tools.invoke(state["messages"])
    return {"messages": [result]}

# Conditional edge
def should_use_tool(state: State):
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tool"
    return END

# Build graph
graph_builder = StateGraph(State)

graph_builder.add_node("llm", call_llm)
graph_builder.add_node("tool", ToolNode(tools))

graph_builder.add_edge(START, "llm")
graph_builder.add_conditional_edges("llm", should_use_tool, {"tool": "tool", END: END})
graph_builder.add_edge("tool", "llm")   # ← after tool, loop back to LLM

graph = graph_builder.compile()

# Run
while True:
    user_input = input("You: ")
    if user_input == "exit":
        break

    result = graph.invoke({
        "messages": [{"role": "user", "content": user_input}]
    })

    print(f"Bot: {result['messages'][-1].content}\n")

# ---

# **Visual of what you just built:**
# ```
# START
#   ↓
# [LLM Node]
#   ↓
# Conditional Edge
#   ↙            ↘
# [Tool Node]    END
#   ↓
# [LLM Node]  ← loops back
# ```

# ---

# **Key new things here:**

# | Thing | What it does |
# |---|---|
# | `llm.bind_tools(tools)` | tells LLM what tools are available |
# | `ToolNode(tools)` | built-in node that executes tool calls |
# | `should_use_tool` | conditional edge — tool or end? |
# | `add_edge("tool", "llm")` | after tool runs → loop back to LLM |

# ---

# Run it and test:
# ```
# You: "What is 25 * 48?"
# You: "What is the weather in Mumbai?"
# You: "Hello how are you?"  ← no tool needed, goes straight to END