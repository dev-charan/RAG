import os
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

load_dotenv()

llm = ChatOpenAI(
    openai_api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile",
    openai_api_base="https://api.groq.com/openai/v1",
)

# 1 — State
class State(TypedDict):
    messages: Annotated[list, add_messages]

# 2 — Node
def call_llm(state: State):
    result = llm.invoke(state["messages"])
    return {"messages": [result]}


graph_builder = StateGraph(State)
# creates an empty graph that uses your State structure

graph_builder.add_node("llm", call_llm)
# registers call_llm function as a node named "llm"

graph_builder.add_edge(START, "llm")
# when graph starts → go to "llm" node

graph_builder.add_edge("llm", END)
# after "llm" node → stop

graph = graph_builder.compile()
# locks the graph and makes it ready to run

# 4 — Run
while True:
    user_input = input("You: ")
    if user_input == "exit":
        break
    
    result = graph.invoke({
        "messages": [{"role": "user", "content": user_input}]
    })
    
    print(f"Bot: {result['messages'][-1].content}\n")
