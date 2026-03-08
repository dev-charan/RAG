import os
import requests

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.tools import tool

load_dotenv()

llm = ChatOpenAI(
    openai_api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile",
    openai_api_base="https://api.groq.com/openai/v1",
)

@tool
def calculator(expression: str) -> str:
    """Use this tool for any math calculation. Input should be a math expression like '25 * 48'"""
    return str(eval(expression))

@tool
def add(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b

@tool
def sub(a: float, b: float) -> float:
    """Subtract b from a."""
    return a - b

@tool
def mul(a: float, b: float) -> float:
    """Multiply two numbers together."""
    return a * b

@tool
def div(a: float, b: float) -> float:
    """Divide a by b. Returns an error if b is zero."""
    if b == 0:
        return "Error: Division by zero"
    return a / b

@tool
def greet(name :str)-> str:
    "Use this to greet a person by name"
    return f"Hello. {name}"


@tool
def get_weather(city : str) -> str:
    "Get current weather for a city. Input should be a city name like 'Mumbai' or 'Londan'"
    geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1"
    geo_response = requests.get(geo_url).json()

    if not geo_response.get("results"):
        return f"City '{city}' not found."
    
    lat = geo_response["results"][0]["latitude"]
    lon = geo_response["results"][0]["longitude"]

    weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
    weather_response = requests.get(weather_url).json()

    current = weather_response["current_weather"]

    return f"{city}: {current['temperature']}°C, wind speed {current['windspeed']} km/h"



    
agent = create_agent(
    model=llm,
    tools=[calculator, add, sub, mul, div, greet, get_weather],
)

inputs = {"messages": [{"role": "user", "content": "What is the weather in Mumbai?"}]}
result = agent.invoke(inputs)
print(result["messages"][-1].content)
