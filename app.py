
import os
import re
import requests

from bs4 import BeautifulSoup
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from pydantic import BaseModel

from typing import TypedDict

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI


# =========================
# LOAD ENV VARIABLES
# =========================

load_dotenv(dotenv_path=".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

print("OPENAI_API_KEY loaded:", bool(OPENAI_API_KEY))
print("OPENAI_MODEL:", OPENAI_MODEL)


# =========================
# FASTAPI APP
# =========================

app = FastAPI(title="Weather Clothing Advisor")


# =========================
# REQUEST MODEL
# =========================

class WeatherRequest(BaseModel):
    location: str


# =========================
# LANGGRAPH STATE
# =========================

class AgentState(TypedDict):
    location: str
    weather_summary: str
    clothing_advice: str


# =========================
# HELPER FUNCTIONS
# =========================

def clean_text(text):

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def scrape_weather(location):

    query = f"current weather in {location} temperature today"

    url = "https://html.duckduckgo.com/html/"

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    response = requests.post(
        url,
        data={"q": query},
        headers=headers,
        timeout=20
    )

    response.raise_for_status()

    soup = BeautifulSoup(response.text, "lxml")

    snippets = []

    for result in soup.select(".result"):

        title = result.select_one(".result__title")

        snippet = result.select_one(".result__snippet")

        title_text = clean_text(title.get_text(" ")) if title else ""

        snippet_text = clean_text(snippet.get_text(" ")) if snippet else ""

        combined = f"{title_text} {snippet_text}".strip()

        if combined:
            snippets.append(combined)

        if len(snippets) >= 5:
            break

    if not snippets:
        return "No reliable weather information found."

    return "\\n".join(snippets)


# =========================
# LANGGRAPH NODES
# =========================

def weather_node(state):

    location = state["location"]

    weather = scrape_weather(location)

    state["weather_summary"] = weather

    return state


def clothing_node(state):

    if not OPENAI_API_KEY:

        state["clothing_advice"] = (
            "OPENAI_API_KEY missing in .env file."
        )

        return state

    try:

        llm = ChatOpenAI(
            model=OPENAI_MODEL,
            temperature=0.2,
            api_key=OPENAI_API_KEY
        )

        prompt = (
            "You are a practical clothing advisor.\\n\\n"

            f"Location:\\n{state['location']}\\n\\n"

            f"Weather Information:\\n"
            f"{state['weather_summary']}\\n\\n"

            "Suggest practical clothing.\\n"
            "Mention:\\n"
            "- clothes\\n"
            "- shoes\\n"
            "- umbrella if needed\\n"
            "- jacket if needed\\n"
            "- sunscreen if needed\\n\\n"

            "Keep answer concise and useful."
        )

        response = llm.invoke(prompt)

        state["clothing_advice"] = response.content

    except Exception as e:

        state["clothing_advice"] = (
            f"OPENAI ERROR: {str(e)}"
        )

    return state


# =========================
# BUILD LANGGRAPH
# =========================

graph = StateGraph(AgentState)

graph.add_node("weather_node", weather_node)

graph.add_node("clothing_node", clothing_node)

graph.set_entry_point("weather_node")

graph.add_edge("weather_node", "clothing_node")

graph.add_edge("clothing_node", END)

agent = graph.compile()


# =========================
# ROUTES
# =========================

@app.get("/")
def home():

    return FileResponse("index.html")


@app.head("/")
def head_home():

    return {"status": "ok"}


@app.post("/weather-clothing")
def weather_clothing(request: WeatherRequest):

    location = request.location.strip()

    if not location:

        raise HTTPException(
            status_code=400,
            detail="Location is required."
        )

    try:

        result = agent.invoke({

            "location": location,
            "weather_summary": "",
            "clothing_advice": ""
        })

        return {

            "location": result["location"],

            "weather_summary": result["weather_summary"],

            "clothing_advice": result["clothing_advice"]
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
