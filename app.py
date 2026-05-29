
import os
import re
import requests

from bs4 import BeautifulSoup
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.responses import HTMLResponse

from pydantic import BaseModel

from typing import TypedDict

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI


load_dotenv()


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


app = FastAPI(title="Weather Clothing Advisor")


class WeatherRequest(BaseModel):
    location: str


class AgentState(TypedDict):
    location: str
    weather_summary: str
    clothing_advice: str


def clean_text(text):

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def scrape_weather(location):

    query = f"current weather in {location}"

    url = "https://html.duckduckgo.com/html/"

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    response = requests.post(
        url,
        data={"q": query},
        headers=headers,
        timeout=15
    )

    soup = BeautifulSoup(response.text, "lxml")

    snippets = []

    for result in soup.select(".result"):

        snippet = result.select_one(".result__snippet")

        if snippet:

            snippets.append(
                clean_text(snippet.get_text())
            )

        if len(snippets) >= 5:
            break

    if not snippets:

        return "Weather information not found."

    return "\\n".join(snippets)


def weather_node(state):

    location = state["location"]

    weather = scrape_weather(location)

    state["weather_summary"] = weather

    return state


def clothing_node(state):

    llm = ChatOpenAI(
        model=OPENAI_MODEL,
        temperature=0.2,
        api_key=OPENAI_API_KEY
    )

    prompt = f"""

You are a clothing advisor.

Location:
{state["location"]}

Weather Information:
{state["weather_summary"]}

Suggest:
- clothes
- shoes
- umbrella if needed
- jacket if needed
- sunscreen if needed

Keep answer practical.
"""

    response = llm.invoke(prompt)

    state["clothing_advice"] = response.content

    return state


graph = StateGraph(AgentState)

graph.add_node("weather_node", weather_node)

graph.add_node("clothing_node", clothing_node)

graph.set_entry_point("weather_node")

graph.add_edge("weather_node", "clothing_node")

graph.add_edge("clothing_node", END)

agent = graph.compile()


@app.get("/")
def home():

    return FileResponse("index.html").read()


@app.post("/weather-clothing")
def weather_clothing(request: WeatherRequest):

    try:

        result = agent.invoke({

            "location": request.location,
            "weather_summary": "",
            "clothing_advice": ""
        })

        return result

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
