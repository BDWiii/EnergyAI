# instead of using BeautifulSoup for example, we will use two agents for that.

import logging
from langchain_ollama import ChatOllama
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    AIMessage,
    ChatMessage,
    AnyMessage,
)
from langgraph.graph import StateGraph, END
from langgraph.types import Command
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from typing import List, Dict, TypedDict

from Scripts.energyAI.tools import search_web
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prompt for generating search queries
SCRAPER_QUERY_PROMPT = (
    "You are a research assistant. Given a topic, generate a diverse list of 10-20 search queries that would help someone comprehensively research this topic. "
    "Queries should cover different aspects, subtopics, and perspectives. "
    "Return ONLY the list of queries, one per line."
)


class ScrappingState(TypedDict):
    task: str
    node_name: str
    next_node: str
    queries: List[str]
    retrieved_content: List[Dict]
    meta_data: List[Dict]


class Query(BaseModel):
    query: List[str]
    max_results: int = 10


class ScraperAgent:
    def __init__(self, llm=None):
        self.llm = llm or ChatOllama()

    def generate_queries(self, state: dict) -> dict:
        topic = state.get("task") or state.get("topic")
        messages = [
            SystemMessage(content=SCRAPER_QUERY_PROMPT),
            HumanMessage(content=f"Topic: {topic}"),
        ]
        response = self.llm.invoke(messages)
        queries = [q.strip() for q in response.content.split("\n") if q.strip()]
        # Validate queries using Query BaseModel
        validated = Query(query=queries)
        # Update state
        state["queries"] = validated.query
        state["task"] = topic
        state["node_name"] = "generate_queries"
        state["next_node"] = "search_queries"
        return state

    def search_queries(self, state: dict) -> dict:
        queries = state.get("queries", [])
        retrieved_content = []
        meta_data = []
        for q in queries:
            results = search_web(query=q, max_results=3)
            for res in results:
                # Extract all available metadata fields
                meta = {}
                for k in ("title", "url", "score", "favicon", "published_date"):
                    if k in res:
                        meta[k] = res[k]
                meta["query"] = q
                meta_data.append(meta)
                # Store the full content
                if "content" in res:
                    retrieved_content.append({"query": q, "content": res["content"]})
        state["retrieved_content"] = retrieved_content
        state["meta_data"] = meta_data
        state["node_name"] = "search_queries"
        state["next_node"] = "END"
        return state

    def build_graph(self):
        builder = StateGraph(ScrappingState)
        builder.add_node("generate_queries", self.generate_queries)
        builder.add_node("search_queries", self.search_queries)
        builder.add_edge("generate_queries", "search_queries")
        builder.set_entry_point("generate_queries")
        builder.add_edge("search_queries", END)
        return builder.compile()

    def run(self, topic: str):
        graph = self.build_graph()
        # Initial state
        state = {
            "task": topic,
            "node_name": "",
            "next_node": "",
            "queries": [],
            "retrieved_content": [],
            "meta_data": [],
        }
        return graph.invoke(state)
