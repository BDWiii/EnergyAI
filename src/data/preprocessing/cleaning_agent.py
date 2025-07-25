from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from typing import List, Dict, TypedDict, Optional
from pydantic import BaseModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CLEANING_PROMPT = (
    "You are a data cleaning assistant. Given a list of raw data entries, clean the data by: "
    "1. Removing duplicate entries. "
    "2. Removing unrelated or irrelevant data. "
    "3. Removing null values or empty entries. "
    "4. Ensuring the output is human-readable and relevant to the task. "
    "Return only the cleaned data, as a list or block of text, with no explanation."
)


class CleaningState(TypedDict):
    task: str
    node_name: str
    next_node: str
    raw_content: List[str]
    cleaned_content: List[str]
    meta_data: List[Dict]


class CleaningInput(BaseModel):
    raw_content: List[str]
    meta_data: Optional[List[Dict]] = None
    task: Optional[str] = None


class CleaningAgent:
    def __init__(self, llm=None):
        self.llm = llm or ChatOllama()

    def clean_data(self, state: dict) -> dict:
        # Validate input
        validated = CleaningInput(
            raw_content=state.get("raw_content", []),
            meta_data=state.get("meta_data", []),
            task=state.get("task", ""),
        )
        # Prepare prompt
        task_str = f"Task: {validated.task}\n" if validated.task else ""
        content_str = "\n".join(validated.raw_content)
        messages = [
            SystemMessage(content=CLEANING_PROMPT),
            HumanMessage(content=task_str + content_str),
        ]
        response = self.llm.invoke(messages)
        cleaned = [
            line.strip() for line in response.content.splitlines() if line.strip()
        ]
        state["cleaned_content"] = cleaned
        state["node_name"] = "clean_data"
        state["next_node"] = "END"
        return state

    def build_graph(self):
        builder = StateGraph(CleaningState)
        builder.add_node("clean_data", self.clean_data)
        builder.set_entry_point("clean_data")
        builder.add_edge("clean_data", END)
        return builder.compile()

    def run(
        self,
        raw_content: List[str],
        meta_data: Optional[List[Dict]] = None,
        task: Optional[str] = None,
    ):
        graph = self.build_graph()
        state = {
            "task": task or "",
            "node_name": "",
            "next_node": "",
            "raw_content": raw_content,
            "cleaned_content": [],
            "meta_data": meta_data or [],
        }
        return graph.invoke(state)
