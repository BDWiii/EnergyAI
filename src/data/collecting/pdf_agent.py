from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from typing import List, Dict, TypedDict
from Scripts.energyAI.tools import load_pdf
from pydantic import BaseModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PDF_CLEAN_PROMPT = (
    "You are a helpful assistant. Given the raw extracted text from a PDF, clean it up to be as readable as possible for a human. "
    "Remove any artifacts, excessive whitespace, and make sure the text is well-formatted and easy to read. "
    "Do not summarize, just clean and format."
)


class PDFParserState(TypedDict):
    task: str
    node_name: str
    next_node: str
    retrieved_content: List[str]
    meta_data: List[Dict]


class PDFInput(BaseModel):
    url: str


class PDFParserAgent:
    def __init__(self, llm=None):
        self.llm = llm or ChatOllama()

    def parse_pdf(self, state: dict) -> dict:
        url = state.get("task") or state.get("url")
        validated = PDFInput(url=url)
        raw_content = load_pdf(validated.url)
        # Clean with LLM
        messages = [
            SystemMessage(content=PDF_CLEAN_PROMPT),
            HumanMessage(content=raw_content),
        ]
        response = self.llm.invoke(messages)
        cleaned = response.content.strip()
        meta = {"url": validated.url}
        state["retrieved_content"] = [cleaned]
        state["meta_data"] = [meta]
        state["node_name"] = "parse_pdf"
        state["next_node"] = "END"
        return state

    def build_graph(self):
        builder = StateGraph(PDFParserState)
        builder.add_node("parse_pdf", self.parse_pdf)
        builder.set_entry_point("parse_pdf")
        builder.add_edge("parse_pdf", END)
        return builder.compile()

    def run(self, url: str):
        graph = self.build_graph()
        state = {
            "task": url,
            "node_name": "",
            "next_node": "",
            "retrieved_content": [],
            "meta_data": [],
        }
        return graph.invoke(state)
