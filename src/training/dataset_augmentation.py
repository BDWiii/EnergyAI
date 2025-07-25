from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from typing import List, Dict, TypedDict, Optional, Any
from pydantic import BaseModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

QA_AUGMENT_PROMPT = (
    "You are a data augmentation assistant. Given a set of retrieved content (as a list of dictionaries or JSON), generate a diverse set of question-answer (QA) pairs that are relevant, factual, and useful for training QA models. "
    "Each QA pair should be a dictionary with 'question' and 'answer' fields. "
    "Do not invent facts; use only the provided content. Return the QA pairs as a JSON list."
)


class DataAugState(TypedDict):
    task: str
    node_name: str
    next_node: str
    retrieved_content: List[Any]
    QA_augment: List[Dict]
    meta_data: List[Dict]


class DataAugInput(BaseModel):
    retrieved_content: List[Any]
    meta_data: Optional[List[Dict]] = None
    task: Optional[str] = None


class DataAugmentationAgent:
    def __init__(self, llm=None):
        self.llm = llm or ChatOllama()

    def generate_qa(self, state: dict) -> dict:
        validated = DataAugInput(
            retrieved_content=state.get("retrieved_content", []),
            meta_data=state.get("meta_data", []),
            task=state.get("task", ""),
        )
        # Prepare prompt
        task_str = f"Task: {validated.task}\n" if validated.task else ""
        content_str = "\n".join([str(item) for item in validated.retrieved_content])
        messages = [
            SystemMessage(content=QA_AUGMENT_PROMPT),
            HumanMessage(content=task_str + content_str),
        ]
        response = self.llm.invoke(messages)
        # Try to parse as JSON, fallback to list of dicts
        import json

        try:
            qa_pairs = json.loads(response.content)
            if isinstance(qa_pairs, dict):
                qa_pairs = [qa_pairs]
        except Exception:
            # Fallback: try to parse line by line
            qa_pairs = []
            for line in response.content.splitlines():
                try:
                    qa = json.loads(line)
                    if isinstance(qa, dict):
                        qa_pairs.append(qa)
                except Exception:
                    continue
        state["QA_augment"] = qa_pairs
        state["node_name"] = "generate_qa"
        state["next_node"] = "END"
        return state

    def build_graph(self):
        builder = StateGraph(DataAugState)
        builder.add_node("generate_qa", self.generate_qa)
        builder.set_entry_point("generate_qa")
        builder.add_edge("generate_qa", END)
        return builder.compile()

    def run(
        self,
        retrieved_content: List[Any],
        meta_data: Optional[List[Dict]] = None,
        task: Optional[str] = None,
    ):
        graph = self.build_graph()
        state = {
            "task": task or "",
            "node_name": "",
            "next_node": "",
            "retrieved_content": retrieved_content,
            "QA_augment": [],
            "meta_data": meta_data or [],
        }
        return graph.invoke(state)
