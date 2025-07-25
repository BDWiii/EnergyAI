# Here we will simply utilize the previous agents we made to generate the benchmarking dataset.
# from Query and search agents we gather the related data, then through QA augmentation agent we will generate our own data.

from src.data.collecting.scrapping_agent import ScraperAgent, ScrappingState
from src.training.dataset_augmentation import DataAugmentationAgent, DataAugState
from langgraph.graph import StateGraph, END
from typing import Dict, Any, TypedDict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BenchmarkState(TypedDict):
    task: str
    node_name: str
    next_node: str
    scrapping_state: ScrappingState  # Nested States
    augmentation_state: DataAugState  # Nested States


class BenchmarkAgent:
    """
    Orchestrates ScraperAgent and DataAugmentationAgent to generate domain-specific QA datasets.
    The state contains the full states of both sub-agents for traceability and analysis.
    """

    def __init__(self, llm=None, scraper_agent=None, augmentation_agent=None):
        # LLM placeholder for future sharing
        self.llm = llm
        self.scraper_agent = scraper_agent or ScraperAgent(llm=llm)
        self.augmentation_agent = augmentation_agent or DataAugmentationAgent(llm=llm)

    def run_scraper(self, state: dict) -> dict:
        topic = state.get("task")
        scrapping_state = self.scraper_agent.run(topic)
        state["scrapping_state"] = scrapping_state
        state["node_name"] = "run_scraper"
        state["next_node"] = "run_augmentation"
        return state

    def run_augmentation(self, state: dict) -> dict:
        scrapping_state = state.get("scrapping_state", {})
        retrieved_content = scrapping_state.get("retrieved_content", [])
        meta_data = scrapping_state.get("meta_data", [])
        augmentation_state = self.augmentation_agent.run(
            retrieved_content, meta_data=meta_data, task=state.get("task")
        )
        state["augmentation_state"] = augmentation_state
        state["node_name"] = "run_augmentation"
        state["next_node"] = "END"
        return state

    def build_graph(self):
        builder = StateGraph(BenchmarkState)
        builder.add_node("run_scraper", self.run_scraper)
        builder.add_node("run_augmentation", self.run_augmentation)
        builder.set_entry_point("run_scraper")
        builder.add_edge("run_scraper", "run_augmentation")
        builder.add_edge("run_augmentation", END)
        return builder.compile()

    def run(self, topic: str):
        graph = self.build_graph()
        state = {
            "task": topic,
            "node_name": "",
            "next_node": "",
            "scrapping_state": {},
            "augmentation_state": {},
        }
        return graph.invoke(state)
