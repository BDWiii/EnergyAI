from src.data.collecting.scrapping_agent import ScraperAgent, ScrappingState
from src.data.collecting.pdf_agent import PDFParserAgent, PDFParserState
from src.data.preprocessing.cleaning_agent import CleaningAgent, CleaningState
from src.training.dataset_augmentation import DataAugmentationAgent, DataAugState
from langgraph.graph import StateGraph, END
from typing import TypedDict, Dict, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MainState(TypedDict):
    task: str
    pdf_url: str
    node_name: str
    next_node: str
    scrapping_state: ScrappingState
    pdf_state: PDFParserState
    cleaning1_state: CleaningState
    augmentation_state: DataAugState
    cleaning2_state: CleaningState


class MainAgent:
    """
    Orchestrates the full pipeline: ScraperAgent -> PDFParserAgent -> CleaningAgent -> DataAugmentationAgent -> CleaningAgent
    Each sub-agent's state is nested in the main state for full traceability.
    """

    def __init__(self, llm=None):
        self.llm = llm
        self.scraper_agent = ScraperAgent(llm=llm)
        self.pdf_agent = PDFParserAgent(llm=llm)
        self.cleaning_agent = CleaningAgent(llm=llm)
        self.augmentation_agent = DataAugmentationAgent(llm=llm)

    def run_scraper(self, state: dict) -> dict:
        topic = state.get("task")
        scrapping_state = self.scraper_agent.run(topic)
        state["scrapping_state"] = scrapping_state
        state["node_name"] = "run_scraper"
        state["next_node"] = "run_pdf"
        return state

    def run_pdf(self, state: dict) -> dict:
        pdf_url = state.get("pdf_url")
        pdf_state = self.pdf_agent.run(pdf_url)
        state["pdf_state"] = pdf_state
        state["node_name"] = "run_pdf"
        state["next_node"] = "run_cleaning1"
        return state

    def run_cleaning1(self, state: dict) -> dict:
        # Combine all retrieved content for cleaning
        scrapping_content = state["scrapping_state"].get("retrieved_content", [])
        pdf_content = state["pdf_state"].get("retrieved_content", [])
        raw_content = []
        # Flatten if needed
        for item in scrapping_content:
            if isinstance(item, dict) and "content" in item:
                raw_content.append(item["content"])
            elif isinstance(item, str):
                raw_content.append(item)
        for item in pdf_content:
            if isinstance(item, str):
                raw_content.append(item)
        cleaning1_state = self.cleaning_agent.run(
            raw_content, meta_data=None, task=state.get("task")
        )
        state["cleaning1_state"] = cleaning1_state
        state["node_name"] = "run_cleaning1"
        state["next_node"] = "run_augmentation"
        return state

    def run_augmentation(self, state: dict) -> dict:
        # Use cleaned content for augmentation
        cleaned_content = state["cleaning1_state"].get("cleaned_content", [])
        augmentation_state = self.augmentation_agent.run(
            cleaned_content, meta_data=None, task=state.get("task")
        )
        state["augmentation_state"] = augmentation_state
        state["node_name"] = "run_augmentation"
        state["next_node"] = "run_cleaning2"
        return state

    def run_cleaning2(self, state: dict) -> dict:
        # Clean the QA pairs (as text)
        qa_pairs = state["augmentation_state"].get("QA_augment", [])
        # Convert QA dicts to strings for cleaning
        raw_content = [str(qa) for qa in qa_pairs]
        cleaning2_state = self.cleaning_agent.run(
            raw_content, meta_data=None, task=state.get("task")
        )
        state["cleaning2_state"] = cleaning2_state
        state["node_name"] = "run_cleaning2"
        state["next_node"] = "END"
        return state

    def build_graph(self):
        builder = StateGraph(MainState)
        builder.add_node("run_scraper", self.run_scraper)
        builder.add_node("run_pdf", self.run_pdf)
        builder.add_node("run_cleaning1", self.run_cleaning1)
        builder.add_node("run_augmentation", self.run_augmentation)
        builder.add_node("run_cleaning2", self.run_cleaning2)
        builder.set_entry_point("run_scraper")
        builder.add_edge("run_scraper", "run_pdf")
        builder.add_edge("run_pdf", "run_cleaning1")
        builder.add_edge("run_cleaning1", "run_augmentation")
        builder.add_edge("run_augmentation", "run_cleaning2")
        builder.add_edge("run_cleaning2", END)
        return builder.compile()

    def run(self, topic: str, pdf_url: str):
        graph = self.build_graph()
        state = {
            "task": topic,
            "pdf_url": pdf_url,
            "node_name": "",
            "next_node": "",
            "scrapping_state": {},
            "pdf_state": {},
            "cleaning1_state": {},
            "augmentation_state": {},
            "cleaning2_state": {},
        }
        return graph.invoke(state)
