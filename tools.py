#!/usr/bin/env python3
import os
from dotenv import load_dotenv
from typing import Optional, Dict, List, Literal
from langchain.tools import tool
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
import arxiv
from pydantic import BaseModel, Field
import json
import requests

# from pypdf import PdfReader
import uuid
import fitz
import logging

from pypdf import PdfReader
from pdf2image import convert_from_path
import pytesseract


logger = logging.getLogger(__name__)


# ====================== Web Search Tool =======================
class SearchInput(BaseModel):
    query: str = Field(..., description="The search query.")
    max_results: Optional[int] = Field(3, description="Number of results to return.")
    include_raw_content: Optional[bool] = Field(
        False, description="Flag to return more content"
    )  # only works for paid plans.


@tool(args_schema=SearchInput)
def search_web(
    query: str, max_results: int = 3, include_raw_content: bool = False
) -> List[Dict]:
    """
    Search the web for real time information and return the top results.
    """
    try:
        load_dotenv()
        search = TavilySearchAPIWrapper(tavily_api_key=os.getenv("TAVILY_API_KEY"))
        results = search.results(
            query=query,
            max_results=max_results,
            include_raw_content=include_raw_content,
        )
        return results
    except Exception as e:
        return [{"ERROR": str(e)}]


# ===================== ArXiv Search Tool ======================
class ArxivSearchInput(BaseModel):
    query: str = Field(..., description="The research paper to search for in arxiv.")
    max_results: Optional[int] = Field(1, description="Number of PDF papers to return.")
    sort_by: Literal["relevance", "lastUpdatedDate", "submittedDate"] = Field(
        "relevance", description="The sorting criteria."
    )


# PDF tool
class LoadPDFInput(BaseModel):
    url: str = Field(..., description="The URL of the PDF to load.")


def _normalize_pdf_url(url: str) -> str:
    # If arxiv.org/abs/XXXX, convert to arxiv.org/pdf/XXXX.pdf
    if "arxiv.org/abs/" in url:
        paper_id = url.rsplit("/", 1)[-1]
        return f"https://arxiv.org/pdf/{paper_id}.pdf"
    # Add more normalization rules here if needed in the future
    return url


@tool(args_schema=LoadPDFInput, return_direct=True)
def load_pdf(url: str) -> str:
    """
    Downloads a PDF from `url`, extracts all text using pypdf (PdfReader).
    If no text is found (scanned PDF), falls back to OCR using pdf2image + pytesseract.
    Returns the concatenated text. Cleans up temp file on exit.
    """
    temp_filename = f"/tmp/{uuid.uuid4().hex}.pdf"
    url = _normalize_pdf_url(url)
    try:
        # Download PDF
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        with open(temp_filename, "wb") as f:
            f.write(resp.content)

        # Try text extraction with pypdf
        try:
            reader = PdfReader(temp_filename)
            all_text = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    all_text.append(text.strip())
            if all_text:
                return "\n\n".join(all_text)
        except Exception as e:
            logger.warning(f"pypdf failed: {e}")

        # Fallback: OCR with pdf2image + pytesseract
        try:
            images = convert_from_path(temp_filename)
            ocr_text = []
            for img in images:
                text = pytesseract.image_to_string(img)
                if text:
                    ocr_text.append(text.strip())
            if ocr_text:
                return "\n\n".join(ocr_text)
        except Exception as e:
            logger.error(f"OCR fallback failed: {e}")

        return ""  # No text found

    except Exception as e:
        logger.error(f"Failed to load or parse PDF at {url}: {e}")
        return ""

    finally:
        try:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
        except OSError as cleanup_err:
            logger.warning(f"Could not delete temp file {temp_filename}: {cleanup_err}")
