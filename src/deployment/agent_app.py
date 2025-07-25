from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, Optional
from main import RunMainAgent

app = FastAPI()
runner = RunMainAgent()


class TaskRequest(BaseModel):
    task: str
    pdf_url: str
    thread_id: Optional[str] = None


class AgentResponse(BaseModel):
    scrapping_state: Dict[str, Any]
    pdf_state: Dict[str, Any]
    cleaned_content: Any
    qa_pairs: Any
    final_cleaned: Any
    thread_id: str


@app.post("/run", response_model=AgentResponse)
async def run_agent(request: TaskRequest):
    task = request.task
    pdf_url = request.pdf_url
    thread_id = request.thread_id

    if thread_id:
        runner.thread_id = thread_id
        runner.config = {"configurable": {"thread_id": thread_id}}
        result = runner.existing_thread(task, pdf_url)
    else:
        result = runner.new_thread(task, pdf_url)

    return {
        "scrapping_state": result["scrapping_state"],
        "pdf_state": result["pdf_state"],
        "cleaned_content": result["cleaning1_state"]["cleaned_content"],
        "qa_pairs": result["augmentation_state"]["QA_augment"],
        "final_cleaned": result["cleaning2_state"]["cleaned_content"],
        "thread_id": runner.thread_id,
    }


@app.get("/state/{thread_id}")
async def get_state(thread_id: str):
    try:
        snapshot = runner.get_current_state(thread_id)
        return snapshot
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "healthy"}
