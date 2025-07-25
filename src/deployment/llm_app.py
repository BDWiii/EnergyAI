import os
import asyncio
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs


# Models
class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    stop: Optional[List[str]] = None


class GenerateResponse(BaseModel):
    text: str
    tokens: int


# Global engine
engine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine

    # Startup - Initialize vLLM engine
    model_path = os.getenv("MODEL_PATH", "/app/model")

    engine_args = AsyncEngineArgs(
        model=model_path,
        tensor_parallel_size=int(os.getenv("TENSOR_PARALLEL_SIZE", "1")),
        max_model_len=int(os.getenv("MAX_MODEL_LEN", "4096")),
        gpu_memory_utilization=float(os.getenv("GPU_MEMORY_UTILIZATION", "0.9")),
        trust_remote_code=True,
    )

    engine = AsyncLLMEngine.from_engine_args(engine_args)
    yield

    # Shutdown
    engine = None


app = FastAPI(lifespan=lifespan)


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not ready")

    sampling_params = SamplingParams(
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
        stop=request.stop or [],
    )

    request_id = f"req-{asyncio.current_task().get_name()}"

    async for output in engine.generate(request.prompt, sampling_params, request_id):
        if output.finished:
            generated_text = output.outputs[0].text
            token_count = len(output.outputs[0].token_ids)

            return GenerateResponse(text=generated_text, tokens=token_count)


@app.get("/health")
async def health():
    return {
        "status": "healthy" if engine else "unhealthy",
        "model": os.getenv("MODEL_PATH", "/app/model"),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
