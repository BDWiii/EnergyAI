import logging
from src.evaluation.benchmark_generator import BenchmarkAgent
from src.evaluation.eval_metrics import compare_llms
from vllm import LLM
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === CONFIGURATION ===
BASELINE_MODEL_PATH = os.environ.get("BASELINE_MODEL_PATH", "path/to/baseline/model")
CANDIDATE_MODEL_PATH = os.environ.get("CANDIDATE_MODEL_PATH", "path/to/candidate/model")
TOPIC = os.environ.get("EVAL_TOPIC", "Smart Grid optimisation with AI")


def main():
    """
    Evaluation pipeline:
    1. Generate QA dataset using BenchmarkAgent.
    2. Load two LLMs (baseline and candidate) with vLLM.
    3. Evaluate both models using compare_llms on the generated QA pairs.
    4. Print a clean comparison report.
    """
    logger.info(f"Generating QA dataset for topic: {TOPIC}")
    benchmark_agent = BenchmarkAgent()
    benchmark_state = benchmark_agent.run(TOPIC)
    qa_pairs = benchmark_state["augmentation_state"].get("QA_augment", [])
    if not qa_pairs:
        logger.error("No QA pairs generated. Exiting.")
        return
    prompts = [qa["question"] for qa in qa_pairs if "question" in qa and "answer" in qa]
    references = [
        qa["answer"] for qa in qa_pairs if "question" in qa and "answer" in qa
    ]
    if not prompts or not references or len(prompts) != len(references):
        logger.error("Mismatch or missing questions/answers in QA pairs. Exiting.")
        return
    logger.info(f"Loading baseline model from {BASELINE_MODEL_PATH}")
    model1 = LLM(model=BASELINE_MODEL_PATH)
    logger.info(f"Loading candidate model from {CANDIDATE_MODEL_PATH}")
    model2 = LLM(model=CANDIDATE_MODEL_PATH)
    logger.info("Running evaluation...")
    report = compare_llms(model1, model2, prompts, references, "Baseline", "Candidate")
    print(report)


if __name__ == "__main__":
    main()
