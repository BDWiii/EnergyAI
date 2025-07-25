import evaluate
from typing import List
from tabulate import tabulate
from vllm import LLM, SamplingParams
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_predictions(
    llm: LLM, prompts: List[str], sampling_params: SamplingParams = None
) -> List[str]:
    """
    Generate predictions from a vLLM model given a list of prompts.
    """
    if sampling_params is None:
        sampling_params = SamplingParams(temperature=0.0, max_tokens=256)
    outputs = llm.generate(prompts, sampling_params)
    # vLLM returns a list of objects with .outputs[0].text
    return [out.outputs[0].text.strip() for out in outputs]


def evaluate_text(predictions: List[str], references: List[str]) -> dict:
    """
    Evaluate predictions against references using ROUGE-L and BLEU metrics.
    Returns a dict of scores.
    """
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have the same length.")
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")
    results = {}
    rouge_result = rouge.compute(
        predictions=predictions, references=references, rouge_types=["rougeL"]
    )
    results["ROUGE-L"] = rouge_result["rougeL"]
    bleu_result = bleu.compute(
        predictions=predictions, references=[[ref] for ref in references]
    )
    results["BLEU"] = bleu_result["bleu"]
    return results


def compare_llms(
    model1: LLM,
    model2: LLM,
    prompts: List[str],
    references: List[str],
    model1_name: str = "Baseline",
    model2_name: str = "Candidate",
    sampling_params: SamplingParams = None,
) -> str:
    """
    Generate predictions from two LLMs and compare their ROUGE-L and BLEU scores.
    Returns a human-readable comparison table.
    """
    logger.info(f"Generating predictions for {model1_name}...")
    preds1 = generate_predictions(model1, prompts, sampling_params)
    logger.info(f"Generating predictions for {model2_name}...")
    preds2 = generate_predictions(model2, prompts, sampling_params)
    logger.info(f"Evaluating {model1_name}...")
    scores1 = evaluate_text(preds1, references)
    logger.info(f"Evaluating {model2_name}...")
    scores2 = evaluate_text(preds2, references)
    # Prepare comparison table
    table = [
        [model1_name, f"{scores1['ROUGE-L']:.4f}", f"{scores1['BLEU']:.4f}"],
        [model2_name, f"{scores2['ROUGE-L']:.4f}", f"{scores2['BLEU']:.4f}"],
    ]
    report = tabulate(table, headers=["Model", "ROUGE-L", "BLEU"], tablefmt="github")
    return f"\nLLM Comparison Results:\n{report}\n"


# Example usage (uncomment and set model paths to test)
# model1 = LLM(model="path/to/baseline/model")
# model2 = LLM(model="path/to/candidate/model")
# prompts = ["What is the capital of France?", "Who wrote 1984?"]
# references = ["Paris", "George Orwell"]
# print(compare_llms(model1, model2, prompts, references, "Baseline", "Candidate"))
