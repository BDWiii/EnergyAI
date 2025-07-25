import logging
import os
from pipelines.data_pipeline import MainAgent
from src.training.dataset_augmentation import DataAugmentationAgent
from src.training.data_formatting import preprocess_openai_schemaـsft
from datasets import Dataset, DatasetDict
import yaml
import argparse
from pathlib import Path
import subprocess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === CONFIGURATION ===
DEFAULT_TOPIC = os.environ.get("TRAIN_TOPIC", "Smart Grid optimisation with AI")
DEFAULT_PDF_URL = os.environ.get(
    "TRAIN_PDF_URL", "https://arxiv.org/pdf/2106.04554.pdf"
)
FORMATTED_DATA_PATH = os.environ.get("FORMATTED_DATA_PATH", "./formatted_data")
FINE_TUNE_CONFIG = os.environ.get("FINE_TUNE_CONFIG", "./finetune_config.yaml")


def pipe():
    """
    Training pipeline:
    1. Generate and clean data using MainAgent.
    2. Generate QA pairs using DataAugmentationAgent.
    3. Format the data for training (OpenAI schema).
    4. Save the formatted dataset.
    5. Call fine-tuning logic to train a model on the formatted data.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--topic", type=str, default=DEFAULT_TOPIC, help="Topic for data generation"
    )
    parser.add_argument(
        "--pdf_url",
        type=str,
        default=DEFAULT_PDF_URL,
        help="PDF URL for data generation",
    )
    parser.add_argument(
        "--formatted_data_path",
        type=str,
        default=FORMATTED_DATA_PATH,
        help="Path to save formatted data",
    )
    parser.add_argument(
        "--finetune_config",
        type=str,
        default=FINE_TUNE_CONFIG,
        help="Path to fine-tune config YAML",
    )
    args = parser.parse_args()

    logger.info(f"Generating and cleaning data for topic: {args.topic}")
    main_agent = MainAgent()
    main_state = main_agent.run(args.topic, args.pdf_url)

    # Use cleaned content from the last cleaning step
    cleaned_content = main_state["cleaning2_state"].get("cleaned_content", [])
    if not cleaned_content:
        logger.error("No cleaned content available for augmentation. Exiting.")
        return

    logger.info("Generating QA pairs using DataAugmentationAgent...")
    augmentation_agent = DataAugmentationAgent()
    aug_state = augmentation_agent.run(cleaned_content, meta_data=None, task=args.topic)
    qa_pairs = aug_state.get("QA_augment", [])
    if not qa_pairs:
        logger.error("No QA pairs generated. Exiting.")
        return

    # Convert QA pairs to HuggingFace Dataset
    logger.info("Formatting data for training (OpenAI schema)...")
    questions = [
        qa["question"] for qa in qa_pairs if "question" in qa and "answer" in qa
    ]
    answers = [qa["answer"] for qa in qa_pairs if "question" in qa and "answer" in qa]
    data_dict = {"user": questions, "assistant": answers}
    dataset = Dataset.from_dict(data_dict)

    # Format using OpenAI schema
    formatted_dataset = preprocess_openai_schemaـsft(
        dataset=dataset,
        system_col=None,
        system_msg="You are a helpful assistant.",
        user_col=["user"],
        response_col=["assistant"],
    )

    # Save formatted dataset
    formatted_data_path = Path(args.formatted_data_path)
    formatted_data_path.mkdir(parents=True, exist_ok=True)
    formatted_dataset.save_to_disk(str(formatted_data_path))
    logger.info(f"Formatted dataset saved to {formatted_data_path}")

    # Call fine-tuning logic (assume fine-tune.py is in src/training/)
    logger.info("Starting fine-tuning...")
    finetune_script = Path(__file__).parent.parent / "src" / "training" / "fine-tune.py"
    cmd = ["python", str(finetune_script), "--config", args.finetune_config]
    logger.info(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    logger.info("Fine-tuning complete. Check output directory specified in config.")


if __name__ == "__main__":
    pipe()
