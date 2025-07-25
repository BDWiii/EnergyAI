import os
from datasets import Dataset, load_dataset
from typing import Optional, List
import argparse
import yaml
import ast


# ========================= Base OpenAI schema system-user-assistant ============================
def preprocess_openai_schemaـsft(
    dataset: Dataset,
    system_col: Optional[List[str]],
    system_msg: Optional[str],
    user_col: List[str],
    response_col: List[str],
) -> Dataset:
    """
    Converts a dataset with specified system, user, and response columns to OpenAI chat format.
    """

    def format_example(example):
        messages = []
        if system_col:
            system_input = "\n\n".join(example[col] for col in system_col)
        else:
            system_input = system_msg

        messages.append({"role": "system", "content": system_input.strip()})

        user_input = "\n\n".join(example[col] for col in user_col)
        messages.append({"role": "user", "content": user_input.strip()})

        response_input = "\n\n".join(example[col] for col in response_col)
        messages.append({"role": "assistant", "content": response_input.strip()})
        return {"messages": messages}

    return dataset.map(format_example, remove_columns=dataset.column_names)
