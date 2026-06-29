# src/evaluation/dataset.py

import json
from dotenv import load_dotenv

# Load variables from .env
load_dotenv()

from langsmith import Client

client = Client()


def create_dataset():

    dataset_name = "Production RAG Evaluation"

    # Check if dataset already exists
    try:
        dataset = client.read_dataset(dataset_name=dataset_name)
        print(f"Dataset '{dataset_name}' already exists.")

    except Exception:
        dataset = client.create_dataset(
            dataset_name=dataset_name,
            description="Evaluation dataset for Production RAG chatbot."
        )
        print(f"Created dataset '{dataset_name}'.")

    # Load local questions
    with open(
        "src/evaluation/questions.json",
        "r",
        encoding="utf-8",
    ) as f:
        questions = json.load(f)

    # Upload examples
    client.create_examples(
        inputs=[
            {"question": q["question"]}
            for q in questions
        ],
        outputs=[
            {"answer": q["answer"]}
            for q in questions
        ],
        dataset_id=dataset.id,
    )

    print(f"Uploaded {len(questions)} examples.")


if __name__ == "__main__":
    create_dataset()