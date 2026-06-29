from dotenv import load_dotenv
load_dotenv()

from langsmith import Client
from langsmith.evaluation import evaluate

from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT

from src.rag_chain import ask_question

client = Client()

DATASET_NAME = "Production RAG Evaluation"

from src.evaluation.evaluator import (
    correctness,
    conciseness,
    helpfulness,
    groundedness,
    retrieval_relevance,
)


def target(inputs):

    result = ask_question(inputs["question"])

    return {
        "answer": result["answer"],
        "contexts": result["contexts"],
    }

correctness_evaluator = create_llm_as_judge(
    prompt=CORRECTNESS_PROMPT,
    feedback_key="correctness",
    model="openai:gpt-4.1-mini",
)


if __name__ == "__main__":

    results = evaluate(
    target,
    data=DATASET_NAME,
    evaluators=[
        correctness,
        conciseness,
        helpfulness,
        groundedness,
        retrieval_relevance,
    ],
    experiment_prefix="CHUNK_SIZE_800_OVERLAP_170",
)

    print(results)