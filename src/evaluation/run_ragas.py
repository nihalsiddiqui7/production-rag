import json
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.llms import llm_factory
# NOTE: ragas 0.4.x `evaluate()` validates against the classic `Metric`
# classes, so we use the same internal metric instances it uses internally.
from ragas.metrics._answer_relevance import answer_relevancy
from ragas.metrics._context_precision import context_precision
from ragas.metrics._context_recall import context_recall
from ragas.metrics._faithfulness import faithfulness

from src.rag_chain import ask_question

QUESTIONS_PATH = Path("src/evaluation/questions.json")
REPORTS_DIR = Path("reports")
CONTEXT_SEPARATOR = "\n\n---\n\n"

METRIC_COLUMNS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
]

# Reference answers that expect a "not in the documents" refusal. These
# questions validate refusal behavior but cannot score on context recall or
# answer relevancy (there is nothing relevant to retrieve).
OUT_OF_DOMAIN_REFUSAL = {
    "The information is not available in the provided documents.",
    "The information is not available in the documents.",
    "The assistant should refuse to use external knowledge and answer only from the provided documents.",
    "The assistant should rely only on retrieved documents and avoid unsupported claims.",
    "The assistant should state that the required information is unavailable in the retrieved context.",
    "If CNNs are not present in the retrieved documents, the assistant should state that the information is unavailable.",
}


def load_questions() -> list[dict]:
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_samples(questions: list[dict]) -> list[SingleTurnSample]:
    samples = []
    for q in questions:
        question = q["question"]
        result = ask_question(question)
        samples.append(
            SingleTurnSample(
                user_input=question,
                retrieved_contexts=result["contexts"].split(CONTEXT_SEPARATOR),
                response=result["answer"],
                reference=q["answer"],
            )
        )
    return samples


def print_scores(df, title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    for metric in METRIC_COLUMNS:
        print(f"  {metric:>20}: {df[metric].mean():.3f}")


def main() -> None:
    evaluator_llm = llm_factory(
        model="gpt-4.1-mini",
        client=OpenAI(),
        temperature=0,
        max_tokens=4000,
    )
    evaluator_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    questions = load_questions()
    samples = build_samples(questions)

    dataset = EvaluationDataset(samples=samples)

    result = evaluate(
        dataset=dataset,
        metrics=[
            answer_relevancy,
            context_precision,
            faithfulness,
            context_recall,
        ],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )

    df = result.to_pandas()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = sys.argv[1] if len(sys.argv) > 1 else "eval"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = REPORTS_DIR / f"ragas_report_{tag}_{timestamp}.csv"
    df.to_csv(csv_path, index=False)

    print_scores(df, "AVERAGE SCORES (all questions)")

    in_domain = [q for q in questions if q["answer"] not in OUT_OF_DOMAIN_REFUSAL]
    in_domain_ids = {q["question"] for q in in_domain}
    in_domain_df = df[df["user_input"].isin(in_domain_ids)]
    if not in_domain_df.empty:
        print_scores(in_domain_df, "AVERAGE SCORES (in-domain only)")

    print(f"\nPer-question report saved to: {csv_path}")

    print("\nLowest 5 faithfulness (potential hallucination):")
    print(
        df.nsmallest(5, "faithfulness")[["user_input", "faithfulness"]].to_string(
            index=False
        )
    )

    print("\nLowest 5 context_recall (retrieval misses):")
    print(
        df.nsmallest(5, "context_recall")[["user_input", "context_recall"]].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
