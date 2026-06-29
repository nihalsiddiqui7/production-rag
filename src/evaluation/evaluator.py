from langsmith.evaluation import run_evaluator

from openevals.llm import create_llm_as_judge
from openevals.prompts import (
    CORRECTNESS_PROMPT,
    CONCISENESS_PROMPT,
    RAG_GROUNDEDNESS_PROMPT,
    RAG_HELPFULNESS_PROMPT,
    RAG_RETRIEVAL_RELEVANCE_PROMPT,
)

MODEL = "openai:gpt-4.1-mini"

_correctness = create_llm_as_judge(
    prompt=CORRECTNESS_PROMPT,
    feedback_key="correctness",
    continuous=True,
    model=MODEL,
)

_conciseness = create_llm_as_judge(
    prompt=CONCISENESS_PROMPT,
    feedback_key="conciseness",
    continuous=True,
    model=MODEL,
)

_groundedness = create_llm_as_judge(
    prompt=RAG_GROUNDEDNESS_PROMPT,
    feedback_key="groundedness",
    continuous=True,
    model=MODEL,
)

_helpfulness = create_llm_as_judge(
    prompt=RAG_HELPFULNESS_PROMPT,
    feedback_key="helpfulness",
    continuous=True,
    model=MODEL,
)

_retrieval = create_llm_as_judge(
    prompt=RAG_RETRIEVAL_RELEVANCE_PROMPT,
    feedback_key="retrieval_relevance",
    continuous=True,
    model=MODEL,
)


@run_evaluator
def correctness(run, example):
    return _correctness(
        inputs=run.inputs,
        outputs={"answer": run.outputs["answer"]},
        reference_outputs=example.outputs,
    )


@run_evaluator
def conciseness(run, example):
    return _conciseness(
        inputs=run.inputs,
        outputs={"answer": run.outputs["answer"]},
    )


@run_evaluator
def helpfulness(run, example):
    return _helpfulness(
        inputs=run.inputs,
        outputs={"answer": run.outputs["answer"]},
    )


@run_evaluator
def groundedness(run, example):
    return _groundedness(
        context=run.outputs["contexts"],
        outputs={"answer": run.outputs["answer"]},
    )


@run_evaluator
def retrieval_relevance(run, example):
    return _retrieval(
        inputs=run.inputs,
        context=run.outputs["contexts"],
    )