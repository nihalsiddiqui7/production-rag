from ragas.metrics.collections import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)


def get_metrics():
    return [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ]