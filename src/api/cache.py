import redis
import json

from langsmith import traceable


redis_client = redis.Redis(
    host="localhost",
    port=6379,
    decode_responses=True
)


@traceable(name="cache_get")
def get_cached_answer(question: str):

    return redis_client.get(question)


@traceable(name="cache_set")
def cache_answer(
    question: str,
    answer: dict
):

    redis_client.set(
        question,
        json.dumps(answer),
        ex=3600
    )