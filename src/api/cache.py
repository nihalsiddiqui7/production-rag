import redis
import json
import os
from dotenv import load_dotenv

load_dotenv()

from langsmith import traceable


import os

print("REDIS_HOST =", os.getenv("REDIS_HOST"))

redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
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