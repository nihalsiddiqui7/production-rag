from src.api.cache import (
    cache_answer,
    get_cached_answer
)

cache_answer(
    "RMSE",
    {
        "answer": "Root Mean Squared Error"
    }
)

print(
    get_cached_answer("RMSE")
)