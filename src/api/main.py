from fastapi import FastAPI
from src.api.routes import router
from src.api.rate_limiter import limiter
from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler

app = FastAPI(
    title="RAG API",
    description="An API for a Retrieval-Augmented Generative (RAG) system",
    version="1.0.0"
)
app.state.limiter = limiter
app.add_exception_handler(
    RateLimitExceeded,
    _rate_limit_exceeded_handler
)

app.include_router(router)


