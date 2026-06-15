from fastapi import APIRouter
from src.api.schemas import QueryRequest, QueryResponse
from src.rag_chain import ask_question
from fastapi import HTTPException
from src.api.logger import logger
from src.api.security import validate_prompt
from src.api.pii import anonymize_pii
from fastapi import Request
from src.api.rate_limiter import limiter
import json
from src.api.cache import (
    get_cached_answer,
    cache_answer
)



router = APIRouter()

@router.post("/ask", 
            response_model=QueryResponse
    )

@limiter.limit("10/minute")

@router.get("/health")
def health():
    return {
        "status": "healthy"
    }



def ask(request: Request, payload: QueryRequest):
    try:
        logger.info(f"Received question: {payload.question}")
        validated_question = validate_prompt(payload.question)
        sanitized_question = anonymize_pii(validated_question)
        logger.info(f"Sanitized question: {sanitized_question}")

        cached_result = get_cached_answer(sanitized_question)

        if cached_result:
            logger.info("Cache hit - returning cached answer")
            return QueryResponse(**json.loads(cached_result))
        
        logger.info("Cache miss - generating answer")
        result = ask_question(sanitized_question)

        cache_answer(sanitized_question, result)

        logger.info("Generated answer successfully")


        return QueryResponse(answer=result["answer"], sources=result["sources"])
    


    except Exception as e:
        logger.error(f"Error occurred while processing question: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
