import re
from fastapi import HTTPException


SUSPICIOUS_PATTERNS = [

    r"ignore previous instructions",

    r"forget previous instructions",

    r"reveal system prompt",

    r"show system prompt",

    r"act as chatgpt",

    r"jailbreak",

    r"developer message",

    r"system message"
]


def validate_prompt(question: str):

    text = question.lower()

    for pattern in SUSPICIOUS_PATTERNS:

        if re.search(pattern, text):

            raise HTTPException(
                status_code=400,
                detail="Potential prompt injection detected"
            )

    return question