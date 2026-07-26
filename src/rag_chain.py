from dotenv import load_dotenv
from langsmith import traceable
from langchain_openai import ChatOpenAI

from src.retriever import retriever
from src.prompt import RAG_PROMPT

import os

load_dotenv()

print("API KEY:", bool(os.getenv("LANGSMITH_API_KEY")))
print("TRACING:", os.getenv("LANGSMITH_TRACING"))
print("PROJECT:", os.getenv("LANGSMITH_PROJECT"))


llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0
)


@traceable
def ask_question(question: str) -> dict:
    """
    RAG pipeline with parent-child chunking.

    ```
    Flow:
        1. Retrieve parent chunks via parent-child retriever
        2. Format context from parent texts
        3. Send to LLM with RAG prompt
        4. Return answer + sources

    The retriever handles the heavy lifting:
        - Searches child chunks in Pinecone (high recall)
        - Resolves parent_id → fetches parent text (rich context)
        - Deduplicates so LLM doesn't see duplicate context
    ```
    """

    # 1. Retrieve parent chunks (not children!)
    docs = retriever.invoke(question)

    # 2. Format context from parent texts
    contexts = "\n\n---\n\n".join(
        doc.page_content
        for doc in docs
    )

    # 3. Build prompt
    final_prompt = RAG_PROMPT.format(
        context=contexts,
        question=question
    )

    # 4. Generate answer
    response = llm.invoke(final_prompt)

    # 5. Return structured output
    return {
        "question": question,
        "contexts": contexts,
        "answer": response.content,
        "sources": [
            {
                "page": doc.metadata.get("page", "Unknown"),
                "title": doc.metadata.get("title", "Unknown"),
                "parent_id": doc.metadata.get("parent_id", "Unknown"),
            }
            for doc in docs
        ]
    }


# ── Quick test ─────────────────────────────────────────────────
# if __name__ == "__main__":

#     question = "What is Adam optimizer?"

#     result = ask_question(question)

#     print("\n" + "=" * 60)
#     print("QUESTION:")
#     print("=" * 60)
#     print(result["question"])

#     print("\n" + "=" * 60)
#     print("ANSWER:")
#     print("=" * 60)
#     print(result["answer"])

#     print("\n" + "=" * 60)
#     print("SOURCES:")
#     print("=" * 60)
#     for s in result["sources"]:
#         print(f"  Page {s['page']} | {s['title']} | parent_id={s['parent_id']}")

#     print("\n" + "=" * 60)
#     print("CONTEXT LENGTH:")
#     print("=" * 60)
#     print(f"{len(result['contexts'])} characters")