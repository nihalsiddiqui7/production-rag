from annotated_types import doc
from dotenv import load_dotenv
from langsmith import traceable
from opentelemetry import context
load_dotenv()

from langchain_openai import ChatOpenAI

from src.retriever import retriever
from src.prompt import RAG_PROMPT
from langsmith import traceable

import os

print("API KEY:", bool(os.getenv("LANGSMITH_API_KEY")))
print("TRACING:", os.getenv("LANGSMITH_TRACING"))
print("PROJECT:", os.getenv("LANGSMITH_PROJECT"))


llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0
)

@traceable
def ask_question(question):

    docs = retriever.invoke(question)

    contexts = "\n\n".join(
        doc.page_content
        for doc in docs
    )
    # for i, doc in enumerate(docs, 1):
    #     print(f"\n--- Document {i} ---")
    #     print(doc.page_content[:500])

    final_prompt = RAG_PROMPT.format(
        context=contexts,
        question=question
    )

    response = llm.invoke(final_prompt)

    return {
        "question": question,
        "contexts": contexts,
        "answer": response.content,
        "sources":[
            {
                "page":doc.metadata.get("page", "Unknown"),
                "title":doc.metadata.get("title", "Unknown"),
            }
            for doc in docs
        ]
    }


# if __name__ == "__main__":

#     question = "What is Adam optimizer?"

#     answer = ask_question(question)

#     print("\nQuestion:")
#     print(question)

#     print("\nAnswer:")
#     print(answer)

   

