from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI

from retriever import retriever
from prompt import RAG_PROMPT


llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0
)


def ask_question(question):

    docs = retriever.invoke(question)

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )
    for i, doc in enumerate(docs, 1):
        print(f"\n--- Document {i} ---")
        print(doc.page_content[:500])

    final_prompt = RAG_PROMPT.format(
        context=context,
        question=question
    )

    response = llm.invoke(final_prompt)

    return response.content


if __name__ == "__main__":

    question = "What is Adam optimizer?"

    answer = ask_question(question)

    print("\nQuestion:")
    print(question)

    print("\nAnswer:")
    print(answer)

   

