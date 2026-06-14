from dotenv import load_dotenv
load_dotenv()

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore


INDEX_NAME = "ml-chatbot"

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = PineconeVectorStore(
    index_name=INDEX_NAME,
    embedding=embedding_model
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5}
)