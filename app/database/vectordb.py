import chromadb

class ChromaVectorDB:
    def __init__(self):
        self.client = chromadb.PersistentClient(
            path="./chroma_db"
        )

        self.collection = self.client.get_or_create_collection(name="ml-book")

    def add_documents(self,ids,texts,embeddings,metadatas):
        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=metadatas
        )
    def search(self,query_embedding,top_k=10):
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
        
        )

        return results