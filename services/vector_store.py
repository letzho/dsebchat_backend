from langchain.vectorstores import Chroma
from langchain.embeddings.base import Embeddings
from langchain.docstore.document import Document
from typing import List
import os

class ChromaManager:
    def __init__(self, collection_name: str):
        self.persist_directory = "vector_store/chroma"
        self.collection_name = collection_name
        
    def add_documents(self, documents: List[Document], embeddings: Embeddings):
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=self.persist_directory,
            collection_name=self.collection_name
        )
        vector_store.persist()
        return vector_store
        
    def query(self, query: str, embeddings: Embeddings, k: int = 4):
        vector_store = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=embeddings,
            collection_name=self.collection_name
        )
        results = vector_store.similarity_search(query, k=k)
        return [doc.page_content for doc in results] 