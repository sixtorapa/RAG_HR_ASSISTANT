from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document

class ChatMemoryStore:
    def __init__(self, persist_path: str):
        self.db = Chroma(
            persist_directory=persist_path,
            embedding_function=OpenAIEmbeddings(model="text-embedding-3-small")
        )

    def add_fact(self, text: str, metadata: dict):
        doc = Document(page_content=text, metadata=metadata)
        self.db.add_documents([doc])

    def recall(self, query: str, k: int = 5):
        return self.db.similarity_search(query, k=k)
