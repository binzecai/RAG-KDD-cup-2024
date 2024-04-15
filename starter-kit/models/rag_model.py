import json

from LLM import LLMPredictor
from embeddings import BGEpeftEmbedding
from web_search_processing import extract_web_sentences
from langchain import FAISS

class RAGModel:
    def __init__(self, model_path, embedding_path) -> None:
        self.model_path = model_path
        self.embedding_path = embedding_path

    
    def generate_answer(self, query: str, search_results: List[Dict]) -> str:
        
        # load model
        self.model = LLMPredictor(self.model_path)
        self.embedding_model = BGEpeftEmbedding(model_path=self.embedding_path)

        # process web search
        docs = extract_web_sentences(search_results)
        db = FAISS.from_documents(docs, self.embedding_model)
        db.save_local(folder_path='./vector', index_name='index_256')

        # inference
        search_docs = db.similarity_search(query, k=5)
        answer = self.model.predict(search_docs, query)
        return answer