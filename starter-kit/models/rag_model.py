import json

from LLM import LLMPredictor
from embeddings import BGEpeftEmbedding
from web_search_processing import extract_web_sentences, extract_web_chunks
from langchain import FAISS

class RAGModel:
    def __init__(self) -> None:
        self.model_path = './models/meta-llama/Llama-2-7b-chat-hf'
        self.embedding_path = './models/BAAI/bge-base-en-v1.5'
        # load model
        self.model = LLMPredictor(self.model_path)
        self.embedding_model = BGEpeftEmbedding(model_path=self.embedding_path)

    def generate_answer(self, query, search_results) -> str:
        try:
            docs = extract_web_chunks(search_results)
            db = FAISS.from_documents(docs, self.embedding_model)
            db.save_local(folder_path='./vector', index_name='index_256')
            search_docs = db.similarity_search(query, k=5)
            references = ""
            for doc in search_docs:
                references += "<DOC>\n" + doc.page_content + "\n</DOC>\n"
            references = " ".join(
                references.split()[:500]
            )
            prediction = self.model.predict(references, query)
            prediction = prediction.strip()
        except:
            prediction = "I don't know"
        return prediction