import json

from LLM import LLMPredictor
from embeddings import BGEpeftEmbedding
from web_search_processing import extract_web_sentences, extract_web_chunks
from langchain import FAISS
from retriever import Retriever
from reranker import Reranker

class RAGModel:
    def __init__(self) -> None:
        self.model_path = './models/meta-llama/Llama-2-7b-chat-hf'
        self.embedding_path = './models/BAAI/bge-base-en-v1.5'
        self.reranker_path = './models/BAAI/bge-reranker-large'
        # load model
        self.model = LLMPredictor(self.model_path)
        self.embedding_model = BGEpeftEmbedding(model_path=self.embedding_path)
        self.reranker = Reranker(rerank_model_name_or_path=self.reranker_path)

    def generate_answer(self, query, search_results) -> str:
        try:
            query_repeat = ' '.join([query,query,query])
            docs = extract_web_chunks(search_results)
            retriever = Retriever(self.embedding_model, docs)
            retrieval_docs = retriever.retrieval(query_repeat, methods=['emb'])
            rerank_docs = self.reranker.rerank(retrieval_docs, query_repeat, k=5)
            references = ""
            for doc in rerank_docs:
                references += "<DOC>\n" + doc + "\n</DOC>\n"
            references = " ".join(
                references.split()[:500]
            )
            prediction = self.model.predict(references, query)
            prediction = prediction.strip()
        except:
            prediction = "I don't know"
        return prediction