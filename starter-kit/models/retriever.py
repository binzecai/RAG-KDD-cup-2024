# from abc import ABC
from transformers import AutoTokenizer, AutoModel
import torch
from langchain.schema.embeddings import Embeddings
from langchain.schema import Document
# from typing import List
import numpy as np
# from rank_bm25 import BM25Okapi
from langchain.vectorstores import FAISS

class Retriever:
    def __init__(self, emb_model=None, docs=None):
        # self.device = device
        # self.langchain_corpus = [Document(page_content=t) for t in corpus]
        # self.corpus = corpus
        # self.lan = lan
      # if lan=='zh':
        # tokenized_documents = [jieba.lcut(doc) for doc in corpus]
      # else:
        # tokenized_documents = [doc.split() for doc in corpus]
      # self.bm25 = BM25Okapi(tokenized_documents)

      # self.emb_model = BGEpeftEmbedding(emb_model_name_or_path=emb_model_name_or_path)
      self.db = FAISS.from_documents(docs, emb_model)

  # def bm25_retrieval(self, query, n=10):

  #     # 此处中文使用jieba分词
  #     query = jieba.lcut(query)  # 分词
  #     res = self.bm25.get_top_n(query, self.corpus, n=n)
  #     return res

    def emb_retrieval(self, query, k=15):
        search_docs = self.db.similarity_search(query, k=k)
        res = [doc.page_content for doc in search_docs]
        return res

    def retrieval(self, query, methods=None):
        if methods is None:
            methods = ['bm25', 'emb']
        search_res = list()
        for method in methods:
          if method == 'bm25':
            bm25_res = self.bm25_retrieval(query)
            for item in bm25_res:
              if item not in search_res:
                search_res.append(item)
          elif method == 'emb':
            emb_res = self.emb_retrieval(query)
            for item in emb_res:
              if item not in search_res:
                search_res.append(item)
        return search_res