from .. import BaseSearch
from typing import Dict
from ...embedding import SentenceTransformerEmbedding, BertEmbedding
from transformers import BertModel, BertTokenizer, BertConfig, BertForPreTraining, load_tf_weights_in_bert
from transformers import RobertaModel, RobertaTokenizer

import numpy as np
from torch import nn, optim
import logging
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import NearestNeighbors
from beir.reranking.models import CrossEncoder
from beir.reranking.rerank import Rerank

device = "cuda"
from sentence_transformers import SentenceTransformer




logger = logging.getLogger(__name__)

class KNNSearch(BaseSearch):
    def __init__(self, model=None, n_neighbors=5, metric='euclidean', emb_model_name=None): # or 'cosine'
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.model = model
        self.embed_model = None
        if emb_model_name == "all-MiniLM-L6-v2" or emb_model_name == "paraphrase-MiniLM-L6-v2": 
            self.embed_model = SentenceTransformer(emb_model_name).to(device)
        elif emb_model_name == "biobert_v1.1_pubmed": 
            self.embed_model = BertModel.from_pretrained(emb_model_name).to(device)
            self.embed_tokenizer = BertTokenizer.from_pretrained(emb_model_name)
        elif emb_model_name == "roberta-base":
            self.embed_model = RobertaModel.from_pretrained(emb_model_name).to(device)
            self.embed_tokenizer = RobertaTokenizer.from_pretrained(emb_model_name)
        

    def fit(self, corpus_embeddings):
        
        if corpus_embeddings is None:
            raise ValueError("Corpus_embeddings is None")
 

        # Assuming corpus_embeddings is a dictionary of embeddings for each document/query
        self.model = NearestNeighbors(n_neighbors=self.n_neighbors, metric=self.metric)
        self.model.fit(corpus_embeddings)
    


    def search(
        self, 
        corpus: Dict[str, Dict[str, str]], 
        queries: Dict[str, str], 
        top_k: int, 
        score_function = str,
        **kwargs
    ) -> Dict[str, Dict[str, float]]:

        self.n_neighbors = top_k
        self.metric = score_function

        if score_function == "cos_sim":
            self.metric = "cosine"
        

        query_embeddings = None
        corpus_embeddings = None

        if isinstance(self.embed_model, BertModel) or isinstance(self.embed_model, RobertaModel):
            BB_Embedder = BertEmbedding(emb_model=self.embed_model, emb_tokenizer = self.embed_tokenizer)
            query_embeddings = BB_Embedder.embed_queries(queries)
            corpus_embeddings = BB_Embedder.embed_corpus(corpus)
        elif isinstance(self.embed_model, SentenceTransformer): 
            ST_Embedder = SentenceTransformerEmbedding(emb_model=self.embed_model)
            query_embeddings = ST_Embedder.embed_queries(queries = queries, model = self.embed_model)
            corpus_embeddings = ST_Embedder.embed_corpus(corpus = corpus, model = self.embed_model)

        
        if self.model is None:
            self.fit(corpus_embeddings)


        results = {}

        indices_and_keys = list(zip(range(len(queries)), queries))

        for i, query_id in indices_and_keys:
            query_embedding = query_embeddings[i].reshape(1,-1)
            distances, indices = self.model.kneighbors(query_embedding)
            neighbors = {}

            for j, idx in enumerate(indices[0]):
                neighbor_id = list(corpus.keys())[idx]
                distance = distances[0][j]
                neighbors[neighbor_id] = distance
            results[query_id] = neighbors
        
        # return serach results should be in format:
        # { PLAIN-3: {'MED-4993': 0.22703276574611664, 'MED-2075': 0.22777065634727478,...}
        

        ################################################
        #### (2) RERANK Top-20(self.topk_emb) docs using Cross-Encoder
        ################################################

        #### Reranking using Cross-Encoder models #####
        #### https://www.sbert.net/docs/pretrained_cross-encoders.html
        cross_encoder_model = CrossEncoder('cross-encoder/ms-marco-electra-base')

        #### Or use MiniLM, TinyBERT etc. CE models (https://www.sbert.net/docs/pretrained-models/ce-msmarco.html)
        # cross_encoder_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        # cross_encoder_model = CrossEncoder('cross-encoder/ms-marco-TinyBERT-L-6')

        reranker = Rerank(cross_encoder_model, batch_size=128)

        # Rerank all results using the reranker provided
        rerank_results = reranker.rerank(corpus, queries, results, top_k=len(results))


        # #### Evaluate your retrieval using NDCG@k, MAP@K ...
        # k_values = [1, 3, 5, 10, 100, 1000]
        # ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(test_qrels, rerank_results, k_values)


        return rerank_results
        # return results
    