from .. import BaseSearch
from typing import Dict
from ...embedding import SentenceTransformerEmbedding

import numpy as np
from torch import nn, optim
import logging
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import NearestNeighbors

device = "cuda"
from sentence_transformers import SentenceTransformer
emb_model_name = "all-MiniLM-L6-v2"



logger = logging.getLogger(__name__)

class KNNSearch(BaseSearch):
    def __init__(self, model=None, n_neighbors=5, metric='euclidean', emb_model = SentenceTransformer(emb_model_name)): # or 'cosine'
        self.n_neighbors = n_neighbors
        self.metric = metric
        if model is not None and not isinstance(model, NearestNeighbors):
            raise TypeError("The 'model' must be an instance of NearestNeighbors.")
        self.model = model
        self.embed_model = emb_model

    def fit(self, corpus_embeddings):
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
        
    
        emb_model = self.embed_model.to(device)
        query_embeddings = SentenceTransformerEmbedding.embed_queries(queries, emb_model)
        corpus_embeddings = SentenceTransformerEmbedding.embed_corpus(corpus, emb_model)

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

        return results
    