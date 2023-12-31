from .. import BaseSearch
from typing import Dict
from ...embedding import SentenceTransformerEmbedding, BertEmbedding
from .prob_index import ProbRankModel, ProbRankModelTrainer, DataLoader
from transformers import BertModel, BertTokenizer, BertConfig, BertForPreTraining, load_tf_weights_in_bert
from transformers import RobertaModel, RobertaTokenizer
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
from torch import nn, optim
import logging
import os
from beir import util
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.datasets.data_loader import GenericDataLoader
from beir.reranking.models import CrossEncoder
from beir.reranking.rerank import Rerank


logger = logging.getLogger(__name__)


device = "cuda"

class ProbIndexSearch(BaseSearch):
    def __init__(self, index_model, emb_model_name, num_clusters, topk_cluster, topk_emb): 
        self.index_model = index_model
        self.num_clusters = num_clusters
        self.topk_cluster = topk_cluster
        self.topk_emb = topk_emb
        if not isinstance(self.index_model, ProbRankModel):
            raise TypeError("The 'model' must be an instance of NearestNeighbors.")
        if emb_model_name == "all-MiniLM-L6-v2" or emb_model_name == "paraphrase-MiniLM-L6-v2": 
            self.embed_model = SentenceTransformer(emb_model_name).to(device)
        elif emb_model_name == "biobert_v1.1_pubmed": 
            self.embed_model = BertModel.from_pretrained(emb_model_name).to(device)
            self.embed_tokenizer = BertTokenizer.from_pretrained(emb_model_name)
        elif emb_model_name == "roberta-base":
            self.embed_model = RobertaModel.from_pretrained(emb_model_name).to(device)
            self.embed_tokenizer = RobertaTokenizer.from_pretrained(emb_model_name)
    

    def search(
        self, 
        corpus: Dict[str, Dict[str, str]], 
        queries: Dict[str, str], 
        top_k: int, 
        score_function = str,
        **kwargs
    ) -> Dict[str, Dict[str, float]]:

        query_embeddings = None


        if isinstance(self.embed_model, BertModel) or isinstance(self.embed_model, RobertaModel):
            BB_Embedder = BertEmbedding(emb_model=self.embed_model, emb_tokenizer = self.embed_tokenizer)
            query_embeddings = BB_Embedder.embed_queries(queries)
        elif isinstance(self.embed_model, SentenceTransformer): 
            ST_Embedder = SentenceTransformerEmbedding(emb_model=self.embed_model)
            query_embeddings = ST_Embedder.embed_queries(queries = queries, model = self.embed_model)

        test_qrels = kwargs.get('test_qrels', 'default_value')
        X_train = kwargs.get('X_train', 'default_value')
        X_dev = kwargs.get('X_dev', 'default_value')
        y_train = kwargs.get('y_train', 'default_value')
        y_dev = kwargs.get('y_dev', 'default_value')
        X_test = kwargs.get('X_test', 'default_value')
        cluster_dict = kwargs.get('cluster_dict', 'default_value')

        # cluster_dicts
        # {190: [{'corpus_id': 'MED-10', 'corpus_embedding': array([ 1.50158629e-03, -1.63566992e-02, -1.00291401e-01,  2.27949563e-02,...])},..., {},...]}


        ##### Rank Model
        model_f = self.index_model.to(device)

        ##### Index Model Training
        # Define the loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model_f.parameters(), lr=0.01)
        # get trainer
        trainer = ProbRankModelTrainer(model_f, criterion=criterion, optimizer=optimizer, batch_size = 500, num_epochs = 60, num_clusters = self.num_clusters)
        # train
        trainer.fit(X_train, y_train, X_dev, y_dev)

        preds, _ = trainer.predict(X_test) # preds - 2d tensor, label 1d tensor

        ###### Reproduing index and search in the first paper starts from here


        '''
        The cluster_dict has the form of 
        {0: [{'corpus_id': 'doc1', 'corpus_embedding': array([1, 2])},
             {'corpus_id': 'doc2', 'corpus_embedding': array([3, 4])}],
         1: [{'corpus_id': 'doc3', 'corpus_embedding': array([5, 6])},
             {'corpus_id': 'doc4', 'corpus_embedding': array([7, 8])}]}

        cluster label 1: [{corpus1}, {corpus2},...],

        '''
        
        '''
        The query_topk_cluster_dict has the form of
        {'query1': [0, 2], 'query2': [2, 1], 'query3': [1, 2]}
         with cluster labels in probability descending order
        '''
        
        query_topk_cluster_dict = {}

        indices_and_keys = list(zip(range(len(queries)), queries))
        for i, query_id in indices_and_keys:
            # Extract the indices of the k largest values
            topk_indices = np.argsort(np.array(preds[i]))[::-1][:self.topk_cluster]
            # Get the corresponding cluster labels
            topk_cluster_labels = topk_indices.tolist()
            # Store the result in the dictionary
            query_topk_cluster_dict[query_id] = topk_cluster_labels


        result_dict = {}

        for query_index, (query_id, topk_clusters) in enumerate(query_topk_cluster_dict.items()):
            query_result_dict = {}
            query_embedding = query_embeddings[query_index]
            for cluster_label in topk_clusters:
                if cluster_label in cluster_dict:
                    cluster = cluster_dict[cluster_label]
                    for corpus_info in cluster:
                        corpus_id = corpus_info['corpus_id']
                        corpus_embedding = corpus_info['corpus_embedding']
                        # we will not use this distance we will rerank all test corpus by cross-encoder reranking
                        distance = np.linalg.norm(query_embedding - corpus_embedding)
                        query_result_dict[corpus_id] = distance
            result_dict[query_id] = query_result_dict

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
        rerank_results = reranker.rerank(corpus, queries, result_dict, top_k=len(result_dict))


        #### Evaluate your retrieval using NDCG@k, MAP@K ...
        k_values = [1, 3, 5, 10, 100, 1000]
        ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(test_qrels, rerank_results, k_values)


        return rerank_results
        # return result_dict






    
    


