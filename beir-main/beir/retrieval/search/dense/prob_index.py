import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import heapq


from sklearn.metrics import pairwise_distances
from sentence_transformers import SentenceTransformer, SentencesDataset, datasets
from sentence_transformers.evaluation import SentenceEvaluator, SequentialEvaluator, InformationRetrievalEvaluator
from sentence_transformers.readers import InputExample
from sklearn.cluster import KMeans
import numpy as np
from tqdm.autonotebook import trange
from typing import Dict, List, Callable, Iterable, Tuple
from beir.retrieval.embedding import SentenceTransformerEmbedding 

STembedder = SentenceTransformerEmbedding()
emb_model = SentenceTransformer('all-MiniLM-L6-v2').to('cuda')

class DataLoader():
    def __init__(self, train_corpus, train_queries, test_queries, test_corpus, train_qrels, test_qrels, knn_centroids = None):
        self.train_corpus = train_corpus 
        self.train_queries = train_queries
        self.test_corpus = test_corpus 
        self.test_queries = test_queries
        self.train_qrels = train_qrels
        self.test_qrels = test_qrels
        self.train_corpus_embeddings = STembedder.embed_corpus(train_corpus, emb_model)
        self.train_queries_embeddings = STembedder.embed_queries(train_queries, emb_model)
        self.test_corpus_embeddings = STembedder.embed_corpus(test_corpus, emb_model)
        self.test_queries_embeddings = STembedder.embed_queries(test_queries, emb_model)
        self.knn_centroids = knn_centroids
    
    
    # Auxiliary function

    # def corpusid2embed(self, corpus_id, flag):
    #     if flag == 'train':
    #         keys_list = list(self.train_corpus.keys())
    #         index = keys_list.index(corpus_id)
    #         return self.train_corpus_embeddings[index]
    #     else:
    #         keys_list = list(self.test_corpus.keys())
    #         index = keys_list.index(corpus_id)
    #         return self.test_corpus_embeddings[index]


    # def queryid2embed(self, query_id, flag):
    #     if flag == "train":
    #         keys_list = list(self.train_queries.keys())
    #         index = keys_list.index(query_id)
    #         return self.train_queries_embeddings[index]
    #     else:
    #         keys_list = list(self.test_queries.keys())
    #         index = keys_list.index(query_id)
    #         return self.test_queries_embeddings[index]

    def create_X(self, queries_embeddings):
        euclidean_distances = self.compute_distance_to_knn_centroids(queries_embeddings, self.knn_centroids)
        distance_matrix = euclidean_distances.reshape((queries_embeddings.shape[0], self.knn_centroids.shape[0]))
        sim_matrix = self.sim_feature(distance_matrix) 
        return sim_matrix
    
    # generate ground truth
    def create_probs(self, queries, qrels, kmeans_model, count_dict):

        query_ids = list(queries.keys())
        probs = []
    
        for query_id in query_ids:
            score_dict = {}
            scores = []
            for corpus_id, score in qrels[query_id].items():
                # get idx for the corpus_id
                idx = self.get_index_for_key(self.train_corpus, corpus_id)
                cluster_label = kmeans_model.labels_[idx]
                if cluster_label not in score_dict:
                    score_dict[cluster_label] = 0
                if score >= 1: # if score = 0, we don't consider for training
                    score_dict[cluster_label] += score
    
            # update the score_dict and store the probability values in list
            for key in list(count_dict.keys()):
                if key in score_dict.keys():
                    score_dict[key] = score_dict[key] / count_dict[key]
                else:
                    score_dict[key] = 0
                
                scores.append(score_dict[key])

            probs.append(scores)

        return probs # list of list

    def get_index_for_key(self, dictionary, key):
        for index, (k, _) in enumerate(dictionary.items()):
            if k == key:
                return index
        return None  # Key not found
    
    def compute_distance_to_knn_centroids(self, embedded_query_data, knn_centroids):
        # Compute Euclidean distance between embedded query data and kNN centroids
        distances = pairwise_distances(embedded_query_data, knn_centroids, metric='euclidean')

        return distances
    

    def sim_feature(self, distance_matrix):
        sim_matrix = np.zeros((distance_matrix.shape[0], distance_matrix.shape[1]))
        max_values = np.max(distance_matrix, axis=1)
        for i in range(distance_matrix.shape[0]):
            for j in range(distance_matrix.shape[1]):
                sim_matrix[i, j] = (max_values[i] - distance_matrix[i, j])/ max_values[i]
        return sim_matrix
    
    # build input X and y(ground truth based on the relevant score - human annotated)
    def load_data(self, num_clusters):

        # Combine train and test corpora with overlapping keys
        combined_corpus = {}

        for key, value in self.train_corpus.items():
            combined_corpus[key] = value

        for key, value in self.test_corpus.items():
            combined_corpus[key] = value

        corpus_ids = list(combined_corpus.keys())
        

        ######Kmeans and Cluastering Information ###############
        # get kmeans model and cluster assignment (that is why not using knn)
        kmeans_model = KMeans(n_clusters=num_clusters, random_state=42) # default metric is euclidean distance
        corpus_embeddings =  STembedder.embed_corpus(combined_corpus, emb_model)
        kmeans_model.fit(corpus_embeddings)

        assert len(corpus_ids) == len(corpus_embeddings)

        self.knn_centroids = kmeans_model.cluster_centers_       
        cluster_counts = dict(zip(*np.unique(kmeans_model.labels_, return_counts=True)))

        # Create a dictionary to store cluster information
        cluster_dict = {}

        # Populate the dictionary with cluster assignments
        for i, cluster_label in enumerate(kmeans_model.labels_):
            corpus_embedding = corpus_embeddings[i]

            if cluster_label not in cluster_dict:
                cluster_dict[cluster_label] = []

            cluster_dict[cluster_label].append({"corpus_id": corpus_ids[i], "corpus_embedding": corpus_embedding})


        count_dict = {}
        # Print the number of embeddings in each cluster
        for cluster, count in cluster_counts.items():
            count_dict[cluster] = count


        ########### Construct ground truth ##############
        train_probs = self.create_probs(self.train_queries, self.train_qrels, kmeans_model, count_dict)
        test_probs = self.create_probs(self.test_queries, self.test_qrels, kmeans_model, count_dict)    

        ############ Build Input X - sim matrix ###########

        # Compute queries distance to kNN centroids
        train_sim_matrix = self.create_X(self.train_queries_embeddings)
        test_sim_matrix = self.create_X(self.test_queries_embeddings)

        return train_sim_matrix, train_probs, test_sim_matrix, test_probs, cluster_dict




class ProbRankModel(nn.Module):
    def __init__(self, input_dimension, num_clusters):
        super(ProbRankModel, self).__init__()
        self.fc1 = nn.Linear(input_dimension, 512)  # Hidden layer H2
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 512)  # Hidden layer H3
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(512, num_clusters)  # Output layer O4
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x
    

   

class ProbRankModelTrainer:
    def __init__(self, model: ProbRankModel, criterion, optimizer, batch_size, num_epochs, num_clusters):
        self.model = model
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_clusters = num_clusters


    def train(self, sim_matrix, probs):

        X_train_tensor = torch.Tensor(sim_matrix)
        y_train_tensor = torch.Tensor(probs) 

        for epoch in range(self.num_epochs):
            # Forward pass
            outputs = self.model(X_train_tensor)
            
            # Compute the loss
            loss = self.criterion(outputs, y_train_tensor)

            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Print the loss every few epochs
            if (epoch + 1) % 5 == 0:
                print(f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {loss.item():.4f}')


    def fit(self, sim_matrix, labels):
        self.train(sim_matrix, labels)

    def get_model(self):
        return self.model
    

    def predict(self, test_sim_matrix):
        self.model.eval()  
        X_test_tensor = torch.Tensor(test_sim_matrix)
        with torch.no_grad():
            preds = self.model(X_test_tensor).cpu()
            labels = torch.argmax(preds, dim = 1)
        
        return preds, labels
    
