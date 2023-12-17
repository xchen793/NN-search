import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import numpy as np
import os

from sklearn.metrics import pairwise_distances
from sentence_transformers import SentenceTransformer, SentencesDataset, datasets
from sentence_transformers.evaluation import SentenceEvaluator, SequentialEvaluator, InformationRetrievalEvaluator
from sentence_transformers.readers import InputExample
from sklearn.cluster import KMeans

from tqdm.autonotebook import trange
from typing import Dict, List, Callable, Iterable, Tuple
from beir.retrieval.embedding import SentenceTransformerEmbedding, BertEmbedding
from transformers import BertModel, BertTokenizer, BertConfig, BertForPreTraining, load_tf_weights_in_bert
from transformers import RobertaModel, RobertaTokenizer

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

device = "cuda"



class DataLoader():
    def __init__(self, emb_model_name, train_corpus, train_queries, dev_corpus, dev_queries,\
                test_corpus, test_queries, train_qrels, dev_qrels, test_qrels, knn_centroids = None):
        self.train_corpus = train_corpus 
        self.train_queries = train_queries

        self.dev_queries = dev_queries
        self.dev_corpus = dev_corpus

        self.test_queries = test_queries
        self.test_corpus = test_corpus 

        self.train_qrels = train_qrels
        self.dev_qrels = dev_qrels
        self.test_qrels = test_qrels
        
        self.knn_centroids = knn_centroids

        if emb_model_name == "all-MiniLM-L6-v2" or emb_model_name == "paraphrase-MiniLM-L6-v2": 
            self.embed_model = SentenceTransformer(emb_model_name).to(device)
            ST_Embedder = SentenceTransformerEmbedding(emb_model=self.embed_model)
            self.embedder = ST_Embedder

            self.train_queries_embeddings = self.embedder.embed_queries(self.train_queries, model = self.embed_model)
            self.dev_queries_embeddings = self.embedder.embed_queries(self.dev_queries, model = self.embed_model)
            self.test_queries_embeddings = self.embedder.embed_queries(self.test_queries, model = self.embed_model)

        elif emb_model_name == "biobert_v1.1_pubmed": 
            self.embed_model = BertModel.from_pretrained(emb_model_name).to(device)
            self.embed_tokenizer = BertTokenizer.from_pretrained(emb_model_name)
            BB_Embedder = BertEmbedding(emb_model=self.embed_model, emb_tokenizer = self.embed_tokenizer)
            self.embedder = BB_Embedder

            self.train_queries_embeddings = self.embedder.embed_queries(self.train_queries)
            self.dev_queries_embeddings = self.embedder.embed_queries(self.dev_queries)
            self.test_queries_embeddings = self.embedder.embed_queries(self.test_queries)

        elif emb_model_name == "roberta-base":
            self.embed_model = RobertaModel.from_pretrained(emb_model_name).to(device)
            self.embed_tokenizer = RobertaTokenizer.from_pretrained(emb_model_name)
            RB_Embedder = BertEmbedding(emb_model=self.embed_model, emb_tokenizer = self.embed_tokenizer)
            self.embedder = RB_Embedder

            self.train_queries_embeddings = self.embedder.embed_queries(self.train_queries)
            self.dev_queries_embeddings = self.embedder.embed_queries(self.dev_queries)
            self.test_queries_embeddings = self.embedder.embed_queries(self.test_queries)
        
        # Combine train and test corpora with overlapping keys
        self.combined_corpus = {}

        for key, value in self.train_corpus.items():
            self.combined_corpus[key] = value

        for key, value in self.test_corpus.items():
            self.combined_corpus[key] = value

        if isinstance(self.embedder, BertEmbedding):
            self.corpus_embeddings = self.embedder.embed_corpus(self.combined_corpus)
        else:
            self.corpus_embeddings =  self.embedder.embed_corpus(self.combined_corpus, self.embed_model)



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
    
    def plot_kmeans_clusters(self, n_clusters):
        # Perform PCA to reduce the embeddings to 2 dimensions for visualization
        pca = PCA(n_components=2)
        reduced_embeddings = pca.fit_transform(self.corpus_embeddings) 

        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(reduced_embeddings)
        cluster_labels = kmeans.labels_

        # Create a scatter plot of the two-dimensional data
        plt.figure(figsize=(10, 8))
        plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=cluster_labels, cmap='viridis', marker='o')

        # Optionally, plot the cluster centers
        centers = kmeans.cluster_centers_
        plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='x')

        # Add axis labels and a title
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('2D PCA Plot of Corpus Embeddings Clusters')
        plt.show()
    
    # build input X and y(ground truth based on the relevant score - human annotated)
    def load_data(self, num_clusters):



    
        ###### Kmeans and Cluastering Information ###############
        # get kmeans model and cluster assignment (that is why not using knn)
        kmeans_model = KMeans(n_clusters=num_clusters, random_state=42) # default metric is euclidean distance
        kmeans_model.fit(self.corpus_embeddings)

        corpus_ids = list(self.combined_corpus.keys())
        assert len(corpus_ids) == len(self.corpus_embeddings)

        self.knn_centroids = kmeans_model.cluster_centers_       
        cluster_counts = dict(zip(*np.unique(kmeans_model.labels_, return_counts=True)))

        # Create a dictionary to store cluster information
        cluster_dict = {}

        # Populate the dictionary with cluster assignments
        for i, cluster_label in enumerate(kmeans_model.labels_):
            corpus_embedding = self.corpus_embeddings[i]

            if cluster_label not in cluster_dict:
                cluster_dict[cluster_label] = []

            cluster_dict[cluster_label].append({"corpus_id": corpus_ids[i], "corpus_embedding": corpus_embedding})


        count_dict = {}
        # Print the number of embeddings in each cluster
        for cluster, count in cluster_counts.items():
            count_dict[cluster] = count


        ########### Construct ground truth ##############
        train_probs = self.create_probs(self.train_queries, self.train_qrels, kmeans_model, count_dict)
        dev_probs = self.create_probs(self.dev_queries, self.dev_qrels, kmeans_model, count_dict)
        test_probs = self.create_probs(self.test_queries, self.test_qrels, kmeans_model, count_dict)    

        ############ Build Input X - sim matrix ###########

        # Compute queries distance to kNN centroids
        train_sim_matrix = self.create_X(self.train_queries_embeddings)
        dev_sim_matrix = self.create_X(self.dev_queries_embeddings)
        test_sim_matrix = self.create_X(self.test_queries_embeddings)

        return train_sim_matrix, train_probs, dev_sim_matrix, dev_probs, test_sim_matrix, test_probs, cluster_dict




class ProbRankModel(nn.Module):
    def __init__(self, input_dimension, num_clusters):
        super(ProbRankModel, self).__init__()
        self.fc1 = nn.Linear(input_dimension, 512)  # Hidden layer H2
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 512)  # Hidden layer H3
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(512, num_clusters)  # Output layer O4
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)

        return x
    


class ProbRankModelTrainer:
    def __init__(self, model, criterion, optimizer, batch_size, num_epochs, num_clusters):
        self.model = model.to(device)
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_clusters = num_clusters

    def train(self, train_sim_matrix, train_probs, dev_sim_matrix, dev_probs):

        ##### trainer
        # Define the loss function and optimizer
        criterion = self.criterion
        optimizer = self.optimizer

        ##### Inputs X ######
        X_train_tensor = torch.Tensor(train_sim_matrix).to(device)
        X_dev_tensor = torch.Tensor(dev_sim_matrix).to(device)

        # mean = torch.mean(X_train_tensor, dim=0)
        # std = torch.std(X_train_tensor, dim=0)
        # X_train_tensor = (X_train_tensor - mean) / std

        # mean = torch.mean(X_dev_tensor, dim=0)
        # std = torch.std(X_dev_tensor, dim=0)
        # X_dev_tensor = (X_dev_tensor - mean) / std

        ######## labels y ##########
        y_train_tensor = torch.Tensor(train_probs).to(device)
        y_dev_tensor = torch.Tensor(dev_probs).to(device)

        labels = torch.argmax(y_train_tensor, dim=1).to(device)# tensor([109,  36,  73,  ...,  49,  39,  23])
        vlabels = torch.argmax(y_dev_tensor, dim=1).to(device)

        # Learning rate scheduler
        # `step_size` is the number of epochs after which the learning rate will be reduced.
        # `gamma` is the factor by which the learning rate will be reduced.
        # scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

        running_loss = 0.
        last_loss = 0.
        train_losses = []
        dev_losses = []
        learning_rates = []


        # l2_lambda = 0.01

        for epoch in range(self.num_epochs):


            for index, inputs in enumerate(X_train_tensor):

                self.model.train(True)
            
                optimizer.zero_grad()
                    
                inputs = inputs.unsqueeze(0)

                outputs = self.model(inputs)  
                loss = criterion(outputs, labels[index].unsqueeze(0))
                # # L2 regularization
                # l2_reg = torch.tensor(0.0, device = 'cuda')
                # for param in model_f.parameters():
                #     l2_reg += torch.norm(param)

                # loss += l2_lambda * l2_reg
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                # Print the loss every few epochs
                if (index + 1) % self.batch_size == 0:
                    last_loss = running_loss / self.batch_size
                    print('batch {} loss: {}'.format(index + 1, last_loss))
                    running_loss = 0.
                
            avg_loss = last_loss
            train_losses.append(last_loss)
            print('Epoch: {}. Train Loss: {}'.format(epoch + 1, avg_loss))  

            self.model.eval()
            running_vloss = 0.

            with torch.no_grad():
                for i, vinputs in enumerate(X_dev_tensor):
                    voutputs = self.model(vinputs.unsqueeze(0))
                    vloss = criterion(voutputs, vlabels[i].unsqueeze(0))
                    running_vloss += vloss
            
            avg_vloss = running_vloss / (i + 1)
            dev_losses.append(avg_vloss)
            print('Epoch: {}. Valid Loss: {}'.format(epoch + 1, avg_vloss))

            current_lr = optimizer.param_groups[0]['lr']
            # scheduler.step()

            learning_rates.append(current_lr)



        
        ###### Plotting ######

        plt.figure()
        plt.plot(learning_rates)
        plt.xlabel('Epochs')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')

        output_path_1 = os.path.join("/nethome/xchen920/Learning-to-Index-for-Nearest-Neighbor-Search-master/outputs", "learning_rate_schedule.png")
        plt.savefig(output_path_1)

        dev_losses = np.array([tensor.cpu() for tensor in dev_losses])
        plt.figure()
        plt.plot(train_losses, label='Training Loss')
        plt.plot(dev_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training/Validation Loss Over Time')
        plt.legend()
        output_path_2 = os.path.join("/nethome/xchen920/Learning-to-Index-for-Nearest-Neighbor-Search-master/outputs", "train_val_loss.png")
        plt.savefig(output_path_2)

       

    def fit(self, train_sim_matrix, train_probs, dev_sim_matrix, dev_probs):
        self.train(train_sim_matrix, train_probs, dev_sim_matrix, dev_probs)

    def get_model(self):
        return self.model
    
    def predict(self, test_sim_matrix):
        self.model.eval()  
        X_test_tensor = torch.Tensor(test_sim_matrix).to(device)
        with torch.no_grad():
            preds = self.model(X_test_tensor).cpu()
            labels = torch.argmax(preds, dim = 1)
        
        return preds, labels
    
