from sentence_transformers import SentenceTransformer
from transformers import BertModel, BertTokenizer, BertConfig, BertForPreTraining, load_tf_weights_in_bert
from transformers import RobertaModel, RobertaTokenizer
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
import time



device = "cuda"

class SentenceTransformerEmbedding:
    def __init__(self, emb_model=None, device='cuda'): # 'paraphrase-MiniLM-L6-v2' or 'all-MiniLM-L6-v2'
        self.model = emb_model


    ## Embedding 
    
    def embed_queries(self, queries, model, batch_size=16):
        self.model = model.to(device)
        queries_embeddings = []
        i = 0
        start_time = time.time()
        for query_id in queries:
            i += 1
            query_embedding = self.model.encode(queries[query_id], convert_to_tensor=True, show_progress_bar=True, batch_size=batch_size)
            queries_embeddings.append(query_embedding.cpu())
        end_time = time.time()
        print(f"Successfully embedded the queries! Average embedding time for each query is { (end_time - start_time)/ i}")

        return np.vstack(queries_embeddings, dtype=np.float64) #numpy.2darray
    
    def embed_corpus(self, corpus, model, batch_size=16):
        self.model = model.to(device)
        corpus_embeddings = []
        i = 0
        start_time = time.time()
        for corpus_id in corpus:
            i += 1
            corpus_embedding = self.model.encode(corpus[corpus_id]['text'], convert_to_tensor=True, show_progress_bar=True, batch_size=batch_size)
            corpus_embeddings.append(corpus_embedding.cpu())
        end_time = time.time()
        print(f"Successfully embedded the corpus! Average embedding time for each corpus is { (end_time - start_time)/ i}")

        return np.vstack(corpus_embeddings, dtype=np.float64)

class BertEmbedding:

    def __init__(self, emb_model = None, emb_tokenizer = None, device='cuda'):
        self.model = emb_model
        self.tokenizer = emb_tokenizer


    # def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, bert_config_file, pytorch_dump_path):
    #     # Initialise PyTorch model
    #     config = BertConfig.from_json_file(bert_config_file)
    #     print("Building PyTorch model from configuration: {}".format(str(config)))
    #     model = BertForPreTraining(config)

    #     # Load weights from tf checkpoint
    #     load_tf_weights_in_bert(model, config, tf_checkpoint_path)

    #     # Save pytorch-model
    #     print("Save PyTorch model to {}".format(pytorch_dump_path))
    #     torch.save(model.state_dict(), pytorch_dump_path)

    # embed queries and returns embeddings as 2d array (N * T): N - number of queries, T - max length
    def embed_queries(self, file):
        
        input_embed = []
        i = 0
        start_time = time.time()
        for key in file:
            i += 1
            input = self.tokenizer(file[key], return_tensors="pt").to("cuda") #, max_length=512, truncation=True

            with torch.no_grad():
                outputs = self.model(**input)

            embeddings = outputs.last_hidden_state.squeeze(0).to("cuda")

            sentence_embedding = torch.mean(embeddings, dim=0)
            input_embed.append(sentence_embedding)

        end_time = time.time()
        print(f"Successfully embedded the queries! Average embedding time for each query is { (end_time - start_time)/ i}")

        # Pad the tensors so they all have the same length
        # `batch_first=True` will make the output tensor of shape (batch, max_length)
        padded_tensors = pad_sequence(input_embed, batch_first=True)
        input_embed = np.vstack(padded_tensors.cpu(), dtype=np.float64)
        return input_embed

    # embed queries and returns embeddings as 2d array (N * 512): N - number of queries
    def embed_corpus(self, file):

        input_embed = []
        i = 0
        start_time = time.time()
        for key in file:
            i += 1
            input = self.tokenizer(file[key]['text'], return_tensors="pt", max_length=512, truncation=True).to("cuda") 

            with torch.no_grad():
                outputs = self.model(**input)

            embeddings = outputs.last_hidden_state.squeeze(0).to("cuda")

            sentence_embedding = torch.mean(embeddings, dim=0)
            input_embed.append(sentence_embedding)

        end_time = time.time()
        print(f"Successfully embedded the corpus! Average embedding time for each corpus is { (end_time - start_time)/ i}")

        # Pad the tensors so they all have the same length
        # `batch_first=True` will make the output tensor of shape (batch, max_length)
        padded_tensors = pad_sequence(input_embed, batch_first=True)
        input_embed = np.vstack(padded_tensors.cpu(), dtype=np.float64)
        return input_embed







  

