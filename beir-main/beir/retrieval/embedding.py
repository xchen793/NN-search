from sentence_transformers import SentenceTransformer
import numpy as np

device = "cuda"

class SentenceTransformerEmbedding:
    def __init__(self, model_name='all-MiniLM-L6-v2', device='cuda'):
        self.model = SentenceTransformer(model_name).to(device)


    ## Embedding 
    
    def embed_queries(self, queries, model, batch_size=16):
        self.model = model.to(device)
        queries_embeddings = []
        for query_id in queries:
            query_embedding = self.model.encode(queries[query_id], convert_to_tensor=True, show_progress_bar=True, batch_size=batch_size)
            queries_embeddings.append(query_embedding.cpu())
        
        print("Successfully embedded the queries!")

        return np.vstack(queries_embeddings, dtype=np.float64) #numpy.2darray
    
    def embed_corpus(self, corpus, model, batch_size=16):
        self.model = model.to(device)
        corpus_embeddings = []
        for corpus_id in corpus:
            corpus_embedding = self.model.encode(corpus[corpus_id]['text'], convert_to_tensor=True, show_progress_bar=True, batch_size=batch_size)
            corpus_embeddings.append(corpus_embedding.cpu())
        
        print("Successfully embedded the corpus!")

        return np.vstack(corpus_embeddings, dtype=np.float64)
    


