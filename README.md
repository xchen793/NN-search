# NN-search

## Week 0(10/23-10/29)
 - Read paper [Learning to Index for Nearest Neighbor Search](https://arxiv.org/pdf/1807.02962.pdf)
 - [Github](https://github.com/AmorntipPrayoonwong/Learning-to-Index-for-Nearest-Neighbor-Search)
## Week 1(10/23-10/29)
 - Read paper [EMBER](https://arxiv.org/pdf/2106.01501.pdf)
 - Find datasets from [BEIR](https://github.com/beir-cellar/beir)
## Week 2(10/30-11/5)
  - Finish problem statement[PS](need to update)
## Week 3(11/6-11/12)
  - Finish the first set of experiment
    - (common embedding + euclidean distance-based ranking + KNN search) and (common embedding + probabiliry-based ranking + KNN search)
    - Embedding Model:
      - Adopted sentence-transformer model `all-MiniLM-L6-v2`
        - Problem: too slow to embed all sentences in NFCorpus since we have (110,000 * 3612) as number of queries * doc pairs.
        - Solution: change the embedding model to BioBERT model "monologg/biobert_v1.1_pubmed" 
      - More models: ([sentence-transformers](https://www.sbert.net))
    - Dataset:
      - [Signal1M(news + related tweets)](https://research.signal-ai.com/datasets/signal1m-tweetir.html)
        - Problem: we only know the tweet-ids in this dataset, to download the corresponding tweets from API, we have to register a twitter developer account and subscribe for one month.
      - [NFCorpus](https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/)
    - Code:
      [Demo](https://colab.research.google.com/drive/1joFab0X8wMZ9PZn32ojT22sF5yM8upXb?usp=sharing)

## Week 4(11/13-11/20)
  - (common embedding + euclidean distance-based ranking + KNN search) - DONE
     - Tasks: Information Retrieval  
     - Dataset: NFCorpus
     - Embedding Model: BioBERT (word embedding -> sentence embedding by avg pooling)
     - euclidean dist-based ranking implemented by scikilearn
     - Metrics:
        - Precision:
           - How do we define the relevant items in our case out of K(i.e., true positives)? [TO-DO 1]
        - Recall@1 and Recall@10
           - For each query, extract most relevant documents and compute their avg euclidean distance as `threshold`.
           - If avg euclidean distances btw top-k results and query < threshold, then mark them as relevant item(1) and 0 vice versa.
        - Query Runtime: Embedding time + Search time
           - Single Query implemented in colab
           - Need to compute avg query runtime [TO-DO 2]
        - More Metrics: [TO-DO 3]
           - [Faiss](https://www.pinecone.io/learn/series/faiss/vector-indexes/) (Still implementing)
             - [Github](https://github.com/facebookresearch/faiss)
             - when/where to use?
   - (common embedding + probability-based ranking + KNN search)
      - Task: Information Retrieval
      - Dataset: NFCorpus
         - Input X: [query] + [SEP] + [document] ('str' type)  {Is there a better way to construct X?}[TO-DO 4]
         - label y: relevant scores (1,2,3)
      - Model:
         - What are deep learning models used for information retrieval? Should we use a simple one to implement the code first?[TO-DO 5]
      - Computing Resources [TO-DO 6]
        
        
          
               
    
