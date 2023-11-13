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
    - For common embedding, we adopted sentence-transformer model `all-MiniLM-L6-v2` ([sentence-transformers](https://www.sbert.net))
    - Dataset:
      - [Signal1M(news + related tweets)](https://research.signal-ai.com/datasets/signal1m-tweetir.html)
        - Problem: we only know the tweet-ids in this dataset, to download the corresponding tweets from API, we have to register a twitter developer account and subscribe for one month.
      - [NFCorpus](https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/)
    - Code:
      [Demo](https://colab.research.google.com/drive/1joFab0X8wMZ9PZn32ojT22sF5yM8upXb?usp=sharing)
    
    
  - 
