# ML Project : Music Recommendation and Genre Classification
Spring 2022 \\
Group Members: Sichen Jin, Jiachen Ren, Antoine Rollet, Adriana Vega Fernández, Robert Ward
## Proposal Video
<iframe width="560" height="315" src="https://www.youtube.com/embed/19_zbvP8lSg" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Introduction & Background 
Spotify has since become the most popular streaming music service, boasting more than 180M paid subscribers. Spotify achieved this position through its high-quality playlists based on listening habits (what you like, share, save, skip) and the listening habits of others with similar tastes. The user experience is personalized with algorithmically curated playlists like Discover Weekly and Release Radar and is now the base of their business model. Our project aims to replicate several tools Spotify uses to curate their playlists.

## Problem Definition
We wish to solve two separate but related problems. First, we will build a music genre classifier that classifies a track based on its audio features.  Next, we aim to build a recommendation system that would recommend songs to complete a playlist (given a list of songs already in the playlist, we predict the rest). The ground truth labels for both are provided by Spotify's Million Playlist Dataset and API. 

## Methods
### Dataset
We will use <a href="https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge" title="MPD">The Million Playlist Dataset (MPD)</a>, a dataset of 1 million playlists from Spotify. In this dataset, each playlist contains a list of song IDs and general information about the song. Using Spotify's [API](https://developer.spotify.com/documentation/web-api/reference/#/operations/get-several-audio-features), we will retrieve precomputed audio features such as `liveness`, `loudness`, `energy`, etc., for supervised and unsupervised learning tasks. In addition, we will use Spotify's API to retrieve raw audio samples of a subset of songs. 

### Unsupervised Learning
Genre classification: We aim to perform K-Means clustering on the selected audio features of songs. Given a set of songs and their corresponding genre label (with k genres total), we will cluster them into k clusters based on audio features in the Euclidean space. We will simply compare the distance between the audio features of new songs and the learned cluster centers to make genre predictions.

Graph-based approaches for playlist completion: We will first derive the songs-playlists bipartite graph from the dataset. Then we will explore some common graph-based methods, such as Random Walk (1), Neural Graph Collaborative Filtering (NGCF) (2), and LightGCN (3), to learn the graph embeddings of songs and playlists for later prediction/recommendation use. 

### Supervised Learning
LSTM/RNN for genre prediction: We will train an LSTM using raw audio samples as features and genre labels as GT labels. The model will consist of several recurrent layers followed by linear layers. The rationale is to use the cell state (or hidden state) from the last recurrent layer as a learned embedding of the audio sample, and then use the subsequent linear layers to classify the embedding into the correct genre label.

Sequential model for playlist completion: We will train a sequential model based on an attentive neural architecture incorporating song features (4).

Graph-based approach for playlist completion: We will train GraphSAGE (5) for playlist completion.

## Conclusion
Our project has two goals - genre classification and playlist completion. We will compare the results of different models that pertain to the same task based on their performance in terms of their precision, accuracy, and other metrics. These evaluations will help us understand what works the best.

## References
1. van Niedek, Timo, and Arjen P. de Vries. "Random walk with restart for automatic playlist continuation and query-specific adaptations." Proceedings of the ACM Recommender Systems Challenge 2018. 2018. 1-6.
2. Wang, Xiang, et al. "Neural graph collaborative filtering." Proceedings of the 42nd international ACM SIGIR conference on Research and development in Information Retrieval. 2019.
3. He, Xiangnan, et al. "Lightgcn: Simplifying and powering graph convolution network for recommendation." Proceedings of the 43rd International ACM SIGIR conference on research and development in Information Retrieval. 2020.
4. Sachdeva, Noveen, Kartik Gupta, and Vikram Pudi. "Attentive neural architecture incorporating song features for music recommendation." Proceedings of the 12th ACM Conference on Recommender Systems. 2018.
5. Hamilton, Will, Zhitao Ying, and Jure Leskovec. "Inductive representation learning on large graphs." Advances in neural information processing systems 30 (2017).

## Appendix
Download GANTT chart [here](https://docs.google.com/spreadsheets/d/1eBinr-KAz04P-j1TpmxQwzl0uz0aNe7P/export?format=xlsx).
