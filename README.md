# Machine Learning Semester Project : Playlist Completion and Music Recommendations
Spring 2022 \\
Group Members: Sichen Jin, Jiachen Ren, Antoine Rollet, Adriana Vega Fernández, Robert Ward
## Proposal Video
<iframe width="560" height="315" src="https://www.youtube.com/embed/M8-o4C4n2p8" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Introduction & Background 
Spotify has since become the most popular streaming music service, boasting more than 180M paid subscribers. How did Spotify become the leader of this highly competitive market?  The app design is centered around playlists, playlists based on your listening habits (what you like, share, save, skip) and the listening habits of others with similar taste. The user experience is personalized with algorithmically curated playlists like Discover Weekly and Release Radar and is now the base of their business model. 

## Problem Definition
The best playlists contain a variety of genres and moods. A good playlist is subjective; it also depends on personal taste. We aim to build a recommendation system that would recommend songs to complete a playlist. Formally put, it means that given a set of songs features, we will output a set of songs represented as their IDs. 
We define the problem as automatic playlist completion. Our ground truth will be the playlists provided by the Spotify Millon Playlist Dataset. Our goal will be to accurately complete them in the same way that the Text Completion task is defined in NLP.

## Methods
### Dataset
Spotify hosted a challenge called <a href="https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge" title="Million Dollar Playlist">Spotify Million Playlists</a> for which they provided a dataset of 1 million playlist, this is the dataset that we will use. In this dataset, each playlist is represented as a list of song ids. We also use Spotify’s API  in order to retrieve general information, precomputed audio features and audio samples from the previous 2.2M songs of the previous dataset.
### Unsupervised Learning
Basic clustering : We aim to perform basic clustering algorithms like K-Means or DBSCAN on the audio features of the dataset provided by Spotify, which can serves as the candidate pool for our later recommendation task \\
Graph-based approaches : We will first derive the songs-playlists bipartite graph from the dataset. Then we will explore some common graph-based methods, such as Random Walk (1), Neural Graph Collaborative Filtering (NGCF) (2) and LightGCN (3), to learn the graph embeddings of songs and playlists for later prediction/recommendation use. 
### Supervised Learning
Genre classification: We will perform Genre Classification, we are hoping to compare the results of this task with the results obtained in our unsupervised learning tasks. \\
Sequence based approach: We will represent playlists as a sequence of song names. Then incorporate extracted song features into an attentive RNN for the prediction (4). \\
We will also employ another graph-based approach, GraphSAGE (5) which can be trained in a fully supervised manner.


## Conclusion
The goal of this project is to recommend songs to complete playlists.  Each of the methods described above will be used to generate a list of recommended songs for each playlist in the test data.  We will then evaluate their absolute performance in terms of their precision, accuracy and other metrics in order to understand which of these methods provides the best recommendations.

## References
1. van Niedek, Timo, and Arjen P. de Vries. "Random walk with restart for automatic playlist continuation and query-specific adaptations." Proceedings of the ACM Recommender Systems Challenge 2018. 2018. 1-6.
2. Wang, Xiang, et al. "Neural graph collaborative filtering." Proceedings of the 42nd international ACM SIGIR conference on Research and development in Information Retrieval. 2019.
3. He, Xiangnan, et al. "Lightgcn: Simplifying and powering graph convolution network for recommendation." Proceedings of the 43rd International ACM SIGIR conference on research and development in Information Retrieval. 2020.
4. Sachdeva, Noveen, Kartik Gupta, and Vikram Pudi. "Attentive neural architecture incorporating song features for music recommendation." Proceedings of the 12th ACM Conference on Recommender Systems. 2018.
5. Hamilton, Will, Zhitao Ying, and Jure Leskovec. "Inductive representation learning on large graphs." Advances in neural information processing systems 30 (2017).

## Appendix
Download GANTT chart [here](https://docs.google.com/spreadsheets/d/1eBinr-KAz04P-j1TpmxQwzl0uz0aNe7P/export?format=xlsx).
