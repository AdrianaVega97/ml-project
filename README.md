# Machine Learning Semester Project : Playlist Completion and Music Recommendations
Spring 2022 \\
Group Members: Sichen Jin, Jiachen Ren, Antoine Rollet, Adriana Vega Fernández, Robert Ward
## Proposal Video

<iframe width="560" height="315" src="https://www.youtube.com/embed/zMEgK9tvkCE" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Introduction & Background 
Spotify has since become the most popular streaming music service, boasting more than 180M paid subscribers. How did Spotify become the leader of this highly competitive market?  The app design is centered around playlists, playlists based on your listening habits (what you like, share, save, skip) and the listening habits of others with similar taste. The user experience is personalized with algorithmically curated playlists like Discover Weekly and Release Radar and is now the base of their business model. 

## Problem Definition
The best playlists contain a variety of genres and moods. A good playlist is subjective; it also depends on personal taste. We aim to build a recommendation system that would recommend songs to complete a playlist. Formally put, it means that given a set of songs features, we will output a set of songs represented as their IDs. 
We define the problem as automatic playlist completion. Our ground truth will be the playlists provided by the Spotify Millon Playlist Dataset. Our goal will be to accurately complete them in the same way that the Text Completion task is defined in NLP.

## Methods
### Dataset
Spotify hosted a challenge called <a href="https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge" title="Million Dollar Playlist">Spotify Million Playlists</a> for which they provided a dataset of 1 million playlist, this is the dataset that we will use. In this dataset, each playlist is represented as a list of song ids. We also use Spotify’s API  in order to retrieve general information, precomputed audio features and audio samples from the previous 2.2M songs of the previous dataset.
### Unsupervised Learning
Basic clustering : We aim to perform basic clustering algorithms like K-Means or DBSCAN on the audio features of the dataset provided by Spotify. \\
Graph-based approaches : We will first derive the songs-playlists bipartite graph from the dataset. Then we will explore some common graph-based methods, such as Random Walk (1), Neural Graph Collaborative Filtering (NGCF) and LightGCN (2), to learn the graph embeddings of songs and playlists for later prediction/recommendation use. 
### Supervised Learning
Genre classification: We will perform Genre Classification, we are hoping to compare the results of this task with the results obtained in our unsupervised learning tasks. \\
Sequence based approach: We will represent playlists as a sequence of song names. Then incorporate extracted song features into an attentive RNN for the prediction. 

## Conclusion
## References
1. Random Walk with Restart for Automatic Playlist Continuation and Query-Specific Adaptations
2. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation
3. Bogdanov, Dmitry, et al. "Essentia: An audio analysis library for music information retrieval." Britto A, Gouyon F, Dixon S, editors. 14th Conference of the International Society for Music Information Retrieval (ISMIR); 2013 Nov 4-8; Curitiba, Brazil.[place unknown]: ISMIR; 2013. p. 493-8.. International Society for Music Information Retrieval (ISMIR), 2013.
4. Attentive Neural Architecture Incorporating Song Features For Music Recommendation
## Appendix
TODO : GANTT chart
