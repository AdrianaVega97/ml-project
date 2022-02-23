# Machine Learning Semester Project : ADD TITLE
Spring 2022 \\
Group Members: Sichen Jin, Jiachen Ren, Antoine Rollet, Adriana Vega Fernández, Robert Ward

## Introduction & Background 
Spotify has since become the most popular streaming music service, boasting more than 180M paid subscribers. How did Spotify become the leader of this highly competitive market?  The app design is centered around playlists, playlists based on your listening habits (what you like, share, save, skip) and the listening habits of others with similar taste. The user experience is personalized with algorithmically curated playlists like Discover Weekly and Release Radar and is now the base of their business model. 

## Problem Definition
The best playlists contain a variety of genres and moods. A good playlist is subjective; it also depends on personal taste. We aim to build a recommendation system that would recommend songs to complete a playlist. Formally put, it means that given a set of songs features, we will output a set of songs represented as their IDs. 
We define the problem as automatic playlist completion. Our ground truth will be the playlists provided by the Spotify Millon Playlist Dataset. Our goal will be to accurately complete them in the same way that the Text Completion task is defined in NLP.

## Methods
### Dataset
### Unsupervised Learning
Basic clustering : We aim to perform basic clustering algorithms like K-Means or DBSCAN on the audio features of the dataset provided by Spotify. \\
Graph-based approaches : We will first derive the songs-playlists bipartite graph from the dataset. Then we will explore some common graph-based methods, such as Random Walk (1), Neural Graph Collaborative Filtering (NGCF) and LightGCN (2), to learn the graph embeddings of songs and playlists for later prediction/recommendation use. 
### Supervised Learning
## Conclusion
## References
## Appendix
TODO : GANTT chart
