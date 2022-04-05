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
#### Midterm Update 

##### Data Collection

### Unsupervised Learning

Genre classification: We aim to perform K-Means clustering on the selected audio features of songs. Given a set of songs and their corresponding genre label (with k genres total), we will cluster them into k clusters based on audio features in the Euclidean space. We will simply compare the distance between the audio features of new songs and the learned cluster centers to make genre predictions.

Graph-based approaches for playlist completion: We will first derive the songs-playlists bipartite graph from the dataset. Then we will explore some common graph-based methods, such as Random Walk (1), Neural Graph Collaborative Filtering (NGCF) (2), and LightGCN (3), to learn the graph embeddings of songs and playlists for later prediction/recommendation use. 

#### Midterm Update

##### Data Exploration
One of the first analysis to perform on a machine learning project is to explore the dataset. Make sure the dataset is clean. That all values are in the expected formats, decide how to handle missing values and more. 

When it comes from the features dataset recovered from the Spotify API, the dataset is very clean with no missing values. The table below is a brief description of each feature according to the Spotify documentation.

| Feature          | Type    |  Description                                                                                           |
| ---------------- | ------- | ------------------------------------------------------------------------------------------------------ |
| acousticness     | float   | A confidence measure from 0.0 to 1.0 of whether the track is acoustic                                  |
| analysis_url     | string  | A URL to access the full audio analysis of this track. An access token is required to access this data |
| danceability     | float   | Danceability describes how suitable a track is for dancing based on a combination of musical elements  |
| duration_ms      | integer | The duration of the track in milliseconds                                                              |
| energy           | float   | Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity      |
| loudness         | float   | The overall loudness of a track in decibels (dB)                                                       |
| id               | string  | The Spotify ID for the track                                                                           |
| instrumentalness | float   | Predicts whether a track contains no vocals                                                            |
| key              | integer | The key the track is in. Integers map to pitches using standard Pitch Class notation                   |
| liveness         | float   | Detects the presence of an audience in the recording                                                   |
| mode             | integer | Mode indicates the modality (major or minor) of a track                                                |
| time_signature   | integer | An estimated time signature, meter. It specifies how many beats are in each bar (measure)              |
| speechiness      | float   | Speechiness detects the presence of spoken words in a track                                            |
| tempo            | float   | The overall estimated tempo of a track in beats per minute                                             |
| track_href       | string  | A link to the Web API endpoint providing full details of the track.                                    |
| type             | string  | Allowed value: "audio_features"                                                                        |
| uri              | string  | The Spotify URI for the track                                                                          |
| valence          | float   | A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track                      |

There are some features that will be useless to our analysis because they are unique or equal to every data point : type, uri, analysis_url, id and track_href are to be removed from our dataset when performing an algorithm. That leaves us with 13 features. We use a MaxMin scaler so that all of the values are within 0 and 1.

We then proceeded to analyze the distribution of each feature and their correlation :
![distribution](https://user-images.githubusercontent.com/37664954/161640857-22656693-93c4-4046-b3f3-1bfbf5a50b5d.png)
We can see that the mode, key and time_signature are only allowed to take few discrete values. This could be a problem when performing dimensionality reduction or any clustering algorithm. 

![correlation](https://user-images.githubusercontent.com/37664954/161641329-1ecd4f35-5591-4db3-bb2d-7b6a0c8bd00b.png)

We can see that some features are positively correlated : danceability and energy, energy and tempo. This was to be expected as per the descriptions above. 

#### PCA
Data visualization is very important in any project. It allows us to better understand our raw data and also our results. PCA is a dimensionality reduction algorithm that seeks to find a new space in which the variance of our data is maximized. After having implemented the PCA algorithm, the total variance explained by the first two principal components (we choose two so that it may be visualized), only explain 0.284 and 0.247 respectively. This means that we are losing a lot of information when only using the first two components. It is not ideal, more research needs to be made. 

Nonetheless, we can observe an interesting phoenomenon when we polt our components. Since our dataset is very large (>2M songs), we chose to sample 1000 points at random for visibility purposes. 

![pca_scatter](https://user-images.githubusercontent.com/37664954/161643656-7eb34767-1c97-444d-8432-0bbccd16d58a.png)
We can see a bimodal distribution in our data and we decided to investigate it further, since the original features no longer exist in the PCA space, we decided to perform a clustering algorithm and then analyse the distribution of our data for each feature, in each cluster. 

#### K-Means
We decided to perform the K-Means algorithm on our entire dataset. Since we have a very big dataset with many dimensions and K-Means is a distance based algorithm this was quite time consuming. Since we saw during the PCA implementation that there was a bimodal distribution in our data, we performed the algorithm to find 2 clusters. 

![kmeans_feat](https://user-images.githubusercontent.com/37664954/161646877-b8ce951b-8eef-40f2-8d1e-f76d079e7d67.png)

We can see that the only feature that seems to be clearly separated by the cluster is the mode. We are still searching for ways to work around this so that we can find other clusters. Removing the feature is one option although maybe a little simplistic. 

### Supervised Learning
##### Song embeddings based on playlists

The idea of this approach is to produce embeddings for each song, and run inference with a Nearest Neighbor search, using an aggregate of the embeddings as the query to find potential new candidates to fill the playlists. Each model is trained using an Embedding layer and a network on various tasks hoping to create interesting features for the songs (the Embedding layer). Data used to train them is simply the playlists data.

Playlists used for example: #1: [1, 2, 4], #2: [3, 5, 6], #3: [7, 8]

| Model name                                                                                                                 | Task Description                                                                                              | Input example | Output example |
|----------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|---------------|----------------|
| Discriminative          | Feeding the model an input of song_ids and outputing whether each belong to a same playlist or not.      | [1, 3, 2, 7] | [1, 0, 1, 0]
| Binary discriminative                                                                                                      | Feeding the model an input of song_ids and outputing if it's a playlist or not                           | [1, 3, 7] / [1, 2, 4]   | 0 / 1
| Last song                                                                                                                  | Predicting a song_id of the playlist based on the other ones                                             |   [1,4] | 2
| Last song with aggregate                                                                                                   | Predicting a song_id of the playlist based on the other ones but here embeddings of input are aggregated |   [6,5] | 3


| Metrics                      | Discriminative<br/>model | Binary<br/> Discriminative<br/>model| Last Song model| Last song <br/>with aggregate |
|------------------------------|--------------------------|--------|
| Task accuracy                | 99.3%             |   97.2%  | 73.1% | 45.7%|
| Inference accuracy (top-20)  | 0.0%                    | 0.0%    | 0.1%| 0.0%|
| Inference accuracy (top-500) | 0.4%                     | 0.0%    | 0.9%| 0.1%|

For now, we have not yet been able to achieve good results. Part of the problem is that the feature we are trying to create is very large (there is over 2M songs) and thus embeddings created are not yet accurate enough. Further improvements will include implementing a model similar to the Nearest Neighbor algorithm and try to make use of RNN layers.  

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
