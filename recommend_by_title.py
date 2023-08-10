import urllib
import zipfile
import time

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from sklearn.metrics.pairwise import cosine_similarity

def create_df():
    raw_df = pd.read_csv('/Users/cindychang/Desktop/aipi540/Nintendo_Switch_Game_Recommendation/switch-games.csv')
    clean_df = raw_df.loc[:,['id','title','developer','genre','rating']].dropna()
    final_df = clean_df[clean_df.rating != 0.0]
    final_df['content'] = final_df['title'].astype(str) + ' ' + final_df['developer'].astype(str) + ' ' + final_df['genre'] + ' ' + final_df['rating'].astype(str) + ' '
    final_df['content'] = final_df['content'].fillna('')
    final_df['tokenized_content'] = final_df['content'].apply(simple_preprocess)
    return final_df


final_df = create_df()
    
model = Word2Vec(vector_size=100, window=5, min_count=1, workers=4)
model.build_vocab(final_df['tokenized_content'])
model.train(final_df['tokenized_content'], total_examples=model.corpus_count, epochs=10)

def average_word_vectors(words, model, vocabulary, num_features):
    feature_vector = np.zeros((num_features,), dtype="float64")
    nwords = 0.

    for word in words:
        if word in vocabulary:
            nwords = nwords + 1.
            feature_vector = np.add(feature_vector, model.wv[word])

    if nwords:
        feature_vector = np.divide(feature_vector, nwords)

    return feature_vector

def averaged_word_vectorizer(corpus, model, num_features):
    vocabulary = set(model.wv.index_to_key)
    features = [average_word_vectors(tokenized_sentence, model, vocabulary, num_features) for tokenized_sentence in corpus]
    return np.array(features)

def get_recommend():
    final_df = create_df()
    # Compute average word vectors for all games
    w2v_feature_array = averaged_word_vectorizer(corpus=final_df['tokenized_content'], model=model, num_features=100)

    # Get the user input
    from streamlit import session_state as session
    user_game = session.options

    # Find the index of the user movie
    game_index = final_df.index.values[final_df['title'] == user_game]

    # Compute the cosine similarities between the user movie and all other movies
    user_game_vector = w2v_feature_array[game_index].reshape(1, -1)
    similarity_scores = cosine_similarity(user_game_vector, w2v_feature_array)

    # Get the top 5 most similar games
    similar_games = list(enumerate(similarity_scores[0]))
    sorted_similar_games = sorted(similar_games, key=lambda x: x[1], reverse=True)[1:6]

    dict = {"game_idx":[],"game_title":[]}
    for i, score in sorted_similar_games:
        dict["game_idx"].append(i)
        dict["game_title"].append(final_df.loc[i, 'title'])
    predict_df = pd.DataFrame.from_dict(dict)
    return predict_df