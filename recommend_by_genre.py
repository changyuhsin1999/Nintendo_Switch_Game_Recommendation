import os
import urllib
import zipfile

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

def genre_get_data():
    raw_df = pd.read_csv('/Users/cindychang/Desktop/aipi540/Nintendo_Switch_Game_Recommendation/switch-games.csv')
    clean_df = raw_df.loc[:,['id','title','game_url','developer','genre','rating']].dropna()
    final_df = clean_df[clean_df.rating != 0.0]
    return final_df
    
df = genre_get_data()
    
def genre_vectorizer():
    # create an object for TfidfVectorizer
    tfidf_vector = TfidfVectorizer(stop_words='english')
    # apply the object to the genres column
    tfidf_matrix = tfidf_vector.fit_transform(df['genre'])
    return tfidf_matrix

    # create the cosine similarity matrix
tfidf = genre_vectorizer()
sim_matrix = linear_kernel(tfidf,tfidf)
indicies = pd.Series(df.index, index = df['title']).drop_duplicates()
    
def get_recommendations_by_genre(title, cosine_sim = sim_matrix):
    idx = indicies[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key = lambda x:x[1], reverse = True)
    sim_scores = sim_scores[1:11]
    game_indicies = [i[0] for i in sim_scores]
    return df['title'].iloc[game_indicies]
