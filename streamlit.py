import pickle
import pandas as pd
import streamlit as st
from streamlit import session_state as session
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from recommend_by_title import create_df, average_word_vectors, averaged_word_vectorizer, get_recommend


def load_data():
    """
    load and cache data
    :return: game data
    """
    
    game_data = pd.read_pickle("/Users/cindychang/Desktop/aipi540/Nintendo_Switch_Game_Recommendation/game_rating.pkl")
    game_data = game_data['title']
    return game_data

st.title("""Nintendo Switch Recommendation System""")

game_list = load_data().values
session.options = st.selectbox(label="Type or select game from the list", options=game_list)

final_df = create_df()
    
model = Word2Vec(vector_size=100, window=5, min_count=1, workers=4)
model.build_vocab(final_df['tokenized_content'])
model.train(final_df['tokenized_content'], total_examples=model.corpus_count, epochs=10)

#with open("./data/movie_list.pickle", "rb") as f:
    #movies = pickle.load(f)

dataframe = None

buffer1, col1, buffer2 = st.columns([1.45, 1, 1])

if st.button('Recommend'):

    st.write(get_recommend())

st.text("")
st.text("")
st.text("")
st.text("")

if dataframe is not None:
    st.table(dataframe)