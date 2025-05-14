import os
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
import sys
import os
import json

SEEN_MOVIES_FILE = "data/seen_movies.json"

def load_seen_movies():
    if not os.path.exists(SEEN_MOVIES_FILE):
        return {}
    with open(SEEN_MOVIES_FILE, "r") as f:
        return json.load(f)

def save_seen_movies(seen_movies_dict):
    with open(SEEN_MOVIES_FILE, "w") as f:
        json.dump(seen_movies_dict, f)

def get_seen_movies_for_user(user_id):
    seen_movies_dict = load_seen_movies()
    return set(seen_movies_dict.get(str(user_id), []))

def add_seen_movies_for_user(user_id, new_seen_movies):
    seen_movies_dict = load_seen_movies()
    user_key = str(user_id)
    seen_movies = set(seen_movies_dict.get(user_key, []))
    seen_movies.update(new_seen_movies)
    seen_movies_dict[user_key] = list(seen_movies)
    save_seen_movies(seen_movies_dict)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

data_path = os.path.join("data", "cleaned", "ratings_clean.csv")
users_df = pd.read_csv(data_path) 

movies_df = pd.read_csv("data/cleaned/items_clean.csv")

ratings = pd.read_csv("data/cleaned/ratings_clean.csv")
merged_df = pd.merge(ratings, movies_df, on="item_id")

average_ratings = merged_df.groupby("item_id")["rating"].mean().reset_index()
average_ratings.columns = ["item_id", "avg_rating"]

average_ratings = average_ratings.merge(movies_df[["item_id", "movie_title"] + list(movies_df.columns[5:])], on="item_id")

@st.cache_resource
def load_vae_model():
    from tensorflow.keras.utils import get_custom_objects # type: ignore
    from model.vae_model import CustomVAE
    get_custom_objects().update({"CustomVAE": CustomVAE})
    loaded_model = tf.keras.models.load_model("saved_models/vae_model.keras", custom_objects={"CustomVAE": CustomVAE}, compile=False)
    return loaded_model

@st.cache_resource
def load_vae_model():
    from tensorflow.keras.utils import get_custom_objects  # type: ignore
    from model.vae_model import CustomVAE
    from model.vae_architecture import sampling  # importa anche la funzione se è in un altro file

    get_custom_objects().update({
        "CustomVAE": CustomVAE,
        "vae_arch>sampling_fn": sampling
    })

    loaded_model = tf.keras.models.load_model(
        "saved_models/vae_model.keras",
        custom_objects={"CustomVAE": CustomVAE, "vae_arch>sampling_fn": sampling},
        compile=False
    )
    return loaded_model

def login_user(user_id, password):
    if password != "pass":
        return False
    try:
        user_id = int(user_id)
    except ValueError:
        return False
    return user_id in users_df["user_id"].values

def generate_initial_movies():
    return ["Film 1", "Film 2", "Film 3", "Film 4"]

def generate_recommendations_VAE(user_id):
    user_vector = ratings_matrix[user_id - 1]

    predicted_ratings = vae.predict(user_vector[np.newaxis, :])[0]
    predicted_ratings[predicted_ratings == 0] = -0.25
    predicted_ratings = predicted_ratings * 4 + 1

    seen_mask = user_vector > 0
    predicted_ratings[seen_mask] = -np.inf

    seen_movies_persistent = get_seen_movies_for_user(user_id)
    for movie_id in seen_movies_persistent:
        predicted_ratings[int(movie_id)] = -np.inf

    top_movie_ids = np.argsort(predicted_ratings)[-4:][::-1]

    add_seen_movies_for_user(user_id, top_movie_ids.tolist())

    recommended_titles = [movie_id_to_title[index_to_movie_id[mid]] for mid in top_movie_ids]

    return recommended_titles

def generate_recommendations_guest(selected_genre):
    if selected_genre == "Any genre":
        top_movies = average_ratings.sort_values("avg_rating", ascending=False).head(4)
    else:
        genre_filtered = average_ratings[average_ratings[selected_genre] == 1]
        top_movies = genre_filtered.sort_values("avg_rating", ascending=False).head(4)

    return list(top_movies["movie_title"])

if "role" not in st.session_state:
    st.session_state.role = "User" 
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "page" not in st.session_state:
    st.session_state.page = "login"
if "current_movies" not in st.session_state:
    st.session_state.current_movies = generate_initial_movies()
if "user_ratings" not in st.session_state:
    st.session_state.user_ratings = {}
if "guest_genre" not in st.session_state:
    st.session_state.guest_genre = "Any genre"

selected_role = st.sidebar.radio("Choose the role", ["User", "Guest"])

if selected_role == "Guest":
    st.session_state.logged_in = False
    st.session_state.page = "guest_rec"
    st.session_state.role = "Guest"
else:
    st.session_state.role = "User"
    if not st.session_state.logged_in:
        st.session_state.page = "login"

def login_page():
    st.title("MyNextMovie - Login")
    st.write("Insert your credentials to log in.")
    user_id = st.text_input("User ID")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        if login_user(user_id, password):
            st.session_state.logged_in = True
            st.session_state.user_id = int(user_id)
            st.session_state.current_movies = generate_recommendations_VAE(int(user_id))
            st.success("Login successfully!")
            st.session_state.page = "rating"
        else:
            st.error("Wrong credentials. Please try again.")

def rating_page():
    st.title("Rate the Movies")
    st.write("Rate the 4 movies you see and press 'Get Recommendations' to receive new recommendations based on your ratings.")
    
    for movie in st.session_state.current_movies:
        default_rating = st.session_state.user_ratings.get(movie, 3)
        rating = st.slider(f"How much did you enjoy '{movie}'?", 0, 5, default_rating, key=movie)
        st.session_state.user_ratings[movie] = rating
    
    if st.button("Get Recommendations"):
        user_id = st.session_state.user_id
        new_movies = generate_recommendations_VAE(user_id)
        st.session_state.current_movies = new_movies
        st.success("New recommendations generated!")
        st.rerun()

def guest_recommendations_page():
    st.title("Guest Recommendations")
    st.write("Select a genre to receive recommendations based on movies with the highest ratings.")
    
    guest_genre = st.selectbox("Select a genre", ["Any genre", "Action", "Comedy", "Drama", "Horror", "Sci-Fi"],
                               index=["Any genre", "Action", "Comedy", "Drama", "Horror", "Sci-Fi"].index(st.session_state.guest_genre))
    st.session_state.guest_genre = guest_genre
    
    if st.button("Recommend Movies"):
        recommendations = generate_recommendations_guest(guest_genre)
        st.session_state.guest_recommendations = recommendations
    
    if "guest_recommendations" in st.session_state:
        st.subheader("Recommended films:")
        for movie in st.session_state.guest_recommendations:
            st.markdown(f"- {movie}")

data_path = os.path.join("data", "cleaned", "ratings_clean.csv")
ratings = pd.read_csv(data_path)

user_item_matrix = ratings.pivot_table(index="user_id", columns="item_id", values="rating", fill_value=0)

index_to_movie_id = list(user_item_matrix.columns)

ratings_matrix = user_item_matrix.to_numpy().astype("float32")

movies_df = pd.read_csv("data/cleaned/items_clean.csv")
movie_id_to_title = dict(zip(movies_df["item_id"], movies_df["movie_title"]))
NUM_MOVIES = len(movies_df)

if st.session_state.role == "User":
    print("Caricamento del modello VAE...")
    vae = load_vae_model()
    print("Modello VAE caricato con successo.")
    if not st.session_state.logged_in:
        login_page()
    else:
        rating_page()
elif st.session_state.role == "Guest":
    guest_recommendations_page()