#streamlit run stuff/recommendation/recommendations.py

import os
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
import sys
import os

# Aggiungi la directory principale del progetto al percorso di ricerca
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# Caricamento del dataset utenti
data_path = os.path.join("data", "cleaned", "ratings_clean.csv")
users_df = pd.read_csv(data_path)  # Aggiorna il path se necessario

# Caricamento dataset dei film
movies_df = pd.read_csv("data/cleaned/items_clean.csv")  # Colonne: item_id, movie_title, ..., generi binari

# Merge con ratings
ratings = pd.read_csv("data/cleaned/ratings_clean.csv")  # Colonne: user_id, item_id, rating, timestamp
merged_df = pd.merge(ratings, movies_df, on="item_id")

# Calcolo del rating medio per ogni film
average_ratings = merged_df.groupby("item_id")["rating"].mean().reset_index()
average_ratings.columns = ["item_id", "avg_rating"]

# Aggiunta del titolo e generi al dataframe dei rating medi
average_ratings = average_ratings.merge(movies_df[["item_id", "movie_title"] + list(movies_df.columns[5:])], on="item_id")

# Carica il modello VAE salvato (una volta sola, fuori dalla funzione per efficienza)
@st.cache_resource
def load_vae_model():
    # Se hai usato una classe custom, registrala prima qui
    from tensorflow.keras.utils import get_custom_objects # type: ignore
    from model.vae_model import CustomVAE
    get_custom_objects().update({"VAE": CustomVAE})
    loaded_model = tf.keras.models.load_model("saved_models/vae_model.keras", custom_objects={"VAE": CustomVAE}, compile=False)
    return loaded_model

# --------------------------
# FUNZIONI DI SIMULAZIONE / DUMMY
# --------------------------

def login_user(user_id, password):
    """
    Verifica se l'user_id esiste nel dataset e la password è corretta.
    """
    if password != "pass":
        return False
    try:
        user_id = int(user_id)  # Assicura che sia un numero
    except ValueError:
        return False
    return user_id in users_df["user_id"].values

def generate_initial_movies():
    """
    Genera in modo dummy la lista iniziale di 4 film da mostrare nella pagina di rating.
    In una versione futura potresti chiamare il tuo modello VAE addestrato
    per ottenere dei suggerimenti iniziali.
    """
    return ["Film 1", "Film 2", "Film 3", "Film 4"]

def generate_recommendations_VAE(user_id):
    """
    Genera raccomandazioni personalizzate per un utente usando il modello VAE.
    """
    # 1. Estrai il vettore rating per l'utente
    user_vector = ratings_matrix[user_id - 1]  # -1 se user_id parte da 1

    # 2. Predici i rating
    predicted_ratings = vae.predict(user_vector[np.newaxis, :])[0]  # shape: (num_movies,)
    predicted_ratings[predicted_ratings == 0] = -0.25
    predicted_ratings = predicted_ratings * 4 + 1

    # 3. Maschera i film già visti (con rating > 0)
    seen_mask = user_vector > 0
    predicted_ratings[seen_mask] = -np.inf

    # 4. Seleziona i 4 film con i rating previsti più alti
    top_movie_ids = np.argsort(predicted_ratings)[-4:][::-1]
    recommended_titles = [movie_id_to_title[mid] for mid in top_movie_ids]

    return recommended_titles

def generate_recommendations_guest(selected_genre):
    """
    Ritorna i 4 film col rating medio più alto per il genere selezionato.
    """
    if selected_genre == "any genre":
        top_movies = average_ratings.sort_values("avg_rating", ascending=False).head(4)
    else:
        # Filtra solo i film che hanno il flag 1 per il genere selezionato
        genre_filtered = average_ratings[average_ratings[selected_genre] == 1]
        top_movies = genre_filtered.sort_values("avg_rating", ascending=False).head(4)

    return list(top_movies["movie_title"])

# --------------------------
# SETTAGGIO INIZIALE DELLO STATO
# --------------------------
if "role" not in st.session_state:
    st.session_state.role = "user"  # Valori possibili: "user" o "guest"
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False  # Per utente "user"
if "page" not in st.session_state:
    st.session_state.page = "login"  # Pagine per user: "login" e "rating"
if "current_movies" not in st.session_state:
    st.session_state.current_movies = generate_initial_movies()  # I film da visualizzare in rating
if "user_ratings" not in st.session_state:
    st.session_state.user_ratings = {}  # Salva i rating inseriti dall'utente
if "guest_genre" not in st.session_state:
    st.session_state.guest_genre = "any genre"

# --------------------------
# GESTIONE DEL RUOLO CON SIDEBAR
# --------------------------
selected_role = st.sidebar.radio("Scegli il ruolo", ["user", "guest"])

# Se l'utente passa da "user" a "guest": esegui il logout
if selected_role == "guest":
    st.session_state.logged_in = False
    st.session_state.page = "guest_rec"
    st.session_state.role = "guest"
else:
    st.session_state.role = "user"
    # Se l'utente non è loggato, forza la pagina di login
    if not st.session_state.logged_in:
        st.session_state.page = "login"

# --------------------------
# DEFINIZIONE DELLE PAGINE PER USER
# --------------------------

def login_page():
    st.title("🎬 Movie Recommender - Login")
    st.write("Inserisci il tuo ID utente per ricevere consigli personalizzati.")
    user_id = st.text_input("User ID")  # Cambiato da username a user_id
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        if login_user(user_id, password):
            st.session_state.logged_in = True
            st.session_state.user_id = int(user_id)  # Salva user_id nella sessione
            st.success("Login effettuato con successo!")
            st.session_state.page = "rating"
        else:
            st.error("Credenziali errate. Riprova.")

def rating_page():
    st.title("⭐ Rate the Movies")
    st.write("Valuta i 4 film che vedi e premi 'Get Recommendations' per ricevere nuovi suggerimenti basati sui tuoi rating.")
    
    # Visualizza i film attualmente suggeriti dal (dummy) VAE
    for movie in st.session_state.current_movies:
        # Se il film non è già stato valutato, imposta un default a 3
        default_rating = st.session_state.user_ratings.get(movie, 3)
        rating = st.slider(f"Quanto ti è piaciuto '{movie}'?", 0, 5, default_rating, key=movie)
        st.session_state.user_ratings[movie] = rating
    
    if st.button("Get Recommendations"):
        user_id = st.session_state.user_id
        new_movies = generate_recommendations_VAE(user_id)
        st.session_state.current_movies = new_movies
        st.success("Nuove raccomandazioni generate!")
        st.rerun()

# --------------------------
# DEFINIZIONE DELLA PAGINA PER GUEST
# --------------------------
def guest_recommendations_page():
    st.title("🎥 Guest Recommendations")
    st.write("Seleziona un genere per ricevere i consigli basati sui film con i rating più alti.")
    
    guest_genre = st.selectbox("Scegli un genere", ["any genre", "Action", "Comedy", "Drama", "Horror", "Sci-Fi"],
                               index=["any genre", "Action", "Comedy", "Drama", "Horror", "Sci-Fi"].index(st.session_state.guest_genre))
    st.session_state.guest_genre = guest_genre
    
    if st.button("Recommend Movies"):
        recommendations = generate_recommendations_guest(guest_genre)
        st.session_state.guest_recommendations = recommendations
    
    if "guest_recommendations" in st.session_state:
        st.subheader("Film Consigliati:")
        for movie in st.session_state.guest_recommendations:
            st.markdown(f"- {movie}")

# --------------------------
# GESTIONE DELLE PAGINE PRINCIPALI
# --------------------------

# Carica la matrice dei rating
data_path = os.path.join("data", "cleaned", "ratings_clean.csv")
ratings = pd.read_csv(data_path)
# Costruisci la matrice utente-film
user_item_matrix = ratings.pivot_table(index="user_id", columns="item_id", values="rating", fill_value=0)
# Converti in numpy array
ratings_matrix = user_item_matrix.to_numpy().astype("float32")

# Mappature: movie_id → title
movies_df = pd.read_csv("data/cleaned/items_clean.csv")  # Deve avere colonne movie_id, title
movie_id_to_title = dict(zip(movies_df["item_id"], movies_df["movie_title"]))
NUM_MOVIES = len(movies_df)

if st.session_state.role == "user":
    print("Caricamento del modello VAE...")
    vae = load_vae_model()
    print("Modello VAE caricato con successo.")
    if not st.session_state.logged_in:
        login_page()
    else:
        rating_page()
elif st.session_state.role == "guest":
    guest_recommendations_page()

# --------------------------
# COMMENTI PER FUTURI AGGIORNAMENTI:
# --------------------------
# - Nella funzione generate_recommendations_VAE():
#     Quando implementerai il VAE:
#       1. Preprocessa i rating degli utenti (normalizzazione, encoding, ecc.).
#       2. Passa i dati pre-elaborati al modello VAE addestrato.
#       3. Ottieni in output 4 film consigliati basati sulla rappresentazione latente.
#
# - La funzione generate_initial_movies():
#     Potrà essere aggiornata per mostrare dei suggerimenti iniziali basati su logiche di default o popolari.
#
# - Per la parte guest (generate_recommendations_guest):
#     Quando avrai il dataset, potrai filtrare i film in base al genere e restituire quelli con i rating più alti.
#
# - Potrai aggiungere immagini (locandine) usando st.image() accanto ai titoli.
# - l'uso di st.experimental_rerun() serve per aggiornare la pagina e mostrare i nuovi film appena generati.
