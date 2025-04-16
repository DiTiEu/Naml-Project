import streamlit as st
from PIL import Image  # Per mostrare locandine (opzionale)

# Simulazione di funzioni backend
def login_user(username, password):
    return username == "user" and password == "pass"

def generate_recommendations_from_ratings(user_ratings):
    return ["Inception", "The Matrix", "Interstellar"]

def generate_recommendations_from_genres(genres):
    return ["Avengers", "Guardians of the Galaxy", "Iron Man"]

# Stato iniziale dell'app
if "page" not in st.session_state:
    st.session_state.page = "login"
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# Sidebar per navigare se loggato
def show_sidebar():
    st.sidebar.title("Menu")
    page = st.sidebar.radio("Vai a", ["Rating", "Guest", "Consigli"])
    st.session_state.page = page

# Login Page
def login_page():
    st.title("🎬 Movie Recommender Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if login_user(username, password):
            st.session_state.logged_in = True
            st.session_state.page = "Rating"
            st.success("Login effettuato!")
        else:
            st.error("Credenziali errate")

# Pagina per valutare film
def rating_page():
    st.title("⭐ Valuta i film che hai visto")
    user_ratings = {}
    for movie in ["Inception", "The Matrix", "Interstellar"]:
        rating = st.slider(f"Quanto ti è piaciuto '{movie}'?", 0, 5, 3)
        user_ratings[movie] = rating

    if st.button("Genera consigli basati sulle valutazioni"):
        st.session_state.recommendations = generate_recommendations_from_ratings(user_ratings)
        st.session_state.page = "Consigli"

# Pagina per utenti ospiti (preferenze di genere)
def guest_page():
    st.title("🎯 Seleziona i tuoi generi preferiti")
    genres = st.multiselect("Generi", ["Action", "Comedy", "Drama", "Horror", "Sci-Fi"])
    if st.button("Genera consigli basati sui generi"):
        st.session_state.recommendations = generate_recommendations_from_genres(genres)
        st.session_state.page = "Consigli"

# Pagina per mostrare le raccomandazioni
def recommendation_page():
    st.title("🎥 Film Consigliati")
    for movie in st.session_state.recommendations:
        st.subheader(movie)
        # st.image("path/to/image.jpg")  # Se vuoi mostrare la locandina

# Main
if st.session_state.logged_in:
    show_sidebar()

if st.session_state.page == "login":
    login_page()
elif st.session_state.page == "Rating":
    rating_page()
elif st.session_state.page == "Guest":
    guest_page()
elif st.session_state.page == "Consigli":
    recommendation_page()
