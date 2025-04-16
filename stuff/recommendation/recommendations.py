import streamlit as st
from PIL import Image

# --------------------------
# FUNZIONI DI SIMULAZIONE / DUMMY
# --------------------------

def login_user(username, password):
    """
    Funzione dummy per il login utente.
    In futuro, sostituisci questa funzione con il controllo delle credenziali sul tuo database o sistema di autenticazione.
    """
    return username == "user" and password == "pass"

def generate_recommendations_VAE(user_ratings, selected_genre):
    """
    Funzione dummy per generare raccomandazioni usando il VAE.
    
    Parametri:
      - user_ratings: dizionario con i rating dei film.
      - selected_genre: genere selezionato dalla select.
    
    In futuro:
      - Qui dovrai processare i rating e il genere in input,
      - Passarli al tuo modello VAE addestrato,
      - E ottenere una lista di film consigliati.
    """
    # Esempio con dati fittizi (dummy).
    return [f"Film VAE 1 ({selected_genre})", f"Film VAE 2 ({selected_genre})", f"Film VAE 3 ({selected_genre})"]

def generate_recommendations_guest(selected_genre):
    """
    Funzione dummy per generare raccomandazioni per gli ospiti.
    
    In futuro:
      - Questa funzione dovrà prendere in considerazione il genere selezionato
      - E restituire i 3 film con i rating più alti per quel genere.
    """
    # Dati dummy per esempio.
    return [f"Film Top 1 ({selected_genre})", f"Film Top 2 ({selected_genre})", f"Film Top 3 ({selected_genre})"]

# --------------------------
# SETTAGGIO INIZIALE DELLO STATO
# --------------------------
if "role" not in st.session_state:
    st.session_state.role = "user"  # ruolo corrente: "user" o "guest"
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False  # per utente "user"
if "page" not in st.session_state:
    st.session_state.page = "login"  # pagine: login, user_rating, user_rec, guest_rec

# --------------------------
# GESTIONE DELLE VARIAZIONI DI RUOLO
# --------------------------
# Sidebar per selezionare il ruolo: "user" o "guest"
selected_role = st.sidebar.radio("Scegli il ruolo", ["user", "guest"])

# Se l'utente era loggato come "user" ma cambia a "guest", allora esegui logout.
if selected_role == "guest":
    st.session_state.logged_in = False
    st.session_state.page = "guest_rec"
    st.session_state.role = "guest"
else:
    # Se viene selezionato "user" mentre era precedentemente ospite, resetta a login se non già loggato.
    st.session_state.role = "user"
    if not st.session_state.logged_in:
        st.session_state.page = "login"

# --------------------------
# DEFINIZIONE DELLE PAGINE
# --------------------------

def login_page():
    st.title("🎬 Movie Recommender - Login")
    st.write("Effettua il login per entrare nella sezione user.")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        if login_user(username, password):
            st.session_state.logged_in = True
            st.session_state.page = "user_rating"
            st.success("Login effettuato con successo!")
        else:
            st.error("Credenziali errate. Riprova.")

def user_rating_page():
    st.title("⭐ Rate Some Movies")
    st.write("Valuta i 4 film (da 0 a 5) per ricevere consigli personalizzati.")
    
    # Dummy films: in futuro potresti iterare su un dataset di film
    movies = ["Inception", "The Matrix", "Interstellar", "Avatar"]
    user_ratings = {}
    for movie in movies:
        rating = st.slider(f"Quanto ti è piaciuto '{movie}'?", 0, 5, 3)
        user_ratings[movie] = rating
    
    if st.button("Get Recommendations"):
        # Salva i rating in session_state se necessario
        st.session_state.user_ratings = user_ratings
        # Passa alla pagina delle raccomandazioni per l'utente
        st.session_state.page = "user_rec"

def user_recommendations_page():
    st.title("🎥 User Recommendations")
    st.write("Seleziona un genere per affinare i consigli.")
    
    # Select per scegliere genere; include opzione "any genre"
    genre = st.selectbox("Scegli un genere", ["any genre", "Action", "Comedy", "Drama", "Horror", "Sci-Fi"])
    
    if st.button("Recommend Movies"):
        # Genera raccomandazioni utilizzando il VAE (dummy per ora)
        recommendations = generate_recommendations_VAE(st.session_state.user_ratings, genre)
        st.session_state.recommendations = recommendations
    
    # Visualizza le raccomandazioni se disponibili
    if "recommendations" in st.session_state:
        st.subheader("Film Consigliati:")
        for movie in st.session_state.recommendations:
            st.markdown(f"- {movie}")
    
    # Pulsante per tornare alla pagina di rating
    if st.button("Back"):
        st.session_state.page = "user_rating"

def guest_recommendations_page():
    st.title("🎥 Guest Recommendations")
    st.write("Seleziona un genere per ricevere i consigli in base ai film con rating più alti.")
    
    genre = st.selectbox("Scegli un genere", ["any genre", "Action", "Comedy", "Drama", "Horror", "Sci-Fi"])
    
    if st.button("Recommend Movies"):
        # Genera raccomandazioni per ospiti basandosi sui film con rating più alti (dummy per ora)
        recommendations = generate_recommendations_guest(genre)
        st.session_state.recommendations = recommendations
    
    if "recommendations" in st.session_state:
        st.subheader("Film Consigliati:")
        for movie in st.session_state.recommendations:
            st.markdown(f"- {movie}")

# --------------------------
# GESTIONE DI NAVIGAZIONE DELLE PAGINE
# --------------------------
if st.session_state.role == "user":
    if not st.session_state.logged_in:
        login_page()
    else:
        if st.session_state.page == "user_rating":
            user_rating_page()
        elif st.session_state.page == "user_rec":
            user_recommendations_page()
        else:
            # Fallback: torna a login se lo stato non è coerente
            st.session_state.page = "login"
            login_page()
elif st.session_state.role == "guest":
    guest_recommendations_page()

# --------------------------
# COMMENTI PER FUTURI AGGIORNAMENTI:
# --------------------------
# - Nella funzione generate_recommendations_VAE:
#   Dovrai implementare la logica per:
#     1. Preprocessare i rating inseriti dall'utente (ad esempio normalizzarli o applicare encoding).
#     2. Passare questi dati al tuo modello di VAE per ottenere una rappresentazione latente.
#     3. Generare le raccomandazioni a partire dalla rappresentazione latente.
#
# - Nella funzione generate_recommendations_guest:
#   Dovrai invece implementare la logica per:
#     1. Filtrare il dataset dei film in base al genere selezionato.
#     2. Ordinarli per rating e prendere i 3 film con il punteggio più alto.
#
# - Quando implementerai il training del modello e avrai il dataset
#   potresti avere un modulo dedicato (es. models/vae_model.py) che contiene
#   funzioni per il preprocessing, addestramento e predizione del modello.
#
# - Eventuali immagini o locandine possono essere aggiunte utilizzando st.image()
#   nel rendering delle raccomandazioni.
