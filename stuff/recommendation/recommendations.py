import streamlit as st

# --------------------------
# FUNZIONI DI SIMULAZIONE / DUMMY
# --------------------------

def login_user(username, password):
    """
    Funzione dummy per il login utente.
    In futuro sostituirai questa funzione con il controllo delle credenziali reale,
    es. confrontando username e password con un database.
    """
    return username == "user" and password == "pass"

def generate_initial_movies():
    """
    Genera in modo dummy la lista iniziale di 4 film da mostrare nella pagina di rating.
    In una versione futura potresti chiamare il tuo modello VAE addestrato
    per ottenere dei suggerimenti iniziali.
    """
    return ["Film 1", "Film 2", "Film 3", "Film 4"]

def generate_recommendations_VAE(user_ratings):
    """
    Funzione dummy che simula il comportamento del VAE.
    
    Parametri:
      - user_ratings: dizionario in cui le chiavi sono i titoli dei film visualizzati e
                      i valori sono i rating assegnati dall'utente.
    
    In futuro:
      - Qui dovrai preprocessare i rating (eventuale normalizzazione o encoding),
      - Passare questi dati al tuo modello VAE addestrato per ottenere una rappresentazione latente
      - E generare 4 nuovi film consigliati in base ai rating appena forniti.
    """
    # Per la demo, restituisce 4 nuovi film basati su una logica dummy.
    return [f"Film VAE {i}" for i in range(1, 5)]

def generate_recommendations_guest(selected_genre):
    """
    Funzione dummy per generare le raccomandazioni per gli ospiti.
    
    In futuro:
      - Potrai filtrare il dataset in base al genere selezionato e
        selezionare i 3 film con i rating più alti.
    """
    if selected_genre == "any genre":
        genre_str = "Generic"
    else:
        genre_str = selected_genre
    # Restituisce 3 film dummy.
    return [f"Film Top {i} ({genre_str})" for i in range(1, 4)]

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
    st.write("Effettua il login per ricevere consigli personalizzati.")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if login_user(username, password):
            st.session_state.logged_in = True
            st.success("Login effettuato con successo!")
            # Dopo il login, si passa direttamente alla pagina di rating
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
        # Passa i rating alla funzione dummy del VAE e aggiorna i film suggeriti
        new_movies = generate_recommendations_VAE(st.session_state.user_ratings)
        st.session_state.current_movies = new_movies
        # Reset dei rating per i nuovi film
        st.session_state.user_ratings = {}
        st.success("Nuove raccomandazioni generate!")
        st.experimental_rerun()

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
if st.session_state.role == "user":
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
