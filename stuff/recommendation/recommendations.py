import streamlit as st

# --------------------------
# FUNZIONI DI SIMULAZIONE / DUMMY
# --------------------------

def login_user(username, password):
    """
    Funzione dummy per il login utente.
    Sostituisci questa funzione con il controllo delle credenziali reale.
    """
    return username == "user" and password == "pass"

def generate_movies_for_rating(selected_genre):
    """
    Genera in modo dummy 4 film da mostrare nella pagina di rating
    in base al genere selezionato.
    
    In futuro:
      - Qui potresti chiamare il tuo modello VAE per generare film consigliati
        a partire dal genere.
    """
    # Se il genere è "any genre", puoi usare una stringa generica oppure uno shuffle.
    if selected_genre == "any genre":
        genre_str = "Generic"
    else:
        genre_str = selected_genre
    # Restituisce 4 film dummy con il genere indicato.
    return [f"Film {i} ({genre_str})" for i in range(1, 5)]

def process_ratings_to_get_final(user_ratings, selected_genre):
    """
    Elabora i rating dati dall’utente per restituire i 3 film migliori.
    
    In questo dummy, ordiniamo i 4 film per rating (maggiore è meglio)
    e restituiamo i primi 3.
    
    In futuro:
      - Qui integrerai il tuo modello VAE che, a partire dai rating
        e dal genere selezionato, genererà le raccomandazioni finali.
    """
    # Ordiniamo i film in base al rating in ordine decrescente.
    sorted_movies = sorted(user_ratings.items(), key=lambda x: x[1], reverse=True)
    top3 = [movie for movie, rating in sorted_movies[:3]]
    return top3

def generate_recommendations_guest(selected_genre):
    """
    Funzione dummy per il caso guest: genera i 3 film con i rating più alti
    per il genere selezionato.
    
    In futuro:
      - Qui filtrerai il dataset per genere e selezionerai quelli con i punteggi
        più alti.
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
    st.session_state.role = "user"  # "user" o "guest"
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False  # per utente "user"
if "page" not in st.session_state:
    st.session_state.page = "login"  # le pagine possibili: login, user_rating, user_rec, guest_rec
if "selected_genre" not in st.session_state:
    st.session_state.selected_genre = "any genre"  # default

# --------------------------
# GESTIONE DEL RUOLO
# --------------------------
# Sidebar per selezionare tra "user" e "guest"
selected_role = st.sidebar.radio("Scegli il ruolo", ["user", "guest"])

# Se l'utente cambia da "user" a "guest", effettuare il logout e impostare la pagina guest.
if selected_role == "guest":
    st.session_state.logged_in = False
    st.session_state.page = "guest_rec"
    st.session_state.role = "guest"
else:
    # Se passa a "user": se non era loggato, reimposta alla pagina di login.
    st.session_state.role = "user"
    if not st.session_state.logged_in:
        st.session_state.page = "login"

# --------------------------
# DEFINIZIONE DELLE PAGINE
# --------------------------

def login_page():
    st.title("🎬 Movie Recommender - Login")
    st.write("Effettua il login per accedere alla sezione user.")
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
    st.write("Seleziona il genere e valuta i 4 film consigliati per ricevere i tuoi consigli personalizzati.")
    
    # Selezione del genere: se l'utente cambia genere, vengono rigenerati i film.
    selected_genre = st.selectbox("Scegli un genere", ["any genre", "Action", "Comedy", "Drama", "Horror", "Sci-Fi"], 
                                  index=["any genre", "Action", "Comedy", "Drama", "Horror", "Sci-Fi"].index(st.session_state.selected_genre))
    st.session_state.selected_genre = selected_genre

    # Generazione (dummy) dei 4 film da valutare in base al genere selezionato.
    movies_for_rating = generate_movies_for_rating(selected_genre)
    st.session_state.movies_for_rating = movies_for_rating  # li salviamo in session_state per coerenza

    # Creiamo un dizionario per salvare i rating.
    if "user_ratings" not in st.session_state:
        st.session_state.user_ratings = {}
    # Per ogni film generato, crea uno slider per il rating.
    for movie in movies_for_rating:
        # Se il film non è già stato valutato, impostiamo un default a 3.
        default_rating = st.session_state.user_ratings.get(movie, 3)
        rating = st.slider(f"Quanto ti è piaciuto '{movie}'?", 0, 5, default_rating, key=movie)
        st.session_state.user_ratings[movie] = rating

    if st.button("Get Recommendations"):
        # Processa i rating per ottenere le raccomandazioni finali (3 film migliori).
        final_recommendations = process_ratings_to_get_final(st.session_state.user_ratings, selected_genre)
        st.session_state.final_recommendations = final_recommendations
        # Passa alla pagina delle raccomandazioni per l'utente.
        st.session_state.page = "user_rec"

def user_recommendations_page():
    st.title("🎥 User Recommendations")
    st.write("I film qui visualizzati sono frutto dei rating che hai dato.")
    
    # Visualizza i film finali (3 film) ottenuti dal processo dei rating.
    if "final_recommendations" in st.session_state:
        st.subheader("Film Consigliati:")
        for movie in st.session_state.final_recommendations:
            st.markdown(f"- {movie}")
    else:
        st.warning("Nessuna raccomandazione disponibile. Prima valuta alcuni film.")
    
    # Pulsante per tornare alla pagina dei rating (in questo caso verranno rigenerati se il genere è stato cambiato).
    if st.button("Back"):
        st.session_state.page = "user_rating"

def guest_recommendations_page():
    st.title("🎥 Guest Recommendations")
    st.write("Seleziona un genere per ricevere i consigli basati sui film con rating più alti.")
    
    guest_genre = st.selectbox("Scegli un genere", ["any genre", "Action", "Comedy", "Drama", "Horror", "Sci-Fi"])
    
    if st.button("Recommend Movies"):
        # Genera le raccomandazioni per guest (3 film con i rating più alti) in base al genere.
        recommendations = generate_recommendations_guest(guest_genre)
        st.session_state.recommendations = recommendations
    
    if "recommendations" in st.session_state:
        st.subheader("Film Consigliati:")
        for movie in st.session_state.recommendations:
            st.markdown(f"- {movie}")

# --------------------------
# GESTIONE DELLA NAVIGAZIONE DELLE PAGINE
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
            # Fallback in caso di stato incoerente: ritorno alla pagina di login.
            st.session_state.page = "login"
            login_page()
elif st.session_state.role == "guest":
    guest_recommendations_page()

# --------------------------
# COMMENTI PER FUTURI AGGIORNAMENTI:
# --------------------------
# - La funzione generate_movies_for_rating():
#     In futuro potresti:
#       1. Preprocessare i dati del genere selezionato.
#       2. Inviare questi dati al tuo modello VAE addestrato.
#       3. Ottenere come output una lista di film da presentare all'utente.
#
# - La funzione process_ratings_to_get_final():
#     Sostituirai la logica dummy con:
#       1. L'elaborazione dei rating (eventuale normalizzazione o encoding).
#       2. L'integrazione del modello VAE per generare raccomandazioni finali.
#
# - Per i dati guest in generate_recommendations_guest():
#     Quando avrai il dataset, potrai:
#       1. Filtrare i film in base al genere.
#       2. Ordinare per rating e restituire i top 3.
#
# - In fase di training del modello, potresti creare un modulo dedicato
#   (ad es. models/vae_model.py) per il preprocessing, training e predizione.
