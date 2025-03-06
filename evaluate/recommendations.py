"""
Movie Recommender System - Script di Valutazione e Raccomandazioni
-------------------------------------------------------------------
Questo script carica il modello VAE addestrato, il dataset dei rating puliti e il file degli item,
e per alcuni utenti casuali stampa una tabella con i film consigliati (top-N) in base al rating predetto.
I rating vengono riconvertiti alla scala originale [1,5] per rendere l'output comprensibile.

Struttura del progetto:
    movie-recommendation-system/
    ├── data/
    │   ├── cleaned/
    │   │   └── ratings_clean.csv     <-- Dataset pulito dei rating (non normalizzato)
    │   └── ml-100k/
    │       └── u.item                <-- File con le informazioni sui film
    ├── models/
    │   ├── __init__.py               <-- File vuoto per definire il package
    │   └── vae_model.py              <-- Modulo con la definizione del modello VAE (incluso beta se necessario)
    ├── evaluate/
    │   └── recommendations.py       <-- Questo file: script per generare le raccomandazioni
    └── train/
        └── vae_train.py             <-- Script di training (già esistente)
"""

import os
import pandas as pd
import numpy as np
import random

import tensorflow as tf
from tensorflow.keras.models import load_model

# Importa la funzione per creare il modello VAE dal modulo models
from model.vae_model import create_vae_architecture

# ------------------------------
# Caricamento dei Dati
# ------------------------------

# Percorso del file dei rating puliti
cleaned_ratings_path = os.path.join("data", "cleaned", "ratings_clean.csv")

# Carica i rating puliti (supponiamo che il file contenga le colonne: user_id, item_id, rating, ...)
ratings_df = pd.read_csv(cleaned_ratings_path)
print("Dati dei rating caricati. Shape:", ratings_df.shape)

# Creazione della matrice utente-item:
# - Pivot: righe = user_id, colonne = item_id, valori = rating
# - I valori mancanti vengono riempiti con 0.
user_item_matrix = ratings_df.pivot(index='user_id', columns='item_id', values='rating').fillna(0)
print("Matrice utente-item creata. Shape:", user_item_matrix.shape)

# Normalizza i rating da [1,5] a [0,1] per essere compatibili con il modello (sigmoid in output)
user_item_matrix_normalized = (user_item_matrix - 1) / 4

# Converti la matrice normalizzata in un array NumPy
data = user_item_matrix_normalized.values.astype('float32')

# Numero di film (item)
n_items = data.shape[1]
print("Numero di film (item):", n_items)

# ------------------------------
# Caricamento delle Informazioni sui Film
# ------------------------------

# Percorso del file degli item (MovieLens u.item)
items_file = os.path.join("data", "ml-100k", "u.item")

# Definisci i nomi delle colonne come specificato (prendiamo almeno 'item_id' e 'movie_title')
items_cols = ['item_id', 'movie_title', 'release_date', 'video_release_date', 'IMDb_URL',
              'unknown', 'Action', 'Adventure', 'Animation', 'Childrens', 'Comedy', 'Crime', 
              'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 
              'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

# Carica il file degli item
items_df = pd.read_csv(items_file, sep='|', names=items_cols, encoding='latin-1')
print("Informazioni sui film caricate. Shape:", items_df.shape)

# Crea un dizionario per mappare l'item_id al movie_title
movie_dict = pd.Series(items_df.movie_title.values, index=items_df.item_id).to_dict()

# ------------------------------
# Caricamento del Modello VAE
# ------------------------------

latent_dim = 50   # Dimensione dello spazio latente (stesso valore usato nel training)
beta = 1.0        # Puoi variare beta se necessario

# Ricrea il modello VAE (lo stesso definito in models/vae_model.py)
vae, encoder, decoder = create_vae_architecture(n_items, latent_dim, beta)

# Se hai salvato i pesi del modello, caricali (devi averli salvati in precedenza durante il training)
# Esempio:
# weights_path = os.path.join("models", "vae_weights.h5")
# vae.load_weights(weights_path)
# In questo esempio assumiamo che il modello sia già addestrato nella sessione corrente.

# ------------------------------
# Predizione e Raccomandazioni
# ------------------------------

# Seleziona alcuni utenti casuali (ad esempio 5)
num_utenti = 5
user_ids = user_item_matrix.index.tolist()  # Lista degli user_id
utenti_selezionati = random.sample(user_ids, num_utenti)

print("\nRaccomandazioni per utenti casuali:")

# Per ogni utente selezionato:
for uid in utenti_selezionati:
    # Estrai il vettore dei rating normalizzati per l'utente (converti in array 2D per il modello)
    user_vector = user_item_matrix_normalized.loc[uid].values.reshape(1, -1)
    
    # Usa il modello VAE per predire il rating (output sarà anch'esso normalizzato in [0,1])
    pred_vector = vae.predict(user_vector)
    
    # Riconverti i rating predetti alla scala originale [1,5]
    pred_ratings = pred_vector[0] * 4 + 1
    
    # Crea un DataFrame per ordinare e visualizzare i risultati
    rec_df = pd.DataFrame({
        'Item_ID': np.arange(1, n_items+1),
        'Predicted_Rating': pred_ratings
    })
    
    # Aggiungi il titolo del film usando il dizionario movie_dict (se l'item_id è presente)
    rec_df['Movie_Title'] = rec_df['Item_ID'].apply(lambda x: movie_dict.get(x, "N/A"))
    
    # Ordina per rating decrescente
    rec_df = rec_df.sort_values(by='Predicted_Rating', ascending=False)
    
    # Prendi i top 10 film con rating più alto
    top_rec = rec_df.head(10)
    
    print(f"\nUtente ID: {uid}")
    print(top_rec[['Item_ID', 'Movie_Title', 'Predicted_Rating']].to_markdown(index=False))