import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from vae_model import create_vae_architecture  # Assicurati che il file si chiami "vae_model.py"

def train_vae(vae, train_data, test_data, epochs, batch_size):
    """
    15 Function train_vae(vae, train_data, test_data, epochs, batch_size)
    16 Initialize EarlyStopping callback with 'val_loss' monitoring and a patience of 8
    17 Fit the VAE model on the train_data with validation on test_data
    18 Apply EarlyStopping callback during training
    19 Return the training history
    20 End Function
    """
    early_stop = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
    
    history = vae.fit(
        train_data,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=test_data,
        callbacks=[early_stop]
    )
    
    return history

def predict_ratings(vae, test_data):
    """
    21 Function predict_ratings(vae, test_data)
    22 Use the VAE model to predict ratings on the test_data
    23 Return the predicted ratings
    24 End Function
    """
    predictions = vae.predict(test_data)
    return predictions

if __name__ == '__main__':
    # ------------------------------
    # Caricamento dei dati reali puliti
    # ------------------------------
    cleaned_data_path = os.path.join("data", "cleaned")
    ratings_file = os.path.join(cleaned_data_path, "ratings_clean.csv")
    items_file = os.path.join(cleaned_data_path, "items_clean.csv")
    
    # Carica i dati dei rating
    ratings_df = pd.read_csv(ratings_file, encoding='latin-1')
    
    # Poiché i rating in MovieLens sono tipicamente da 1 a 5, li scalare in [0,1]
    # Inoltre, per costruire la matrice utente-item, si assume che gli item_id siano numerici
    # Spostiamo gli item_id in modo che siano 0-indexed
    ratings_df['item_id'] = ratings_df['item_id'].astype(int) - 1

    # Costruisci la matrice utente-item: righe = user_id, colonne = item_id, valori = rating
    user_item_matrix = ratings_df.pivot(index='user_id', columns='item_id', values='rating')
    user_item_matrix = user_item_matrix.fillna(0)
    
    # Scala i rating (da 1-5 a 0-1)
    user_item_matrix = user_item_matrix / 5.0

    # Converti in array numpy
    data_matrix = user_item_matrix.values
    print("Forma della matrice utente-item:", data_matrix.shape)

    # ------------------------------
    # Split dei dati in training e test (80/20)
    # ------------------------------
    train_data, test_data = train_test_split(data_matrix, test_size=0.2, random_state=42)
    print("Training data shape:", train_data.shape)
    print("Test data shape:", test_data.shape)

    # ------------------------------
    # Creazione del modello VAE con i veri dati
    # ------------------------------
    # Il numero di item è dato dal numero di colonne della matrice utente-item
    n_items = data_matrix.shape[1]
    latent_dim = 32  # Puoi modificare questo parametro se lo desideri

    vae_model, encoder_model, decoder_model = create_vae_architecture(n_items, latent_dim)
    
    # Stampa della struttura del modello
    vae_model.summary()

    # ------------------------------
    # Training del modello VAE
    # ------------------------------
    epochs = 50
    batch_size = 32
    history = train_vae(vae_model, train_data, test_data, epochs, batch_size)

    # ------------------------------
    # Predizione dei rating sui dati di test
    # ------------------------------
    predicted_ratings = predict_ratings(vae_model, test_data)
    print("Forma dei rating predetti:", predicted_ratings.shape)
