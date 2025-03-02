"""
Movie Recommender System - VAE Training Script
----------------------------------------------
Questo script addestra il modello VAE utilizzando il dataset dei rating puliti.
Operazioni eseguite:
  1. Caricamento e preparazione dei dati:
     - Creazione della matrice utente-item dal file CSV (data/cleaned/ratings_clean.csv)
     - Normalizzazione dei rating da [1,5] a [0,1]
  2. Divisione dei dati in training e validation (80/20)
  3. Creazione del modello VAE tramite la funzione create_vae_architecture definita in models/vae_model.py
  4. Addestramento del modello con EarlyStopping per evitare overfitting
  5. Salvataggio dei pesi del modello e della cronologia del training

Struttura del progetto:
    movie-recommendation-system/
    ├── data/
    │   └── cleaned/
    │       └── ratings_clean.csv
    ├── models/
    │   └── vae_model.py
    ├── train/
    │   └── vae_train.py   <-- Questo file
    └── notebooks/
        └── 02_vae_training.ipynb
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

# Importa la funzione per creare il modello VAE dal modulo models
from model.vae_model import create_vae_architecture

# ------------------------------
# 1. Caricamento e Preparazione dei Dati
# ------------------------------

data_path = os.path.join("data", "cleaned", "ratings_clean.csv")
ratings_df = pd.read_csv(data_path)
print("Dataset dei rating caricato. Shape:", ratings_df.shape)

# Crea la matrice utente-item: righe = user_id, colonne = item_id, valori = rating
user_item_matrix = ratings_df.pivot(index='user_id', columns='item_id', values='rating').fillna(0)
print("Matrice utente-item creata. Shape:", user_item_matrix.shape)

# Normalizza i rating da [1,5] a [0,1]
user_item_matrix_normalized = (user_item_matrix - 1) / 4

# Converti la matrice in array NumPy
data = user_item_matrix_normalized.values.astype('float32')
n_items = data.shape[1]
print("Numero di film (item):", n_items)

# ------------------------------
# 2. Divisione dei Dati in Training e Validation
# ------------------------------

train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
print("Training data shape:", train_data.shape)
print("Validation data shape:", val_data.shape)

# Poiché il modello usa un train_step personalizzato, passiamo come target il medesimo input.
train_targets = train_data.copy()
val_targets = val_data.copy()

# ------------------------------
# 3. Creazione del Modello VAE
# ------------------------------

latent_dim = 50
vae, encoder, decoder = create_vae_architecture(n_items, latent_dim)
print("Modello VAE creato correttamente.")

# ------------------------------
# 4. Addestramento del Modello VAE
# ------------------------------

epochs = 50
batch_size = 64
early_stopping = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)

history = vae.fit(
    train_data,           # Input
    train_targets,        # Target (dummy, uguale all'input)
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(val_data, val_targets),
    callbacks=[early_stopping]
)

# ------------------------------
# 5. Salvataggio dei Pesi e della Cronologia di Training
# ------------------------------

# Modifica il filename in modo che termini con ".weights.h5"
weights_path = os.path.join("models", "vae_weights.weights.h5")
vae.save_weights(weights_path)
print("Pesi del modello salvati in:", weights_path)

history_df = pd.DataFrame(history.history)
history_path = os.path.join("data", "cleaned", "vae_training_history.csv")
history_df.to_csv(history_path, index=False)
print("Cronologia del training salvata in:", history_path)
