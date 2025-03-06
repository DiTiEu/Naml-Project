"""
Movie Recommender System - VAE Training Script
----------------------------------------------
This script trains the VAE model using the cleaned ratings dataset.
Operations performed:
  1. Data loading and preparation:
     - Creation of the user-item matrix from the CSV file (data/cleaned/ratings_clean.csv)
     - Normalization of ratings from [1,5] to [0,1]
  2. Splitting the data into training and validation (80/20)
  3. Creation of the VAE model using the create_vae_architecture function defined in models/vae_model.py
  4. Training the model with EarlyStopping to avoid overfitting
  5. Saving the model weights and training history
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

# Import the function to create the VAE model from the models module
from model.vae_model import create_vae_architecture

# ------------------------------
# 1. Data Loading and Preparation
# ------------------------------

data_path = os.path.join("data", "cleaned", "ratings_clean.csv")
ratings_df = pd.read_csv(data_path)
print("Ratings dataset loaded. Shape:", ratings_df.shape)

# Create the user-item matrix: rows = user_id, columns = item_id, values = rating
user_item_matrix = ratings_df.pivot(index='user_id', columns='item_id', values='rating').fillna(0)
print("User-item matrix created. Shape:", user_item_matrix.shape)

# Normalize the ratings from [1,5] to [0,1]
user_item_matrix_normalized = (user_item_matrix - 1) / 4

# Convert the matrix to a NumPy array
data = user_item_matrix_normalized.values.astype('float32')
n_items = data.shape[1]
print("Number of movies (items):", n_items)

# ------------------------------
# 2. Splitting the Data into Training and Validation
# ------------------------------

train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
print("Training data shape:", train_data.shape)
print("Validation data shape:", val_data.shape)

# Since the model uses a custom train_step, we pass the same input as the target.
train_targets = train_data.copy()
val_targets = val_data.copy()

# ------------------------------
# 3. Creation of the VAE Model
# ------------------------------

latent_dim = 50
vae, encoder, decoder = create_vae_architecture(n_items, latent_dim)
print("VAE model created successfully.")

# ------------------------------
# 4. Training the VAE Model
# ------------------------------

epochs = 50
batch_size = 64
early_stopping = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)

history = vae.fit(
    train_data,           # Input
    train_targets,        # Target (dummy, same as input)
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(val_data, val_targets),
    callbacks=[early_stopping]
)

# ------------------------------
# 5. Saving the Model Weights and Training History
# ------------------------------

# Build the model by calling it on some data
vae.build(input_shape=(None, n_items))

# Modify the filename to end with ".weights.h5"
weights_path = os.path.join("model", "vae_weights.weights.h5")
vae.save_weights(weights_path)
print("Model weights saved at:", weights_path)

history_df = pd.DataFrame(history.history)
history_path = os.path.join("data", "cleaned", "vae_training_history.csv")
history_df.to_csv(history_path, index=False)
print("Training history saved at:", history_path)