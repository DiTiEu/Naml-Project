"""
Movie Recommender System - VAE Model Training Script
------------------------------------------------------
This script trains the Variational Autoencoder (VAE) model on the MovieLens 100K dataset (cleaned version).

Overall Process:
  1. Load the cleaned ratings dataset from 'data/cleaned/ratings_clean.csv'.
  2. Convert the long-format ratings data into a user-item matrix where each row represents a user and 
     each column represents a movie. Missing ratings are filled with zeros.
  3. Split the user-item matrix into training and validation sets (80/20 split).
  4. Create the VAE model using the number of items (movies) and a specified latent dimension.
  5. Train the model using the training data and validate on the validation set. EarlyStopping is applied
     to monitor the validation loss and prevent overfitting.
  6. Optionally, save the training history for further analysis.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

# Import the VAE model creation function from our models module
from model.vae_model import create_vae_architecture

# ------------------------------
# Load and Prepare the Data
# ------------------------------

# Define the path to the cleaned ratings CSV file
cleaned_ratings_path = os.path.join("data", "cleaned", "ratings_clean.csv")

# Load the cleaned ratings dataset using pandas
ratings_df = pd.read_csv(cleaned_ratings_path)
print("Cleaned ratings data loaded. Shape:", ratings_df.shape)

# Convert the ratings DataFrame into a user-item matrix:
# Rows represent users, columns represent items (movies), and values are the ratings.
# Missing ratings are filled with zeros.
user_item_matrix = ratings_df.pivot(index='user_id', columns='item_id', values='rating').fillna(0)
print("User-item matrix created. Shape:", user_item_matrix.shape)
user_item_matrix = (user_item_matrix - 1) / 4
print("_______________________-")
print(user_item_matrix.head())
print("_______________________-")
# Convert the DataFrame to a NumPy array for training.
data = user_item_matrix.values.astype('float32')

# Number of items (movies) is the number of columns in the user-item matrix
n_items = data.shape[1]
print("Number of items (movies):", n_items)

# ------------------------------
# Split the Data into Training and Validation Sets
# ------------------------------
# We split on the user dimension; each row is a complete rating vector for one user.
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
print("Training data shape:", train_data.shape)
print("Validation data shape:", val_data.shape)

# ------------------------------
# Create the VAE Model
# ------------------------------
latent_dim = 50  # Set the latent space dimension (can be tuned as needed)

# Create the VAE model using the previously defined function in models/vae_model.py
vae, encoder, decoder = create_vae_architecture(n_items, latent_dim)
print("VAE model created.")

# ------------------------------
# Train the VAE Model
# ------------------------------
# Define training hyperparameters
epochs = 50       # Number of epochs to train
batch_size = 64   # Batch size for training

# Create an EarlyStopping callback to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)

# Train the VAE model using the training data and validate on the validation data.
# Note: The VAE model computes its own loss internally, so we do not need to provide labels.
history = vae.fit(
    train_data,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(val_data, None),  # 'None' because the model calculates loss from inputs and reconstruction
    callbacks=[early_stopping]
)

# ------------------------------
# Save the Training History (Optional)
# ------------------------------
# Save the training history to a CSV file for further analysis if desired.
history_df = pd.DataFrame(history.history)
history_csv_path = os.path.join("data", "cleaned", "vae_training_history.csv")
history_df.to_csv(history_csv_path, index=False)
print("Training history saved to:", history_csv_path)

# ------------------------------
# Overall Explanation:
# ------------------------------
# This script loads the preprocessed and cleaned ratings data, converts it into a user-item matrix,
# and splits it into training and validation sets. The VAE model is built using the number of items and a specified
# latent dimension. The model is trained end-to-end using the training data with an EarlyStopping callback to monitor
# validation loss and prevent overfitting. Finally, the training history is saved for later review.