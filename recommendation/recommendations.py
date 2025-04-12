"""
Movie Recommender System - Evaluation and Recommendations Script
-------------------------------------------------------------------
This script loads the trained VAE model, the cleaned ratings dataset, and the items file,
and for some random users prints a table with the recommended movies (top-N) based on the predicted rating.
The ratings are converted back to the original scale [1,5] to make the output understandable.
"""

import os
import pandas as pd
import numpy as np
import random

import tensorflow as tf
from tensorflow.keras.models import load_model

# Import the function to create the VAE model from the models module
from model.vae_model import create_vae_architecture

# ------------------------------
# Data Loading
# ------------------------------

# Path to the cleaned ratings file
cleaned_ratings_path = os.path.join("data", "cleaned", "ratings_clean.csv")

# Load the cleaned ratings (assuming the file contains the columns: user_id, item_id, rating, ...)
ratings_df = pd.read_csv(cleaned_ratings_path)
print("Ratings data loaded. Shape:", ratings_df.shape)

# Create the user-item matrix:
# - Pivot: rows = user_id, columns = item_id, values = rating
# - Missing values are filled with 0.
user_item_matrix = ratings_df.pivot(index='user_id', columns='item_id', values='rating').fillna(0)
print("User-item matrix created. Shape:", user_item_matrix.shape)

# Normalize the ratings from [1,5] to [0,1] to be compatible with the model (sigmoid output)
user_item_matrix_normalized = (user_item_matrix - 1) / 4

# Convert the normalized matrix to a NumPy array
data = user_item_matrix_normalized.values.astype('float32')

# Number of movies (items)
n_items = data.shape[1]
print("Number of movies (items):", n_items)

# ------------------------------
# Loading Movie Information
# ------------------------------

# Path to the items file (MovieLens u.item)
items_file = os.path.join("data", "ml-100k", "u.item")

# Define the column names as specified (we take at least 'item_id' and 'movie_title')
items_cols = ['item_id', 'movie_title', 'release_date', 'video_release_date', 'IMDb_URL',
              'unknown', 'Action', 'Adventure', 'Animation', 'Childrens', 'Comedy', 'Crime', 
              'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 
              'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

# Load the items file
items_df = pd.read_csv(items_file, sep='|', names=items_cols, encoding='latin-1')
print("Movie information loaded. Shape:", items_df.shape)

# Create a dictionary to map item_id to movie_title
movie_dict = pd.Series(items_df.movie_title.values, index=items_df.item_id).to_dict()

# ------------------------------
# Loading the VAE Model
# ------------------------------

latent_dim = 50   # Latent space dimension (same value used in training)

# Recreate the VAE model (the same defined in models/vae_model.py)
vae, encoder, decoder = create_vae_architecture(n_items, latent_dim)

# If you have saved the model weights, load them (you must have saved them previously during training)
# Example:
# weights_path = os.path.join("models", "vae_weights.h5")
# vae.load_weights(weights_path)
# In this example, we assume the model is already trained in the current session.

# ------------------------------
# Prediction and Recommendations
# ------------------------------

# Select some random users (e.g., 5)
num_users = 5
user_ids = user_item_matrix.index.tolist()  # List of user_ids
selected_users = random.sample(user_ids, num_users)

print("\nRecommendations for random users:")

# For each selected user:
for uid in selected_users:
    # Extract the normalized rating vector for the user (convert to 2D array for the model)
    user_vector = user_item_matrix_normalized.loc[uid].values.reshape(1, -1)
    
    # Use the VAE model to predict the rating (output will also be normalized in [0,1])
    pred_vector = vae.predict(user_vector)
    
    # Convert the predicted ratings back to the original scale [1,5]
    pred_ratings = pred_vector[0] * 4 + 1
    
    # Create a DataFrame to sort and display the results
    rec_df = pd.DataFrame({
        'Item_ID': np.arange(1, n_items+1),
        'Predicted_Rating': pred_ratings
    })
    
    # Add the movie title using the movie_dict (if the item_id is present)
    rec_df['Movie_Title'] = rec_df['Item_ID'].apply(lambda x: movie_dict.get(x, "N/A"))
    
    # Sort by descending rating
    rec_df = rec_df.sort_values(by='Predicted_Rating', ascending=False)
    
    # Take the top 10 movies with the highest rating
    top_rec = rec_df.head(10)
    
    print(f"\nUser ID: {uid}")
    print(top_rec[['Item_ID', 'Movie_Title', 'Predicted_Rating']].to_markdown(index=False))