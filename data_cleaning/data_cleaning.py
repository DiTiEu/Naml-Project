"""
Movie Recommender System - Data Acquisition and Cleaning Module
-----------------------------------------------------------------
This script demonstrates the process of:
  1. Automatically downloading and extracting the MovieLens 100K dataset if not already present.
  2. Splitting the dataset into its three components: ratings, user details, and movie items.
  3. Performing data cleaning steps:
      - Filtering out users with fewer than 20 ratings.
      - Handling missing values in the user and item datasets.
  4. Saving the cleaned datasets for later use.
"""

import os
import requests
import zipfile
import pandas as pd

# ------------------------------
# Define Dataset URL and Paths
# ------------------------------

DATASET_URL = "http://files.grouplens.org/datasets/movielens/ml-100k.zip"
DATA_DIR = os.path.join("data", "ml-100k")
ZIP_PATH = os.path.join("data", "ml-100k.zip")

# ------------------------------
# Automatic Download & Extraction
# ------------------------------

def download_and_extract_dataset():
    """
    Checks if the dataset folder exists. If not, downloads the MovieLens 100K zip file,
    extracts it to the 'data' folder, and removes the zip file.
    """
    # If the dataset folder already exists, no need to download again
    if os.path.exists(DATA_DIR):
        print("Dataset already exists at:", DATA_DIR)
        return

    # Ensure the data directory exists
    if not os.path.exists("data"):
        os.makedirs("data")
    
    print("Dataset not found. Downloading from:", DATASET_URL)
    # Download the dataset zip file
    response = requests.get(DATASET_URL)
    with open(ZIP_PATH, "wb") as f:
        f.write(response.content)
    print("Download completed. Extracting files...")

    # Extract the zip file into the 'data' folder
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall("data")
    print("Extraction completed. Dataset is now available in:", DATA_DIR)

    # Remove the zip file after extraction
    os.remove(ZIP_PATH)
    print("Zip file removed.")

# Call the download and extraction function
download_and_extract_dataset()

# ------------------------------
# Define File Paths for Dataset Files
# ------------------------------

# According to the paper, the dataset contains three files: u.data, u.item, and u.user
ratings_file = os.path.join(DATA_DIR, "u.data")
users_file = os.path.join(DATA_DIR, "u.user")
items_file = os.path.join(DATA_DIR, "u.item")

# ------------------------------
# Define Column Names (Matching the Paper)
# ------------------------------

rating_cols = ['user_id', 'item_id', 'rating', 'timestamp']
users_cols  = ['user_id', 'age', 'gender', 'occupation', 'zip']
items_cols  = ['item_id', 'movie_title', 'release_date', 'video_release_date', 'IMDb_URL',
               'unknown', 'Action', 'Adventure', 'Animation', 'Childrens', 'Comedy', 'Crime', 
               'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 
               'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

# ------------------------------
# Load the Datasets
# ------------------------------

# Load ratings data (tab-separated)
ratings_df = pd.read_csv(ratings_file, sep='\t', names=rating_cols, encoding='latin-1')
print("Ratings data loaded. Shape:", ratings_df.shape)

# Load user data (pipe-separated)
users_df = pd.read_csv(users_file, sep='|', names=users_cols, encoding='latin-1')
print("User data loaded. Shape:", users_df.shape)

# Load item (movie) data (pipe-separated)
items_df = pd.read_csv(items_file, sep='|', names=items_cols, encoding='latin-1')
print("Items data loaded. Shape:", items_df.shape)

# ------------------------------
# Data Cleaning Process
# ------------------------------

# Step 1: Filter out users with fewer than 20 ratings
ratings_count = ratings_df.groupby('user_id').size().reset_index(name='ratings_count')
valid_users = ratings_count[ratings_count['ratings_count'] >= 20]['user_id']
clean_ratings_df = ratings_df[ratings_df['user_id'].isin(valid_users)]
print("Ratings after filtering users with <20 ratings:", clean_ratings_df.shape)

# Step 2: Handle missing values in user data
if users_df.isnull().values.any():
    users_df = users_df.fillna(0)
    print("Missing values in user data have been filled with zeros.")
else:
    print("No missing values found in user data.")

# Step 3: Handle missing values in items data
if items_df.isnull().values.any():
    items_df = items_df.fillna(0)
    print("Missing values in items data have been filled with zeros.")
else:
    print("No missing values found in items data.")

# ------------------------------
# Save Cleaned Datasets
# ------------------------------

# Create a directory for cleaned data if it doesn't exist
cleaned_data_path = os.path.join("data", "cleaned")
if not os.path.exists(cleaned_data_path):
    os.makedirs(cleaned_data_path)

# Save the cleaned data to CSV files
clean_ratings_df.to_csv(os.path.join(cleaned_data_path, "ratings_clean.csv"), index=False)
users_df.to_csv(os.path.join(cleaned_data_path, "users_clean.csv"), index=False)
items_df.to_csv(os.path.join(cleaned_data_path, "items_clean.csv"), index=False)
print("Cleaned datasets saved to:", cleaned_data_path)

# ------------------------------
# Final Overview and Statistics
# ------------------------------

num_users = clean_ratings_df['user_id'].nunique()
print("Number of valid users after filtering:", num_users)
print("Sample of cleaned ratings data:")
print(clean_ratings_df.head())
