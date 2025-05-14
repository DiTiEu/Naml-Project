import os
import requests
import zipfile
import pandas as pd

DATASET_URL = "http://files.grouplens.org/datasets/movielens/ml-100k.zip"
DATA_DIR = os.path.join("data", "ml-100k")
ZIP_PATH = os.path.join("data", "ml-100k.zip")

def download_and_extract_dataset():
    if os.path.exists(DATA_DIR):
        print("Dataset already exists at:", DATA_DIR)
        return

    if not os.path.exists("data"):
        os.makedirs("data")
    
    print("Dataset not found. Downloading from:", DATASET_URL)
    
    response = requests.get(DATASET_URL)
    with open(ZIP_PATH, "wb") as f:
        f.write(response.content)
    print("Download completed. Extracting files...")

    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall("data")
    print("Extraction completed. Dataset is now available in:", DATA_DIR)

    os.remove(ZIP_PATH)
    print("Zip file removed.")

download_and_extract_dataset()

ratings_file = os.path.join(DATA_DIR, "u.data")
users_file = os.path.join(DATA_DIR, "u.user")
items_file = os.path.join(DATA_DIR, "u.item")

rating_cols = ['user_id', 'item_id', 'rating', 'timestamp']
users_cols  = ['user_id', 'age', 'gender', 'occupation', 'zip']
items_cols  = ['item_id', 'movie_title', 'release_date', 'video_release_date', 'IMDb_URL',
               'unknown', 'Action', 'Adventure', 'Animation', 'Childrens', 'Comedy', 'Crime', 
               'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 
               'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

ratings_df = pd.read_csv(ratings_file, sep='\t', names=rating_cols, encoding='latin-1')
print("Ratings data loaded. Shape:", ratings_df.shape)

users_df = pd.read_csv(users_file, sep='|', names=users_cols, encoding='latin-1')
print("User data loaded. Shape:", users_df.shape)

items_df = pd.read_csv(items_file, sep='|', names=items_cols, encoding='latin-1')
print("Items data loaded. Shape:", items_df.shape)

ratings_count = ratings_df.groupby('user_id').size().reset_index(name='ratings_count')
valid_users = ratings_count[ratings_count['ratings_count'] >= 20]['user_id']
clean_ratings_df = ratings_df[ratings_df['user_id'].isin(valid_users)]
print("Ratings after filtering users with <20 ratings:", clean_ratings_df.shape)

if users_df.isnull().values.any():
    users_df = users_df.fillna(0)
    print("Missing values in user data have been filled with zeros.")
else:
    print("No missing values found in user data.")

if items_df.isnull().values.any():
    items_df = items_df.fillna(0)
    print("Missing values in items data have been filled with zeros.")
else:
    print("No missing values found in items data.")

cleaned_data_path = os.path.join("data", "cleaned")
if not os.path.exists(cleaned_data_path):
    os.makedirs(cleaned_data_path)

clean_ratings_df.to_csv(os.path.join(cleaned_data_path, "ratings_clean.csv"), index=False)
users_df.to_csv(os.path.join(cleaned_data_path, "users_clean.csv"), index=False)
items_df.to_csv(os.path.join(cleaned_data_path, "items_clean.csv"), index=False)
print("Cleaned datasets saved to:", cleaned_data_path)

num_users = clean_ratings_df['user_id'].nunique()
print("Number of valid users after filtering:", num_users)
print("Sample of cleaned ratings data:")
print(clean_ratings_df.head())