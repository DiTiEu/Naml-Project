# import os
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# import tensorflow as tf
# from tensorflow.keras.callbacks import EarlyStopping

# from model.vae_model import create_vae_architecture

# # --- Evaluation Utilities ---
# def predict_ratings(vae_model, data):
#     return vae_model(data).numpy()

# def compute_precision_recall_f1(true_data, predicted_ratings, k=5):
#     precisions, recalls, f1s = [], [], []
#     for true_row, pred_row in zip(true_data, predicted_ratings):
#         top_k_pred = np.argsort(pred_row)[::-1][:k]
#         top_k_true = np.argsort(true_row)[::-1][:k]
#         true_set = set(top_k_true)
#         pred_set = set(top_k_pred)

#         intersection = len(true_set & pred_set)
#         precision = intersection / k
#         recall = intersection / len(true_set) if len(true_set) > 0 else 0
#         f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

#         precisions.append(precision)
#         recalls.append(recall)
#         f1s.append(f1)

#     return np.mean(precisions), np.mean(recalls), np.mean(f1s)

# def evaluate_model(true_data, predicted_ratings):
#     mse = np.mean((true_data - predicted_ratings) ** 2)
#     mae = np.mean(np.abs(true_data - predicted_ratings))
#     rmse = np.sqrt(mse)
#     print(f"Evaluation Metrics:\n  RMSE: {rmse:.4f}\n  MAE: {mae:.4f}")

# # --- Main ---
# if __name__ == "__main__":
#     # Step 1: Load and preprocess data
#     data_path = os.path.join("data", "cleaned", "ratings_clean.csv")
#     ratings_df = pd.read_csv(data_path)
#     user_item_matrix = ratings_df.pivot(index='user_id', columns='item_id', values='rating').fillna(0)
#     user_item_matrix_normalized = (user_item_matrix - 1) / 4
#     data = user_item_matrix_normalized.values.astype('float32')
#     n_users, n_items = data.shape

#     # Step 2: Split the data
#     train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

#     # Step 3: Create VAE architecture
#     latent_dim = 50
#     vae, encoder, decoder = create_vae_architecture(n_items, latent_dim=latent_dim)

#     # Step 4: Train the model
#     epochs = 10
#     batch_size = 64
#     early_stopping = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)

#     vae.fit(
#         train_data, train_data,
#         validation_data=(test_data, test_data),
#         epochs=epochs,
#         batch_size=batch_size,
#         callbacks=[early_stopping],
#     )

#     # Step 5: Predict and evaluate
#     predicted_ratings = predict_ratings(vae, test_data)
#     precision, recall, f1 = compute_precision_recall_f1(test_data, predicted_ratings, k=5)

#     print(f"\nPrecision@5: {precision:.4f}\nRecall@5: {recall:.4f}\nF1-score@5: {f1:.4f}")
#     evaluate_model(test_data, predicted_ratings)

#     # Step 6: Save model
#     os.makedirs("model", exist_ok=True)
#     vae.save_weights(os.path.join("model", "vae_weights.weights.h5"))

# main.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

from model.vae_model import create_vae_architecture

# ------------------------------
# 1. Caricamento e preparazione dei dati
# ------------------------------

def load_and_prepare_data():
    data_path = os.path.join("data", "cleaned", "ratings_clean.csv")
    ratings_df = pd.read_csv(data_path)
    print("Dataset delle valutazioni caricato. Forma:", ratings_df.shape)

    user_item_matrix = ratings_df.pivot(index='user_id', columns='item_id', values='rating').fillna(0)
    print("Matrice utente-elemento creata. Forma:", user_item_matrix.shape)

    user_item_matrix_normalized = (user_item_matrix - 1) / 4
    data = user_item_matrix_normalized.values.astype('float32')

    return data, user_item_matrix

# ------------------------------
# 2. Creazione e addestramento del modello VAE
# ------------------------------

def train_vae_model(data, n_items, latent_dim=50, epochs=50, batch_size=64):
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
    print("Dati di addestramento:", train_data.shape)
    print("Dati di validazione:", val_data.shape)

    vae, encoder, decoder = create_vae_architecture(n_items, latent_dim)
    print("Modello VAE creato con successo.")

    early_stopping = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)

    history = vae.fit(
        train_data,
        train_data,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(val_data, val_data),
        callbacks=[early_stopping]
    )

    return vae, encoder, decoder, history, val_data

# ------------------------------
# 3. Predizione delle valutazioni
# ------------------------------

def predict_ratings(vae, data):
    predicted_ratings = vae.predict(data)
    return predicted_ratings

# ------------------------------
# 4. Valutazione delle prestazioni
# ------------------------------

def compute_precision_recall_f1(true_ratings, predicted_ratings, k=5):
    precisions = []
    recalls = []
    f1s = []

    for true_user, pred_user in zip(true_ratings, predicted_ratings):
        true_items = np.where(true_user > 0)[0]
        pred_items = np.argsort(pred_user)[-k:][::-1]

        if len(true_items) == 0:
            continue

        hits = len(set(pred_items) & set(true_items))
        precision = hits / k
        recall = hits / len(true_items)
        f1 = (2 * precision * recall) / (precision + recall + 1e-8)

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    avg_f1 = np.mean(f1s)

    return avg_precision, avg_recall, avg_f1

# ------------------------------
# 5. Visualizzazione dei risultati
# ------------------------------

def plot_training_history(history):
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Loss di addestramento')
    plt.plot(history.history['val_loss'], label='Loss di validazione')
    plt.title('Andamento della loss durante l\'addestramento')
    plt.xlabel('Epoche')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('training_loss_plot.png')
    plt.show()

# ------------------------------
# Main
# ------------------------------

if __name__ == "__main__":
    # Passo 1: Caricamento e preparazione dei dati
    data, user_item_matrix = load_and_prepare_data()
    n_items = data.shape[1]
    n_users = data.shape[0]

    # Passo 2: Creazione e addestramento del modello VAE
    vae, encoder, decoder, history, val_data = train_vae_model(data, n_items)

    # Passo 3: Predizione delle valutazioni
    predicted_ratings = predict_ratings(vae, val_data)

    # Passo 4: Valutazione delle prestazioni
    precision, recall, f1 = compute_precision_recall_f1(val_data, predicted_ratings, k=5)
    print(f"Precision@5: {precision:.4f}")
    print(f"Recall@5: {recall:.4f}")
    print(f"F1-score@5: {f1:.4f}")

    # Passo 5: Visualizzazione dei risultati
    plot_training_history(history)
