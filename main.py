import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping # type: ignore

from model.vae_architecture import create_encoder, create_decoder
from model.vae_model import CustomVAE

def load_and_prepare_data():
    data_path = os.path.join("data", "cleaned", "ratings_clean.csv")
    #ratings_df = pd.read_csv(data_path)
    #print("Dataset delle valutazioni caricato. Forma:", ratings_df.shape)

    #user_item_matrix = ratings_df.pivot(index='user_id', columns='item_id', values='rating').fillna(0)
    #print("Matrice utente-elemento creata. Forma:", user_item_matrix.shape)

    #user_item_matrix_normalized = (user_item_matrix - 1) / 4
    #data = user_item_matrix_normalized.values.astype('float32')

    #return data, user_item_matrix

    ################

    ratings = pd.read_csv(data_path)

    user_item_matrix = ratings.pivot_table(index="user_id", columns="item_id", values="rating", fill_value=0)

    data_matrix = user_item_matrix.to_numpy().astype("float32")
    data_matrix = (data_matrix - 1) / 4
    data_matrix[data_matrix == -0.25] = 0
    return data_matrix, user_item_matrix

def train_vae_model(data, n_items, latent_dim=64, epochs=200, batch_size=128):
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
    print("Dati di addestramento:", train_data.shape)
    print("Dati di validazione:", val_data.shape)

    encoder = create_encoder(n_items, latent_dim)
    decoder = create_decoder(n_items, latent_dim)
    
    vae = CustomVAE(encoder, decoder)
    vae.compile(optimizer=tf.keras.optimizers.Adam())
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

def predict_ratings(vae, data):
    predicted_ratings = vae.predict(data)
    predicted_ratings = (predicted_ratings * 4) + 1
    predicted_ratings[predicted_ratings < 1] = 1
    predicted_ratings[predicted_ratings > 5] = 5
    
    return predicted_ratings

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

def plot_training_history(history):
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Loss di addestramento')
    plt.plot(history.history['val_loss'], label='Loss di validazione')
    plt.title('Andamento della loss durante l\'addestramento')
    plt.xlabel('Epoche [n]')
    plt.ylabel('Loss [MSE]')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('training_loss_plot.png')
    plt.show()

if __name__ == "__main__":
    data, user_item_matrix = load_and_prepare_data()
    n_items = data.shape[1]
    n_users = data.shape[0]

    vae, encoder, decoder, history, val_data = train_vae_model(data, n_items)

    os.makedirs("saved_models", exist_ok=True)
    vae.save("saved_models/vae_model.keras", save_format="keras")

    predicted_ratings = predict_ratings(vae, val_data)

    precision, recall, f1 = compute_precision_recall_f1(val_data, predicted_ratings, k=5)
    print(f"Precision@5: {precision:.4f}")
    print(f"Recall@5: {recall:.4f}")
    print(f"F1-score@5: {f1:.4f}")

    plot_training_history(history)

    os.makedirs("model", exist_ok=True)