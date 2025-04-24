import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping # type: ignore

from model.vae_architecture import create_encoder, create_decoder
from model.vae_model import CustomVAE

# ------------------------------
# 1. Caricamento e preparazione dei dati
# ------------------------------

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

    # Costruisci la matrice utente-film
    user_item_matrix = ratings.pivot_table(index="user_id", columns="item_id", values="rating", fill_value=0)

    # Converti in numpy array
    data_matrix = user_item_matrix.to_numpy().astype("float32")

    return data_matrix, user_item_matrix

# ------------------------------
# 2. Creazione e addestramento del modello VAE
# ------------------------------

def train_vae_model(data, n_items, latent_dim=10, epochs=100, batch_size=8):
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
    print("Dati di addestramento:", train_data.shape)
    print("Dati di validazione:", val_data.shape)

    # Creazione dell'encoder e del decoder
    encoder = create_encoder(n_items, latent_dim)
    decoder = create_decoder(n_items, latent_dim)
    
    # Creazione del modello VAE personalizzato
    vae = CustomVAE(encoder, decoder)
    vae.compile(optimizer=tf.keras.optimizers.Adam())
    print("Modello VAE creato con successo.")

    # EarlyStopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)

    # Addestramento del modello
    history = vae.fit( #attenzione a righe o colonne
        train_data, #matrice X
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

    # Salvataggio del modello intero dopo il training
    os.makedirs("saved_models", exist_ok=True)
    vae.save("saved_models/vae_model.keras", save_format="keras")

    # Passo 3: Predizione delle valutazioni
    predicted_ratings = predict_ratings(vae, val_data)

    # Passo 4: Valutazione delle prestazioni
    precision, recall, f1 = compute_precision_recall_f1(val_data, predicted_ratings, k=5)
    print(f"Precision@5: {precision:.4f}")
    print(f"Recall@5: {recall:.4f}")
    print(f"F1-score@5: {f1:.4f}")

    # Passo 5: Visualizzazione dei risultati
    plot_training_history(history)

    # Passo 6: Salvataggio del modello
    os.makedirs("model", exist_ok=True)