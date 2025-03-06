import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Importa le funzioni dai moduli già creati
from vae_model import create_vae_architecture
from vae_train import train_vae, predict_ratings

##########################################
# Funzione: compute_precision_recall_f1
##########################################
def compute_precision_recall_f1(test_data, predicted_ratings, k, threshold=0.5):
    """
    Calcola la precision, recall e F1 score medi su tutti gli utenti, considerando
    i top-k item predetti per ciascun utente.
    
    Args:
      test_data: matrice reale dei rating (dimensione: num_users x n_items)
      predicted_ratings: matrice dei rating predetti (stesse dimensioni)
      k: numero di item da considerare per il top-k
      threshold: soglia per definire un item come rilevante (default 0.5)
      
    Returns:
      avg_precision, avg_recall, avg_f1: media delle metriche su tutti gli utenti
    """
    num_users = test_data.shape[0]
    precision_list = []
    recall_list = []
    f1_list = []
    
    for i in range(num_users):
        # Estrai i rating reali e predetti per l'utente i-esimo
        actual = test_data[i]
        predicted = predicted_ratings[i]
        
        # Calcola gli indici dei top-k item (maggiore rating predetto)
        top_k_indices = np.argsort(predicted)[::-1][:k]
        
        # Determina gli item rilevanti: consideriamo rilevante un item se il rating reale >= threshold
        relevant_indices = np.where(actual >= threshold)[0]
        
        # Calcola l'intersezione tra top-k e item rilevanti
        hit_count = len(set(top_k_indices) & set(relevant_indices))
        
        precision = hit_count / k
        recall = hit_count / len(relevant_indices) if len(relevant_indices) > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
    
    avg_precision = np.mean(precision_list)
    avg_recall = np.mean(recall_list)
    avg_f1 = np.mean(f1_list)
    
    return avg_precision, avg_recall, avg_f1

##########################################
# Funzione: evaluate_model
##########################################
def evaluate_model(test_data, predicted_ratings):
    """
    Calcola e visualizza le metriche di valutazione:
      - MSE (Mean Squared Error)
      - MAE (Mean Absolute Error)
      - RMSE (Root Mean Squared Error)
    
    Args:
      test_data: matrice dei rating reali
      predicted_ratings: matrice dei rating predetti
    """
    mse = np.mean((test_data - predicted_ratings) ** 2)
    mae = np.mean(np.abs(test_data - predicted_ratings))
    rmse = np.sqrt(mse)
    
    print("\nEvaluation Metrics:")
    print("MSE:  {:.4f}".format(mse))
    print("MAE:  {:.4f}".format(mae))
    print("RMSE: {:.4f}".format(rmse))

##########################################
# Main
##########################################
def main():
    # ------------------------------
    # Caricamento dei dati puliti
    # ------------------------------
    cleaned_data_path = os.path.join("data", "cleaned")
    ratings_file = os.path.join(cleaned_data_path, "ratings_clean.csv")
    items_file = os.path.join(cleaned_data_path, "items_clean.csv")
    
    # Carica il file dei rating
    ratings_df = pd.read_csv(ratings_file, encoding='latin-1')
    # Assicurati che gli item_id siano 0-indexed
    ratings_df['item_id'] = ratings_df['item_id'].astype(int) - 1

    # Costruisci la matrice utente-item: righe = user_id, colonne = item_id, valori = rating
    user_item_matrix = ratings_df.pivot(index='user_id', columns='item_id', values='rating')
    user_item_matrix = user_item_matrix.fillna(0)
    # Scala i rating da [1,5] a [0,1]
    user_item_matrix = user_item_matrix / 5.0
    data_matrix = user_item_matrix.values
    print("User-item matrix shape:", data_matrix.shape)

    # ------------------------------
    # Split in dati di training e test (80/20)
    # ------------------------------
    train_data, test_data = train_test_split(data_matrix, test_size=0.2, random_state=42)
    print("Train data shape:", train_data.shape)
    print("Test data shape:", test_data.shape)

    # ------------------------------
    # Creazione del modello VAE
    # ------------------------------
    n_items = data_matrix.shape[1]         # numero di item (colonne della matrice)
    latent_dim = 50                        # dimensione dello spazio latente come specificato
    vae, encoder, decoder = create_vae_architecture(n_items, latent_dim)
    
    # Stampa della struttura del modello
    vae.summary()

    # ------------------------------
    # Training del modello VAE
    # ------------------------------
    epochs = 10
    batch_size = 64
    history = train_vae(vae, train_data, test_data, epochs, batch_size)

    # ------------------------------
    # Predizione dei rating sui dati di test
    # ------------------------------
    predicted_ratings = predict_ratings(vae, test_data)
    
    # ------------------------------
    # Calcolo di precision, recall e F1 score per k=5
    # ------------------------------
    precision, recall, f1 = compute_precision_recall_f1(test_data, predicted_ratings, k=5)
    print("\nRecommendation Metrics (Top-5):")
    print("Precision: {:.4f}".format(precision))
    print("Recall:    {:.4f}".format(recall))
    print("F1 Score:  {:.4f}".format(f1))
    
    # ------------------------------
    # Valutazione del modello (MSE, MAE, RMSE)
    # ------------------------------
    evaluate_model(test_data, predicted_ratings)

if __name__ == '__main__':
    main()
