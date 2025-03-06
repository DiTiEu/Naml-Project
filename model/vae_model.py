import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

# Funzione di campionamento per il reparameterization trick
def sampling(args):
    """
    Applica il reparameterization trick.
    Args:
      args: tuple contenente (z_mean, z_log_var)
    Returns:
      z: campione ottenuto dalla distribuzione definita da z_mean e z_log_var
    """
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # Campiona epsilon da una distribuzione normale standard
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def create_vae_architecture(n_items, latent_dim):
    """
    Crea l'architettura di un Variational Autoencoder (VAE).
    
    Args:
      n_items: dimensione dell'input (es. numero di item/features)
      latent_dim: dimensione dello spazio latente
    
    Returns:
      vae: modello VAE compilato
      encoder: modello encoder
      decoder: modello decoder
    """
    # 2. Definire il layer di input per l'encoder con forma (n_items,)
    encoder_input = Input(shape=(n_items,), name='encoder_input')
    
    # 3. Aggiungere dense layers all'encoder: 1024, 512, e 256 unità con attivazione 'relu'
    x = Dense(1024, activation='relu')(encoder_input)
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    
    # 4. Creare le variabili latenti z_mean e z_log_var con 'latent_dim' unità
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)
    
    # 5. Applicare il reparameterization trick per ottenere il vettore latente z
    z = Lambda(sampling, name='z')([z_mean, z_log_var])
    
    # 9. Istanziare il modello encoder che restituisce z_mean, z_log_var e z
    encoder = Model(encoder_input, [z_mean, z_log_var, z], name='encoder')
    
    # 6. Definire il layer di input per il decoder con forma (latent_dim,)
    decoder_input = Input(shape=(latent_dim,), name='decoder_input')
    
    # 7. Aggiungere dense layers al decoder: 256, 512, e 1024 unità con attivazione 'relu'
    x_dec = Dense(256, activation='relu')(decoder_input)
    x_dec = Dense(512, activation='relu')(x_dec)
    x_dec = Dense(1024, activation='relu')(x_dec)
    
    # 8. Definire il layer di output del decoder con 'n_items' unità e attivazione 'sigmoid'
    decoder_output = Dense(n_items, activation='sigmoid')(x_dec)
    
    # 9. Istanziare il modello decoder
    decoder = Model(decoder_input, decoder_output, name='decoder')
    
    # 10. Definire il modello VAE collegando l'encoder e il decoder
    vae_output = decoder(z)
    vae = Model(encoder_input, vae_output, name='vae')
    
    # 11. Definire la funzione di perdita come combinazione di MSE e KL divergence
    # Loss di ricostruzione (MSE)
    reconstruction_loss = tf.reduce_mean(tf.square(encoder_input - vae_output), axis=-1)
    reconstruction_loss *= n_items  # Scala la loss in base al numero di unità in input
    
    # Calcolo della KL Divergence Loss
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = -0.5 * K.sum(kl_loss, axis=-1)
    
    # Loss totale: somma di reconstruction loss e KL divergence loss
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    
    # 12. Compilare il VAE con l'ottimizzatore Adam
    vae.compile(optimizer='adam')
    
    # 13. Restituire il modello VAE compilato, l'encoder e il decoder
    return vae, encoder, decoder

# Esempio di utilizzo con i veri dati (opzionale)
if __name__ == '__main__':
    import os
    import pandas as pd

    # Carica il file cleaned degli item per ottenere il numero di item
    cleaned_data_path = os.path.join("data", "cleaned")
    items_df = pd.read_csv(os.path.join(cleaned_data_path, "items_clean.csv"), encoding='latin-1')
    # Si assume che il file contenga 1682 item; in alternativa:
    n_items = items_df.shape[0]  # oppure: items_df['item_id'].max() se gli id non sono consecutivi
    latent_dim = 32

    vae_model, encoder_model, decoder_model = create_vae_architecture(n_items, latent_dim)
    
    # Stampa dei sommari dei modelli per verificare la struttura
    encoder_model.summary()
    decoder_model.summary()
    vae_model.summary()