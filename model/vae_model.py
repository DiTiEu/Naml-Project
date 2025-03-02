"""
Movie Recommender System - VAE Model Architecture Module (Traditional Implementation)
----------------------------------------------------------------------------------------
Questo modulo implementa il Variational Autoencoder (VAE) in modo tradizionale, 
calcolando la loss (reconstruction loss + KL divergence) nel metodo train_step e test_step 
della sottoclasse del modello. Questo approccio rispetta lo pseudocodice del paper.

Struttura:
1. Encoder:
   - Input di dimensione (n_items,)
   - Dense layers: 1024, 512, 256 unità, attivazione ReLU
   - Genera z_mean e z_log_var (con dimensione latent_dim)
   - Applica il reparameterization trick per campionare z
2. Decoder:
   - Input di dimensione (latent_dim,)
   - Dense layers: 256, 512, 1024 unità, attivazione ReLU
   - Output layer: n_items unità con attivazione sigmoid
3. VAE Model:
   - Sottoclasse di tf.keras.Model che integra encoder e decoder.
   - Il metodo call() restituisce la ricostruzione.
   - I metodi train_step() e test_step() calcolano la reconstruction loss (MSE) e la KL divergence,
     ne sommano i valori e aggiornano il modello.
   
Project Folder Structure:
    movie-recommendation-system/
    ├── models/
    │   └── vae_model.py      <-- Questo file: definizione del modello VAE
    ├── notebooks/
    │   └── 02_vae_training.ipynb
    ├── data/
    ├── data_cleaning/
    └── README.md
"""

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

# Funzione per il reparameterization trick
def sampling(args):
    """
    Applica il reparameterization trick.
    
    Parameters:
      args (tuple): contiene (z_mean, z_log_var)
    
    Returns:
      z (tensor): campione dallo spazio latente, calcolato come:
                  z_mean + exp(0.5 * z_log_var) * epsilon, con epsilon ~ N(0,1)
    """
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.random.normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Funzione per costruire l'encoder
def build_encoder(n_items, latent_dim):
    """
    Costruisce l'encoder del VAE.
    
    Parameters:
      - n_items (int): dimensione dell'input (numero di film)
      - latent_dim (int): dimensione dello spazio latente
    
    Returns:
      - encoder: modello Keras che mappa l'input al vettore latente e ai suoi parametri
    """
    encoder_inputs = layers.Input(shape=(n_items,), name="encoder_input")
    x = layers.Dense(1024, activation='relu', name="encoder_dense_1")(encoder_inputs)
    x = layers.Dense(512, activation='relu', name="encoder_dense_2")(x)
    x = layers.Dense(256, activation='relu', name="encoder_dense_3")(x)
    
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    
    z = layers.Lambda(sampling, output_shape=(latent_dim,), name="z")([z_mean, z_log_var])
    
    encoder = models.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    return encoder

# Funzione per costruire il decoder
def build_decoder(n_items, latent_dim):
    """
    Costruisce il decoder del VAE.
    
    Parameters:
      - n_items (int): dimensione dell'output (numero di film)
      - latent_dim (int): dimensione dello spazio latente
    
    Returns:
      - decoder: modello Keras che mappa lo spazio latente alla ricostruzione dell'input
    """
    decoder_inputs = layers.Input(shape=(latent_dim,), name="decoder_input")
    x = layers.Dense(256, activation='relu', name="decoder_dense_1")(decoder_inputs)
    x = layers.Dense(512, activation='relu', name="decoder_dense_2")(x)
    x = layers.Dense(1024, activation='relu', name="decoder_dense_3")(x)
    decoder_outputs = layers.Dense(n_items, activation='sigmoid', name="decoder_output")(x)
    
    decoder = models.Model(decoder_inputs, decoder_outputs, name="decoder")
    return decoder

# Definizione del modello VAE come sottoclasse di tf.keras.Model
class VAE(tf.keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        """
        Inizializza il modello VAE con l'encoder e il decoder.
        """
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        """
        Esegue il forward pass del modello.
        """
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstruction = self.decoder(z)
        return reconstruction

    def train_step(self, data):
        """
        Definisce il training step personalizzato.
        Calcola:
          - reconstruction loss (MSE)
          - KL divergence loss
        e aggiorna i pesi.
        """
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(tf.square(data - reconstruction))
            kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return {"loss": total_loss, "reconstruction_loss": reconstruction_loss, "kl_loss": kl_loss}

    def test_step(self, data):
        """
        Definisce il test/validation step.
        Calcola le stesse loss del training senza aggiornare i pesi.
        """
        if isinstance(data, tuple):
            data = data[0]
        z_mean, z_log_var, z = self.encoder(data)
        reconstruction = self.decoder(z)
        reconstruction_loss = tf.reduce_mean(tf.square(data - reconstruction))
        kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        total_loss = reconstruction_loss + kl_loss
        return {"loss": total_loss, "reconstruction_loss": reconstruction_loss, "kl_loss": kl_loss}

def create_vae_architecture(n_items, latent_dim=50):
    """
    Crea l'architettura del VAE.
    
    Steps:
      1. Costruisce l'encoder con input (n_items,) e dense layers (1024, 512, 256) con attivazione ReLU,
         generando z_mean e z_log_var e campionando z tramite il reparameterization trick.
      2. Costruisce il decoder con input (latent_dim,) e dense layers (256, 512, 1024) con attivazione ReLU,
         generando un output con n_items unità e attivazione sigmoid.
      3. Crea il modello VAE unendo encoder e decoder.
      4. Compila il modello con l'ottimizzatore Adam e una loss dummy (lambda y_true, y_pred: 0), 
         poiché la loss viene calcolata nel train_step/test_step.
    
    Returns:
      - vae: il modello VAE compilato
      - encoder: il modello encoder
      - decoder: il modello decoder
    """
    encoder = build_encoder(n_items, latent_dim)
    decoder = build_decoder(n_items, latent_dim)
    vae = VAE(encoder, decoder)
    vae.compile(optimizer=optimizers.Adam(), loss=lambda y_true, y_pred: 0)
    return vae, encoder, decoder

# Per testing: se eseguito direttamente, costruisce e stampa i sommari dei modelli.
if __name__ == "__main__":
    n_items = 1682    # Esempio: numero di film
    latent_dim = 50   # Dimensione dello spazio latente
    vae_model, encoder_model, decoder_model = create_vae_architecture(n_items, latent_dim)
    print("Modello VAE, encoder e decoder creati correttamente.")
    encoder_model.summary()
    decoder_model.summary()
    vae_model.summary()
