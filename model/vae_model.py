"""
Movie Recommender System - VAE Model Architecture Module (Traditional Implementation)
----------------------------------------------------------------------------------------
This module implements the Variational Autoencoder (VAE) in a traditional way,
calculating the loss (reconstruction loss + KL divergence) in the train_step and test_step
methods of the model subclass. This approach follows the pseudocode of the paper.

Structure:
1. Encoder:
   - Input of size (n_items,)
   - Dense layers: 1024, 512, 256 units, ReLU activation
   - Generates z_mean and z_log_var (with latent_dim size)
   - Applies the reparameterization trick to sample z
2. Decoder:
   - Input of size (latent_dim,)
   - Dense layers: 256, 512, 1024 units, ReLU activatio
   - Output layer: n_items units with sigmoid activation
3. VAE Model:
   - Subclass of tf.keras.Model that integrates encoder and decoder.
   - The call() method returns the reconstruction.
   - The train_step() and test_step() methods calculate the reconstruction loss (MSE) and KL divergence,
     sum their values, and update the model.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

# Function for the reparameterization trick
def sampling(args):
    """
    Applies the reparameterization trick.
    
    Parameters:
      args (tuple): contains (z_mean, z_log_var)
    
    Returns:
      z (tensor): sample from the latent space, calculated as:
                  z_mean + exp(0.5 * z_log_var) * epsilon, with epsilon ~ N(0,1)
    """
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.random.normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Function to build the encoder
def build_encoder(n_items, latent_dim):
    """
    Builds the VAE encoder.
    
    Parameters:
      - n_items (int): input size (number of movies)
      - latent_dim (int): latent space size
    
    Returns:
      - encoder: Keras model that maps the input to the latent vector and its parameters
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

# Function to build the decoder
def build_decoder(n_items, latent_dim):
    """
    Builds the VAE decoder.
    
    Parameters:
      - n_items (int): output size (number of movies)
      - latent_dim (int): latent space size
    
    Returns:
      - decoder: Keras model that maps the latent space to the input reconstruction
    """
    decoder_inputs = layers.Input(shape=(latent_dim,), name="decoder_input")
    x = layers.Dense(256, activation='relu', name="decoder_dense_1")(decoder_inputs)
    x = layers.Dense(512, activation='relu', name="decoder_dense_2")(x)
    x = layers.Dense(1024, activation='relu', name="decoder_dense_3")(x)
    decoder_outputs = layers.Dense(n_items, activation='sigmoid', name="decoder_output")(x)
    
    decoder = models.Model(decoder_inputs, decoder_outputs, name="decoder")
    return decoder

# Definition of the VAE model as a subclass of tf.keras.Model
class VAE(tf.keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        """
        Initializes the VAE model with the encoder and decoder.
        """
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        """
        Executes the forward pass of the model.
        """
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstruction = self.decoder(z)
        return reconstruction

    def train_step(self, data):
        """
        Defines the custom training step.
        Calculates:
          - reconstruction loss (MSE)
          - KL divergence loss
        and updates the weights.
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
        Defines the test/validation step.
        Calculates the same losses as training without updating the weights.
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
    Creates the VAE architecture.
    
    Steps:
      1. Builds the encoder with input (n_items,) and dense layers (1024, 512, 256) with ReLU activation,
         generating z_mean and z_log_var and sampling z via the reparameterization trick.
      2. Builds the decoder with input (latent_dim,) and dense layers (256, 512, 1024) with ReLU activation,
         generating an output with n_items units and sigmoid activation.
      3. Creates the VAE model by combining encoder and decoder.
      4. Compiles the model with the Adam optimizer and a dummy loss (lambda y_true, y_pred: 0),
         since the loss is calculated in the train_step/test_step.
    
    Returns:
      - vae: the compiled VAE model
      - encoder: the encoder model
      - decoder: the decoder model
    """
    encoder = build_encoder(n_items, latent_dim)
    decoder = build_decoder(n_items, latent_dim)
    vae = VAE(encoder, decoder)
    vae.compile(optimizer=optimizers.Adam(), loss=lambda y_true, y_pred: 0)
    return vae, encoder, decoder

# For testing: if executed directly, builds and prints the summaries of the models.
if __name__ == "__main__":
    n_items = 1682    # Example: number of movies
    latent_dim = 50   # Latent space size
    vae_model, encoder_model, decoder_model = create_vae_architecture(n_items, latent_dim)
    print("VAE model, encoder, and decoder created successfully.")
    encoder_model.summary()
    decoder_model.summary()
    vae_model.summary()