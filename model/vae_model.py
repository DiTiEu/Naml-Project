"""
Movie Recommender System - VAE Model Architecture Module (Corrected)
----------------------------------------------------------------------
This module implements the Variational Autoencoder (VAE) architecture as described in the paper.
It builds the following components:

1. Encoder:
   - Accepts an input of shape (n_items,) where n_items is the number of movies.
   - Applies several dense layers (with 1024, 512, and 256 units) using ReLU activation.
   - Creates two outputs: z_mean and z_log_var, representing the latent space parameters.

2. Reparameterization Trick:
   - Implements a sampling function that uses the latent mean and log variance to sample a latent vector,
     ensuring a smooth and continuous latent space.

3. Decoder:
   - Accepts a latent vector of shape (latent_dim,).
   - Applies dense layers (with 256, 512, and 1024 units) using ReLU activation.
   - Outputs a reconstructed vector with the same number of units as the input (n_items) using a sigmoid activation.

4. VAE Loss Layer:
   - A custom layer that computes the reconstruction loss (Mean Squared Error) and KL divergence loss,
     then adds the total loss to the model.
   - This approach wraps the loss computation in a Keras layer so that symbolic tensors are properly handled.

5. VAE Integration:
   - Connects the encoder, decoder, and loss layer into one model.
   - Compiles the model with the Adam optimizer.
"""

import tensorflow as tf
import keras
from tensorflow.keras import layers, models

def sampling(args):
    """
    Sampling function using the reparameterization trick.
    This function takes as input the mean and log variance of the latent distribution and returns a sampled latent vector.
    
    Parameters:
        args (tuple): (z_mean, z_log_var)
    
    Returns:
        z (tensor): A sample from the latent space computed as z_mean + exp(0.5 * z_log_var) * epsilon.
    """
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    # Generate random noise from a normal distribution
    epsilon = tf.random.normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class VAELossLayer(layers.Layer):
    """
    Custom layer to compute and add the VAE loss.
    It computes the reconstruction loss (MSE) and KL divergence,
    then adds the total loss to the model using self.add_loss().
    """
    def __init__(self, beta=1.0, **kwargs):
        super(VAELossLayer, self).__init__(**kwargs)
        self.beta = beta  # Fattore di bilanciamento per la KL divergence
    
    def call(self, inputs):
        encoder_inputs, reconstructed, z_mean, z_log_var = inputs
        # Compute reconstruction loss: mean squared error between input and reconstruction
        reconstruction_loss = tf.reduce_mean(tf.square(encoder_inputs - reconstructed))
        # Compute KL divergence loss to regularize the latent space
        kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        # Total loss is the sum of reconstruction loss and KL divergence loss
        total_loss = reconstruction_loss + kl_loss
        # Add the loss to the layer (and hence to the model)
        self.add_loss(total_loss)
        # Return the reconstructed output (for the rest of the model)
        return reconstructed

def create_vae_architecture(n_items, latent_dim=50, beta=1.0):
    """
    Creates the VAE architecture based on the given number of items (input features) and latent dimension.
    
    Parameters:
        n_items (int): The number of items (movies) in the dataset (input dimension).
        latent_dim (int): The dimension of the latent space (default is 50).
    
    Returns:
        vae (tf.keras.Model): The compiled VAE model.
        encoder (tf.keras.Model): The encoder model.
        decoder (tf.keras.Model): The decoder model.
    """
    # ----------------------
    # Encoder Architecture
    # ----------------------
    # Define the input layer for the encoder with shape (n_items,)
    encoder_inputs = layers.Input(shape=(n_items,), name="encoder_input")
    
    # Add dense layers with decreasing number of units using ReLU activation
    x = layers.Dense(1024, activation='relu', name="encoder_dense_1")(encoder_inputs)
    x = layers.Dense(512, activation='relu', name="encoder_dense_2")(x)
    x = layers.Dense(256, activation='relu', name="encoder_dense_3")(x)
    
    # Create latent variables: z_mean and z_log_var with 'latent_dim' units each
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    
    # Use a Lambda layer to implement the sampling function (reparameterization trick)
    z = layers.Lambda(sampling, output_shape=(latent_dim,), name="z")([z_mean, z_log_var])
    
    # Instantiate the encoder model mapping the inputs to the latent vector and its parameters
    encoder = models.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    
    # ----------------------
    # Decoder Architecture
    # ----------------------
    # Define the input layer for the decoder with shape (latent_dim,)
    decoder_inputs = layers.Input(shape=(latent_dim,), name="decoder_input")
    
    # Add dense layers mirroring the encoder structure (in reverse order) using ReLU activation
    x = layers.Dense(256, activation='relu', name="decoder_dense_1")(decoder_inputs)
    x = layers.Dense(512, activation='relu', name="decoder_dense_2")(x)
    x = layers.Dense(1024, activation='relu', name="decoder_dense_3")(x)
    
    # Define the output layer with 'n_items' units and a sigmoid activation to reconstruct the original input
    decoder_outputs = layers.Dense(n_items, activation='sigmoid', name="decoder_output")(x)
    
    # Instantiate the decoder model mapping latent space to the reconstructed input
    decoder = models.Model(decoder_inputs, decoder_outputs, name="decoder")
    
    # ----------------------
    # VAE Model: Connecting Encoder, Decoder, and Loss Layer
    # ----------------------
    # Get the latent representations from the encoder
    z_mean, z_log_var, z = encoder(encoder_inputs)
    # Obtain the reconstructed input from the decoder
    reconstructed = decoder(z)
    
    # Pass the inputs, reconstruction, and latent parameters through the custom loss layer
    loss_output = VAELossLayer(beta=beta, name="vae_loss")([encoder_inputs, reconstructed, z_mean, z_log_var])
    
    # Create the final VAE model that maps encoder_inputs to the output of the loss layer
    vae = models.Model(encoder_inputs, loss_output, name="vae")
    
    # ----------------------
    # Compile the VAE Model
    # ----------------------
    vae.compile(optimizer=keras.optimizers.Adam(), loss=None)
    
    return vae, encoder, decoder

# For testing purposes: when running this file directly, build a sample model.
if __name__ == "__main__":
    # Example usage: Assume there are 1682 items (movies) and use a latent dimension of 50.
    n_items = 1682
    latent_dim = 50
    beta = 1.0
    vae_model, encoder_model, decoder_model = create_vae_architecture(n_items, latent_dim, beta)
    print("VAE, encoder, and decoder models have been successfully created.")
    # Optionally, print model summaries to verify the architecture
    encoder_model.summary()
    decoder_model.summary()
    vae_model.summary()