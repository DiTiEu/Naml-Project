from tensorflow.keras import Model # type: ignore
from tensorflow.keras.layers import Input, Dense # type: ignore
import tensorflow as tf
from keras.saving import serialize_keras_object # type: ignore

class CustomVAE(Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(CustomVAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        # Codifica
        z_mean, z_log_var, z = self.encoder(inputs)

        # Decodifica
        reconstructed = self.decoder(z)

        recon_loss = self.reconstruction_loss(inputs, reconstructed)
        kl = self.kl_loss(z_mean, z_log_var)

        # Aggiungo la perdita totale (puoi anche loggarla)
        total_loss = recon_loss + kl
        self.add_loss(total_loss)

        return reconstructed
    
    def get_config(self):
        return {
            "encoder": serialize_keras_object(self.encoder),
            "decoder": serialize_keras_object(self.decoder),
        }

    @classmethod
    def from_config(cls, config):
        encoder = tf.keras.models.deserialize(config.pop("encoder"))
        decoder = tf.keras.models.deserialize(config.pop("decoder"))
        return cls(encoder, decoder, **config)

    def kl_loss(self, z_mean, z_log_var):
        # Perdita di Kullback-Leibler per il VAE
        kl_loss = -0.5 * tf.reduce_mean(
            z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1
        )
        return kl_loss

    def reconstruction_loss(self, true, reconstructed):
        # Perdita di ricostruzione (MSE per il VAE)
        reconstruction_loss = tf.reduce_mean(
            tf.square(true - reconstructed)
        )
        return reconstruction_loss