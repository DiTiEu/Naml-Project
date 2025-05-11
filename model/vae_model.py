from tensorflow.keras import Model # type: ignore
from tensorflow.keras.layers import Input, Dense # type: ignore
import tensorflow as tf
from keras.saving import serialize_keras_object # type: ignore
import keras

@keras.saving.register_keras_serializable(package="CustomVAE")
class CustomVAE(Model):
    def __init__(self, encoder, decoder, beta=0.1, kl_weight=1.0, **kwargs):
        super(CustomVAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta
        self.kl_weight = kl_weight
        self.kl_weight_tensor = tf.Variable(kl_weight, trainable=False, name='kl_weight')

    def call(self, inputs):
        # Codifica
        z_mean, z_log_var, z = self.encoder(inputs)

        # Decodifica
        reconstructed = self.decoder(z)

        # Calcolo delle loss
        recon_loss = self.reconstruction_loss(inputs, reconstructed)
        kl = self.kl_loss(z_mean, z_log_var)

        # Aggiungo la perdita totale con beta e kl_weight
        # total_loss = recon_loss + self.beta * self.kl_weight_tensor * kl # Loss calcolata con i coefficienti
        total_loss = recon_loss + kl
        self.add_loss(total_loss)

        return reconstructed
    
    def get_config(self):
        base_config = super().get_config()
        config  = {
            "encoder": keras.saving.serialize_keras_object(self.encoder), 
            "decoder": keras.saving.serialize_keras_object(self.decoder),
            "beta": self.beta,
            "kl_weight": self.kl_weight
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        encoder_config = config.pop("encoder")
        encoder = keras.saving.deserialize_keras_object(encoder_config)
        decoder_config = config.pop("decoder")
        decoder = keras.saving.deserialize_keras_object(decoder_config)
        beta = config.pop("beta", 0.1)
        kl_weight = config.pop("kl_weight", 1.0)
        return cls(encoder=encoder, decoder=decoder, beta=beta, kl_weight=kl_weight, **config)

    def kl_loss(self, z_mean, z_log_var):
        # Perdita di Kullback-Leibler per il VAE
        kl_loss = -0.5 * tf.reduce_mean(
            z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1
        )
        return kl_loss

    # Essendo che dobbiamo verificare che la matrice ricostruita non si allontani troppo da quella originale,
    # calcoliamo la perdita di ricostruzione solamente sui valori non nulli (ovvero quelli che ci interessano).
    def reconstruction_loss(self, true, reconstructed):
        # Create mask for non-zero entries
        mask = tf.cast(tf.not_equal(true, 0), tf.float32)
        
        # Calculate squared error only for non-zero entries
        squared_error = tf.square(true - reconstructed)
        masked_squared_error = mask * squared_error
        
        # Sum of squared errors for non-zero entries
        sum_squared_error = tf.reduce_sum(masked_squared_error)
        
        # Count of non-zero entries (with small epsilon to prevent division by zero)
        count_non_zero = tf.reduce_sum(mask) + 1e-8
        
        # Calculate mean squared error only for non-zero entries
        reconstruction_loss = sum_squared_error / count_non_zero
        
        return reconstruction_loss