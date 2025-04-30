import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras import backend as K # type: ignore

def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def create_encoder(n_items, latent_dim):
    inputs = Input(shape=(n_items,))
    x = Dense(1024, activation='relu')(inputs)
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)
    z = Lambda(sampling, name='z')([z_mean, z_log_var])
    return Model(inputs, [z_mean, z_log_var, z], name='encoder')

def create_decoder(n_items, latent_dim):
    latent_inputs = Input(shape=(latent_dim,))
    x = Dense(256, activation='relu')(latent_inputs)
    x = Dense(512, activation='relu')(x)
    x = Dense(1024, activation='relu')(x)
    outputs = Dense(n_items, activation='sigmoid')(x)
    return Model(latent_inputs, outputs, name='decoder')