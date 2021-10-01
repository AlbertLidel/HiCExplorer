import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses, models
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model

class Autoencoder(Model):
    
    def __init__(self, latent_dim, feature_size,activation_enc='relu',activation_dec='sigmoid'):
        
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.feature_size = feature_size
        self.encoder = tf.keras.Sequential([
        layers.Dense(latent_dim, activation=activation_enc),
        ])
        self.decoder = tf.keras.Sequential([
        layers.Dense(feature_size, activation=activation_dec)
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def build_and_fit(x_train,x_test,outfile,epochs=10,activation_enc='relu',activation_dec='sigmoid'):
    '''initialize a new model with test set and save at outfile'''
    
    assert x_train.shape[1] == x_test.shape[1]
    latent_dim = x_train.shape[1]
    autoencoder = Autoencoder(latent_dim,x_train.shape[1],activation_enc,activation_dec)
    
    autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
    autoencoder.fit(x_train, x_train,
                    epochs=epochs,
                    shuffle=True,
                    validation_data=(x_test, x_test))
    
    tf.saved_model.save(autoencoder, outfile)
    #autoencoder.save(outfile,save_format="tf")
    
def dimensionality_reduction(features,autoencoder_file):
    '''perform dimensionality reduction with autoencoder on given features'''
    
    autoencoder = tf.saved_model.load(export_dir=autoencoder_file,tags=None, options=None)
    #autoencoder = models.load_model(autoencoder_file)

    assert autoencoder.feature_size == features.shape[1]
    
    encoded_features = autoencoder.encoder(features).numpy()

    return encoded_features