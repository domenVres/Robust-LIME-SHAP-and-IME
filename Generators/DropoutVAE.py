import numpy as np

from keras import backend as K
from keras.models import Model
from keras.losses import binary_crossentropy
from keras.layers import Input, Dense, Dropout
from keras.regularizers import l2
from sklearn.model_selection import train_test_split

class DropoutVAE:
    def __init__(self, original_dim, input_shape, 
                 intermediate_dim=32, latent_dim=3, dropout=0.05, 
                 summary=False):
        
        self._build_model(original_dim, input_shape,
                         intermediate_dim, 
                          latent_dim, summary,
                          dropout)

    def _build_model(self, original_dim, input_shape, intermediate_dim, latent_dim,
                    summary=False, dropout=0.05):
        inputs = Input(shape=input_shape, name='encoder_input')
        x = inputs
        x = Dense(intermediate_dim, activation='relu')(x)
        x = Dense(intermediate_dim//2, activation='relu')(x)
        
        z_mean = Dense(latent_dim, name='z_mean')(x)
        z_log_var = Dense(latent_dim, name='z_log_var')(x)
        
        # We remove the z layer ( z layer is used in VAE but not here)
        self.encoder = Model(inputs, [z_mean, z_log_var], 
                        name='encoder')
        
        latent_inputs = Input(shape=(latent_dim,), 
                              name='z_sampling')
        x = latent_inputs
        x = Dense(intermediate_dim//2, activation='relu',
                 kernel_regularizer=l2(1e-4),
                 bias_regularizer=l2(1e-4))(x)
        x = Dropout(dropout)(x)
        x = Dense(intermediate_dim, activation='relu',
                 kernel_regularizer=l2(1e-4),
                 bias_regularizer=l2(1e-4))(x)
        x = Dropout(dropout)(x)
        outputs = Dense(original_dim, activation='sigmoid',
                       kernel_regularizer=l2(1e-4),
                       bias_regularizer=l2(1e-4))(x)

        self.decoder = Model(latent_inputs, 
                             outputs, 
                             name='decoder')
        
        # Here we take the mean (not the z-layer) 
        outputs = self.decoder(self.encoder(inputs)[0])
        self.vae = Model(inputs, outputs, 
                         name='vae_mlp')
        
        reconstruction_loss = binary_crossentropy(inputs, outputs)
        reconstruction_loss *= original_dim
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        
        vae_loss = K.mean(reconstruction_loss + kl_loss)	
        
        self.vae.add_loss(vae_loss)
        self.vae.compile(optimizer='adam')
        if summary: 
            print(self.vae.summary())
        
    def fit(self, x_train, x_test, epochs=100, batch_size=100,
           verbose=1):
        self.vae.fit(x_train, 
            shuffle=True,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            validation_data=(x_test, None))

    def fit_unsplit(self, X, epochs=100, batch_size=100, verbose=1):
        x_train, x_test = train_test_split(X, test_size = 0.5)
        self.fit(x_train, x_test, epochs, batch_size, verbose)
    
    def encoder_predict(self, x_test, batch_size=100):
        return self.encoder.predict(x_test,
                                   batch_size=batch_size)
    
    def generate(self, latent_val, batch_size=100):
        return self.decoder.predict(latent_val)
    
    def predict(self, x_test, batch_size=1, nums=1000):
        Yt_hat = []
        for _ in range(nums):
            Yt_hat.extend(self.vae.predict(x_test))
                          
        return np.asarray(Yt_hat)
                          
    def mean_predict(self, x_test, batch_size=1, nums=1000):
        predict_stochastic = K.function([self.decoder.layers[0].input,
                                K.learning_phase()],
                                [self.decoder.get_output_at(0)])
        latents = self.encoder.predict(x_test)[0]
        Yt_hat = []
        for _ in range(nums):
            Yt_hat.append(predict_stochastic([latents, 1])) 
        return np.asarray(Yt_hat)