from keras import backend as K
from keras.models import Model
from keras.losses import binary_crossentropy
from keras.layers import Lambda, Input, Dense
from sklearn.model_selection import train_test_split

class VAE:
    def __init__(self, original_dim, input_shape, 
                 intermediate_dim=128, latent_dim=2, summary=False):
        
        self._build_model(original_dim, input_shape,
                         intermediate_dim, 
                          latent_dim, summary)
    
    def _build_model(self, original_dim, input_shape, intermediate_dim, latent_dim,
                    summary=False):
        inputs = Input(shape=input_shape, name='encoder_input')
        x = inputs
        x = Dense(intermediate_dim, activation='relu')(x)
        x = Dense(intermediate_dim//2, activation='relu')(x)
        
        z_mean = Dense(latent_dim, name='z_mean')(x)
        z_log_var = Dense(latent_dim, name='z_log_var')(x)

        z = Lambda(self.sampling, output_shape=(latent_dim,), 
                   name='z')([z_mean, z_log_var])

        self.encoder = Model(inputs, [z_mean, z_log_var, z], 
                        name='encoder')
        
        latent_inputs = Input(shape=(latent_dim,), 
                              name='z_sampling')
        x = latent_inputs
        x = Dense(intermediate_dim//2, activation='relu')(x)
        x = Dense(intermediate_dim, activation='relu')(x)
        outputs = Dense(original_dim, activation='sigmoid')(x)

        self.decoder = Model(latent_inputs, outputs, name='decoder')
        outputs = self.decoder(self.encoder(inputs)[2])
        self.vae = Model(inputs, outputs, name='vae_mlp')
        
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
        
    def sampling(self, args):
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    def fit(self, x_train, x_test, epochs=100, batch_size=100,
           verbose=1):
        self.vae.fit(x_train, 
            shuffle=True,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            validation_data=(x_test, None))

    '''
    Function that first splits the data into training and evaluation set and then fits the
    generator to the training set.
    args:
    X: numpy arrray, data to be splited
    Others are the same as in fit.
    '''
    def fit_unsplit(self, X, epochs=100, batch_size=100, verbose=1):
        x_train, x_test = train_test_split(X, test_size = 0.5)
        self.fit(x_train, x_test, epochs, batch_size, verbose)
    
    def encoder_predict(self, x_test, batch_size=100):
        return self.encoder.predict(x_test,
                                   batch_size=batch_size)
    
    def generate(self, latent_val, batch_size=100):
        return self.decoder.predict(latent_val)
    
    def predict(self, x_test, batch_size=1):
        prediction = self.vae.predict(x_test)
        return prediction