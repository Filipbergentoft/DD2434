import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfpl = tfp.layers
tfk = tf.keras
tfkl = tf.keras.layers


class VAE_MNIST(tfk.Model):
    
    def __init__(self, dim_z, kl_weight=1, name="autoencoder", **kwargs):
        super(VAE_MNIST, self).__init__(name=name, **kwargs)
        self.dim_x = (28, 28, 1)
        self.dim_z = dim_z
        self.encoder = self.encoder_z()
        self.decoder = self.decoder_x()
        self.kl_weight = kl_weight
        
    # Sequential API encoder
    def encoder_z(self):
        layers = [tfkl.InputLayer(input_shape=self.dim_x)]
        layers.append(tfkl.Conv2D(filters=32, kernel_size=3, strides=(2,2), 
                                  padding='valid', activation='relu'))
        layers.append(tfkl.Conv2D(filters=64, kernel_size=3, strides=(2,2), 
                                  padding='valid', activation='relu'))
        layers.append(tfkl.Flatten())
        # *2 because number of parameters for both mean and (raw) standard deviation
        layers.append(tfkl.Dense(self.dim_z*2, activation=None))
        return tfk.Sequential(layers)
    
    def encode(self, x_input):
        mu, rho = tf.split(self.encoder(x_input), num_or_size_splits=2, axis=1)
        sd = tf.math.log(1+tf.math.exp(rho))
        z_sample = mu + sd * tf.random.normal(shape=(self.dim_z,))
        return z_sample, mu, sd
    
    # Sequential API decoder
    def decoder_x(self):
        layers = [tfkl.InputLayer(input_shape=self.dim_z)]
        layers.append(tfkl.Dense(7*7*32, activation=None))
        layers.append(tfkl.Reshape((7,7,32)))
        layers.append(tfkl.Conv2DTranspose(filters=64, kernel_size=3, strides=2, 
                                           padding='same', activation='relu'))
        layers.append(tfkl.Conv2DTranspose(filters=32, kernel_size=3, strides=2, 
                                           padding='same', activation='relu'))
        layers.append(tfkl.Conv2DTranspose(filters=1, kernel_size=3, strides=1, 
                                           padding='same'))
        return tfk.Sequential(layers, name='decoder')
    
    def call(self, x_input):
        z_sample, mu, sd = self.encode(x_input)
        kl_divergence = tf.math.reduce_mean(- 0.5 * 
                tf.math.reduce_sum(1+tf.math.log(
                tf.math.square(sd))-tf.math.square(mu)-tf.math.square(sd), axis=1))
        x_logits = self.decoder(z_sample)
        # VAE_MNIST is inherited from tfk.Model, thus have class method add_loss()
        self.add_loss(self.kl_weight * kl_divergence)
        return x_logits
    
# custom loss function with tf.nn.sigmoid_cross_entropy_with_logits
def custom_sigmoid_cross_entropy_loss_with_logits(x_true, x_recons_logits):
    raw_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
                                            labels=x_true, logits=x_recons_logits)
    neg_log_likelihood = tf.math.reduce_sum(raw_cross_entropy, axis=[1, 2, 3])
    return tf.math.reduce_mean(neg_log_likelihood)

  
####################   The following code shows how to train the model   ####################
# set hyperparameters
epochs = 10
batch_size = 32
lr = 0.0001
latent_dim=16
kl_w=3
# compile and train tfk.Model
vae = VAE_MNIST(dim_z=latent_dim, kl_weight=kl_w)
vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), 
            loss=custom_sigmoid_cross_entropy_loss_with_logits)
train_history = vae.fit(x=train_images, y=train_images, batch_size=batch_size, epochs=epochs, 
                        verbose=1, validation_data=(test_images, test_images), shuffle=True)