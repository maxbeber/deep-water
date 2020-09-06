import tensorflow as tf
from tensorflow.keras import models, layers

class DeepAutoEncoder(tf.keras.Model):
    """ Represents an autoencoder for image denoising. """
    def __init__(self, image_size):
        super().__init__()
        self.image_size = image_size
        self._initialize_layers()
        self._build_graph()


    def call(self, x):
        z = self._encode(x)
        x_prime = self._decode(z)
        return x_prime

    
    def _build_graph(self):
        input_shape = (self.image_size, self.image_size, 1)
        self.build((None,) + input_shape)
        encoder_input = tf.keras.Input(shape=input_shape)
        _ = self.call(encoder_input)


    def _decode(self, z):
        x_prime = self.conv2d_4(z)
        x_prime = self.up_sampling2d_1(x_prime)
        x_prime = self.conv2d_5(x_prime)
        x_prime = self.up_sampling2d_2(x_prime)
        x_prime = self.conv2d_6(x_prime)
        x_prime = self.up_sampling2d_3(x_prime)
        x_prime = self.conv2d_7(x_prime)
        return x_prime


    def _encode(self, x):
        z = self.conv2d_1(x)
        z = self.max_pooling2d_1(z)
        z = self.conv2d_2(z)
        z = self.max_pooling2d_2(z)
        z = self.conv2d_3(z)
        z = self.max_pooling2d_3(z)
        return z


    def _initialize_layers(self):
        # encoder
        self.conv2d_1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')
        self.max_pooling2d_1 = layers.MaxPooling2D((2, 2), padding='same')
        self.conv2d_2 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')
        self.max_pooling2d_2 = layers.MaxPooling2D((2, 2), padding='same')
        self.conv2d_3 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')
        self.max_pooling2d_3 = layers.MaxPooling2D((2, 2), padding='same')
        # decoder
        self.conv2d_4 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')
        self.up_sampling2d_1 = layers.UpSampling2D((2, 2))
        self.conv2d_5 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')
        self.up_sampling2d_2 = layers.UpSampling2D((2, 2))
        self.conv2d_6 = layers.Conv2D(32, (3, 3), activation='relu', padding='valid')
        self.up_sampling2d_3 = layers.UpSampling2D((2, 2))
        self.conv2d_7 = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')

if __name__ == '__main__':
    image_size = 124
    dae = DeepAutoEncoder(image_size)
    dae.summary()