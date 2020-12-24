from tensorflow.keras import Input, layers
from tensorflow.keras.models import Model

class DeepAutoEncoder:
    """
    Represents an autoencoder for image denoising.

    Parameters
    ----------
    image_size : the size of the image for the input layer.
    """
    def __init__(self, image_size):
        # encoder
        image_shape=(image_size, image_size, 1)
        input_encoder = Input(shape=image_shape)
        z = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_encoder)
        z = layers.MaxPooling2D((2, 2), padding='same')(z)
        z = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(z)
        z = layers.MaxPooling2D((2, 2), padding='same')(z)
        z = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(z)
        # bottle neck
        x_prime = layers.MaxPooling2D((2, 2), padding='same')(z)
        # decoder
        x_prime = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x_prime)
        x_prime = layers.UpSampling2D((2, 2))(x_prime)
        x_prime = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x_prime)
        x_prime = layers.UpSampling2D((2, 2))(x_prime)
        x_prime = layers.Conv2D(32, (3, 3), activation='relu')(x_prime)
        x_prime = layers.UpSampling2D((2, 2))(x_prime)
        output_decoder = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x_prime)
        # create the model
        self.model = Model(input_encoder, output_decoder)
        self.model.compile(optimizer='rmsprop', loss='binary_crossentropy')
    

    def get_model_summary(self):
        return self.model.summary()


if __name__ == '__main__':
    image_size = 124
    dae = DeepAutoEncoder(image_size)
    dae.get_model_summary()