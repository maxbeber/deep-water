import tensorflow as tf
from metric import mean_iou
from tensorflow.keras import Input, layers
from tensorflow.keras.models import Model

class Unet():
    """
    Represents a deep neural network using an U-Net architecture.

    Parameters
    ----------
    image_size : the size of the image for the input layer.
    """
    def __init__(self, image_size):
        input_encoder = Input(image_size)
        z = input_encoder
        f = 8
        ff2 = 64
        # encoder
        skip_connections = []
        for _ in range(6):
            z = layers.Conv2D(f, 3, activation='relu', padding='same')(z)
            z = layers.Conv2D(f, 3, activation='relu', padding='same')(z)
            skip_connections.append(z)
            z = layers.MaxPooling2D((2, 2), padding='same')(z)
            f = f * 2
        # bottleneck
        j = len(skip_connections) - 1
        z = layers.Conv2D(f, 3, activation='relu', padding='same')(z)
        z = layers.Conv2D(f, 3, activation='relu', padding='same')(z)
        z = layers.Conv2DTranspose(ff2, 2, strides=(2, 2), padding='same')(z)
        x_prime = layers.Concatenate(axis=3)([z, skip_connections[j]])
        j = j - 1 
        # decoder
        for _ in range(5):
            ff2 = ff2 // 2
            f = f // 2 
            x_prime = layers.Conv2D(f, 3, activation='relu', padding='same')(x_prime)
            x_prime = layers.Conv2D(f, 3, activation='relu', padding='same')(x_prime)
            x_prime = layers.Conv2DTranspose(ff2, 2, strides=(2, 2), padding='same')(x_prime)
            x_prime = layers.Concatenate(axis=3)([x_prime, skip_connections[j]])
            j = j - 1 
        # classification
        x_prime = layers.Conv2D(f, 3, activation='relu', padding='same')(x_prime)
        x_prime = layers.Conv2D(f, 3, activation='relu', padding='same')(x_prime)
        output_decoder = layers.Conv2D(1, 1, activation='sigmoid')(x_prime)
        # create the model
        self._model = Model(input_encoder, output_decoder)
        self._model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = [mean_iou])
    

    def get_model_summary(self):
        return self._model.summary()


if __name__ == '__main__':
    image_size = (512, 512, 3)
    dae = Unet(image_size)
    dae.get_model_summary()