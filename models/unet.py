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
        image_width, image_height = image_size
        image_shape = (image_width, image_height, 3)
        input_encoder = Input(shape=image_shape)
        f = 8
        ff2 = 64
        skip_connections = []
        # encoder
        z = input_encoder
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
        self.model = Model(input_encoder, output_decoder)
    

    def get_model_summary(self):
        return self.model.summary()


if __name__ == '__main__':
    image_size = (512, 512)
    dae = Unet(image_size)
    dae.get_model_summary()