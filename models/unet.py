from models.unetBase import UnetBase
from tensorflow.keras import Input, layers
from tensorflow.keras.models import Model

class Unet(UnetBase):
    """
    Represents a deep neural network using an U-Net architecture.

    Parameters
    ----------
    model_name: the name of the model.
    image_size : the size of the image for the input layer.
    """
    def __init__(self, model_name, image_size, version=1, dropout_rate=0.25):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.model_name = model_name
        self.image_size = image_size
        self.model = self.pre_build(version)


    def pre_build(self, version):
        try:
            if version == 1:
                model = self.build(self.image_size, 8, 64, False, False)
            elif version == 2:
                model = self.build(self.image_size, 16, 128, True, False)
            elif version == 3:
                model = self.build(self.image_size, 8, 64, True, True)
            elif version == 4:
                model = self.build(self.image_size, 16, 128, True, True)
            else:
                raise Exception(f'version {version} is not valid.')
            return model
        except Exception as error:
            print('Input error:', error)


    def build(self, image_size, f, ff2, flag_dropout, flag_batch_normalization):
        image_width, image_height = image_size
        image_shape = (image_width, image_height, 3)
        input_tensor = Input(shape=image_shape)
        skip_connections = []
        # encoder
        z = input_tensor
        for _ in range(6):
            z = layers.Conv2D(f, 3, activation='relu', padding='same')(z)
            z = layers.Conv2D(f, 3, activation='relu', padding='same')(z)
            if flag_batch_normalization:
                z = layers.BatchNormalization()(z)
            skip_connections.append(z)
            z = layers.MaxPooling2D((2, 2), padding='same')(z)
            if flag_dropout:
                z = layers.Dropout(self.dropout_rate)(z)
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
            if flag_dropout:
                x_prime = layers.Dropout(self.dropout_rate)(x_prime)
            j = j - 1 
        # classification
        x_prime = layers.Conv2D(f, 3, activation='relu', padding='same')(x_prime)
        x_prime = layers.Conv2D(f, 3, activation='relu', padding='same')(x_prime)
        output_tensor = layers.Conv2D(1, 1, activation='sigmoid')(x_prime)
        # create the model
        model = Model(input_tensor, output_tensor)
        return model