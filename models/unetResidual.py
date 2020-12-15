from models.unetBase import UnetBase
from tensorflow.keras import Input, layers
from tensorflow.keras.models import Model

class UnetResidual(UnetBase):
    """
    Represents a deep neural network using an U-Net architecture.

    Parameters
    ----------
    model_name: the name of the model.
    image_size : the size of the image for the input layer.
    """
    def __init__(self, model_name, image_size, version=1):
        super().__init__()
        self.image_size = image_size
        self.model_name = model_name
        self.model = self.pre_build(version)
        
    
    def batchnorm_act(self, x):
        x = layers.BatchNormalization()(x)
        return layers.Activation("relu")(x)


    def conv_block(self, x, filters, kernel_size=(3, 3), padding="same", strides=1):
        conv = self.batchnorm_act(x)
        return layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(conv)


    def bottleneck_block(self, x, filters, kernel_size=(3, 3), padding="same", strides=1):
        conv = layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
        conv = self.conv_block(conv, filters, kernel_size=kernel_size, padding=padding, strides=strides)
        bottleneck = layers.Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
        bottleneck = self.batchnorm_act(bottleneck)
        return layers.Add()([conv, bottleneck])


    def res_block(self, x, filters, kernel_size=(3, 3), padding="same", strides=1):
        res = self.conv_block(x, filters, kernel_size=kernel_size, padding=padding, strides=strides)
        res = self.conv_block(res, filters, kernel_size=kernel_size, padding=padding, strides=1)
        bottleneck = layers.Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
        bottleneck = self.batchnorm_act(bottleneck)
        return layers.Add()([bottleneck, res])


    def upsamp_concat_block(self, x, xskip):
        u = layers.UpSampling2D((2, 2))(x)
        return layers.Concatenate()([u, xskip])


    def build(self, image_size, f):
        image_width, image_height = image_size
        image_shape = (image_width, image_height, 3)
        input_tensor = Input(image_shape)
        ## downsample  
        e1 = self.bottleneck_block(input_tensor, f)
        f = int(f * 2)
        e2 = self.res_block(e1, f, strides=2)
        f = int(f * 2)
        e3 = self.res_block(e2, f, strides=2)
        f = int(f * 2)
        e4 = self.res_block(e3, f, strides=2)
        f = int(f * 2)
        _ = self.res_block(e4, f, strides=2)
        ## bottleneck
        b0 = self.conv_block(_, f, strides=1)
        _ = self.conv_block(b0, f, strides=1)
        ## upsample
        _ = self.upsamp_concat_block(_, e4)
        _ = self.res_block(_, f)
        f //= 2
        _ = self.upsamp_concat_block(_, e3)
        _ = self.res_block(_, f)
        f //= 2
        _ = self.upsamp_concat_block(_, e2)
        _ = self.res_block(_, f)
        f //= 2
        _ = self.upsamp_concat_block(_, e1)
        _ = self.res_block(_, f)
        ## classify
        output_tensor = layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(_)
        #model creation 
        model = Model(input_tensor, output_tensor)
        return model


    def pre_build(self, version):
        try:
            if version == 1:
                model = self.build(self.image_size, 16)
            elif version == 2:
                model = self.build(self.image_size, 32)
            else:
                raise Exception(f'version {version} is not valid.')
            return model
        except Exception as error:
            print('Input error:', error)