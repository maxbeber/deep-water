import numpy as np
import tensorflow as tf

class DataAugmentationLoader:
    """
    Represents the augmentation process on the training dataset.
    Parameters
    ----------
    image_folder : folder containing the training images
    mask_folder : folder containing the respective mask for each image
    batch_size : size of the batch
    image_size : size of each image
    """
    def __init__(self, image_folder, mask_folder, batch_size, image_size):
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.batch_size = batch_size
        self.image_size = image_size
        self.seed = 111
        self.datagenerator_image_args = {
            "featurewise_center": False,
            "featurewise_std_normalization": False,
            "shear_range": 0,
            "zoom_range": 0.2,
            "rotation_range": 45,
            "horizontal_flip": True,
            "vertical_flip": True,
            "rescale":1.0/255
        }
        self.datagenerator_mask_args = {
            "featurewise_center": False,
            "featurewise_std_normalization": False,
            "shear_range": 0,
            "zoom_range": 0.2,
            "rotation_range": 45,
            "horizontal_flip": True,
            "vertical_flip": True,
            "rescale":1.0/255,
            "preprocessing_function":self._clip_mask,
        }


    def __call__(self):
        image_loader = tf.keras.preprocessing.image.ImageDataGenerator(**self.datagenerator_image_args)
        mask_loader = tf.keras.preprocessing.image.ImageDataGenerator(**self.datagenerator_mask_args)
        image_generator = image_loader.flow_from_directory(\
            self.image_folder,
            batch_size=self.batch_size,
            class_mode=None,
            seed=self.seed,
            shuffle=False,
            target_size=self.image_size)
        mask_generator = mask_loader.flow_from_directory(\
            self.mask_folder,
            batch_size=self.batch_size,
            class_mode=None,
            color_mode = 'grayscale',
            seed=self.seed,
            shuffle=False,
            target_size=self.image_size)
        data_generator = (pair for pair in zip(image_generator, mask_generator))

        return data_generator
    

    def get_pair(self, x, y):
        image = x
        mask = y.squeeze()
        mask = np.stack((mask,) * 3, axis=-1)
        return np.concatenate([image, mask], axis = 1)
        
    
    def _clip_mask(self, mask):
        mask[mask < 100] = 0
        mask[mask >= 150] = 255
        return mask