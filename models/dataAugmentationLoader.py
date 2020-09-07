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
        self.datagenerator_args = {
            "featurewise_center": True,
            "featurewise_std_normalization": True,
            "shear_range": 0,
            "zoom_range": 0,
            "rotation_range": 90,
            "horizontal_flip": True
        }


    def __call__(self):
        image_loader = tf.keras.preprocessing.image.ImageDataGenerator(**self.datagenerator_args)
        mask_loader = tf.keras.preprocessing.image.ImageDataGenerator(**self.datagenerator_args)
        image_generator = image_loader.flow_from_directory(\
            self.image_folder,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode=None,
            seed=self.seed,
            shuffle=False)
        mask_generator = mask_loader.flow_from_directory(\
            self.mask_folder,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode=None,
            seed=self.seed,
            shuffle=False)
        train_generator = (pair for pair in zip(image_generator, mask_generator))
        return train_generator
    

    def get_pair(self, X, Y, i):
        image = X[i].astype('uint8') / 255
        mask = np.max(Y[i], axis=2) / 255
        mask[mask >= 0.5]  = 1
        mask[mask < 0.5] = 0
        mask = np.stack((mask,) * 3, axis=-1)
        return np.concatenate([image, mask], axis = 1)