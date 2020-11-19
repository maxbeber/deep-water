import glob
import matplotlib
import numpy as np
import os
from models.dataAugmentationLoader import DataAugmentationLoader

class DataAugmentationGenerator:
    """
    Generate augmented data for a given dataset.
    Parameters
    ----------
    augmented_folder : folder where the images shall be saved
    image_size : size of each image
    """
    def __init__(self, image_folder, augmented_image_folder, mask_folder, augmented_mask_folder, image_size):
        self.image_folder = image_folder
        self.augmented_image_folder = augmented_image_folder
        self.mask_folder = mask_folder
        self.augmented_mask_folder = augmented_mask_folder
        self.image_size = image_size
        self.data_folder = 'data'
        self.augmented_prefix = 'augmented'
        self.batch_size = 1


    def generate(self, samples, initial_count = 1):
        counter = 0
        next_sample = initial_count
        self._create_augmented_image_folder()
        self._create_augmented_mask_folder()
        data_augmentation_loader = DataAugmentationLoader(self.image_folder, self.mask_folder, self.batch_size, self.image_size)
        data_augmentation_generator = data_augmentation_loader()
        while counter < samples:
            x, y = next(data_augmentation_generator)
            raw_image, raw_mask = np.squeeze(x), np.squeeze(y)
            file_name = f'{self.augmented_prefix}_{next_sample:03d}.jpg'
            image_file_name = os.path.join(self.augmented_image_folder, self.data_folder, file_name)
            mask_file_name = os.path.join(self.augmented_mask_folder, self.data_folder, file_name)
            matplotlib.image.imsave(image_file_name, raw_image)
            matplotlib.image.imsave(mask_file_name, raw_mask, cmap='gray')
            next_sample += 1
            counter += 1
        print(f'{counter} image(s) have been genearated with success!')


    def _create_augmented_image_folder(self):
        folder_exist = os.path.exists(self.augmented_image_folder)
        if not folder_exist:
            try:
                os.mkdir(self.augmented_image_folder)
                data_folder = os.path.join(self.augmented_image_folder, self.data_folder)
                os.mkdir(data_folder)
            except Exception as error:
                print(f'An error occurred: {error}.')

    
    def _create_augmented_mask_folder(self):
        folder_exist = os.path.exists(self.augmented_mask_folder)
        if not folder_exist:
            try:
                os.mkdir(self.augmented_mask_folder)
                data_folder = os.path.join(self.augmented_mask_folder, self.data_folder)
                os.mkdir(data_folder)
            except Exception as error:
                print(f'An error occurred: {error}.')