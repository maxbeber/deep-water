import glob
import numpy as np
import os
import random
from PIL import Image

class BatchLoader:
    """
    Represents a batch containing pairs of images and labels.
    Parameters
    ----------
    base_folder : folder where the images are located
    batch_size : size of the batch
    image_size : size of each image
    """
    def __init__(self, images, image_folder, mask_folder, batch_size, image_size, threshold_water_pixel=100):
        self.image_folder = os.path.join(image_folder, 'data')
        self.mask_folder = os.path.join(mask_folder, 'data')
        self.batch_size = batch_size
        self.image_size = image_size
        self.threshold_water_pixel = threshold_water_pixel
        self.images = [image.split(os.sep)[-1] for image in images]
    

    def __call__(self):
        random.shuffle(self.images)
        while True:
            batch_x = []
            batch_y = []
            images = np.random.choice(self.images, size=self.batch_size)
            for image in images:
                raw_image = self._generate_image(image)
                n = raw_image.shape[0]
                batch_x.append(raw_image)
                mask_image = self._generate_mask(image, n)
                batch_y.append(mask_image)
            batch_x = np.array(batch_x) / 255.0
            batch_y = np.array(batch_y)
            batch_y = np.expand_dims(batch_y, 3)
            yield (batch_x, batch_y)


    def get_pair(self, x, y):
        image = x
        mask = y.squeeze()
        mask = np.stack((mask,) * 3, axis=-1)
        pair = np.concatenate([image, mask, image * mask], axis=1)
        return pair
    
    
    def _generate_image(self, image):
        raw_file = os.path.join(self.image_folder, image)
        raw_image = Image.open(raw_file)
        raw_image = raw_image.resize(self.image_size)
        raw_image = np.array(raw_image)
        if raw_image.ndim == 2:
            raw_image = np.stack((raw_image,) * 3, axis=-1)
        else:
            raw_image = raw_image[:, :, :3]
        nx, ny, _ = np.shape(raw_image)
        n = np.minimum(nx, ny)
        raw_image = raw_image[:n, :n, :]
        return raw_image

    
    def _generate_mask(self, image, n):
        mask_file = os.path.join(self.mask_folder, image)
        mask_image = Image.open(mask_file)
        mask_image = mask_image.resize(self.image_size)
        mask_image = np.array(mask_image)
        if mask_image.ndim == 2:
            mask_image = np.stack((mask_image,) * 3, axis=-1)
        else:
            mask_image = mask_image[:, :, :3]
        mask_image = np.max(mask_image, axis=2)
        mask_image = (mask_image > self.threshold_water_pixel).astype('int')
        mask_image = mask_image[:n, :n]
        return mask_image