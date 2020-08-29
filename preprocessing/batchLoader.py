import glob
import numpy as np
import os
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
    def __init__(self, base_folder, batch_size, image_size):
        self.base_folder = base_folder
        self.data_folder = os.path.join(self.base_folder, 'data')
        self.mask_folder = os.path.join(self.base_folder, 'mask')
        self.batch_size = batch_size
        self.image_size = image_size
        self.threshold_water_pixel = 100
    

    def generate(self):
        all_images = self._get_images()
        while True:
            batch_x = []
            batch_y = []
            images = np.random.choice(all_images, size=self.batch_size)
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
        raw_file = os.path.join(self.data_folder, image)
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
        mask_image = np.max(mask_image, axis=2)
        mask_image = (mask_image > self.threshold_water_pixel).astype('int')
        mask_image = mask_image[:n, :n]
        return mask_image


    def _get_images(self):
        image_list = []
        images = glob.glob(f'{self.mask_folder}/*.jpg')
        for image in images:
            image_file_name = image.split(os.sep)[-1]
            image_list.append(image_file_name)
        return image_list