import argparse
import glob
import numpy as np
import os
import rasterio
import tensorflow as tf


class SmartResizer:
    """
    Represents a class that resize an image folder using
    the smart resize function available in tensorflow.
    Parameters
    ----------
    input_folder : folder where the images are located
    """
    def __init__(self, input_folder, output_folder):
        self.output_folder = output_folder
        self.search_pattern = os.path.join(input_folder, '*.jpg')
        self.image_paths = list(glob.glob(self.search_pattern))
        self._create_output_folder_if_necessary()


    def resize(self, image_size=(256, 256)):
        image_count = len(self.image_paths)
        print(f'{image_count} images have been found.')
        for image_path in self.image_paths:
            image_name = image_path.split(os.sep)[-1]
            output_path = os.path.join(self.output_folder, image_name)
            raw_image = self._load_image(image_path)
            raw_tensor = tf.Variable(raw_image)
            new_image = tf.keras.preprocessing.image.smart_resize(raw_tensor, image_size)
            tf.keras.preprocessing.image.save_img(output_path, new_image, scale=True)
        print(f'{image_count} images have been resized with success.')


    def _create_output_folder_if_necessary(self):
        output_folder_exist = os.path.exists(self.output_folder)
        if not output_folder_exist:
            try:
                os.mkdir(self.output_folder)
            except Exception as error:
                print(f'An error has occurred: {error}.')


    def _load_image(self, file_path):
        with rasterio.open(file_path) as dataset:
            image = self._reshape_dataset_as_image(dataset)
        return image


    def _reshape_dataset_as_image(self, dataset):
        bands = dataset.read()
        image = np.ma.transpose(bands, [1, 2, 0])
        return image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_folder', default=None, help='a folder that contains the original images')
    parser.add_argument('output_folder', default=None, help='a output folder where the images will be saved')
    parser.add_argument('--width', default=0, type=int, help='desired image width')
    parser.add_argument('--height', default=0, type=int, help='desired image height')
    args = parser.parse_args()
    input_folder = args.input_folder
    output_folder = args.output_folder
    image_width = args.width
    image_height = args.height
    try:
        image_size = (256, 256)
        smartResizer = SmartResizer(input_folder, output_folder)
        if image_width > 0 and image_height > 0:
            image_size = (image_width, image_height)
        smartResizer.resize(image_size)
    except Exception as error:
        print(f'An error has occurred: {error}.')