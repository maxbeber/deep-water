import glob
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import rasterio
from matplotlib.image import imsave
from PIL import Image, ImageDraw

class WaterMask:
    """
    Represents a set of functions to work with satellite imagery
    and generate segmentation masks.
    """
    def __init__(self, base_folder):
        self.base_folder = base_folder
        self.data_folder = os.path.join(base_folder, 'data')
        self.mask_folder = os.path.join(base_folder, 'mask')
        self._create_mask_folder()


    def create_mask(self, annotation, image_name):
        file_path = os.path.join(self.data_folder, image_name)
        image = self._load_image(file_path)
        nx, ny, _ = np.shape(image)
        mask = np.zeros((nx, ny))
        X, Y = self._get_annotation_points_from_vgg_json(annotation, image_name)
        for x, y in zip(X, Y):
            polygon = np.vstack((x, y)).reshape((-1), order='F').tolist()
            img = Image.new('L', (nx, ny), 0)
            ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
            m = np.flipud(np.rot90(np.array(img)))
            mask = mask + m
        file_name = os.path.join(self.mask_folder, image_name)
        imsave(file_name, mask.astype('uint8'))


    def create_masks(self, annotations):
        all_dicts = {}
        for annotation in annotations:
            all_dicts.update(annotation)
        images = sorted(all_dicts.keys())
        for image in images:
            self.create_mask(annotations, image)


    def display_image_with_annotations(self, image_name, annotations, annotation_color='r'):
        image_path = self._get_image_path(image_name)
        image = self._load_image(image_path)
        X, Y = self._get_annotation_points_from_vgg_json(annotations, image_name)
        plt.imshow(image)
        plt.title(image_name)
        plt.axis('off')
        for i in range(len(X)):
            plt.plot(Y[i], X[i], annotation_color)


    def display_mask(self, image_name):
        image_path = self._get_mask_path(image_name)
        image = self._load_image(image_path)
        plt.imshow(image)
        plt.title(image_name)
        plt.axis('off')


    def _create_mask_folder(self):
        folder_exists = os.path.exists(self.mask_folder)
        if not folder_exists:
            try:
                os.mkdir(self.mask_folder)
            except OSError as os_error:
                print (f'Creation of the directory failed. {os_error}')

    
    def _get_all_images(self):
        files = glob.glob(f'{self.data_folder}/*.jpg')
        return files


    def _get_image_path(self, image_name):
        file_path = os.path.join(self.data_folder, image_name)
        return file_path


    def _get_mask_path(self, image_name):
        file_path = os.path.join(self.mask_folder, image_name)
        return file_path


    def _get_annotation_points_from_vgg_json(self, annotations, image_name):
        """
        Retrive all anotation pairs for a given image.
        Note: JSON coordinates are flippled relative to the image.
        """
        X = []
        Y= []
        for i in annotations[image_name]['regions']:
            x = annotations[image_name]['regions'][i]['shape_attributes']['all_points_x']
            y = annotations[image_name]['regions'][i]['shape_attributes']['all_points_y']
            X.append(x)
            Y.append(y)
        assert len(X) == len(Y)
        return Y, X


    def _load_image(self, file_path):
        with rasterio.open(file_path) as dataset:
            image = self._reshape_dataset_as_image(dataset)
        return image


    def _reshape_dataset_as_image(self, dataset):
        """
        Returns the source array reshaped into the order
        expected by image processing libraries by swapping
        the axes order from (bands, rows, columns)
        to (rows, columns, bands).
            
        Parameters
        ----------
        raster: array-like of shape (bands, rows, columns).
    
        Returns the source array reshaped.
        """
        bands = dataset.read()
        image = np.ma.transpose(bands, [1, 2, 0])
        return image