import glob
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from tensorflow.keras.callbacks import Callback


class PlotLearning(Callback):
    """
    Display the model's progress during training at the end of each epoch.
    """
    def __init__(self, model, test_images, image_size=(256, 256), threshold_water_pixel=100, crf_model=None):
        self.crf_model = crf_model
        self.image_size = image_size
        self.model = model
        self.test_images = test_images
        self.threshold_water_pixel = threshold_water_pixel


    def on_train_begin(self, logs={}):
        super().on_train_begin(logs=logs)


    def on_epoch_end(self, epoch, logs={}):
        super().on_epoch_end(epoch, logs=logs)
        self._display_triplet()


    def _display_triplet(self):
        image_path = np.random.choice(self.test_images)
        raw_image = Image.open(image_path)
        raw_image = np.array(raw_image.resize(self.image_size)) / 255.0
        raw_image = raw_image[:, :, :3]
        prediction = 255 * self.model.predict(np.expand_dims(raw_image, 0)).squeeze()
        mask = (prediction > self.threshold_water_pixel).astype('int')
        if self.crf_model:
            mask = self.crf_model.refine(raw_image, mask)
        mask = np.stack((mask,) * 3, axis= -1)
        triplet = np.concatenate([raw_image, mask, raw_image * mask], axis = 1)
        plt.axis('off')
        plt.imshow(triplet)
        plt.show()