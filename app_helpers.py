import numpy as np
import os
import pandas as pd
import pyproj
import rasterio
import tensorflow as tf
from functools import partial
from models import UnetResidual
from shapely.geometry import Polygon
from shapely.ops import transform


def calculate_water(predicted_mask):
  white = len(predicted_mask[predicted_mask >= 0.5])
  black = len(predicted_mask[predicted_mask < 0.5])
  water_percentage = white / (white+black)

  return round(water_percentage, 5)


def get_geom(df):
    longit = [
        df["min_longitude"], 
        df["max_longitude"], 
        df["max_longitude"],
        df["min_longitude"] 
    ]
    latit = [
        df["min_latitude"], 
        df["min_latitude"], 
        df["max_latitude"],
        df["max_latitude"]
    ]
    coordinates = [[lat, long] for lat, long in zip (latit, longit)]
    polygon = Polygon(coordinates)

    return polygon


def get_sqkm(bounding_box):
    proj = partial(
        pyproj.transform, 
        pyproj.Proj(init='epsg:4326'),
        pyproj.Proj(init='epsg:4088')
    )
    bounding_box_new = transform(proj, bounding_box)
    sqm = bounding_box_new.area / 1000000

    return sqm


def get_water_land_per_year(fraction, area):
    water_sqkm = area * fraction
    land_sqkm = area - water_sqkm
    
    return (water_sqkm, land_sqkm)


def load_image(image):
    dataset = rasterio.open(image)
    bands = dataset.read()
    image = np.ma.transpose(bands, [1, 2, 0])
    
    return image
    

def load_model():
    model_file_name = 'saved_models/unet-residual-large-dice.h5'
    model_name = 'foo'
    image_size = (256, 256)
    unet_residual = UnetResidual(model_name, image_size, version=2)
    unet_residual.restore(model_file_name)
    
    return unet_residual


def slct_image(df, lake, year):
    country = df.loc[lake, "country"]
    name = df.loc[lake, "name"].replace(" ", "_").lower()
    folder = "assets/lakes"
    image_path = f"{folder}/{country}_{name}_s2cloudless_{year}.jpg"
    
    return image_path


def load_dataset(file_path):
    df = pd.read_json(file_path).T
    df["lat"] = (df['min_latitude'] + df['max_latitude']) / 2.0
    df["lon"] = (df['min_longitude'] + df['max_longitude']) / 2.0

    return df


def model_prediction(X, model):
    image_size = (256, 256)
    with rasterio.open(X) as dataset:
        bands = dataset.read()
        raw_image = np.ma.transpose(bands, [1, 2, 0])
        original_image = tf.Variable(raw_image)
        resized_image = tf.keras.preprocessing.image.smart_resize(original_image, image_size)
        y_pred = model.predict(np.expand_dims(resized_image, axis=0))
    
    return y_pred