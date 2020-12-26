import numpy as np
import os
import pandas as pd
import pyproj
import rasterio
import tensorflow as tf
from functools import partial
from models import UnetResidual
from preprocessing import CrfLabelRefiner
from shapely.geometry import Polygon
from shapely.ops import transform


def calculate_water(predicted_mask):
  white = len(predicted_mask[predicted_mask >= 0.5])
  black = len(predicted_mask[predicted_mask < 0.5])
  water_percentage = white / (white+black)

  return round(water_percentage, 5)


def ensemble_predict(models, raw_image):
    model1, model2 = models
    image = np.expand_dims(raw_image, axis=0)
    y_pred_1 = model1.predict(image)
    y_pred_2 = model2.predict(image)
    combined_mask = _get_ensemble_mask(image, y_pred_1, y_pred_2)
    
    return combined_mask


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
    

def load_models():
    model_name = 'unet-residual-large-dice'
    model_file_name = 'unet-residual-large-dice.h5'
    unet_residual = _load_model(model_name, model_file_name)
    model_name = 'unet-residual-large-crf-dice'
    model_file_name = 'unet-residual-large-crf-dice.h5'
    unet_residual_crf = _load_model(model_name, model_file_name)

    return (unet_residual, unet_residual_crf)


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


def _get_ensemble_mask(raw_image, y_pred_1, y_pred_2):
    #crf_model = CrfLabelRefiner()
    #image = np.squeeze(raw_image, axis=0)
    pred_1 = np.squeeze(y_pred_1, axis=0)
    pred_2 = np.squeeze(y_pred_2, axis=0)
    mask = np.maximum(pred_1, pred_2)
    mask[mask >= 0.5] = 1 
    mask[mask < 0.5] = 0
    #mask = crf_model.refine(image, mask)
    mask = np.squeeze(mask, -1)

    return mask


def _load_model(model_name, model_file_name, image_size=(256, 256)):
    model_file_path = f'saved_models/{model_file_name}'
    unet_residual = UnetResidual(model_name, image_size, version=2)
    unet_residual.restore(model_file_path)
    
    return unet_residual