import numpy as np
import os
import pandas as pd
import tensorflow as tf
from functools import partial
from models import UnetResidual
from PIL import Image
from preprocessing import CrfLabelRefiner
from pyproj import CRS, Geod
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


def get_bounding_box(df):
    longitude = [
        df["min_longitude"], 
        df["min_longitude"], 
        df["max_longitude"],
        df["max_longitude"] 
    ]
    latitude = [
        df["min_latitude"], 
        df["max_latitude"], 
        df["max_latitude"],
        df["min_latitude"]
    ]
    coordinates = [[lat, long] for lat, long in zip (latitude, longitude)]
    polygon = Polygon(coordinates)

    return polygon


def get_bounding_box_area(bounding_box):
    crs_4326 = CRS('epsg:4326')
    geod_wgs84 = crs_4326.get_geod()
    polygon_area_m2, _ = geod_wgs84.geometry_area_perimeter(bounding_box)
    polygon_area_km2 = polygon_area_m2 / 1000000.0

    return polygon_area_km2


def get_water_land_per_year(fraction, area):
    water_sqkm = area * fraction
    land_sqkm = area - water_sqkm
    
    return (water_sqkm, land_sqkm)


def load_image(image_path):
    raw_image = Image.open(image_path)
    raw_image = np.array(raw_image)
    if raw_image.ndim == 2:
        raw_image = np.stack((raw_image,) * 3, axis=-1)
    else:
        raw_image = raw_image[:, :, :3]
    raw_image = raw_image / 255.0
    
    return raw_image
    

def load_models():
    model_name = 'unet-residual-large-dice'
    model_file_name = 'unet-residual-large-dice.h5'
    unet_residual = _load_model(model_name, model_file_name, version=2)
    model_name = 'unet-residual-large-crf-dice'
    model_file_name = 'unet-residual-large-crf-dice.h5'
    unet_residual_crf = _load_model(model_name, model_file_name, version=2)

    return (unet_residual, unet_residual_crf)


def get_image_path(df, lake, year):
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
    crf_model = CrfLabelRefiner()
    image = np.squeeze(raw_image, axis=0)
    pred_1 = np.squeeze(y_pred_1, axis=0)
    pred_2 = np.squeeze(y_pred_2, axis=0)
    mask = np.maximum(pred_1, pred_2)
    mask[mask >= 0.5] = 1 
    mask[mask < 0.5] = 0
    image = image.copy(order='C')
    mask = mask.copy(order='C')
    mask = crf_model.refine(image, mask)

    return mask


def _load_model(model_name, model_file_name, image_size=(256, 256), version=1):
    model_file_path = f'saved_models/{model_file_name}'
    unet_residual = UnetResidual(model_name, image_size, version=version)
    unet_residual.restore(model_file_path)
    
    return unet_residual