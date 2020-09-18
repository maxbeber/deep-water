import pandas as pd
import requests
import shutil
import rasterio
import numpy as np
import json
import os
import tensorflow as tf
from models.unetResidual import UnetResidual
from models.unet import Unet




#--------------------------------------------------------------------
def download_image(data_frame, slct_lake, size="256"):
    dff = data_frame.loc[slct_lake, :]
    
    min_longitude = dff['min_longitude']
    min_latitude = dff['min_latitude']
    max_longitude = dff['max_longitude']
    max_latitude = dff['max_latitude']
    bounding_box = f"{min_longitude},{min_latitude},{max_longitude},{max_latitude}"

    payload = {
        "version": '1.1.1',
        "layers": "s2cloudless-2019",
        "width": size,
        "height": size,
        "srs": 'epsg:4326',
        "bbox": bounding_box
    }

    base_url = 'https://tiles.maps.eox.at/wms'
    base_url_parameters = '?service=wms&request=getmap'

    url = base_url + base_url_parameters
    r = requests.get(url, params=payload, stream=True)

    filename = "sample_image.jpg"

    if r.status_code == 200:
        # Set decode_content value to True, otherwise the downloaded image file's size will be zero.
        r.raw.decode_content = True
        
        with open(filename,'wb') as f:
            shutil.copyfileobj(r.raw, f)


#--------------------------------------------------------------------
def import_image(im = "sample_image.jpg"):
    dataset = rasterio.open(im)
    bands = dataset.read()
    image = np.ma.transpose(bands, [1, 2, 0])
    
    return image
    
#--------------------------------------------------------------------
def slct_image(data_frame, slct_lake, slct_year):
        dff = data_frame.loc[slct_lake, :]
    
        country = dff["country"]
        name = dff["name"]
        year = slct_year
    
        folder = "lakes_cropped"
        image_path = f"{folder}/{country}_{name}_s2cloudless_{year}.jpg"
    
        return image_path



#--------------------------------------------------------------------
def lat_long(data_frame, slct_lake):
    dff = data_frame.loc[slct_lake, :]

    longit = [
        dff["min_longitude"], 
        dff["max_longitude"], 
        dff["max_longitude"],
        dff["min_longitude"] 
        ]
    
    latit = [
        dff["min_latitude"], 
        dff["min_latitude"], 
        dff["max_latitude"],
        dff["max_latitude"]
    ]
    return longit, latit


#--------------------------------------------------------------------   
def import_annotation(data_frame, slct_lake):
    dff = data_frame.loc[slct_lake, :]
    query = dff["country"] + "_" +dff["name"]
    
    path = "../annotations/s2cloudless"
    files = os.listdir(path)
    item = [query in i for i in files].index(True)
    file_ = files[item]
    query_path = os.path.join(path, file_)

    with open(query_path) as f:
        annotation = json.load(f)

    im_name = list(annotation.keys())[0]


    X = []
    Y = []

    for i in annotation[im_name]['regions']:
        x = annotation[im_name]["regions"][i]["shape_attributes"]["all_points_x"]
        y = annotation[im_name]["regions"][i]["shape_attributes"]["all_points_y"]
        X.append(x)
        Y.append(y)
        assert len(X) == len(Y)

    return X, Y

#--------------------------------------------------------------------
def load_model():
    model_file_name = 'unet-residual-dice.h5'
    model_name = 'foo'
    image_size = (256, 256)
    unet_residual = UnetResidual(model_name, image_size, version=1)
    #unet_residual = Unet(model_name, image_size, version=1)
    unet_residual.restore(model_file_name)
    
    return unet_residual

#--------------------------------------------------------------------
def model_prediction(X, model):
    image_size = (256, 256)
    with rasterio.open(X) as dataset:
        bands = dataset.read()
        raw_image = np.ma.transpose(bands, [1, 2, 0])
        original_image = tf.Variable(raw_image)
        resized_image = tf.keras.preprocessing.image.smart_resize(original_image, image_size)
        y = model.predict(np.expand_dims(resized_image, axis=0))
    
    return y


#--------------------------------------------------------------------
def model_prediction_stationary(X, model):
    with rasterio.open(X) as dataset:
        bands = dataset.read()
        raw_image = np.ma.transpose(bands, [1, 2, 0])
        original_image = tf.Variable(raw_image)
        resized_image = tf.keras.preprocessing.image.smart_resize(original_image, image_size)
        y = model.predict(np.expand_dims(resized_image, axis=0))
    
    return y

#--------------------------------------------------------------------
def blkwhte_rgb(mask):
    mask_sq = mask.squeeze()
    mask_st = np.stack((mask_sq,)*3, axis=2)
    mask_rgb = mask_st*250

    return mask_rgb
