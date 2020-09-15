import pandas as pd
import requests
import shutil
import rasterio
import numpy as np

#--------------------------------------------------------------------
def download_image(data_frame, slct_lake):
    dff = data_frame.loc[slct_lake, :]
    
    min_longitude = dff['min_longitude']
    min_latitude = dff['min_latitude']
    max_longitude = dff['max_longitude']
    max_latitude = dff['max_latitude']
    bounding_box = f"{min_longitude},{min_latitude},{max_longitude},{max_latitude}"

    payload = {
        "version": '1.1.1',
        "layers": "s2cloudless-2019",
        "width": "512",
        "height": "512",
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
def import_image():
    dataset = rasterio.open("sample_image.jpg")
    bands = dataset.read()
    image = np.ma.transpose(bands, [1, 2, 0])
    
    return image
    

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