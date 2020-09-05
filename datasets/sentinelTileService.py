import argparse
import json
import os
import shutil
from requests import Session
from requests.adapters import HTTPAdapter
from requests.exceptions import ConnectionError
from requests.exceptions import HTTPError
from requests.exceptions import Timeout

class SentinelTileService:
    """
    Represents the Sentinel-2 Tile Service.

    Parameters
    ----------
    name : the name of the river
    country : the country where the river is located
    min_longitude : the upper-left longitude of the bounding box
    max_longitude : the lower-right longitude of the bounding box
    min_latitude : the upper-left latitude of the bounding box
    max_latitude : the lower-right latitude of the bounding box
    """
    def __init__(self, name, country, layers, min_longitude, min_latitude, max_longitude, max_latitude, width=4096, height=4096):
        self.base_url = 'https://tiles.maps.eox.at/wms'
        self.base_url_parameters = '?service=wms&request=getmap'
        self.country = country
        self.data_folder = os.path.join('s2cloudless_imagery', 'data')
        self.default_height = height
        self.default_widht = width
        self.flag_map_service_url = False
        self.flag_open_street_map_url = False
        self.file_extension = 'jpg'
        self.http_success = 200
        self.layers = map(self._get_layer_name, layers)
        self.min_longitude = min_longitude
        self.max_longitude = max_longitude
        self.min_latitude = min_latitude
        self.max_latitude = max_latitude
        self.name = name
        self.open_street_map_url = 'http://openstreetmap.org/'
        self.service_version = '1.1.1'
        self.srs = 'epsg:4326'
        self.output_folder = 's2cloudless_imagery'
        self._create_output_directory()


    def download_all_layers(self):
        http_adapter = HTTPAdapter(max_retries=3)
        if self.flag_open_street_map_url:
            open_street_map_url = self._get_open_street_map_url()
            print('url:', open_street_map_url)
        with Session() as session:
            session.mount(self.base_url, http_adapter)
            for layer in self.layers:
                url = self._get_service_url(layer)
                file_name = self._get_file_name(layer)
                print(f'downloading {file_name}... Done.')
                if self.flag_map_service_url:
                    print('url: ', url)
                self._download_url(session, file_name, url)


    def _create_output_directory(self):
        folder_exists = os.path.exists(self.output_folder)
        if not folder_exists:
            try:
                os.mkdir(self.output_folder)
            except OSError as os_error:
                print (f'Creation of the directory failed. {os_error}')
        subfolder_exists = os.path.exists(self.data_folder)
        if not subfolder_exists:
            try:
                os.mkdir(self.data_folder)
            except OSError as os_error:
                print (f'Creation of the sub-directory failed. {os_error}')


    def _download_url(self, session, file_name, url):
        try:
            response = session.get(url, stream=True, timeout=(3, 30))
            response.raise_for_status()
            if response.status_code == self.http_success:
                self._save_file_content(response.raw, file_name)
        except ConnectionError as connection_error:
            print(f'Connection error occurred: {connection_error}.')
        except HTTPError as http_error:
            print(f'HTTP error occurred: {http_error}.')
        except Timeout:
            print('The request timed out.')
        except Exception as error:
            print(f'Other error occurred: {error}.')

    
    def _format_layer_name(self, layer):
        if '-' in layer:
            formatted_layer = layer.replace('-', '_')
        else:
            formatted_layer = f'{layer}_2016'
        return formatted_layer


    def _get_bounding_box(self):
        bounding_box = f"{self.min_longitude},{self.min_latitude},{self.max_longitude},{self.max_latitude}"
        return bounding_box


    def _get_open_street_map_url(self):
        url = f'{self.open_street_map_url}?minlon={self.min_longitude}&minlat={self.min_latitude}&maxlon={self.max_longitude}&maxlat={self.max_latitude}'
        return url


    def _get_service_url(self, layer):
        bounding_box = self._get_bounding_box()
        url = f'{self.base_url}{self.base_url_parameters}&version={self.service_version}&layers={layer}&width={self.default_widht}&height={self.default_height}&srs={self.srs}&bbox={bounding_box}'
        return url


    def _get_file_name(self, layer):
        formated_layer = self._format_layer_name(layer)
        file_name = f'{self.data_folder}{os.sep}{self.country}_{self.name}_{formated_layer}.{self.file_extension}'
        return file_name

    
    def _get_layer_name(self, year):
        layers = {
            2016: 's2cloudless',
            2017: 's2cloudless-2017',
            2018: 's2cloudless-2018',
            2019: 's2cloudless-2019'
        }
        layer = layers.get(year, 'unknown')
        return layer


    def _save_file_content(self, raw_content, file_name):
        with open(file_name, 'wb') as file:
            raw_content.decode_content = True
            shutil.copyfileobj(raw_content, file)


def get_all_layers_from_tile(lake_properties, flag_map_service_url, flag_open_street_map_url):
    sentinel_service = SentinelTileService(**lake_properties)
    sentinel_service.flag_map_service_url = flag_map_service_url
    sentinel_service.flag_open_street_map_url = flag_open_street_map_url
    sentinel_service.download_all_layers() 


if __name__== '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', default=None, help='a path for a JSON file containing water bodies names and its coordinates')
    parser.add_argument('-i', '--id', default=None, help='the id of a given lake.')
    parser.add_argument('-m', '--ms', action='store_true', help='display the map service url.')
    parser.add_argument('-o', '--os', action='store_true', help='display the location in OpenStreetMap website.')
    args = parser.parse_args()
    file_path = args.path
    lake_id = args.id
    flag_map_service_url = args.ms
    flag_open_street_map_url = args.os
    try:
        tiles = json.load(open(file_path))
        if lake_id:
            if lake_id in tiles.keys():
                lake_properties = tiles[lake_id]
                get_all_layers_from_tile(lake_properties, flag_map_service_url, flag_open_street_map_url)
            else:
                print('error: the id could not be found.')
        else:
            for _, lake_properties in tiles.items():
                get_all_layers_from_tile(lake_properties, flag_map_service_url, flag_open_street_map_url)
    except Exception as error:
        print(f'An error has occurred: {error}.')