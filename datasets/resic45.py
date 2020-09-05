import argparse
import os
import shutil
import zipfile
from requests import Session
from requests.adapters import HTTPAdapter
from requests.exceptions import ConnectionError
from requests.exceptions import HTTPError
from requests.exceptions import Timeout

class Resic45:
    """
    Represents the Resic45 dataset.
    """
    def __init__(self):
        self.base_url = 'https://docs.google.com/uc?export=download'
        self.chunk_size = 32768
        self.http_success = 200
        self.lake_directory = 'lake'
        self.data_directory = 'data'
        self.zip_file_directory = 'images'
        self.base_directory = 'nwpu_images'
        self.directory_listing_file = os.path.join(self.base_directory, 'dirs.txt')


    def download_file_from_google_drive(self, file_id, file_name):
        http_adapter = HTTPAdapter(max_retries=3)
        with Session() as session:
            session.mount(self.base_url, http_adapter)
            try:
                params = {'id': file_id}
                response = session.get(self.base_url, params=params, stream=True)
                token = self._get_confirm_token(response)
                if token:
                    params = {'id': file_id, 'confirm': token}
                    response = session.get(self.base_url, params=params, stream=True)
                if response.status_code == self.http_success:
                    self._save_response_content(self.chunk_size, response, file_name)
                    print(f'downloading {file_name}... Done.')
            except ConnectionError as connection_error:
                print(f'Connection error occurred: {connection_error}.')
            except HTTPError as http_error:
                print(f'HTTP error occurred: {http_error}.')
            except Timeout:
                print('The request timed out.')
            except Exception as error:
                print(f'Other error occurred: {error}.')


    def extract_and_clean_up(self, file_name):
        file_exists = os.path.exists(file_name)
        if not file_exists:
            print(f'{file_name} could not be found.')
            return
        folder_exists = os.path.exists(self.base_directory)
        if folder_exists:
            print(f'{self.base_directory} folder already exists.')
            return
        self._unzip_dataset(file_name)
        self._rename_zip_folder()
        self._clean_up()
        self._rename_lake_folder()
        self._delete_directory_listing_file()

    
    def _clean_up(self):
        sub_directories = [x[0] for x in os.walk(self.base_directory)][1:]
        folders_to_delete = [s for s in sub_directories if self.lake_directory not in s]
        for folder in folders_to_delete:
            shutil.rmtree(folder, ignore_errors=True)
        print('house cleaning... Done.')

    
    def _delete_directory_listing_file(self):
        try:
            os.remove(self.directory_listing_file)
        except Exception as error:
            print(f'An error has occurred: {error}.')


    def _get_confirm_token(self, response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None


    def _rename_lake_folder(self):
        try:
            current_directory = os.path.join(self.base_directory, self.lake_directory)
            new_directory = os.path.join(self.base_directory, self.data_directory)
            os.rename(current_directory, new_directory)
        except Exception as error:
            print(f'An error has occurred: {error}.')

    
    def _rename_zip_folder(self):
        try:
            os.rename(self.zip_file_directory, self.base_directory)
        except Exception as error:
            print(f'An error has occurred: {error}.')

    
    def _save_response_content(self, chunk_size, response, file_name):
        with open(file_name, 'wb') as file:
            for chunk in response.iter_content(chunk_size):
                if chunk:
                    file.write(chunk)
    

    def _unzip_dataset(self, file_name):
        with zipfile.ZipFile(file_name, 'r') as zip_file:
            zip_file.extractall()
        print(f'uncompressing {file_name}... Done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file_id', default=None, help='the file id in Google Drive.')
    parser.add_argument('file_name', default=None, help='the name of the file to be saved.')
    parser.add_argument('-d', '--download', action='store_true', help='whether or not download the dataset from google drive.')
    args = parser.parse_args()
    file_id = args.file_id
    file_name = args.file_name
    flag_download_dataset = args.download
    try:
        resic45 = Resic45()
        if(flag_download_dataset):
            resic45.download_file_from_google_drive(file_id, file_name)
        resic45.extract_and_clean_up(file_name)
    except Exception as error:
        print(f'An error has occurred: {error}.')