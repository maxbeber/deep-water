import argparse
import zipfile
from requests import Session
from requests.adapters import HTTPAdapter
from requests.exceptions import ConnectionError
from requests.exceptions import HTTPError
from requests.exceptions import Timeout

class GoogleDriveDownloader:
    """
    Represents a class that downloads and can also extract files
    from Google Drive.
    """
    def __init__(self):
        self.base_url = 'https://docs.google.com/uc?export=download'
        self.chunk_size = 32768
        self.http_success = 200
        self.max_number_attempts = 3


    def download(self, file_id, file_name, extract_file=False):
        http_adapter = HTTPAdapter(max_retries=self.max_number_attempts)
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
                    if(extract_file):
                        self._unzip_file(file_name)
                    print(f'downloading {file_name}... Done.')
            except ConnectionError as connection_error:
                print(f'Connection error occurred: {connection_error}.')
            except HTTPError as http_error:
                print(f'HTTP error occurred: {http_error}.')
            except Timeout:
                print('The request timed out.')
            except Exception as error:
                print(f'Other error occurred: {error}.')


    def _get_confirm_token(self, response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None


    def _save_response_content(self, chunk_size, response, file_name):
        with open(file_name, 'wb') as file:
            for chunk in response.iter_content(chunk_size):
                if chunk:
                    file.write(chunk)


    def _unzip_file(self, file_name):
        with zipfile.ZipFile(file_name, 'r') as zip_file:
            zip_file.extractall()
        print(f'uncompressing {file_name}... Done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file_id', default=None, help='the file id in Google Drive.')
    parser.add_argument('file_name', default=None, help='the name of the file to be saved.')
    parser.add_argument('-e', '--extract', action='store_true', help='whether or not extract the file after download.')
    args = parser.parse_args()
    file_id = args.file_id
    file_name = args.file_name
    extract_file = args.extract
    try:
        google_drive_downloader = GoogleDriveDownloader()
        google_drive_downloader.download(file_id, file_name, extract_file)
    except Exception as error:
        print(f'An error has occurred: {error}.')