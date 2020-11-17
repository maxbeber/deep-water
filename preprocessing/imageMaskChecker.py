import glob
import os

class ImageMaskChecker:
    """
    The purpose of this class is to delete images that
    does not contain an annotation file in order to avoid
    errors caused by the image augmentation generator.
    """
    def __init__(self):
        self.annotation_base_folder = 'annotations'
        self.annotation_extension = 'json'
        self.annotation_wildcard = f'*.{self.annotation_extension}'
        self.data_folder = 'data'
        self.image_extension = 'jpg'
        self.image_wildcard= f'*.{self.image_extension}'
        self.datasets = {
            'nwpu': 'nwpu_images',
            's2cloudless': 's2cloudless_imagery'
        }


    def clean_up(self):
        for annotation_folder, dataset_name in self.datasets.items():
            annotations = self._get_annotations(annotation_folder)
            images = self._get_images(dataset_name)
            self._delete_orphan_images(dataset_name, images, annotations)
            print(f'{dataset_name}: # orphan images: {len(images) - len(annotations)}')

    
    def _delete_orphan_images(self, dataset_name, images, annotations):
        orphan_files = set(images) - set(annotations)
        for orphan_file in orphan_files:
            image_file_path = os.path.join(dataset_name, self.data_folder, orphan_file)
            os.remove(image_file_path)


    def _get_annotations(self, folder):
        annotations = []
        query = os.path.join(self.annotation_base_folder, folder, self.annotation_wildcard)
        file_paths = glob.glob(query)
        for file_path in file_paths:
            file_name = file_path.split(os.sep)[-1].replace(self.annotation_extension, self.image_extension)
            annotations.append(file_name)
        return annotations


    def _get_images(self, dataset_name):
        images = []
        query = os.path.join(dataset_name, self.data_folder, self.image_wildcard)
        file_paths = glob.glob(query)
        for file_path in file_paths:
            file_name = file_path.split(os.sep)[-1]
            images.append(file_name)
        return images