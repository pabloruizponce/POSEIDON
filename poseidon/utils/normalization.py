import os
import json
import shutil
import pandas as pd
from tqdm import tqdm
from PIL import Image
from utils.auxiliar import ignore_extended_attributes, NpEncoder


class COCONormalization:

    def __init__(self):
        self.base_path = os.getenv('POSEIDON_DATASET_PATH')
        if self.base_path is None:
            raise EnvironmentError("Environment variable 'POSEIDON_DATASET_PATH' not found")

    
    def normalize_image(self, img_row, min_width):
        
        annotations = pd.DataFrame(self.train_annotations['annotations'])
        instances = annotations[annotations['image_id'] == img_row['id']]

        normalize_factor = min_width / img_row['width']

        img_path = os.path.join(self.images_path, 'train', img_row['file_name'])
        image = Image.open(img_path)
        img_row['width'] *= normalize_factor
        img_row['height'] *= normalize_factor
        image.resize((int(img_row['width']), int(img_row['height'])))
        image.save(img_path)

        for i, instance in instances.iterrows():
            x, y, w, h = instance['bbox']
            center = (x + w / 2, y + h / 2)
            w = w * normalize_factor
            h = h * normalize_factor
            x = center[0] - w / 2
            y = center[1] - h / 2
            instance['bbox'] = [x, y, w, h]

            """
            instance["bbox"][0] *= normalize_factor
            instance["bbox"][1] *= normalize_factor
            instance["bbox"][2] *= normalize_factor
            instance["bbox"][3] *= normalize_factor
            instance["area"] *= normalize_factor
            """

        self.train_annotations['annotations'] = annotations.to_dict('records')

        return


    def normalize(self, output_path):

        # Create output directory
        if os.path.exists(output_path):
            print("Removing previous dataset in the specified path")
            shutil.rmtree(output_path, onerror=ignore_extended_attributes)

        print("Copying the original dataset")
        # Generate copy of all the images
        shutil.copytree(self.base_path, 
                os.path.join(output_path),
                ignore=shutil.ignore_patterns('.*'))


        self.base_path = output_path

        # Get path from the images and the annotation files
        self.train_annotations_path = os.path.join(self.base_path, "annotations", "instances_train.json")
        self.images_path = os.path.join(self.base_path, "images")

        # Read annotations as a dictionary
        with open(self.train_annotations_path) as f:
            self.train_annotations = json.load(f)


        images = pd.DataFrame(self.train_annotations['images'])
        min_width = images['width'].min()

        tqdm.pandas()

        # Generate labels of the train set
        print("Normalizing images from the train set")
        images.progress_apply(lambda x: self.normalize_image(x, min_width=min_width), axis=1)


        with open(self.train_annotations_path, 'w') as f:
            json.dump(self.train_annotations, f,  cls=NpEncoder)


        os.environ['POSEIDON_DATASET_PATH'] = output_path

        return

