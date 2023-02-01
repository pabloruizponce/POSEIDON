import json
import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from generation.instance_generator import InstanceGenerator
import random
import shutil
from filecmp import dircmp

# Numpy converting things without being asked to do it
# Stolen from: https://stackoverflow.com/a/57915246
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

# MAC shitty file system related
# Stolen from: https://stackoverflow.com/a/70355470
def ignore_extended_attributes(func, filename, exc_info):
    is_meta_file = os.path.basename(filename).startswith("._")
    if not (func is os.unlink and is_meta_file):
        raise

class COCOInstanceGenerator(InstanceGenerator):

    def __init__(self):
        
        # Get base path from the dataset
        super().__init__()

        # Get path from the images and the annotation files
        self.train_annotations_path = os.path.join(self.base_path, "annotations", "instances_train.json")
        self.a_train_annotations_path = os.path.join(self.base_path, "annotations", "instances_train_augmented.json")
        self.images_path = os.path.join(self.base_path, "images")
        self.images_a_train_path = os.path.join(self.base_path, "images", "train_augmented")

        shutil.copyfile(self.train_annotations_path, 
                        self.a_train_annotations_path)
        
        if os.path.exists(self.images_a_train_path):
            shutil.rmtree(self.images_a_train_path, onerror=ignore_extended_attributes)

        shutil.copytree(os.path.join(self.images_path, "train"), 
                        self.images_a_train_path,
                        ignore=shutil.ignore_patterns('.*'))

        print("Copy Created")

        # Read annotations as a dictionary
        with open(self.a_train_annotations_path) as f:
            self.train_annotations = json.load(f)

    def dataset_stats(self):

        # Basic Information about the dataset
        print("Dataset Stats")
        print("Base Path: ", self.base_path, "\n")

        # Instances on the Training Set
        print("Instances Training Set") 
        print("______________________")
        # Load annotations into a dataframe
        df = pd.DataFrame(self.train_annotations['annotations'])
        # Obtain pd.Series with the count of the different rows
        df_counts = df['category_id'].value_counts()
        # Print
        for category in self.train_annotations['categories']:
            print(category['name'], ": ",sep="", end="")
            if category['id'] in df_counts.index:
                print(df_counts[category['id']])   
            else:
                print("0")
        # Save the count distance between the minirity classes and the majority class
        self.class_count_difference = df_counts.max() - df_counts

    def box_collider(self, x1, x2):
        return(x1.x < x2.x + x2.w and
               x1.x + x1.w > x2.x and
               x1.y < x2.y + x2.h and
               x1.h + x1.y > x2.y)


    def check_instance_collider(self, img_row, bg_window, bboxs):
        bboxs = np.array([i for i in bboxs])
        bboxs = pd.DataFrame(bboxs, columns=['x', 'y', 'w', 'h'])
        window_has_collision = bboxs.apply(lambda x: self.box_collider(bg_window, x), axis=1).any()
        if not window_has_collision:
            return True
        return False


    # Get the maximun Y of the instances of a given image
    def get_max_y_image(self, img_row):

        annotations = pd.DataFrame(self.train_annotations['annotations'])
        bboxs =  annotations[annotations['image_id'] == img_row['id']]['bbox']
        bboxs = np.array([i for i in bboxs])
        bboxs = pd.DataFrame(bboxs, columns=['x', 'y', 'w', 'h'])
        return bboxs['y'].max()

    def add_instance_image(self, img_row, instances_path):
        population = self.class_count_difference.keys().to_numpy()
        weights = self.class_count_difference.to_numpy()
        choice = random.choices(population, weights=weights)[0]

        if img_row["meta"] is not None and "gimbal_heading(degrees)" in img_row["meta"]:
            angle_camera = img_row["meta"]["gimbal_heading(degrees)"]
            # TODO: El 8 esta metido a palo y eso no esta bien
            angle_camera_bin = int(angle_camera) % 8
            instances_path = os.path.join(instances_path, str(choice), str(angle_camera_bin))
            instance_path = os.path.join(instances_path, random.choice(os.listdir(instances_path)))
            
            #img_path = os.path.join(self.images_path, 'train', img_row['file_name'])
            img_path = os.path.join(self.images_a_train_path, img_row['file_name'])
            
            instance = Image.open(instance_path)
            image = Image.open(img_path)
            
            max_y = self.get_max_y_image(img_row)
            annotations = pd.DataFrame(self.train_annotations['annotations'])
            
            instance_id = annotations['id'].max() + 1
            instance_image = img_row['id']
            instance_bbox = [
                                random.randint(0, img_row['width']),
                                random.randint(max_y, img_row['height']),
                                instance.width,
                                instance.height
                            ]
            instance_area = instance.width * instance.height
            instance_category = choice

            instance_dict = {
                "id": instance_id, 
                "image_id": instance_image, 
                "bbox": instance_bbox, 
                "area": instance_area, 
                "category_id": instance_category
            }

            image.paste(instance, box=(random.randint(0, img_row['width']), 
                                       random.randint(max_y, img_row['height'])))

            # Update everything
            image.save(os.path.join(self.images_a_train_path,img_row['file_name']))
            self.train_annotations['annotations'].append(instance_dict)
            self.class_count_difference[choice] = self.class_count_difference[choice] - 1

        return

    def balance(self, instances_path):
        self.dataset_stats()
        # Iterate until the dataset has been balanced
        while self.class_count_difference.sum() != 0:
            print("Class Balance")
            print(self.class_count_difference)
            
            images = pd.DataFrame(self.train_annotations['images'])
            # Fancier
            print("Extracting Instances:")
            tqdm.pandas()
            # New instances generation
            images.progress_apply(lambda x: self.add_instance_image(x, instances_path), axis=1)
            
            with open(self.a_train_annotations_path, 'w') as f:
                json.dump(self.train_annotations, f,  cls=NpEncoder)
        
        return