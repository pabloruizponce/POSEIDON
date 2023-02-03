import os
import json
import shutil
import pandas as pd
from tqdm import tqdm
from utils.auxiliar import ignore_extended_attributes




class COCO2YOLO:

    def __init__(self):
        self.base_path = os.getenv('POSEIDON_DATASET_PATH')
        if self.base_path is None:
            raise EnvironmentError("Environment variable 'POSEIDON_DATASET_PATH' not found")

        # Get path from the images and the annotation files
        self.train_annotations_path = os.path.join(self.base_path, "annotations", "instances_train.json")
        self.val_annotations_path = os.path.join(self.base_path, "annotations", "instances_val.json")
        self.images_path = os.path.join(self.base_path, "images")

        # Read annotations as a dictionary
        with open(self.train_annotations_path) as f:
            self.train_annotations = json.load(f)

        with open(self.val_annotations_path) as f:
            self.val_annotations = json.load(f)


    
    def generate_labels_image(self, img_row, output_path, set):
        

        if set == "train":
            annotations = pd.DataFrame(self.train_annotations['annotations'])
            output_path_label = os.path.join(output_path, "labels", "train", (str(img_row['id']) + ".txt"))
        elif set == "val":
            annotations = pd.DataFrame(self.val_annotations['annotations'])
            output_path_label = os.path.join(output_path, "labels", "val", (str(img_row['id']) + ".txt"))

        instances = annotations[annotations['image_id'] == img_row['id']]

        with open(output_path_label, "w") as file:
            for i, instance in instances.iterrows():
                
                instance_x = (instance["bbox"][0] + (instance["bbox"][2] / 2)) / img_row['width']
                instance_y = (instance["bbox"][1] + (instance["bbox"][3] / 2)) / img_row['height']
                instance_w = instance["bbox"][2] / img_row['width']
                instance_h = instance["bbox"][3] / img_row['height']

                file.write(
                    str(instance['category_id']) + " " + 
                    str(instance_x) + " " + 
                    str(instance_y) + " " + 
                    str(instance_w) + " " + 
                    str(instance_h) + "\n"
                )

        return


    def convert(self, output_path, name):

        # Create output directory
        if os.path.exists(output_path):
            print("Removing previous dataset in the specified path")
            shutil.rmtree(output_path, onerror=ignore_extended_attributes)

        os.mkdir(output_path)
        os.mkdir(os.path.join(output_path, "labels"))
        os.mkdir(os.path.join(output_path, "labels", "train"))
        os.mkdir(os.path.join(output_path, "labels", "val"))

        print("Copying the images from the original dataset")
        # Generate copy of all the images
        shutil.copytree(self.images_path, 
                os.path.join(output_path, "images"),
                ignore=shutil.ignore_patterns('.*'))

        print("Generating YAML configuration file")
        with open(os.path.join(output_path, (name + ".yaml")), "w") as file:
            file.write("path: " + output_path + "\n")
            file.write("train: " + "images/train" + "\n")
            file.write("val: " + "images/val" + "\n")

            file.write("\nnames: " + "\n")
            for category in self.train_annotations['categories']:
                file.write("\t" + str(category['id']) + ": " + category['name'] + "\n")


        tqdm.pandas()

        # Generate labels of the train set
        print("Converting labels from the train set")
        images_train = pd.DataFrame(self.train_annotations['images'])
        images_train.progress_apply(lambda x: self.generate_labels_image(x, output_path=output_path, set="train"), axis=1)

        # Generate labels of the validation set
        print("Converting labels from the validation set")
        images_val = pd.DataFrame(self.val_annotations['images'])
        images_val.progress_apply(lambda x: self.generate_labels_image(x, output_path=output_path, set="val"), axis=1)

        return

