import json
import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from extraction.instance_extractor import InstanceExtractor

class COCOInstanceExtractor(InstanceExtractor):

    def __init__(self):
        
        # Get base path from the dataset
        super().__init__()

        # Get path from the images and the annotation files
        self.train_annotations_path = os.path.join(self.base_path, "annotations", "instances_train.json")
        self.val_annotations_path = os.path.join(self.base_path, "annotations", "instances_val.json")
        self.images_path = os.path.join(self.base_path, "images")

        # Read annotations as a dictionary
        with open(self.train_annotations_path) as f:
            self.train_annotations = json.load(f)

        with open(self.val_annotations_path) as f:
            self.val_annotations = json.load(f)


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

        print("")
        # Instances on the Validation Set
        print("Instances Validation Set") 
        print("________________________")
        df = pd.DataFrame(self.val_annotations['annotations'])
        df_counts = df['category_id'].value_counts()
        for category in self.val_annotations['categories']:
            print(category['name'], ": ",sep="", end="")
            if category['id'] in df_counts.index:
                print(df_counts[category['id']])   
            else:
                print("0")        


    # Extract and save a particular instance from an image
    def extract_instance_image(self, img, bbox, output_path):
        output_path = output_path + "_" + str(bbox['id']) + "_" + str(bbox['category_id']) + ".png"
        bbox = bbox['bbox']
        x, y, w, h = bbox
        instance = Image.fromarray(img[y:y+h, x:x+w])
        instance.save(output_path)
        return 

    # Extract all instances from an image
    def extract_instances_image(self, annotations, img_row, output_path):
        output_path = os.path.join(output_path, str(img_row['id']))
        bboxs =  annotations[annotations['image_id'] == img_row['id']]
        img_path = os.path.join(self.images_path, 'train', img_row['file_name'])
        img = Image.open(img_path) 
        img = np.array(img)
        bboxs.apply(lambda x: self.extract_instance_image(img, x, output_path) ,axis=1)
        return

    # Extract and save all the intances from all the images in the training set 
    def extract(self, output_path='./outputs'):
        # Create output directory
        if not os.path.exists(output_path):
            os.mkdir(output_path)
            print("Output directory creted: ", output_path)
        # Get annotations and images information
        annotations = pd.DataFrame(self.train_annotations['annotations'])
        images = pd.DataFrame(self.train_annotations['images'])
        # Fancier
        print("Extracting Instances:")
        tqdm.pandas()
        # Extraction
        images.progress_apply(lambda x: self.extract_instances_image(annotations, x, output_path), axis=1)
        return

    




        




