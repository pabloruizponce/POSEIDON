import json
import os
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





