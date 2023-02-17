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
import swifter
from utils.auxiliar import ignore_extended_attributes, NpEncoder


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
        
        self.annotations = pd.DataFrame(self.train_annotations['annotations'])
        self.original_annotations = pd.DataFrame(self.train_annotations['annotations'])
        self.images = pd.DataFrame(self.train_annotations['images'])
        self.original_images = pd.DataFrame(self.train_annotations['images'])


    def dataset_stats(self):

        # Basic Information about the dataset
        print("Dataset Stats")
        print("Base Path: ", self.base_path, "\n")

        # Instances on the Training Set
        print("Instances Training Set") 
        print("______________________")

        # Obtain pd.Series with the count of the different rows
        df_counts = self.annotations['category_id'].value_counts()
        # Print
        for category in self.train_annotations['categories']:
            print(category['name'], ": ",sep="", end="")
            if category['id'] in df_counts.index:
                print(df_counts[category['id']])   
            else:
                print("0")
        # Save the count distance between the minirity classes and the majority class
        self.class_count_difference = df_counts.max() - df_counts
        return
    
    def update_class_count_difference(self): 
        # Obtain pd.Series with the count of the different rows
        df_counts = self.annotations['category_id'].value_counts()
        # Save the count distance between the minirity classes and the majority class
        self.class_count_difference = df_counts.max() - df_counts
        return

    def box_collider(self, x1, x2):
        return(x1.x < x2.x + x2.w and
               x1.x + x1.w > x2.x and
               x1.y < x2.y + x2.h and
               x1.h + x1.y > x2.y)


    def check_instance_collider(self, instance, bboxs):
        bboxs = np.array([i for i in bboxs])
        bboxs = pd.DataFrame(bboxs, columns=['x', 'y', 'w', 'h'])
        window_has_collision = bboxs.apply(lambda x: self.box_collider(instance.iloc[0], x), axis=1)
        if not window_has_collision.any():
            return True
        return False


    # Get the maximun Y of the instances of a given image
    def get_max_y_image(self, img_row):
        bboxs =  self.annotations[self.annotations['image_id'] == img_row['id']]['bbox']
        bboxs = np.array([i for i in bboxs])
        bboxs = pd.DataFrame(bboxs, columns=['x', 'y', 'w', 'h']).astype('int')
        return bboxs['y'].max()


    def add_instance_image(self, img_row, instances_path, image, max_y, angle_camera_bin):
        population = self.class_count_difference.keys().to_numpy()
        weights = self.class_count_difference.to_numpy()
        choice = random.choices(population, weights=weights)[0]

        self.class_count_difference[choice] = self.class_count_difference[choice] - 1

        instances_path = os.path.join(instances_path, str(choice), str(angle_camera_bin))
        instance_path = os.path.join(instances_path, random.choice(os.listdir(instances_path)))
        instance = Image.open(instance_path)
                
        instance_id = self.annotations['id'].max() + 1
        instance_image = img_row['id']
        
        # Just to debug the number of collisions
        iters_colliding = 0

        while True:
            instance_x = random.randint(0, img_row['width'] - instance.width)
            instance_y = random.randint(max_y - instance.height, img_row['height'] - instance.height)

            instance_bbox = [
                                instance_x,
                                instance_y,
                                instance.width,
                                instance.height
                            ]

            instance_bbox_df = pd.DataFrame([instance_bbox], columns=['x', 'y', 'w', 'h'])
            # TODO
            bboxs = (self.annotations[self.annotations['image_id'] == img_row['id']]['bbox']).copy()

            if len(bboxs) == 0:
                break

            # The terminology has to be changed because is a litle bit ambiguous
            # The function returns True if there is NO collisions
            if self.check_instance_collider(instance_bbox_df, bboxs):
                break
            iters_colliding += 1
            #if iters_colliding == 10:
                #print(bboxs)
                #print(instance_bbox)
                #image.show()
            
        instance_area = instance.width * instance.height
        instance_category = choice

        instance_dict = {
            "id": instance_id, 
            "image_id": instance_image, 
            "bbox": instance_bbox, 
            "area": instance_area, 
            "category_id": instance_category
        }

        instance_df = pd.Series([instance_id, instance_image, instance_bbox, instance_area, instance_category],
                                     index=['id', 'image_id', 'bbox', 'area', 'category_id'])

        image.paste(instance, box=(instance_x, instance_y))
        self.annotations.loc[len(self.annotations)] = instance_df

        return instance_dict
    
    def get_new_instance_id(self):
        return (self.annotations['id'].max() + 1)

    def add_instances_image(self, img_row, instances_path, iteration, max_tolerance):
        if self.class_count_difference.sum() > len(self.annotations) * max_tolerance and (img_row["meta"] is not None and "gimbal_heading(degrees)" in img_row["meta"]):

            angle_camera = img_row["meta"]["gimbal_heading(degrees)"]
            # TODO: El 8 esta metido a palo y eso no esta bien
            angle_camera_bin = int(angle_camera) % 8
            img_path = os.path.join(self.images_a_train_path, img_row['file_name'])
            image = Image.open(img_path)
            max_y = self.get_max_y_image(img_row)

            # Update new image metadata
            new_image_metadata = img_row.copy()
            new_image_id = self.images['id'].max() + 1
            new_image_metadata['id'] = new_image_id
            new_image_metadata['file_name'] =  str(iteration) + "_" + img_row['file_name']
            self.images.loc[len(self.images)] = new_image_metadata

            # Add the new instances
            annotations_img =  self.original_annotations[self.original_annotations['image_id'] == img_row['id']].copy()
            annotations_img['image_id'] = new_image_id
            annotations_img['id'] = annotations_img['id'].apply(lambda x: self.get_new_instance_id())

            majority_class = self.class_count_difference.idxmin()
            majority_annotations = annotations_img[annotations_img['category_id'] == majority_class]
            
            # Blackout majority classes intances
            image_np = np.array(image)
            for i, r in majority_annotations.iterrows():
                image_np[r['bbox'][1]:r['bbox'][1]+r['bbox'][3], r['bbox'][0]:r['bbox'][0]+r['bbox'][2]] = 0
                 
            # Convert to PIL image
            image = Image.fromarray(image_np)

            # Only keep the annotations from the minority classes
            annotations_img = annotations_img[annotations_img['category_id'] != majority_class]

            # Update annotations
            self.annotations = pd.concat([self.annotations, annotations_img], ignore_index=True)
            self.update_class_count_difference()

            # Gets the number of instances that will be generated for each image
            n_instances = random.randint(1, 10)

            for i in range(n_instances):
                instance_metadata = self.add_instance_image(new_image_metadata, instances_path, image, max_y, angle_camera_bin)
                #self.annotations.append(instance_metadata)

            # Update everything
            image.save(os.path.join(self.images_a_train_path,new_image_metadata['file_name']))

        return

    def balance(self, instances_path, max_tolerance=0.03):
        self.dataset_stats()
        # Iterate until the dataset has been balanced
        iteration = 1

        while self.class_count_difference.sum() > len(self.annotations) * max_tolerance:
            print("Class Balance")
            print(self.class_count_difference)

            # Fancier
            print(f"Iteration {iteration}")
            tqdm.pandas()
            # New instances generation
            self.original_images.progress_apply(lambda x: self.add_instances_image(x, instances_path, iteration, max_tolerance), axis=1)
            # Progress Bar not appearing I dunno why
            #images.swifter.progress_bar(True, bar_format='{l_bar}{bar}| elapsed: {elapsed}s').apply(lambda x: self.add_instances_image(x, instances_path, iteration), axis=1)

            self.train_annotations['images'] = self.images.to_dict('records')
            self.train_annotations['annotations'] = self.annotations.to_dict('records')

            with open(self.a_train_annotations_path, 'w') as f:
                json.dump(self.train_annotations, f,  cls=NpEncoder)
            
            iteration = iteration + 1
        
        return