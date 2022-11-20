import os

"""Skeleton for the different InstanceStractors"""
class InstanceExtractor:

    def __init__(self):
        self.base_path = os.getenv('POSEIDON_DATASET_PATH')
        if self.base_path is None:
            raise EnvironmentError("Environment variable 'POSEIDON_DATASET_PATH' not found")

    def dataset_stats():
        pass

    def extract():
        pass

    def extract_background():
        pass

    def visualize(img_path=None):
        pass