import os

"""Skeleton for the different InstanceStractors"""
class InstanceGenerator:

    def __init__(self):
        self.base_path = os.getenv('POSEIDON_DATASET_PATH')
        if self.base_path is None:
            raise EnvironmentError("Environment variable 'POSEIDON_DATASET_PATH' not found")

    def dataset_stats(self):
        pass

    def balance(self):
        pass