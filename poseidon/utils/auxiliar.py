import numpy as np
import os
import json

# MAC shitty file system related
# Stolen from: https://stackoverflow.com/a/70355470
def ignore_extended_attributes(func, filename, exc_info):
    is_meta_file = os.path.basename(filename).startswith("._")
    if not (func is os.unlink and is_meta_file):
        raise


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