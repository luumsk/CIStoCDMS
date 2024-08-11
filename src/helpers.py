import os
import json
import pickle
import joblib
from collections import defaultdict
import numpy as np

def load_json(path):
    data = None
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def save_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

def load_pickle(path):
    data = None
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def save_pickle(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def reorder_dict_by_key(original_dict, reverse=True):
    sorted_items = sorted(original_dict.items(), reverse=reverse)
    return {key: value for key, value in sorted_items}

def append_json(path, data):
    current_data = None
    if os.path.exists(path):
        current_data = load_json(path)
    
    if not isinstance(current_data, dict):
        current_data = {}

    current_data.update(data)
    current_data = reorder_dict_by_key(current_data)
    save_json(path, current_data)

def load_joblib(path):
    return joblib.load(path)

def save_joblib(path, data):
    joblib.dump(data, path)

def get_mean_by_key(list_of_dictionaries):
    # Initialize defaultdict to store lists of values for each key
    values_by_key = defaultdict(list)

    # Collect values for each key
    for d in list_of_dictionaries:
        for key, value in d.items():
            values_by_key[key].append(value)

    # Calculate mean for each key
    mean_by_key = {
        f'mean_{key}': np.mean(values) for key, values in values_by_key.items()
    }
    return mean_by_key