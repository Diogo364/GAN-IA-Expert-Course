import numpy as np

def normalize(data):
    flatten_data = np.ravel(data)
    max_value, min_value = flatten_data.max(), flatten_data.min()
    middle = (max_value + min_value) / 2
    return (data - middle) / middle