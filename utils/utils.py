import numpy as np
from dotenv import load_dotenv

def normalize(data):
    flatten_data = np.ravel(data)
    max_value, min_value = flatten_data.max(), flatten_data.min()
    middle = (max_value + min_value) / 2
    return (data - middle) / middle

def load_env():
    load_dotenv('.env')