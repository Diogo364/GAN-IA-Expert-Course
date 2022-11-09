import numpy as np
from sklearn.preprocessing import minmax_scale
from dotenv import load_dotenv

def normalize(data, feature_range=(0, 1)):
    input_shape = data.shape
    flatten_data = np.ravel(data)
    return minmax_scale(flatten_data, feature_range=feature_range).reshape(input_shape)
    
def load_env():
    load_dotenv('.env')