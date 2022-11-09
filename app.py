import os
import os.path as osp
import cv2
from utils import normalize
import streamlit as st
import tensorflow as tf
from utils import load_env
from models.nn import DCGANGenerator

load_env()
def generate_random_input():
    return tf.random.normal((1, 100))

if __name__ == '__main__':
    st.title('Generator Evaluation')
    
    saved_generator = os.listdir(os.environ['ASSETS'])

    selected_generator = st.selectbox('Select the Generator model', saved_generator)

    if selected_generator is not None:
        full_path_generator = osp.join(os.environ['ASSETS'], selected_generator)
        generator = DCGANGenerator(None, None, from_path=full_path_generator)

        button_clicked = st.button('Generate input')
        if button_clicked:
            random_input = generate_random_input()
            out_image = generator.predict(random_input, training=False)
            img = cv2.resize(normalize(out_image[0].numpy()), (250, 250))
            st.image(img)