from time import sleep
import streamlit as st
import numpy as np
from PIL import Image
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.applications import inception_resnet_v2

from pathlib import Path

st.set_page_config(page_title="Trash Classification")
st.title("Classifying Garbage into various titles based on type of recycling")

info_md = Path("app.md").read_text()
st.markdown(info_md)

classes = {0: 'Cardboard', 1: 'Glass', 2: 'Metal', 3: 'Paper',  4: 'Plastic',5: 'Trash'}

TFLITE_MODEL = Path("model.tflite")

with st.spinner("Please wait initialising the model"):
    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL.__str__())
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]


uploader = st.file_uploader("", type=['jpg', 'png'])


def predict(image):
    with st.spinner("Computing ..."):
        interpreter.set_tensor(input_details["index"], image)
        interpreter.invoke()
        out = interpreter.get_tensor(output_details["index"])
    prediction = np.argmax(out, axis=1)[0]
    return classes[prediction]


if uploader is not None:
    file = uploader.read()
    img = Image.open(uploader)
    img = inception_resnet_v2.preprocess_input(
        tf.image.resize(img, [384, 512]).numpy())
    img = np.expand_dims(img, axis=0)
    p = predict(img)
    st.image(
        file, caption=f"{p}", width=500)
