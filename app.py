import streamlit as st
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
from PIL import Image


from tensorflow.keras.models import load_model

st.set_page_config(page_title = 'Sentiment Analysis Bitcoin',
                  initial_sidebar_state = "expanded",
                  menu_items = {
                      'About' : 'Milestone 2 Fase 2'
                  })

image = Image.open('bitcoin.png')

# load model
model = keras.models.load_model("model_bitcoin")


label = ['Negative', 'Neutral', 'Positive']

st.title("Sentiment Analysis Bitcoin")
st.image(image)

news_title = st.text_input('Enter a Tweet Bitcoin')
new_data = pd.DataFrame([news_title])
res = model.predict(new_data)
res = res.argmax()
press = st.button('Predict')
if press:
   st.title(label[res])