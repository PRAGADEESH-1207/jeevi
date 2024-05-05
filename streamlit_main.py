import streamlit as st
import requests
from bs4 import BeautifulSoup
import numpy as np
import tensorflow as tf
import pandas as pd

#Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_model.h5")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(64, 64))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions) #return index of max element

#function form sheets
def prices(predicted_item):
    
    predictions =predicted_item
    excel_file = 'pricelist2.xlsx'
    df = pd.read_excel(excel_file)
    plst= df['price'].tolist()
    lst = df['vegatables and fruits '].tolist()
    for i in lst:
    #print(type(i))
    
        if predictions == i:
            idx = lst.index(i)
            print(i)
            print(idx)
        
    price = plst[idx]
    return price

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About Project", "Prediction"])

# Main Page
if app_mode == "Home":
    st.header("FRUITS & VEGETABLES RECOGNITION SYSTEM")
    image_path = "thomas-le-pRJhn4MbsMM-unsplash.jpg"
    st.image(image_path)

# About Project
elif app_mode == "About Project":
    st.header("About Project")
    st.subheader("About Dataset")
    st.text("This dataset contains images of the following food items:")
    st.code("fruits- banana, apple, pear, grapes, orange, kiwi, watermelon, pomegranate, pineapple, mango.")
    st.code("vegetables- cucumber, carrot, capsicum, onion, potato, lemon, tomato, raddish, beetroot, cabbage, lettuce, spinach, soy bean, cauliflower, bell pepper, chilli pepper, turnip, corn, sweetcorn, sweet potato, paprika, jalepeño, ginger, garlic, peas, eggplant.")
    st.subheader("Content")
    st.text("This dataset contains three folders:")
    st.text("1. train (100 images each)")
    st.text("2. test (10 images each)")
    st.text("3. validation (10 images each)")

# Prediction Page
elif app_mode == "Prediction":
    st.header("Model Prediction")
    test_image = st.file_uploader("Choose an Image:")
    if st.button("Show Image") and test_image is not None:
        st.image(test_image)

    # Predict button
    if st.button("Predict") and test_image is not None:
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        # Reading Labels
        with open("labels.txt") as f:
            content = f.readlines()
        labels = [i.strip() for i in content]
        predicted_item = labels[result_index]
        st.success("Model is Predicting it's a {}".format(predicted_item))

        # Scrape market prices
        prices = prices(predicted_item)
        
        st.write("market price in Rs")

        # Display market price if available
        st.write(prices)
