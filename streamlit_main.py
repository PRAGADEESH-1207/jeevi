import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd

# Tensorflow Model Prediction
@st.cache(allow_output_mutation=True)
def load_model():
    return tf.keras.models.load_model("trained_model.h5")

def model_prediction(model, test_image):
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(64, 64))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return index of max element

# Function for fetching prices from sheet
def fetch_prices(predicted_item):
    df = pd.read_excel('vf/pricelist2.xlsx')
    idx = df.index[df['vegatables and fruits '] == predicted_item].tolist()
    if idx:  # Check if index is not empty
        return df.loc[idx[0], 'price']
    return None

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About Project", "Prediction"])

# Main Page
if app_mode == "Home":
    st.header("FRUITS & VEGETABLES RECOGNITION SYSTEM")
    st.image('vf/thomas-le-pRJhn4MbsMM-unsplash.jpg')

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
        model = load_model()
        result_index = model_prediction(model, test_image)
        # Reading Labels
        with open("vf/labels.txt") as f:
            content = f.readlines()
        labels = [i.strip() for i in content]
        predicted_item = labels[result_index]
        st.success("Model is Predicting it's a {}".format(predicted_item))

        # Fetch market prices
        market_price = fetch_prices(predicted_item)
        if market_price:
            st.write("Market price in Rs:")
            st.write(market_price)
        else:
            st.warning("Market price not available for {}".format(predicted_item))
