import streamlit as st
import joblib
import tensorflow as tf

@st.cache_resource
def load_models():
    preprocessor = joblib.load('preprocessor.joblib')
    nn_model = tf.keras.models.load_model('neural_network_model.h5')
    return preprocessor, nn_model

@st.cache_data
def load_data():
    prepared_data = joblib.load('prepared_data.joblib')
    return prepared_data