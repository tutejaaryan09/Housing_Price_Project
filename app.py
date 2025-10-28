import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.base import BaseEstimator, TransformerMixin

# --- Define the Custom Transformer ---
# This class definition MUST be in the script to unpickle the pipeline
# These column indices are based on the original dataframe structure
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


# --- Load Model and Pipeline ---
@st.cache_resource
def load_resources():
    # Load the preprocessing pipeline
    with open('preprocessing_pipeline.pkl', 'rb') as f_pipeline:
        pipeline = pickle.load(f_pipeline)
    
    # Load the trained model
    with open('model.pkl', 'rb') as f_model:
        model = pickle.load(f_model)
        
    return pipeline, model

try:
    full_pipeline, model = load_resources()
except FileNotFoundError:
    st.error("Model or pipeline not found. Please ensure 'model.pkl' and 'preprocessing_pipeline.pkl' are in the same directory.")
    st.stop()
except AttributeError as e:
    st.error(f"An attribute error occurred: {e}")
    st.error("This usually means a custom class (like CombinedAttributesAdder) is missing from the script. Ensure it's defined correctly.")
    st.stop()


# --- Dashboard Title ---
st.title('California Housing Price Prediction')
st.write("Enter the property details below to get a price prediction.")


# --- User Input Fields ---
st.header("Property Details")

# Create columns for better layout
col1, col2 = st.columns(2)

with col1:
    longitude = st.number_input('Longitude', min_value=-125.0, max_value=-114.0, value=-122.23, format="%.4f")
    latitude = st.number_input('Latitude', min_value=32.0, max_value=42.0, value=37.88, format="%.4f")
    housing_median_age = st.slider('Housing Median Age', 1, 52, 29)
    total_rooms = st.number_input('Total Rooms', min_value=1, max_value=40000, value=2571)
    total_bedrooms = st.number_input('Total Bedrooms', min_value=1, max_value=7000, value=538)

with col2:
    population = st.number_input('Population', min_value=1, max_value=36000, value=1425)
    households = st.number_input('Households', min_value=1, max_value=6100, value=501)
    median_income = st.number_input('Median Income (in tens of thousands)', min_value=0.0, max_value=15.0, value=3.87, format="%.4f")
    income_cat=st.number_input('Category', min_value=1, max_value=6, value=1)
    ocean_proximity = st.selectbox('Ocean Proximity', ('<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'NEAR BAY', 'ISLAND'))


# --- Prediction Button ---
if st.button('**Predict Price**', use_container_width=True):

    # 1. Create a DataFrame from user input
    input_data = {
        'longitude': [longitude],
        'latitude': [latitude],
        'housing_median_age': [housing_median_age],
        'total_rooms': [total_rooms],
        'total_bedrooms': [total_bedrooms],
        'population': [population],
        'households': [households],
        'median_income': [median_income],
        'income_cat' :[income_cat],
        'ocean_proximity': [ocean_proximity]
    }
    input_df = pd.DataFrame(input_data)
    
    st.write("---")
    st.subheader("Raw Input Data")
    st.dataframe(input_df)

    # 2. Transform the input data using the loaded pipeline
    try:
        input_prepared = full_pipeline.transform(input_df)
        
        # 3. Make the prediction
        predicted_price = model.predict(input_prepared)
        
        # Display the result
        st.success(f"**Predicted House Price: ${predicted_price[0]:,.2f}**")
        
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.warning("Please ensure your input values are reasonable and that the model pipeline is loaded correctly.")

