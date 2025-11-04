from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

app = Flask(__name__)

# Custom Transformer (must match your pipeline pipeline)
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

# Load pipeline and model
with open('preprocessing_pipeline.pkl', 'rb') as f_pipeline:
    pipeline = pickle.load(f_pipeline)

with open('model.pkl', 'rb') as f_model:
    model = pickle.load(f_model)

@app.route('/')
def home():
    return "Housing Price Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Input JSON example expected keys
        data = request.get_json()
        
        # Convert JSON to DataFrame just like in Streamlit
        input_df = pd.DataFrame({
            'longitude': [data['longitude']],
            'latitude': [data['latitude']],
            'housing_median_age': [data['housing_median_age']],
            'total_rooms': [data['total_rooms']],
            'total_bedrooms': [data['total_bedrooms']],
            'population': [data['population']],
            'households': [data['households']],
            'median_income': [data['median_income']],
            'income_cat': [data['income_cat']],
            'ocean_proximity': [data['ocean_proximity']]
        })

        # Transform input
        input_prepared = pipeline.transform(input_df)
        
        # Predict
        predicted_price = model.predict(input_prepared)

        return jsonify({'predicted_price': float(predicted_price[0])})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
