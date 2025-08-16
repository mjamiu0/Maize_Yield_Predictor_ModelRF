import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load assets with error handling
try:
    loaded_model = joblib.load('Maize_yield_prediction_models04.pkl')
    columns = joblib.load('model_columns03.pkl')
   
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# Feature configuration
soil_columns = [col for col in columns if 'SOIL TYPE PERCENT1' in col]
soil_types = [col.replace('SOIL TYPE PERCENT1 (Percent)_', '') for col in soil_columns]

state_columns = [col for col in columns if 'State Name' in col]
state_names = [col.replace('State Name_', '') for col in state_columns]

numerical_features = [
    'NITROGEN PER HA OF GCA (Kg per ha)',
    'PHOSPHATE PER HA OF GCA (Kg per ha)',
    'POTASH PER HA OF GCA (Kg per ha)',
    'AVERAGE RAINFALL (Millimeters)',
    'AVERAGE TEMPERATURE (Centigrate)',
    'AVERAGE PERCIPITATION (Millimeters)',
    'Year'
]
# Ensure we're using the exact column names from the trained model
features = columns  # Use the loaded columns directly

# Streamlit UI
st.title('Maize Yield Prediction Model')
st.markdown("### Predict the maize yield based on environmental and farming factors.")

# Input widgets
st.header('Input Parameters')
year = st.slider('Year', min_value=1966, max_value=2025, value=2023)
avg_temp = st.number_input('Average Temperature (Centigrate)', value=25.0)
nitrogen = st.number_input('Nitrogen per ha of GCA (Kg per ha)', value=10.0)
phosphate = st.number_input('Phosphate per ha of GCA (Kg per ha)', value=5.0)
potash = st.number_input('Potash per ha of GCA (Kg per ha)', value=3.0)
rainfall = st.number_input('Average Rainfall (Millimeters)', value=150.0)
precipitation = st.number_input('Average Precipitation (Millimeters)', value=100.0)
selected_soil_type = st.selectbox('Soil Type', soil_types)
selected_state_name = st.selectbox('State Name', state_names)

if st.button('Predict Maize Yield'):
    # Create input dictionary with all features initialized to 0
    input_data = {col: 0 for col in features}
    
    # Set numerical features
    input_data['Year'] = year
    input_data['AVERAGE TEMPERATURE (Centigrate)'] = avg_temp
    input_data['NITROGEN PER HA OF GCA (Kg per ha)'] = nitrogen
    input_data['PHOSPHATE PER HA OF GCA (Kg per ha)'] = phosphate   
    input_data['POTASH PER HA OF GCA (Kg per ha)'] = potash
    input_data['AVERAGE RAINFALL (Millimeters)'] = rainfall
    input_data['AVERAGE PERCIPITATION (Millimeters)'] = precipitation
    
    # Set categorical features (one-hot encoded)
    input_data[f'SOIL TYPE PERCENT1 (Percent)_{selected_soil_type}'] = 1
    input_data[f'State Name_{selected_state_name}'] = 1
    
    # Create DataFrame ensuring correct column order
    input_df = pd.DataFrame([input_data])[features]
    
    try:
        prediction = loaded_model.predict(input_df)
        st.success(f'The predicted maize yield is: **{prediction[0]:.2f} Kg per ha**')
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        st.write("Input DataFrame columns:", input_df.columns.tolist())
        st.write("Scaler expects features:", scaler.feature_names_in_)