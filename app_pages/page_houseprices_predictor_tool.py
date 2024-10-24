import streamlit as st
import pandas as pd
import numpy as np
import joblib
from src.data_management import load_pkl_file

def page_houseprices_predictor_tool_body():
    st.write("### Houseprice Predictor Interface")
    st.info(
        f"* The client is interested in determining the likely sale price for 4 inherited houses."
        f"Through using Machine Learning techniques a model was developed to provide estimates,"
        f" and based on the existing data for house prices in Ames, the predicted values houseprices are:\n"
        f"* House 1: **$171,704.51**\n"
        f"* House 2: **$178,962.06**\n"
        f"* House 3: **$194,297.98**\n"
        f"* House 4: **$184,463.36**\n"
        f"\nThe client also wanted a tool for predicting the likely houseprice of other properties."
        f" A tool has therefore been provided, enabling the estimation of houseprice based on a smaller"
        f"set of characteristics."
    )
    st.write("---")

    st.write("### The Houseprice Predictor Tool for estimating house prices")

    st.info("Enter the values for the features below to estimate the house price.")

    # Load the model pipeline
    version = 'v1'
    pipeline_path = f'outputs/ml_pipeline/regression_analysis/{version}/pipeline_best.pkl'
    houseprice_pipeline = load_pkl_file(pipeline_path)

    # Mapping for OverallQual from text to numeric values
    overall_qual_mapping = {
        "Very Poor": 1,
        "Poor": 2,
        "Fair": 3,
        "Below Average": 4,
        "Average": 5,
        "Above Average": 6,
        "Good": 7,
        "Very Good": 8,
        "Excellent": 9,
        "Very Excellent": 10
    }

    # Create input fields for each feature
    def draw_input_widgets():
        X_live = pd.DataFrame([], index=[0])  # Create an empty dataframe to store inputs

        # User input for each feature
        overall_qual_label = st.selectbox(
            'Overall Quality (OverallQual)',
        options=list(overall_qual_mapping.keys())  # Show the text options in the dropdown
        )

        # Map the selected label to the corresponding numeric value
        X_live['OverallQual'] = overall_qual_mapping[overall_qual_label]
        X_live['GrLivArea'] = st.number_input('Above ground living area (GrLivArea)', min_value=0, value=1500)
        X_live['GarageArea'] = st.number_input('Garage Area (GarageArea)', min_value=0, value=500)
        X_live['YearBuilt'] = st.number_input('Year Built (YearBuilt)', min_value=1800, max_value=2023, value=2000)
        X_live['TotalBsmtSF'] = st.number_input('Total Basement Area (TotalBsmtSF)', min_value=0, value=1000)
        X_live['LotArea'] = st.number_input('Lot Area (LotArea)', min_value=0, value=8000)

        return X_live

    # Draw input widgets for live prediction
    X_live = draw_input_widgets()

    # Load the existing pipeline
    pipeline_path = 'outputs/ml_pipeline/regression_analysis/v1/pipeline_best.pkl'
    houseprice_pipeline = joblib.load(pipeline_path)

    X_live_filtered = X_live[['OverallQual', 'GrLivArea', 'GarageArea', 'YearBuilt', 'TotalBsmtSF', 'LotArea']]

    # Predict on live data when the user clicks the button
    if st.button("Predict House Price"):
        # Ensure the correct input features are provided
        if not X_live.isnull().values.any():
            prediction = houseprice_pipeline.predict(X_live_filtered)[0]  # Get the prediction
            st.success(f"The estimated house price is: ${np.round(prediction, 2)}")
        else:
            st.error("Please fill in all the fields before predicting.")

# Call the function to load the predictor page
page_houseprices_predictor_tool_body()
