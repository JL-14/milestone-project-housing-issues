import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.data_management import load_pkl_file

# Load model performance metrics from regression evaluation
def load_regression_metrics():
    return {
        "MAE": 23102.253,
        "RMSE": 40308.223
    }

def page_houseprices_predictor_tool_body():
    st.write("### Houseprice Predictor Interface")
    st.info(
        f"* The client is interested in determining the likely sale price for 4 inherited houses."
        f"Through using Machine Learning techniques a model was developed to provide estimates,"
        f" and based on the existing data for house prices in Ames. The Houseprice"
        f" Predictor Tool produces a main estimate, and shows the range within which"
        f" we can be 95% certain that the actual price will fall (95% Confidence Interval).\n"
        f" The houseprice predictions for the inherited houses are:\n"
        f"* House 1: ** $128065.75** (95% CI: $82,785.33 - $173,346.16)\n"
        f"* House 2: **$154,095.91** (95% CI: $108,815.49 - $199,376.32)\n"
        f"* House 3: **$184,205.29** (95% CI: $138,924.87 - $229,485.70)\n"
        f"* House 4: ** $182,389.97** (95% CI: $137,109.55 - $227,670.38)\n"
        f"\nThe client also wanted a tool for predicting the likely houseprice of other properties."
        f" A tool has therefore been provided, enabling the estimation of houseprice based on a smaller"
        f" set of characteristics."
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
        "1. Very Poor": 1,
        "2. Poor": 2,
        "3. Fair": 3,
        "4. Below Average": 4,
        "5. Average": 5,
        "6. Above Average": 6,
        "7. Good": 7,
        "8. Very Good": 8,
        "9. Excellent": 9,
        "10. Very Excellent": 10
    }

    # Create input fields for each feature
    def draw_input_widgets():
        X_live = pd.DataFrame([], index=[0])  # Create an empty dataframe to store inputs

        # User input for each feature
        overall_qual_label = st.selectbox(
            'Overall Quality (OverallQual -Select option)',
        options=list(overall_qual_mapping.keys()) 
        )

        # Map the selected label to the corresponding numeric value
        X_live['OverallQual'] = overall_qual_mapping[overall_qual_label]
        X_live['YearBuilt'] = st.number_input('Year Built (YearBuilt -Square Feet)', min_value=1800, max_value=2023, value=1800)
        X_live['GrLivArea'] = st.number_input('Above ground living area (GrLivArea -Square Feet)', min_value=0, value=0)
        X_live['TotalBsmtSF'] = st.number_input('Total Basement Area (TotalBsmtSF -Square Feet)', min_value=0, value=0)
        X_live['GarageArea'] = st.number_input('Garage Area (GarageArea -Square Feet)', min_value=0, value=0) 

        return X_live

    # Draw input widgets for live prediction
    X_live = draw_input_widgets()

    # Ensure input features are in the correct order
    X_live_filter = X_live[['GarageArea', 'GrLivArea', 'OverallQual', 'TotalBsmtSF', 'YearBuilt']]

    # Load model performance metrics
    metrics = load_regression_metrics()
    mae = metrics["MAE"]

    # Predict on live data when the user clicks the button
    if st.button("Predict House Price"):
        # Ensure no missing values
        if not X_live_filter.isnull().values.any():
            prediction = houseprice_pipeline.predict(X_live_filter)[0]  # Get the prediction

            # Calculate 95% confidence interval based on MAE
            lower_bound = prediction - 1.96 * mae
            upper_bound = prediction + 1.96 * mae

            st.success(
                f"The estimated house price is: **${np.round(prediction, 2):,}** "
                f"(95% CI: ${np.round(lower_bound, 2):,} - ${np.round(upper_bound, 2):,})"
            )
        else:
            st.error("Please fill in all the fields before predicting.")

# Call the function to load the predictor page
page_houseprices_predictor_tool_body()