import streamlit as st
from app_pages.multipage import MultiPage
from app_pages import page_summary
from app_pages import page_descriptive_analytics
from app_pages import page_houseprices_predictor_tool
from app_pages import page_project_hypothesis
from app_pages import page_regression_original
from app_pages import page_regression_transformed

# Set the page configuration as the first Streamlit command
# st.set_page_config(page_title="House Price Predictor", layout="centered")

# Initialize the app
app = MultiPage(app_name="House Price Predictor")

# load pages scripts
from app_pages.page_summary import page_summary_body
from app_pages.page_descriptive_analytics import page_descriptive_analytics_body
from app_pages.page_houseprices_predictor_tool import page_houseprices_predictor_tool_body
from app_pages.page_project_hypothesis import page_project_hypothesis_body
from app_pages.page_regression_original import page_regression_original_body
from app_pages.page_regression_transformed import page_regression_transformed_body


app = MultiPage(app_name= "House Price Predictor") # Create an instance of the app

# Add your app pages here using .add_page()
app.add_page("Quick Project Summary", page_summary_body)
app.add_page("House Characteristics Study", page_descriptive_analytics_body)
app.add_page("Houseprice Predictor Tool", page_houseprices_predictor_tool_body)
app.add_page("Project Hypothesis and Validation", page_project_hypothesis_body)
app.add_page("ML: Regression Model with Original Dataset", page_regression_original_body)
app.add_page("ML: Regression Model with Feature Engineered Dataset", page_regression_transformed_body)
# app.add_page("ML: Regression Analysis", page_regression_analysis_body)
# app.add_page("ML: Data Cleaning and Feature Engineering", page_data_cleaning_and_feature_engineering_body)

app.run() # Run the app
