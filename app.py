import streamlit as st
from app_pages.multipage import MultiPage
from app_pages import page_summary
# from app_pages import page_descriptive_analytics

# load pages scripts
from app_pages.page_summary import page_summary_body


app = MultiPage(app_name= "House Price Predictor") # Create an instance of the app

# Add your app pages here using .add_page()
app.add_page("Quick Project Summary", page_summary_body)
# app.add_page("House Price Characteristics Study", page_churned_customer_study_body)
# app.add_page("Houseprice Predictor Tool", page_houseprices_predictor_tool_body)
# app.add_page("Project Hypothesis and Validation", page_project_hypothesis_body)
# app.add_page("ML: Regression Analysis", page_regression_analysis_body)
# app.add_page("ML: Data Cleaning and Feature Engineering", page_data_cleaning_and_feature_engineering_body)

app.run() # Run the app
