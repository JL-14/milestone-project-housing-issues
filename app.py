import streamlit as st
from app_pages.multipage import MultiPage
from app_pages import page_summary
from app_pages import page_descriptive_analytics
from app_pages import page_houseprices_predictor_tool
from app_pages import page_project_hypothesis
from app_pages import page_ml_regression_analysis

# Loading pages
from app_pages.page_summary import page_summary_body
from app_pages.page_descriptive_analytics import page_descriptive_analytics_body
from app_pages.page_houseprices_predictor_tool import page_houseprices_predictor_tool_body
from app_pages.page_project_hypothesis import page_project_hypothesis_body
from app_pages.page_ml_regression_analysis import page_ml_regression_analysis_body

app = MultiPage(app_name= "House Price Predictor") # Create an instance of the app

# Adding app pages using .add_page()
app.add_page("Quick Project Summary", page_summary_body)
app.add_page("Correlation and Predictive Power Score Study", page_descriptive_analytics_body)
app.add_page("Houseprice Predictor Tool", page_houseprices_predictor_tool_body)
app.add_page("Project Hypothesis and Validation", page_project_hypothesis_body)
app.add_page("ML: Regression Model", page_ml_regression_analysis_body)

app.run() # Run the app
