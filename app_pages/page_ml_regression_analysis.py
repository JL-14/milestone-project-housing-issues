import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.data_management import load_pkl_file
from src.evaluate_model import regression_performance

def page_ml_regression_analysis_body():

    st.write(
        """
        A number of regression and classification models were developed in
        order to identify the best model and machine learning pipeline to
        predict the price of a house in Ames, Iowa. The approaches included:
        * Regression analysis of the full dataset\n
        * Regression analysis of the full dataset with Principal Component
        Analysis (PCA)\n
        * Classification analysis of the full dataset (with the sale price
        grouped into bands)\n
        * Regression analysis of the full dataset with only the variables
        identified as most important\n
        * Regression analysis of the dataset with further feature engineering
        and data cleaning (including power transformations, Box Cox
        transformations and Winsorization of outliers)\n
        The requirement for the model was that it should have an R2 score
        (or f1 score for the classifier) of at least 0.75 for both the
        Train Set and Test Set.
        Of the models tested, the following met the requirement:\n
        * Regression analysis of the full dataset\n
        * Regression analysis of the full dataset with Principal Component Analysis (PCA)\n
        * Regression analysis of the full dataset with only the variables identified as most important\n
        The performance of these models was similar, so the regression analysis of
        the full dataset with only the variables identified as most important was
        selected. This model was particularly useful in lending itself to the
        development of the Houseprice Predictor Tool. The model performance on the
        Train Set was 0.88, and the Test Set was 0.77, both above the required
        performance level.\n
        All models showed signs of over-fitting (with a higher score for the Train
        Set than the Test Set), which is an issue that can be investigated should
        further refinement of the models be deemed necessary.
        The details of the selected model are:\n
        """
    )

    # load pipeline files
    version = 'v1'
    pipeline_best = load_pkl_file(
        f"outputs/ml_pipeline/regression_analysis/{version}/pipeline_best.pkl")
    feature_importance = plt.imread(
        f"outputs/ml_pipeline/regression_analysis/{version}/feature_importance.png")
    X_train = pd.read_csv(
        f"outputs/ml_pipeline/regression_analysis/{version}/X_train.csv")
    X_test = pd.read_csv(
        f"outputs/ml_pipeline/regression_analysis/{version}/X_test.csv")
    y_train = pd.read_csv(
        f"outputs/ml_pipeline/regression_analysis/{version}/y_train.csv")
    y_test = pd.read_csv(
        f"outputs/ml_pipeline/regression_analysis/{version}/y_test.csv")

    st.write("---")

    # Display pipeline and feature importance
    st.write("**ML Pipeline Steps**")
    st.code(pipeline_best, language='python')

    st.write("---")

    st.write("**Feature Importance**")
    st.write(X_train.columns.to_list())
    st.image(feature_importance)

    st.write("---")

    # Evaluate and display model performance
    st.write("**Performance on Train and Test Sets**")
    performance_results = regression_performance(X_train, y_train, X_test, y_test, pipeline_best)
    st.write("_**Train Set**_")
    st.write(f"R2 Score: {performance_results['Train R2 Score']:.3f}")
    st.write(f"Mean Absolute Error: {performance_results['Train Mean Absolute Error']:.3f}")
    st.write(f"Mean Squared Error: {performance_results['Train Mean Squared Error']:.3f}")
    st.write(f"Root Mean Squared Error: {performance_results['Train Root Mean Squared Error']:.3f}")

    st.write("_**Test Set**_")
    st.write(f"R2 Score: {performance_results['Test R2 Score']:.3f}")
    st.write(f"Mean Absolute Error: {performance_results['Test Mean Absolute Error']:.3f}")
    st.write(f"Mean Squared Error: {performance_results['Test Mean Squared Error']:.3f}")
    st.write(f"Root Mean Squared Error: {performance_results['Test Root Mean Squared Error']:.3f}")
