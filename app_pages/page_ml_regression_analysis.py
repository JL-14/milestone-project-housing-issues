import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.data_management import load_pkl_file
from src.evaluate_model import regression_performance

def page_ml_regression_analysis_body():
    st.write(
        """
        A number of regression and classification models were developed to
        identify the best model and machine learning pipeline to predict the
        price of a house in Ames, Iowa. The approaches included:
        
        * Regression analysis of the full dataset
        * Regression analysis of the full dataset with Principal Component
          Analysis (PCA)
        * Classification analysis of the full dataset (with the sale price
          grouped into bands)
        * Regression analysis of the full dataset with only the most
          important variables
        * Regression analysis of the dataset with additional feature
          engineering and data cleaning (e.g. power transformations, 
          Box-Cox transformations, and Winsorization of outliers)
        
        The requirement was for a model with an R2 score (or f1 score for the
        classifier) of at least 0.75 for both the Train and Test sets.
        The following models met this criterion:
        
        * Regression analysis of the full dataset
        * Regression analysis of the full dataset with PCA
        * Regression analysis of the full dataset with only the most important variables
        
        The model performance was similar across these approaches, so the model
        using the full dataset with only the most important variables was selected.
        This model was particularly suitable for the development of the 
        Houseprice Predictor Tool. The model performance was 0.88 on the 
        Train Set and 0.77 on the Test Set.
        
        The details of the selected model are below:
        """
    )

    # Loading pipeline data files
    version = 'v1'
    pipeline_best = load_pkl_file(
        f"outputs/ml_pipeline/regression_analysis/{version}/pipeline_best.pkl"
    )
    feature_importance = plt.imread(
        f"outputs/ml_pipeline/regression_analysis/{version}/feature_importance.png"
    )
    X_train = pd.read_csv(
        f"outputs/ml_pipeline/regression_analysis/{version}/X_train.csv"
    )
    X_test = pd.read_csv(
        f"outputs/ml_pipeline/regression_analysis/{version}/X_test.csv"
    )
    y_train = pd.read_csv(
        f"outputs/ml_pipeline/regression_analysis/{version}/y_train.csv"
    )
    y_test = pd.read_csv(
        f"outputs/ml_pipeline/regression_analysis/{version}/y_test.csv"
    )

    st.write("---")

    # Displaying pipeline and feature importance
    st.write("**ML Pipeline Steps**")
    st.code(pipeline_best, language='python')

    st.write("---")

    st.write("**Feature Importance**")
    st.write(X_train.columns.to_list())
    st.image(feature_importance)

    st.write("---")

    # Evaluating and displaying model performance
    st.write("**Performance on Train and Test Sets**")
    performance_results = regression_performance(
        X_train, y_train, X_test, y_test, pipeline_best
    )
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
