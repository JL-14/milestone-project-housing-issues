import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.data_management import load_pkl_file
from src.evaluate_model import regression_performance




def page_ml_regression_analysis_body():

    st.info(
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
        Of the models tested, the following met the requirement:
        * Regression analysis of the full dataset
        * Regression analysis of the full dataset with Principal Component Analysis (PCA)
        * Regression analysis of the full dataset with only the variables identified as most important
        The performance of these models was similar, so the regression analysis of
        the full dataset with only the variables identified as most important was
        selected. This model was particularly useful in lending itself to the
        development of the Houseprice Predictor Tool. The model performance on the
        Train Set was 0.88, and the Test Set was 0.77, both above the required
        performance level.
        All models showed signs of over-fitting (with a higher score for the Train
        Set than the Test Set), which is an issue that can be investigated should
        further refinement of the models be deemed necessary.
        The details of the selected model are:
        """
    )

    # load pipeline files
    version = 'v1'
    pipeline_best = load_pkl_file(
        f"outputs/ml_pipeline/regression_analysis/{version}/pipeline_best.pkl")
    feature_importance = plt.imread(
        f"outputs/ml_pipeline/predict_tenure/{version}/features_importance.png")
    X_train = pd.read_csv(
        f"outputs/ml_pipeline/regression_analysis/{version}/X_train.csv")
    # X_test = pd.read_csv(
    #     f"outputs/ml_pipeline/regression_analysis/{version}/X_test.csv")
    # y_train = pd.read_csv(
    #     f"outputs/ml_pipeline/regression_analysis/{version}/y_train.csv")
    # y_test = pd.read_csv(
    #     f"outputs/ml_pipeline/regression_analysis/{version}/y_test.csv")


    st.write("**ML Pipeline Steps**")
    st.write(pipeline_best)

    st.write("**Feature Importance**")
    st.write(X_train.columns.to_list())
    st.image(feature_importance)

    st.write("**Performance on Train and Test Sets**")
    st.write(regression_performance)
