import streamlit as st


def page_project_hypothesis_body():

    st.write("### Project Hypothesis and Validation")

    # conclusions taken from "02 - Churned Customer Study" notebook
    st.success(
    """
    The project had three hypotheses that were validated through the analysis and modelling:
    1)	Based on information from the credit checking agency Experian, the five factors that most affect a home’s value
    are: Prices of Compatible Properties; The Neighbourhood; The Home’s Age and Condition; the Property Size;
    and the State of the Housing Market.\n
    The dataset available for house prices in Ames, Iowa, does not contain information about compatible properties,
    neighbourhood, or the housing market, so the first hypothesis is that:\n
    The Age, Condition, and Size of the house will be the key predictors of house prices from the available dataset.\n 
    **Validation: True**. Correlation analysis and analysis show that the most important characteristics for predicting
    the value of a house are:\n
    * The overall quality of the house and kitchen (variables: OverallQual, KitchenQual)\n
    * The size of the living area above ground, 1st floor square feet, and basement size (square feet)
    (variables: GrLivArea, 1stFlrSF, TotalBsmtSF)\n
    * The year the house and garage was built (variables: YearBuilt, GarageYrBlt)\n

    2)	The second hypothesis is that based on the first hypothesis it is possible to predict the prices of four
    specific houses in Ames, Iowa, using data provided by the client.\n
    **Validation: True**. Based on a regression model developed from the overall dataset of house prices, an estimated
    price has been produced for each of the four properties.

    3)	The third hypothesis is that it is possible to develop a predictor tool, which based on a user inputting the
    key characteristics identified in the first hypothesis will produce an estimated house price for any house with data
    about the relevant characteristics.\n
    **Validation: True**. Based on the machine learning model developed to answer the first two hypotheses, a tool has
    been produced which will provide an estimated house price for any house based on the key
    characteristics of the house. 
    """
    )
