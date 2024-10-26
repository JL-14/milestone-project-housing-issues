# House Predictor Tool for Properties in Ames, Iowa
## Project Overview
The Houseprice Predictor Tool is an app developed through using machine learning approaches to analyse and predict houseprices based on a dataset of houseprices in Ames, Iowa, USA. The tool has been developed to meet the requirements of a client looking for an estimated sale price for four inherited properties, and a way to estimate houseprices for other properties. It takes the shape of an on-line app, backed up by descriptive analysis and a machine learning model.

## Dataset Content
* The dataset is sourced from [Kaggle](https://www.kaggle.com/codeinstitute/housing-prices-data).
* The dataset consists of 1,460 rows, and represents the characteristics of houses in Ames, Iowa, including house profile (such as Floor Area, Basement, Garage, Kitchen, Lot, Porch, Wood Deck, Year Built) and its respective sale price for houses built between 1872 and 2010.

|Variable|Meaning|Units|
|:----|:----|:----|
|1stFlrSF|First Floor square feet|334 - 4692|
|2ndFlrSF|Second-floor square feet|0 - 2065|
|BedroomAbvGr|Bedrooms above grade (does NOT include basement bedrooms)|0 - 8|
|BsmtExposure|Refers to walkout or garden level walls|Gd: Good Exposure; Av: Average Exposure; Mn: Minimum Exposure; No: No Exposure; None: No Basement|
|BsmtFinType1|Rating of basement finished area|GLQ: Good Living Quarters; ALQ: Average Living Quarters; BLQ: Below Average Living Quarters; Rec: Average Rec Room; LwQ: Low Quality; Unf: Unfinshed; None: No Basement|
|BsmtFinSF1|Type 1 finished square feet|0 - 5644|
|BsmtUnfSF|Unfinished square feet of basement area|0 - 2336|
|TotalBsmtSF|Total square feet of basement area|0 - 6110|
|GarageArea|Size of garage in square feet|0 - 1418|
|GarageFinish|Interior finish of the garage|Fin: Finished; RFn: Rough Finished; Unf: Unfinished; None: No Garage|
|GarageYrBlt|Year garage was built|1900 - 2010|
|GrLivArea|Above grade (ground) living area square feet|334 - 5642|
|KitchenQual|Kitchen quality|Ex: Excellent; Gd: Good; TA: Typical/Average; Fa: Fair; Po: Poor|
|LotArea| Lot size in square feet|1300 - 215245|
|LotFrontage| Linear feet of street connected to property|21 - 313|
|MasVnrArea|Masonry veneer area in square feet|0 - 1600|
|EnclosedPorch|Enclosed porch area in square feet|0 - 286|
|OpenPorchSF|Open porch area in square feet|0 - 547|
|OverallCond|Rates the overall condition of the house|10: Very Excellent; 9: Excellent; 8: Very Good; 7: Good; 6: Above Average; 5: Average; 4: Below Average; 3: Fair; 2: Poor; 1: Very Poor|
|OverallQual|Rates the overall material and finish of the house|10: Very Excellent; 9: Excellent; 8: Very Good; 7: Good; 6: Above Average; 5: Average; 4: Below Average; 3: Fair; 2: Poor; 1: Very Poor|
|WoodDeckSF|Wood deck area in square feet|0 - 736|
|YearBuilt|Original construction date|1872 - 2010|
|YearRemodAdd|Remodel date (same as construction date if no remodelling or additions)|1950 - 2010|
|SalePrice|Sale Price|34900 - 755000|

## Business Requirements
The client for the app has inherited four houses in Ames, Iowa, and is looking at ways to maximise the sale price for the inherited properties. Understanding which characteristics are most closely linked to house prices, and being able to use these through a bespoke tool will help identify which alterations or modifications will add the most value to the properties, as well as provide an informed estimate of the likely sale price for the houses.
The particular requirements for the app are:
* 1 - Identifying how house attributes/ characteristics correlate with sale price with data visualisations of the correlated variables against the sale price to illustrate the relationship between sale price and house characteristics.
* 2 - The client is interested in predicting the house sale price both from the four specific inherited houses, and also any other house in Ames, Iowa.

## Hypothesis and how to validate?
 Based on information from the credit checking agency Experian (https://www.experian.com/blogs/ask-experian/factors-that-affect-home-value/), the five factors that most affect a home’s value are: 
* Prices of Compatible Properties
* The Neighbourhood
* The Home’s Age and Condition
* The Property Size
* The State of the Housing Market.
The dataset available for house prices in Ames, Iowa, does not contain information about compatible properties, neighbourhood, or the housing market.
1. The first hypothesis is that the age, condition, and size of the house will be the key predictors of house prices from the available dataset.
    - Validation: A correlations study with a Predictive Power Score (PPS) element will be used to examine the relationship between features/ characteristics and sale price, identifying the most salient characteristics of a house/ property.
2. The second hypothesis is that based on the identified key characteristics it is possible to predict the prices of four specific houses in Ames, Iowa, using data provided by the client for house prices in the area.
    - Validation: Based on a regression model developed from the overall dataset of house prices, an estimated sale price can been produced for each of the four properties.
3. The third hypothesis is that a machine learning model will be able to predict the price of any house in Ames, Iowa, based on information from a historic dataset of house sales in Ames.
    - Validation: This hypothesis will be validated by the training of a machine learning model, using different logarithms and hyperparameters to achieve the outcome of an accurate prediction tool (with R2 performance scores of no less than 0.75).

## The rationale to map the business requirements to the Data Visualisations and ML tasks
#### Business Requirement 1: Data Visualization and Correlation study
- The relationship between the variables in the dataset, and importantly with the target variable (sale price), can be effectively established through correlation (using both Spearman and Pearson) and predictive power score analysis (PPS).
- The results of the correlation and PPS analysis can be reported both numerically and visually (using a mixture of scatter plots, bar charts, and heatmaps) to give the client a full overview.

#### Business Requirement 2: Predicted sale prices for four inherited houses
- A machine learning model of sale prices based on key characteristics will be trained through identifying a range of logarithms and hyperparameters, and using several different models (including regression analysis, regression with principal component analysis, and a classification model).
- These will be applied to the dataset as a whole (with fewer transformations applied prior to training) and on a dataset that has undergone substantial transformation prior to training to address variation, distribution, and missing values in the dataset.
- As part of the machine learning process, the most important features for predicting house prices will be identified (through assessing feature importance). This analysis is not the same as the correlation and PPS analysis (which examines relationships between features rather than contribution to the machine learning model).
- The machine learning model will enable the prediction of the house prices of the four specific properties, and will be produced as a predicted number with a range based on the 95% confidence interval attached to the figure.
#### Business Requirement 3: A tool to predict house prices of other properties in Ames, Iowa
- A Household Prediction Tool will be developed from the machine learning analysis, whereby based on the best fitting model and the features identified as most important, the user will be able to enter certain key characteristics into a tool and produce a predicted house price for any property in Ames, Iowa.
- The Houseprice Predictor Tool will be deployed and hosted on Heroku.

## ML Business Case
### Houseprice Predictor
#### Regression Model with full dataset
- In order to develop a machine learning model to predict houseprices, a regression model will be built to be trained on a supervised dataset with a continuous numeric target variable.
- The dataset has 1,460 records of house prices with associated characteristics.
- The dataset which will be used will not have undergone extensive feature engineering prior to the training, other than the removal of missing values and non-valid data (such as where data has been defaulted to the lowest value in the dataset, leading to distortion of the distribution of data).
- The success metric for the regression analysis will be:
    - At least 0.75 R2 scores both for the Train and Test Set.
    - The output is an identified model with hyperparameters and feature engineering steps.
    - There will also be an assessment of feature importance, to identify the features that account for the most variance in the dataset.
    - The outcome of the analysis is a predicted sale price, with 95% confidence intervals in order to produce a range.
- The target for the model is SalePrice, with all features present except EnclosedPorch and WoodDeckSF, both of which have more than 80% missing data.
#### Regression Model with full dataset and Principal Component Analysis
- A second machine learning model will be built to predict houseprices, consisting of a regression model with a Principal Component Analysis (PCA) element. This will be built to be trained on a supervised dataset with a continuous numeric target variable.
- The dataset has 1,460 records of house prices with associated characteristics.
- The dataset which will be used will not have undergone extensive feature engineering prior to the training, other than the removal of missing values and non-valid data (such as where data has been defaulted to the lowest value in the dataset, leading to distortion of the distribution of data).
- The success metric for the regression analysis with PCA will be:
    - At least 0.75 R2 scores both for the Train and Test Set.
    - The output is an identified model with hyperparameters and feature engineering steps.
    - The outcome of the analysis is a predicted sale price, with 95% confidence intervals in order to produce a range.
- The target for the model is SalePrice, with all features present except EnclosedPorch and WoodDeckSF, both of which have more than 80% missing data.
#### Classification Model
- A technique to convert the ML task from Regression to Classification was employed in addition to the regression analysis for use with the full dataset. The continuous numerical target (SalePrice) with a range from 34,900-755,000 was recoded into a new feature with 4 bands: '129,975 and under', '129,975 to 163,000', '163,000 to 214,000', and '214,000 and over'.
- As a result the target is categorical and contains 4 classes. This was used to develop a classification model, which is supervised and uni-dimensional.
- The target for the model is a categorised feature SalePrice, with all features present except EnclosedPorch and WoodDeckSF, both of which have more than 80% missing data.
- The model performance is measured with recall, precision, and f1-scores.
- The model success metrics are:
    - At least 0.75 Recall on train and test set
    - The outcome of the analysis is a predicted sale price band.
#### Regression Model with most important features only
- In order to develop a machine learning model to predict houseprices, a regression model will be built to be trained on a supervised dataset with a continuous numeric target variable.
- The dataset has 1,460 records of house prices with associated characteristics.
- The dataset which will be used will not have undergone extensive feature engineering prior to the training, other than the removal of missing values and non-valid data (such as where data has been defaulted to the lowest value in the dataset, leading to distortion of the distribution of data).
- The features included in the analysis were those identified through feature importance analysis of the full dataset:
    - OverallQual
    - GrLivArea
    - TotalBsmtSF
    - GarageArea
    - YearBuilt
- The success metric for the regression analysis will be:
    - At least 0.75 R2 scores both for the Train and Test Set.
    - The output is an identified model with hyperparameters and feature engineering steps.
    - The outcome of the analysis is a predicted sale price, with 95% confidence intervals in order to produce a range.
- The target for the model is SalePrice, with only the selected features present.
#### Regression Model with feature engineered dataset
- In order to develop a machine learning model to predict houseprices, a regression model will be built to be trained on a supervised and feature engineered dataset with a continuous numeric target variable.
- The dataset has 1,460 records of house prices with associated characteristics.
- The dataset which will be used will be used for the regression model with feature engineered dataset has undergone a range of transformations before being trained with appropriate models and hyperparameters. The feature engineering techniques employed are:
    - Winsorization: To address outliers affecting the overall distribution of the data. Applied to:
        - 1stFlrSF
        - 2ndFlrSF
        - BsmtFinSF1
        - BsmtUnfSF
        - GarageArea
        - GrLivArea
        - LotArea
        - LotFrontage
        - MasVnrArea
        - OpenPorchSF
        - TotalBsmtSF
    - Power transformations: To address non-normal distribution of data, applied to:
        - 1stFlrSF
        - GrLivArea
        - MasVnrArea
    - Box Cox transformations: To address non-normal distribution of data, applied to:
        - BsmtFinSF1
        - BsmtUnfSF
        - GarageArea
        - OpenPorchSF
    - SmartCorrelation: To identify variables that are strongly collinear with other variables, leading to the dropping of:
        - 1stFlrSF
        - GarageYrBlt
        - YearBuilt
- The success metric for the regression analysis analysis with feature engineered features will be:
    - At least 0.75 R2 scores both for the Train and Test Set.
    - The output is an identified model with hyperparameters and feature engineering steps.
    - The outcome of the analysis is a predicted sale price, with 95% confidence intervals in order to produce a range.
- The target for the model is SalePrice, with feature engineered variables except EnclosedPorch, WoodDeckSF (both of which were excluded due to over 80% of data missing), 1stFlrSF, GarageYrBlt, and YearBuilt.

## Dashboard Design
### Page 1: Quick project summary
- Quick project summary
    - Project Terms & Jargon
    - Describe Project Dataset
    - State Business Requirements

### Page 2: Correlation and Predictive Power Score Study
- Overview of business requirement answered 
    - Checkbox to inspect the full dataset (first 10 rows)
- Statement of variables found to be most strongly correlated with the target feature (SalePrice)
    - Checkbox: Individual plots showing the distribution of SalePrice for each correlated variable 
    - Checkbox: Heatmap showing the correlations (Spearman and Pearson) between all features, and predictive power score heatmap showing the strength of the predictive power between features 

### Page 3: Houseprice Predictor Tool
- State business requirement 2 and 3
- Provide predicted house prices for the 4 inherited houses
- 5 widget inputs, which relates to the most important features for predicting house price:
    - Overall Quality
    - Year Built
    - Above Ground Living Area
    - Total Basement Area
    - Garage Area
- "Predict House Price" button that serves the input data to the ML pipeline and predicts the house price with 95% confidence intervals providing a range.

### Page 4: Project Hypothesis and Validation
- Overview of factors generally known to affect house prices, and introduction to three hypotheses:
1. The Age, Condition, and Size of the house will be the key predictors of house prices from the available dataset.

**Validation: True.** Correlation analysis and analysis show that the most important characteristics for predicting the value of a house are:
    - The overall quality of the house and kitchen (variables: OverallQual, KitchenQual)
    - The size of the living area above ground, 1st floor square feet, and basement size (square feet) (variables: GrLivArea, 1stFlrSF, TotalBsmtSF)
    - The year the house and garage was built (variables: YearBuilt, GarageYrBlt)
2. The second hypothesis is that based on the first hypothesis it is possible to predict the prices of four specific houses in Ames, Iowa, using data provided by the client.

**Validation: True.** Based on a regression model developed from the overall dataset of house prices, an estimated price has been produced for each of the four properties.
3. The third hypothesis is that it is possible to develop a predictor tool, which based on a user inputting the key characteristics identified in the first hypothesis will produce an estimated house price for any house with data about the relevant characteristics.

**Validation: True.** Based on the machine learning model developed to answer the first two hypotheses, a tool has been produced which will provide an estimated house price for any house based on its key characteristics.

### Page 5: ML: Regression Model
- Description of the models trained for the tools and which one was chosen
- Chosen ML pipeline steps
- Feature importance chart and detail
- Pipeline performance metrics

## Unfixed Bugs
- There is a bug whereby repeated entries in the Houseprice Predictor Tool redirects the user back to the Summary Page
- There is a bug whereby the Summary Page also displays the Houseprice Predictor Tool on loading (and after redirect)

## Deployment
### Heroku

* The App live link is: <https://YOUR_APP_NAME.herokuapp.com/>
* Set the runtime.txt Python version to a [Heroku-20](https://devcenter.heroku.com/articles/python-support#supported-runtimes) stack currently supported version.
* The project was deployed to Heroku using the following steps.

1. Log in to Heroku and create an App
2. At the Deploy tab, select GitHub as the deployment method.
3. Select your repository name and click Search. Once it is found, click Connect.
4. Select the branch you want to deploy, then click Deploy Branch.
5. The deployment process should happen smoothly if all deployment files are fully functional. Click the button Open App on the top of the page to access your App.
6. If the slug size is too large then add large files not required for the app to the .slugignore file.

## Main Data Analysis and Machine Learning Libraries
The machine learning libraries used during this project were:
- Python -The programming language employed throughout the project
- Jupyter -Provided the infrastructure for the project through Jupyter Notebooks
- Pandas -Data analysis and manipulation library, used to format and analyse dataframes throughout the project
- Numpy -Used for data analysis and feature engineering throughout the project, such as in the pipeline and hyperparameter searches and in lambda functions
- Matplotlib -Visualisation library used for the production of charts and visualisation throughout the project
- Seaborn -A statistical data library for data visualisation, used to produce bar charts, scatter plots and other visualisations
- Streamlit -Used to develop the App for the project, with inbuilt styles and formatting features
- Joblib -Used for pipelining, especially linking the app and the sources (datasets and pipelines/ pkl-files)
- ydata_profiling -A library used to produce quick data summaries for data, primarily used in the data cleaning and feature engineering notebooks to provide a comprehensive overview of the data
- feature_engine – Used as part of the feature-engineering process, providing functions for engineering and transforming data
- Scikit-learn -Used for predictive data analysis, especially as part of the model pipelining to explore the most suitable models for the dataset (especially in the regression analysis and modelling notebooks)

## Credits
### Content
* The layout of the Streamlit app was inspired by the Code Institute Churnometer Walkthrough project
* The information about factors generally considered linked to house prices was from Experian (https://www.experian.com/blogs/ask-experian/factors-that-affect-home-value/)
* Background information about Ames, Iowa, was sourced from the official Ames, Iowa website (https://www.cityofames.org/home) 
* The dataset used was sourced from Kaggle (https://www.kaggle.com/datasets/codeinstitute/housing-prices-data?resource=download)

### Code

* Custom code snippets were provided by Code Institute (such as PipelineOptimization, HyperparameterOptimization, DataCleaningEffect, DisplayCorrAndPPS, FeatureEngineeringAnalysis), all adapted by me for the project.
* The coders of the world assisted with hints and pointers from other queries on StackOverflow and Reddit
* ChatGPT was used to solve particularly tricky aspects of the code, especially when the ordering of variables for the App became an issue.
* A number of books were consulted to aid the development of the code and project, including:
    - Muller, A. C. & Guido, S. (2018): Introduction to Machine Learning with Python. O’Reilly Media.
    - Huyen, C. (2022): Designing Machine Learning Systems. O’Reilly Media.
    - Nelson, H. (2023): Essential Math for AI. O’Reilly Media.
    - Ozoemena, S. (2022): Machine Learning Explained the Simple Way. Simple Code Publishing.

## Acknowledgements
* A huge thank you goes to my wife, Joanne Lovbakke, for her unwavering support throughout the project.
