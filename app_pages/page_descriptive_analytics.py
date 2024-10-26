import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
from feature_engine.discretisation import ArbitraryDiscretiser
import streamlit as st
import seaborn as sns
import pandas as pd
import ppscore as pps
from src.data_management import load_houseprice_data

sns.set_style("whitegrid")


def page_descriptive_analytics_body():

    st.write("### Descriptive analysis of house sales in Ames, Iowa")

    # load data
    df = load_houseprice_data()

    # hard copied from churned customer study notebook
    vars_to_study = ['1stFlrSF', 'GrLivArea', 'KitchenQual',
                    'OverallQual', 'TotalBsmtSF', 'YearBuilt',
                    'GarageYrBlt']

    st.write("### Correlation and Predictive Power Score Study")
    st.info(
        f"* The client is interested in understanding how house saleprice is related to the"
        f"characteristics of a house. A correlation study has therefore been conducted to"
        f"examine the relationship between characteristics and house price."
        f"Predictive Power Score analysis has also been conducted to identify the"
        f"characteristics that are the most powerful predictors of house price.")

    # inspect data
    if st.checkbox("Inspect Full Houseprice Dataset"):
        st.write(
            f"* The dataset has {df.shape[0]} rows and {df.shape[1]} columns, "
            f"find below the first 10 rows.")

        st.write(df.head(10))

    st.write("---")

    # Correlation Study Summary
    st.write(
        f"* Following the correlation study, the following variables were found to be most"
        f"strongly correlated with house price.\n"
        f"The most correlated variable were found to be: **{vars_to_study}**"
    )

    # Text based on "02 - Churned Customer Study" notebook - "Conclusions and Next steps" section
    st.info(
        "The correlation study found that houseprice is related to: \n"
        "- The overall quality of the house (variable: OverallQual) \n"
        "- The size of the living area above ground, 1st floor square feet, and basement size (square feet) \n"
        "(variables: GrLivArea, 1stFlrSF, TotalBsmtSF) \n"
        "- The quality of the kitchen (variable: KitchenQual) \n"
        "- The year the house and garage was built (variables: YearBuilt, GarageYrBlt)\n"
        "\n \n Some of the characteristics are clearly linked (such as the year the house was built"
        "and the year the garage was built, and the size of the 1st floor, living area above ground, and the basement size),"
        "as can be seen in the Spearman correlation heatmap below."
    )

    # Interactive checkboxes
    df_eda = df.filter(vars_to_study + ['SalePrice'])

    # Individual plots per variable
    if st.checkbox("Houseprice per Correlated Variable"):
        houseprice_per_variable(df_eda)

    # Heatmap
    if st.checkbox("Correlation Heatmap"):
        st.write(
            "* Boxes that are lighter green or yellow show more highly correlated variables" 
            "(the heatmaps may take a moment to load)")
        correlation_heatmap(df)


## Variable Distribution by SalePrice" section
def houseprice_per_variable(df):
    vars_to_study = ['1stFlrSF', 'GrLivArea', 'KitchenQual',
                    'OverallQual', 'TotalBsmtSF', 'YearBuilt',
                    'GarageYrBlt']
    palette = sns.color_palette("coolwarm", 10)
    target_var = 'SalePrice'
    
    # Ensure the dataframe contains the required variables
    df_eda = df.filter(vars_to_study + [target_var])

    # Defining the label mappings for KitchenQual and OverallQual
    kitchen_qual_mapping = {'Ex': 'Excellent', 'Gd': 'Good', 'TA': 'Typical', 'Fa': 'Fair', 'Po': 'Poor'}
    overall_qual_mapping = {1: 'Very Poor', 2: 'Poor', 3: 'Fair', 4: 'Below Average',
                            5: 'Average', 6: 'Above Average', 7: 'Good',
                            8: 'Very Good', 9: 'Excellent', 10: 'Very Excellent'}

    # Creating SalePriceBand for visualisation
    min_price = df_eda[target_var].min()
    max_price = df_eda[target_var].max()

    bin_width = (max_price - min_price) / 10
    bins = [min_price + i * bin_width for i in range(11)]
    df_eda['SalePriceBand'] = pd.cut(df_eda[target_var], bins=bins, labels=range(10), include_lowest=True)

    # Ensuring full range of categories (even with zero values)
    kitchen_qual_order = sorted(kitchen_qual_mapping.keys())
    overall_qual_order = sorted(overall_qual_mapping.keys())

    def plot_categorical(df_eda, col, target_var, label_mapping=None, categories_order=None):
        fig, ax = plt.subplots(figsize=(12, 5))  # Create a figure and axis object
        sns.countplot(data=df_eda, hue='SalePriceBand', x=col, palette=palette, order=categories_order, ax=ax)

        # Setting x-tick labels using the label mapping for KitchenQual and OverallQual
        if label_mapping:
            tick_labels = [label_mapping.get(val, val) for val in categories_order]
            ax.set_xticks(range(len(categories_order)))
            ax.set_xticklabels(tick_labels, rotation=90)
        
        ax.set_title(f"{col}", fontsize=20, y=1.05)
        
        st.pyplot(fig)

    def plot_numerical(df_eda, col, target_var, label_mapping=None):
        fig, ax = plt.subplots(figsize=(8, 5))

        # Plot histogram for the single numerical variable
        sns.histplot(data=df_eda, x=col, kde=True, hue='SalePriceBand', element="step", palette=palette, ax=ax)
        
        ax.set_title(f"{col}", fontsize=20, y=1.05)
        
        st.pyplot(fig)

    def plot_scatter_with_correlation(df_eda, x_col, y_col):
        """Plots scatter plot between two numerical variables with a regression line"""
        
        # Ensure that x_col and y_col are passed as column names (strings), not as Series
        if x_col not in df_eda.columns or y_col not in df_eda.columns:
            st.error(f"Columns {x_col} and {y_col} must exist in the dataframe.")
            return
        
        # Drop rows where x_col or y_col have NaN values
        df_clean = df_eda[[x_col, y_col]].dropna()

        # Ensure both columns are numeric
        if not pd.api.types.is_numeric_dtype(df_clean[x_col]) or not pd.api.types.is_numeric_dtype(df_clean[y_col]):
            st.error(f"Columns {x_col} and {y_col} must be numeric.")
            return
        
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Create a scatter plot with a regression line
        sns.regplot(data=df_clean, x=x_col, y=y_col, scatter_kws={"s": 20}, line_kws={"color": "red"}, ci=None, ax=ax)
        
        # Set the x and y axis labels
        ax.set_xlabel(x_col, fontsize=12)
        ax.set_ylabel(y_col, fontsize=12)
        
        # Set the plot title
        ax.set_title(f"Scatter Plot of {x_col} vs {y_col}", fontsize=16, y=1.05)
        
        # Render the figure in Streamlit
        st.pyplot(fig)

    # Visualisation loop
    target_var = 'SalePrice'
    vars_to_study = ['1stFlrSF', 'GrLivArea', 'KitchenQual',
                    'OverallQual', 'TotalBsmtSF', 'YearBuilt',
                    'GarageYrBlt']
    for col in vars_to_study:
        if col == 'KitchenQual':
            plot_categorical(df_eda, col, target_var, label_mapping=kitchen_qual_mapping, categories_order=kitchen_qual_order)
        elif col == 'OverallQual':
            plot_categorical(df_eda, col, target_var, label_mapping=overall_qual_mapping, categories_order=overall_qual_order)
        elif df_eda[col].dtype == 'object':
            plot_categorical(df_eda, col, target_var)
        else:
            # plot_numerical(df_eda, col, target_var)

            # Add scatter plot with correlation line for numerical variables (e.g., SalePrice vs another numerical column)
            plot_scatter_with_correlation(df_eda, col, target_var)

## Correlation Heatmap section, Notebook 3
def correlation_heatmap(df):

    def heatmap_corr(corr_matrix, threshold, figsize=(20, 12), font_annot=8):
        if len(corr_matrix.columns) > 1:
            # Create a mask for the upper triangle of the heatmap
            mask = np.zeros_like(corr_matrix, dtype=np.bool)
            mask[np.triu_indices_from(mask)] = True
            mask[abs(corr_matrix) < threshold] = True

            # Create a figure and axis object
            fig, axes = plt.subplots(figsize=figsize)
            
            # Generate the heatmap
            sns.heatmap(corr_matrix, annot=True, xticklabels=True, yticklabels=True,
                        mask=mask, cmap='viridis', annot_kws={"size": font_annot}, ax=axes,
                        linewidth=0.5)
            
            # Optional: remove y-tick rotation
            axes.set_yticklabels(corr_matrix.columns, rotation=0)
            
            # Render the figure in Streamlit
            st.pyplot(fig)  # Pass the figure to st.pyplot to render it in the dashboard

    def CalculateCorrAndPPS(df):
        # Calculate correlation matrices for all variables in df
        df_corr_spearman = df.corr(method="spearman")
        df_corr_pearson = df.corr(method="pearson")

        # Calculate PPS matrix for all variables in df
        pps_matrix_raw = pps.matrix(df)
        pps_matrix = pps_matrix_raw.filter(['x', 'y', 'ppscore']).pivot(columns='x', index='y', values='ppscore')

        # Display PPS score statistics
        pps_score_stats = pps_matrix_raw.query("ppscore < 1").filter(['ppscore']).describe().T
        print("PPS threshold - check PPS score IQR to decide threshold for heatmap \n")
        print(pps_score_stats.round(3))

        return df_corr_pearson, df_corr_spearman, pps_matrix

    def DisplayCorrAndPPS(df_corr_pearson, df_corr_spearman, pps_matrix, CorrThreshold, PPS_Threshold,
            figsize=(20, 12), font_annot=8):

        st.write("*** Heatmap: Spearman Correlation ***")
        st.write("It evaluates monotonic relationships \n")
        
        # Display the heatmap for Spearman correlation
        heatmap_corr(df_corr_spearman, threshold=CorrThreshold, figsize=figsize, font_annot=font_annot)

        st.write("*** Heatmap: Pearson Correlation ***")
        st.write("It evaluates linear relationships \n")
        heatmap_corr(df_corr_pearson, threshold=CorrThreshold, figsize=figsize, font_annot=font_annot)

        st.write("*** Heatmap: PPS (Predictive Power Score) ***")
        st.write("It evaluates predictive relationships \n")
        heatmap_corr(pps_matrix, threshold=PPS_Threshold, figsize=figsize, font_annot=font_annot)

    # Calculate correlation and PPS matrices for all variables in df
    df_corr_pearson, df_corr_spearman, pps_matrix = CalculateCorrAndPPS(df)

    # Display correlation and PPS heatmaps for all variables in df
    DisplayCorrAndPPS(df_corr_pearson=df_corr_pearson,
                      df_corr_spearman=df_corr_spearman, 
                      pps_matrix=pps_matrix,
                      CorrThreshold=0.4, PPS_Threshold=0.2,
                      figsize=(12, 10), font_annot=10)


