import plotly.express as px
import numpy as np
from feature_engine.discretisation import ArbitraryDiscretiser
import streamlit as st
from src.data_management import load_houseprice_data

# import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")


def page_descriptive_analytics_body():

    st.write("### Descriptive analysis of house sales in Ames, Iowa")

    st.info(
        f"This is where the descriptive data and correlation/ PPS charts will appear"
    )

    # load data
    df = load_houseprice_data()

    # hard copied from churned customer study notebook
    vars_to_study = ['1stFlrSF', 'GrLivArea', 'KitchenQual',
                    'OverallQual', 'TotalBsmtSF', 'YearBuilt',
                    'GarageYrBlt']

    st.write("### Churned Customer Study")
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
        f"The correlation study found that houseprice is related to: \n"
        f"- The overall quality of the house (variable: OverallQual)"
        f"- The size of the living area above ground, 1st floor square feet, and basement size (square feet)"
        f"(variables: GrLivArea, 1stFlrSF, TotalBsmtSF)"
        f"- The quality of the kitchen (variable: KitchenQual)"
        f"- The year the house and garage was built (variables: YearBuilt, GarageYrBlt)"
        f"\n \n Some of the characteristics are clearly linked (such as the year the house was built"
        f"and the year the garage was built, and the size of the 1st floor, living area above ground, and the basement size),"
        f"as can be seen in the Spearman correlation heatmap below."
    )

    # Code copied from "02 - Churned Customer Study" notebook - "EDA on selected variables" section
    df_eda = df.filter(vars_to_study + ['SalePrice'])

    # Individual plots per variable
    if st.checkbox("Houseprice per Correlated Variable"):
        houseprice_per_variable(df_eda)

    # Parallel plot
    if st.checkbox("Correlation Heatmap"):
        st.write(
            f"* Boxes that are lighter green or yellow show more highly correlated variables)")
        parallel_plot_churn(df_eda)


# # function created using "02 - Churned Customer Study" notebook code - "Variables Distribution by Churn" section
# def churn_level_per_variable(df_eda):
#     target_var = 'Churn'

#     for col in df_eda.drop([target_var], axis=1).columns.to_list():
#         if df_eda[col].dtype == 'object':
#             plot_categorical(df_eda, col, target_var)
#         else:
#             plot_numerical(df_eda, col, target_var)


# # code copied from "02 - Churned Customer Study" notebook - "Variables Distribution by Churn" section
# def plot_categorical(df, col, target_var):
#     fig, axes = plt.subplots(figsize=(12, 5))
#     sns.countplot(data=df, x=col, hue=target_var,
#                   order=df[col].value_counts().index)
#     plt.xticks(rotation=90)
#     plt.title(f"{col}", fontsize=20, y=1.05)
#     st.pyplot(fig)  # st.pyplot() renders image, in notebook is plt.show()


# # code copied from "02 - Churned Customer Study" notebook - "Variables Distribution by Churn" section
# def plot_numerical(df, col, target_var):
#     fig, axes = plt.subplots(figsize=(8, 5))
#     sns.histplot(data=df, x=col, hue=target_var, kde=True, element="step")
#     plt.title(f"{col}", fontsize=20, y=1.05)
#     st.pyplot(fig)  # st.pyplot() renders image, in notebook is plt.show()


# # function created using "02 - Churned Customer Study" notebook code - Parallel Plot section
# def parallel_plot_churn(df_eda):

#     # hard coded from "disc.binner_dict_['tenure']"" result,
#     tenure_map = [-np.Inf, 6, 12, 18, 24, np.Inf]
#     # found at "02 - Churned Customer Study" notebook
#     # under "Parallel Plot" section
#     disc = ArbitraryDiscretiser(binning_dict={'tenure': tenure_map})
#     df_parallel = disc.fit_transform(df_eda)

#     n_classes = len(tenure_map) - 1
#     classes_ranges = disc.binner_dict_['tenure'][1:-1]
#     LabelsMap = {}
#     for n in range(0, n_classes):
#         if n == 0:
#             LabelsMap[n] = f"<{classes_ranges[0]}"
#         elif n == n_classes-1:
#             LabelsMap[n] = f"+{classes_ranges[-1]}"
#         else:
#             LabelsMap[n] = f"{classes_ranges[n-1]} to {classes_ranges[n]}"

#     df_parallel['tenure'] = df_parallel['tenure'].replace(LabelsMap)
#     fig = px.parallel_categories(
#         df_parallel, color="Churn", width=750, height=500)
#     # we use st.plotly_chart() to render, in notebook is fig.show()
#     st.plotly_chart(fig)
