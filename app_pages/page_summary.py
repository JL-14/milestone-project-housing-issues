import streamlit as st

def page_summary_body():

    st.write("### Summary of a House Price Predictor for Ames, Iowa")

    st.info(
    """**Project Terms & Jargon**

- A **house** is a property in Ames, Iowa, irrespective of its characteristics.\n\n
- **Sale price** is the price which was paid when the house was sold.\n\n
- **Characteristics** refer to both objective and subjective descriptions
of the house. Objective characteristics include space in square feet, number
of bedrooms, etc., whilst subjective characteristics include judgment of overall quality and
condition of the house, quality of the kitchen, etc.\n\n

**Project Dataset**

The dataset is a publicly available record of 1,460 house sales in Ames, Iowa,
containing a range of information about each house sold, including
the sale price. The information available does not provide a complete picture
of each house sold, with certain pertinent information not included, such as
location and neighbourhood characteristics, date of sale, and other factors
likely to affect house prices.

Nonetheless, there is sufficient information available in the dataset for
the Houseprice Predictor tool to provide accurate estimates for house prices
based on the available characteristics.
    """
)


    # Link to README file, so the users can have access to full project documentation
    st.write(
        """* Additional information about the Houseprice Predictor app can be
        found in the
        [Project README file](https://github.com/Code-Institute-Solutions/churnometer).
        """
    )

    st.success(
"""The project has 3 business requirements:
- **1** - The **client** is interested in **discovering how the characteristics 
of houses correlate with the sale price.**
Therefore, the client expects data visualisations of the correlated variables 
against the sale price to show how they are linked.\n\n
- **2** - The **client** is interested in **predicting the house sale price 
for her four inherited houses.**\n\n
- **3** - The **client** would also like to be able to **estimate sale prices 
for other houses in Ames, Iowa,** based on the relevant and pertinent 
characteristics identified through the Machine Learning process."""
    )
