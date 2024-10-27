import streamlit as st

def page_summary_body():
    st.write("### Summary of a House Price Predictor for Ames, Iowa")

    st.info(
        """**Project Terms & Jargon**

- A **house** is a property in Ames, Iowa, irrespective of its characteristics.
- **Sale price** is the price that was paid when the house was sold.
- **Characteristics** refer to both objective and subjective descriptions 
of the house. Objective characteristics include space in square feet, number 
of bedrooms, etc., while subjective characteristics include judgments of overall quality, 
condition of the house, quality of the kitchen, etc.

**Project Dataset**

The dataset is a publicly available record of 1,460 house sales in Ames, Iowa,
containing a range of information about each house sold, including
the sale price. The available information does not provide a complete picture
of each house sold, with certain pertinent information not included, such as
location and neighborhood characteristics, date of sale, and other factors
likely to affect house prices.

Nonetheless, there is sufficient information available in the dataset for
the Houseprice Predictor tool to provide accurate estimates for house prices
based on the available characteristics.
"""
    )

    # Link to README file for full project documentation
    st.write(
        """* Additional information about the Houseprice Predictor app can be
        found in the
        [Project README file](https://github.com/JL-14/milestone-project-housing-issues/blob/main/README.md).
        """
    )

    st.success(
        """The project has 3 business requirements:
        
- **1** - The **client** is interested in **discovering how the characteristics  
of houses correlate with the sale price.** Therefore, the client expects data visualizations of the 
correlated variables against the sale price to show how they are linked.

- **2** - The **client** is interested in **predicting the house sale price  
for her four inherited houses.**

- **3** - The **client** would also like to be able to **estimate sale prices  
for other houses in Ames, Iowa,** based on the relevant and pertinent 
characteristics identified through the Machine Learning process.
        """
    )
