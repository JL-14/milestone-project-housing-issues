import streamlit as st

def page_summary_body():

    st.write("### Summary of a House Price Predictor for Ames, Iowa")

    # text based on README file - "Dataset Content" section
    st.info(
        f"**Project Terms & Jargon**\n"
        f"* A **house** is a property in Ames, Iowa, irrespective of its characteristics.\n"
        f"* **Sale price** is the price which was paid when the house was sold.\n"
        f"* **Characteristics** refer to both objective and subjectivedescriptions"
        f"of the house. Objective characteristics include space in square feet, number"
        f"of bedrooms, etc, whilst subjective include judgment of overall quality and"
        f"condition of the house, quality of the kitchen, etc.\n "
        f"**Project Dataset**\n"
        f"* The dataset is a publicly available record of 1,460 house sales in Ames, Iowa,"
        f"containing a range of information about each house sold, including"
        f"the sale price. The information available does not provide a complete picture"
        f"of each house sold, with certain pertinent information not included, such as"
        f"location and neighbourhood characteristics, date of sale, and other factors"
        f"likely to affect house prices.\n"
        f"Nonetheless, there is sufficient information available in the dataset for"
        f"the Houseprice Predictor tool to provide accurate estimates for house prices"
        f"based on the available characteristics."
        )

    # Link to README file, so the users can have access to full project documentation
    st.write(
        f"* Additional information about the Houseprice Predictor app can be"
        f"found in the"
        f"[Project README file](https://github.com/Code-Institute-Solutions/churnometer).")    

    # copied from README file - "Business Requirements" section
    st.success(
        f"The project has 3 business requirements:\n"
        f"* **1** - The **client** is interested in discovering how the characteristics of houses correlate with the sale price." 
        f"Therefore, the client expects data visualisations of the correlated variables against the sale price to show how"
        f"they are linked.\n"
        f"* **2** - The client is interested in predicting the house sale price for her four inherited houses.\n"
        f"* **3** - The client would also like to be able to estimate sale prices for other houses in Ames, Iowa, based on"
        f"the relevant and pertinent characteristics identified through the Machine Learning process."
        )
