# streamlit_app/app.py

import streamlit as st
import pandas as pd
from src.utils import predict_price
from src.filter import property_types,compounds,delivery_term,cities,compound_city_dict
import warnings
# Suppress all warnings
# warnings.filterwarnings("ignore")

import os
import warnings

# Set logical CPU cores manually
os.environ["LOKY_MAX_CPU_COUNT"] = "8"

# Suppress specific UserWarning
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=".*Could not find the number of physical cores.*"
)




def main():
    

    st.set_page_config(page_title="House Price Prediction", page_icon="🏠")
    
    # REMOVE default top padding
    st.markdown(
        """
        <style>
            .block-container {
                padding-top: 1rem;  /* Default is ~6rem. Smaller means move up */
            }
        </style>
        """,
        unsafe_allow_html=True
    )


    
    st.title("House Price Prediction")
    st.markdown("### Enter Property Details")


    st.image("Images/house.jpeg", width=50)



    

    
    


    # Sidebar - model choice
    st.sidebar.title("🔍 Select Prediction Model")
    model_choice = st.sidebar.selectbox("Choose Model", ("Lightgbm", "SVR", "Xgboost"))

    
    # --- Property Form ---
    col1, col2 = st.columns(2)

    with col1:
        Type = st.selectbox("Property Type",property_types )  # Example types
        Furnished = st.radio("Furnished", ["Yes", "No"])
        Compound = st.selectbox("Compound (or type 'Not in Compound')", compounds)
        Payment_Option = st.selectbox("Payment Option", ["Cash","Installment"])
        Delivery_Term = st.selectbox("Delivery Term", delivery_term)

        # city_options =cities
        if Compound != "Not in Compound":
            city_options = compound_city_dict[Compound]

        else :
            city_options =cities
        # --- Location selection ---
        City = st.selectbox("Choose Place (or type 'Unknown')", city_options)
 
        
    with col2:

        Bedrooms = st.slider("Bedrooms", min_value=1, max_value=10, step=1)
        Bathrooms = st.slider("Bathrooms", min_value=1, max_value=10, step=1)
        Area = st.slider("Area (sqm)", min_value=20, max_value=1000, step=1)
        Level = st.slider("Floor Level", min_value=0, max_value=10, step=1)
        Delivery_Date = st.slider("Delivery in (months)", min_value=0, max_value=72, step=1)

    

    


    

    input_data = {
        "Type": [Type],
        "Bedrooms": [Bedrooms],
        "Bathrooms": [Bathrooms],
        "Area": [Area],
        "Furnished": [Furnished],
        "Level": [Level],
        "Compound": [Compound],
        "Payment_Option": [Payment_Option],
        "Delivery_Date": [Delivery_Date],
        "Delivery_Term": [Delivery_Term],
        "City": City,
        
    }


    if st.button("Predict Price"):
        
        
        
        predicted_price = predict_price(input_data,model_choice)
        
        
        # st.success(f"Predicted Price EGP")
        
        
        st.markdown(f"<h2 style='color: green;'> 💰 EGP {predicted_price:,.2f}</h2>", unsafe_allow_html=True)


if __name__ == '__main__':
    import subprocess
    from streamlit import runtime
    if runtime.exists():
        main()
    else:
        process = subprocess.Popen(["streamlit", "run", "./main.py"])
