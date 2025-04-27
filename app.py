import streamlit as st
import utils
import filter


def main():
    st.title("🏠 House Price Prediction")

    st.markdown("### Enter Property Details")

    Type = st.selectbox("Property Type",filter.property_types)
    Bedrooms = st.number_input("Bedrooms", min_value=1, step=1)
    Bathrooms = st.number_input("Bathrooms", min_value=1, step=1)
    Area = st.number_input("Area (sqm)", min_value=20.0, step=5.0)
    Furnished = st.radio("Furnished", ["Yes", "No"])
    Level = st.number_input("Floor Level", min_value=0, step=1)
    Compound = st.selectbox("Compound (or type 'Unknown')", filter.compounds)
    Payment_Option = st.selectbox("Payment Option", ["Cash", "Cash or Installment","Installment"])
    Delivery_Date = st.number_input("Delivery in (months)", min_value=0, step=1)
    Delivery_Term = st.selectbox("Delivery Term", filter.delivery_term)
    City = st.selectbox("City", filter.cities)

    # Area_Category = st.selectbox("Area Category", ["Very Large", "Large", "Medium", "Small"])
    # in_Compound = st.radio("Inside Compound?", ["Yes", "No"])
    # Immediate_Move = st.radio("Ready to Move?", ["Yes", "No"])

    # Convert binary inputs
    # in_Compound = 1 if in_Compound == "Yes" else 0
    # Immediate_Move = 1 if Immediate_Move == "Yes" else 0

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
        # "Area_Category": [Area_Category],
        # "in_Compound": [in_Compound],
        # "Immediate_Move": [Immediate_Move]
    }

    if st.button("Predict Price"):

        predicted_price = utils.predict_price(input_data)
        
        st.markdown(f"<h2 style='color: green;'>💰 ${predicted_price:,.2f}</h2>", unsafe_allow_html=True)


if __name__ == '__main__':
    import subprocess
    from streamlit import runtime
    if runtime.exists():
        main()
    else:
        process = subprocess.Popen(["streamlit", "run", "./main.py"])
