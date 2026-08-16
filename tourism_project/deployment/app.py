import os
import streamlit as st
import pandas as pd
import joblib

# Load the model committed by the pipeline
model_path = os.path.join(os.path.dirname(__file__), "best_tourism_model_v1.joblib")
model = joblib.load(model_path)

st.title("Tourism Package Prediction App")
st.write("""
This application predicts whether a customer will purchase a tourism package.
Enter the customer details below to get a prediction.
""")

# Input widgets for features
Age = st.number_input("Age", min_value=18, max_value=99, value=35)
TypeofContact = st.selectbox("Type of Contact", ["Self Inquiry", "Company Invited"])
CityTier = st.selectbox("City Tier", [1, 2, 3])
DurationOfPitch = st.number_input("Duration of Pitch (minutes)", min_value=0, max_value=60, value=10)
Occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer", "Government Sector"])
Gender = st.selectbox("Gender", ["Male", "Female"])
NumberOfPersonVisiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
PreferredPropertyStar = st.selectbox("Preferred Property Star", [3, 4, 5])
MaritalStatus = st.selectbox("Marital Status", ["Married", "Single", "Divorced"])
NumberOfTrips = st.number_input("Number of Trips Annually", min_value=0, max_value=50, value=5)
Passport = st.selectbox("Has Passport?", [0, 1])
OwnCar = st.selectbox("Owns a Car?", [0, 1])
NumberOfChildrenVisiting = st.number_input("Number of Children Visiting", min_value=0, max_value=5, value=0)
Designation = st.selectbox("Designation", ["Manager", "Executive", "Senior Manager", "AVP", "VP", "President"])
MonthlyIncome = st.number_input("Monthly Income", min_value=0.0, value=50000.0)
PitchSatisfactionScore = st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
ProductPitched = st.selectbox("Product Pitched", ["Deluxe", "Basic", "Standard", "Super Deluxe", "King"])
NumberOfFollowups = st.number_input("Number of Follow-ups", min_value=0, max_value=10, value=3)

input_data = pd.DataFrame([{
    "Age": Age,
    "TypeofContact": TypeofContact,
    "CityTier": CityTier,
    "DurationOfPitch": DurationOfPitch,
    "Occupation": Occupation,
    "Gender": Gender,
    "NumberOfPersonVisiting": NumberOfPersonVisiting,
    "PreferredPropertyStar": PreferredPropertyStar,
    "MaritalStatus": MaritalStatus,
    "NumberOfTrips": NumberOfTrips,
    "Passport": Passport,
    "OwnCar": OwnCar,
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "Designation": Designation,
    "MonthlyIncome": MonthlyIncome,
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "ProductPitched": ProductPitched,
    "NumberOfFollowups": NumberOfFollowups
}])

if st.button("Predict Purchase"):    
    prediction = model.predict(input_data)[0]
    result = "Customer WILL purchase the package!" if prediction == 1 else "Customer will NOT purchase the package."
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
