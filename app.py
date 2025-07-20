import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Load the dataset
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    columns = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", 
               "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
    data = pd.read_csv(url, names=columns)
    return data

# Train model
@st.cache_resource
def train_model(data):
    X = data.drop("Outcome", axis=1)
    y = data["Outcome"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    return model, X.columns

# Main Streamlit app
def main():
    st.title("🧬 Diabetes Prediction App")
    st.write("Enter medical details to predict if the patient has diabetes.")

    data = load_data()
    model, features = train_model(data)

    # Collect user input
    user_input = {}
    for feature in features:
        user_input[feature] = st.number_input(feature, value=float(data[feature].mean()))

    input_df = pd.DataFrame([user_input])

    if st.button("Predict"):
        prediction = model.predict(input_df)[0]
        result = "🟢 No Diabetes" if prediction == 0 else "🔴 Diabetes Detected"
        st.subheader(f"Prediction: {result}")

        # Feature importance
        importance = model.feature_importances_
        fig, ax = plt.subplots()
        ax.barh(features, importance)
        ax.set_title("Feature Importance")
        st.pyplot(fig)

if __name__ == "__main__":
    main()