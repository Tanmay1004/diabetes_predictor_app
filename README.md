Diabetes Prediction App - Local Deployment Guide
-------------------------------------------------

This is a Streamlit-based machine learning app that predicts whether a person has diabetes based on medical input features.

How to Run the App Locally
--------------------------

1. Make sure Python is installed (version 3.7+ recommended).

2. Open a terminal or command prompt and navigate to the folder where this ZIP was extracted.

3. (Optional) Create and activate a virtual environment:
   Windows:
       python -m venv venv
       venv\Scripts\activate
   macOS/Linux:
       python3 -m venv venv
       source venv/bin/activate

4. Install required libraries using:
       pip install -r requirements.txt

5. Launch the app using:
       streamlit run app.py

6. A browser window should automatically open.
   If not, manually go to: http://localhost:8501

Files Included
--------------
- app.py             → Main Streamlit application
- requirements.txt   → List of required Python libraries
- README.txt         → This instruction file

Dataset Used
------------
Pima Indians Diabetes dataset from the UCI Machine Learning Repository.

