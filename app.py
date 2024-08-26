import pickle
import numpy as np
import streamlit as st

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

# Set up the title and layout
st.title("ⴍage - Salary Prediction App")
st.markdown("<h6 style='text-align: center;'>A simple web app to predict annual salary</h6>", unsafe_allow_html=True)

# User inputs
gender = st.radio('Pick your gender', ["Female", "Male"])
age = st.slider('Pick your age', 21, 55)
education = st.selectbox('Pick your education level', ["Bachelor's", "Master's", "PhD"])
job = st.selectbox('Pick your job title', ["Director of Marketing", "Director of Operations", "Senior Data Scientist", "Senior Financial Analyst", "Senior Software Engineer"])
experience = st.slider('Pick your years of experience', 0.0, 25.0, 0.0, 0.5, "%1f")

# Prediction
if st.button('Predict Salary'):
    X = [
        int(age),
        int(["Female", "Male"].index(gender)),
        int(["Bachelor's", "Master's", "PhD"].index(education)),
        int(["Director of Marketing", "Director of Operations", "Senior Data Scientist", "Senior Financial Analyst", "Senior Software Engineer"].index(job)),
        float(experience)
    ]
    salary = model.predict([X])[0]
    st.write(f"Estimated salary: ${int(salary)}")
