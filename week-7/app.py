import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Load the trained model
with open('iris_model.pkl', 'rb') as file:
    model = pickle.load(file)

# App title
st.title("Iris Flower Species Predictor")
st.write("Input flower measurements and get prediction from trained ML model.")

# Input fields
sepal_length = st.slider("Sepal length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.slider("Sepal width (cm)", 2.0, 4.5, 3.5)
petal_length = st.slider("Petal length (cm)", 1.0, 7.0, 1.4)
petal_width = st.slider("Petal width (cm)", 0.1, 2.5, 0.2)

# Predict
input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                          columns=['sepal length (cm)', 'sepal width (cm)', 
                                   'petal length (cm)', 'petal width (cm)'])

if st.button("Predict"):
    prediction = model.predict(input_data)
    proba = model.predict_proba(input_data)
    
    species = load_iris().target_names[prediction[0]]
    st.success(f"Predicted species: **{species}**")
    
    # Show prediction probabilities
    st.subheader("Prediction Probability")
    proba_df = pd.DataFrame(proba, columns=load_iris().target_names)
    st.bar_chart(proba_df.T)

# Show raw data
if st.checkbox("Show sample data"):
    iris = load_iris(as_frame=True)
    df = iris.frame
    st.dataframe(df.head())

    # Plot pairplot
    st.subheader("Feature Distribution")
    fig, ax = plt.subplots()
    sns.pairplot(df, hue='target', palette='bright')
    st.pyplot(fig)
