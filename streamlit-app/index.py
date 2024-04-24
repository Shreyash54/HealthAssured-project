import pickle
import streamlit as st
import pandas as pd
import pickle
from utils import *
import numpy as np


# Sidebar for navigation
with st.sidebar:
    selected = st.selectbox(
        "HealthAssured: Unmasking Misinformation in Healthcare Domain",
        ["About the System", "Accuracy", "Prediction"],
    ) 

# Define function to display "About the System" section
def display_about_section():
    st.title("About the System")

    st.markdown(
        "In today's world flooded with health info, HealthAssured is your trusted guide. With so much misleading stuff out there, it's hard to know what's true. HealthAssured uses smart tech to check health articles and make sure they're accurate. By giving you reliable info, HealthAssured helps you make better choices about your health. In a time when bad info can be dangerous, HealthAssured is here to keep you safe and healthy."
    )

  
# Define function to display "Accuracy" section
def display_accuracy_section():
    st.title("Accuracy")
    st.subheader("Information on the Accuracy via Bar Graph ")
    data = {"name": ["SVM", "LR", "RF", "LSTM", "RNN", "GCN"], "Accuracy": [82, 80, 79, 86, 82, 75]}
    data = pd.DataFrame(data)
    data = data.set_index("name")
    st.bar_chart(data)

# Define function to display "Prediction" section
def display_prediction_section():
    st.title("HealthAssured: Unmasking Misinformation in Healthcare Domain")
    st.subheader("Enter Text : ")
    news_data = st.text_area("", value="", height=100)
    select_notebook_file = st.selectbox(
        "Select the model versions to run over",
        [       
            "rnn_model",
            "lstm_model",
            "gnn_model"
        ],
    )

    if select_notebook_file == "rnn_model":
        select_model = st.selectbox(
            "Select model",
            ["rnn"],
        )
        if select_model == "rnn":
            rnn_model_tokenizer_path = "rnn_tokenizer.pkl"
            rnn_model_path = "rnn_model/cached_pmi_model.p"

    submit_button = st.button("Check")
    
    if submit_button and select_notebook_file == "rnn_model":
        if select_model == "rnn":
            prediction = predict_classes(news_data)
            #label_arr = ["TRUE", "FALSE"]
            st.success("Checking done", icon="✅")
        # Convert the elements of the NumPy array to strings and capitalize them
            #prediction_str = str(prediction).capitalize()
        # Map prediction to 0 or 1 based on label_arr
            #prediction_index = label_arr.index(prediction_str)
            if prediction == ['FALSE']:
                st.title("We found this news to be False")
            else:
                st.title("We found this news to be True")

            #st.title("We found this news to be " + str(prediction))
        else:
            prediction = predict_classes(news_data)
            st.success("Checking done", icon="✅")
    

        # If prediction is already an integer, use it directly

# Main section to route to different sections based on selection
if selected == "About the System":
    display_about_section()


elif selected == "Accuracy":
    display_accuracy_section()

elif selected == "Prediction":
    display_prediction_section()




