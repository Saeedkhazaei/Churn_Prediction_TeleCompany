import streamlit as st
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer , make_column_transformer
import numpy as np
import pandas as pd
import joblib
from PIL import Image
import time

model = joblib.load('LGBMpipe.joblib')
col1,col2 = st.columns([2,1])
col1.header('*Whether the Costumer Existed or not in the Company*') 
col1.write("Learn More >> [link](https://github.com/Saeedkhazaei)")
photo = Image.open("Churn.jpg")
photo.resize((800,600))
col2.image(photo)
st.write("---")

#gender = st.radio(" Select the Gender of Customer",['Male','Female'])
SeniorCitizen = st.radio(" Select the Status of Citizenship",['Yes','No']) 
Partner = st.radio("Does the customer Have Partner ?", ['Yes','No'])
Dependents = st.radio("Is the costumer Dependent ?", ['Yes','No'])
tenure = st.slider("the length of time the person has been a costumer ? ",1,75) 
PhoneService = st.radio ("Does the cosumer have phone service?", ['Yes','No'])
MultipleLines = st.radio("Does the cosumer have multiple lines",['Yes','No'])
InternetService = st.selectbox("Does the cosumer have internet service?",['DSL','Fiber Optic', 'No'])
OnlineSecurity = st.radio("Does the cosumer have Online Security?", ['Yes','No'])
OnlineBackup = st.radio("Does the cosumer have Online Backup", ['Yes','No'])
DeviceProtection = st.radio("Does the cosumer have Device Protection?", ['Yes','No'])
TechSupport = st.radio("Does the cosumer have Tech Support?", ['Yes','No'])
StreamingTV = st.radio("Does the cosumer have Streaming TV?", ['Yes','No'])
StreamingMovies = st.radio("Does the cosumer have Streaming Movies?", ['Yes','No'])
Contract = st.selectbox("Which Contract does the Cosumer use?", ['Month-to-month','One year','Two year'])
PaperlessBilling = st. radio("Does the cosumer Choose Paperless Billing?", ['Yes','No'])
PaymentMethod = st. selectbox("Which pement method does the Cosumer use?", ['Electronic check','Mailed check','Bank transfer (automatic)'
 'Credit card (automatic)'])
MonthlyCharges = st.slider ("How much does the cosumer pay monthly ($)?",10,120 )
TotalCharges = st.slider("How much does the Costumer Pay Totally? ($)", 18,8700)
 


def predict(): 
    columns=[ 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
       'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
       'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
       'PaperlessBilling', 'MonthlyCharges', 'TotalCharges', 'InternetService_DSL',
        'InternetService_Fiber optic',
       'InternetService_No', 'Contract_Month-to-month', 'Contract_One year',
       'Contract_Two year', 'PaymentMethod_Bank transfer (automatic)',
       'PaymentMethod_Credit card (automatic)',
       'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check']
    row = np.zeros(len(columns))
    columns1 = [ 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
       'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
       'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
       'MonthlyCharges', 'TotalCharges']
    row1=[ SeniorCitizen, Partner, Dependents, tenure,
       PhoneService, MultipleLines, InternetService, OnlineSecurity,
       OnlineBackup, DeviceProtection, TechSupport, StreamingTV,
       StreamingMovies, Contract, PaperlessBilling, PaymentMethod,
       MonthlyCharges, TotalCharges]
    X = pd.DataFrame([row], columns = columns)
    X1=pd.DataFrame([row1], columns = columns1)

    yes_no_columns= ["PaperlessBilling","StreamingMovies", "StreamingTV","TechSupport","DeviceProtection","OnlineBackup",
                 "OnlineSecurity", "MultipleLines","PhoneService","PhoneService", "Dependents", "Partner"]
    for col in yes_no_columns:
        X1[col].replace({"Yes":1 , "No":0},inplace=True)
    
    X1['tenure'] = pd.to_numeric(X1['tenure'],errors='coerce')
    X1['MonthlyCharges'] = pd.to_numeric(X1['MonthlyCharges'],errors='coerce')
    X1['TotalCharges'] = pd.to_numeric(X1['TotalCharges'],errors='coerce')
    cols_to_scale=["tenure","TotalCharges","MonthlyCharges"]
    scaler=MinMaxScaler()
    X1[cols_to_scale]=scaler.fit_transform(X1[cols_to_scale])
    #X1["gender"].replace({"Male":1 , "Female":0},inplace=True)
    X1=pd.get_dummies(X1, columns=["InternetService","Contract","PaymentMethod"])
    for col in X1[col]:
        if col in X.columns:
            X[col]=X1[col]


    prediction = model.predict(X)
    progress_bar = col1.progress(0)
    for perdiction_complited in range(100):
        time.sleep(0.05)
        progress_bar.progress(perdiction_complited+1)

    if prediction[0] == 1: 
        st.error('The Costumer will be Exited :thumbsdown:')
    else: 
        st.success('The Costumer will Stay :thumbsup:') 

trigger = st.button('Predict', on_click=predict) 
st.write("---")

    
