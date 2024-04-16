import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
# from DataModel import AirModel
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle 
import os

df = pd.read_csv(r".\Airline Staisfaction Data\train.csv")
sc = ['Flight Distance', 'Inflight wifi service', 'Online boarding', 'Seat comfort', 'Inflight entertainment',
      'On-board service', 'Leg room service', 'Baggage handling', 'Checkin service', 'Inflight service',
      'Cleanliness', 'Type of Travel_enc', 'Class_enc']

def wrangle(df):

        df.fillna(0, inplace=True)
        
        satisfaction_columns = df.columns[6:-4]
        df['overall_rating'] = round(df[satisfaction_columns].mean(axis=1), 2)

        # Instantiate LabelEncoder
        enc = LabelEncoder()
        # Encoding the categorical features in the training and test data
        for column in df.select_dtypes("O").columns:
            df[f'{column}_enc'] = enc.fit_transform(df[column])
            df.drop(column, axis=1, inplace=True)

        # Check that all columns are numeric
        if df.select_dtypes("O").columns.any():
            print("Non-numeric columns found:", df.select_dtypes("O").columns)
            return None  # or handle this case as you find suitable
        
        # Inistantiating the scaler object
        scaler = StandardScaler()
        # Fitting the scaler to the data and transforming it.
        df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
        
        return df



# Construct the absolute path to your file
base_dir = os.path.dirname(__file__)  # Gets the directory where the script is located
model_path = os.path.join(base_dir, "Airline Satisfaction Data", "model.pth")

with open(r"C:\Users\khale\Documents\VS code\Projects\Airplane Customer Satisfaction\Airline Staisfaction Data\model.pth", "rb") as file:
    model = pickle.load(file)

st.title("Airplane Satisfaction survey")


row = {}

with st.form("Survey Qs"):
    st.subheader("Survey Data")
    for col in df.columns[1:-1]:
        row[f"{col}_value"] = st.text_input(f"Enter {col}:", "")

    submitted = st.form_submit_button("Submit")
    if submitted:
        cus_df = pd.DataFrame(row, columns=row.keys(), index=[0])
        cus_df = wrangle(cus_df)
        pred = model.predict(cus_df[sc])
        st.dataframe(cus_df)
        if pred == 1:
          cus_sat = "Satisfied"
        else:
          cus_sat = "Unsatisfied"
        print(f"This customer is: {cus_sat}")
        

        

