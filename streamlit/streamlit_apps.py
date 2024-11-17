import streamlit as st
import xgboost as xgb
import pandas as pd
import joblib
import numpy as np

# # Load model dan kategori yang digunakan pada data pelatihan
# model = xgb.Booster(model_file='fraud_detection.json')
# category_categories = list(set(joblib.load('category.joblib')))  # Pastikan unik
# city_categories = list(set(joblib.load('city.joblib')))          # Pastikan unik
# job_categories = list(set(joblib.load('job.joblib')))            # Pastikan unik
# X_train = joblib.load('X_train.pkl')

import os
st.write("Current Working Directory:", os.getcwd())
st.write("Available Files:", os.listdir('.'))

def get_input():
    amt = st.number_input('Jumlah Transaksi', min_value=0.0, value=0.0)
    job = st.selectbox('Job', sorted(job_categories))
    category = st.selectbox('Kategori', sorted(category_categories))
    city = st.selectbox('City', sorted(city_categories))
    
    return pd.DataFrame(
        {
            'category': [category],
            'amt': [amt],
            'city': [city],
            'job': [job],
        }
    )

def preprocess_input(dataframe):
    _ = dataframe.copy()
    
    # Membuat DataFrame akhir dengan fitur yang sesuai dengan model
    df_dmatrix = pd.DataFrame()
    # df_dmatrix['trans_day_of_year'] = _.date.dt.dayofyear
    df_dmatrix['amt'] = _['amt']
    # df_dmatrix['trans_day_of_month'] = _.date.dt.day.astype('category').cat.set_categories(X_train['trans_day_of_month'].cat.categories)
    # df_dmatrix['trans_hour'] = _.hour.dt.hour.astype('category').cat.set_categories(X_train['trans_hour'].cat.categories)
    # df_dmatrix['trans_month'] = _.date.dt.month.astype('category').cat.set_categories(X_train['trans_month'].cat.categories)
    df_dmatrix['job'] = _['job'].astype('category').cat.set_categories(X_train['job'].cat.categories)
    df_dmatrix['category'] = _['category'].astype('category').cat.set_categories(X_train['category'].cat.categories)
    df_dmatrix['city'] = _['city'].astype('category').cat.set_categories(X_train['city'].cat.categories)
    
    return df_dmatrix

st.title('Prediction App')
input_data = get_input()
preprocessed_data = preprocess_input(input_data)

prediction = model.predict(xgb.DMatrix(preprocessed_data, enable_categorical=True))

st.write('Prediction:', prediction)
