import streamlit as st
import xgboost as xgb
import pandas as pd
import joblib
import numpy as np

# Load model dan kategori yang digunakan pada data pelatihan
model = xgb.Booster(model_file='Classification/fraud_detection.json')
category_categories = list(set(joblib.load('joblib/category.joblib')))  # Pastikan unik
city_categories = list(set(joblib.load('joblib/city.joblib')))          # Pastikan unik
job_categories = list(set(joblib.load('joblib/job.joblib')))            # Pastikan unik
X_train = joblib.load('Classification/X_train.pkl')

# Fungsi untuk mengambil input dari pengguna
def get_input():
    st.subheader("Input Data Transaksi")
    amt = st.number_input(
        'Masukkan Jumlah Transaksi (USD)',
        min_value=0.0,
        value=0.0,
        step=0.01,
        help="Jumlah total transaksi dalam mata uang USD."
    )
    job = st.selectbox(
        'Pilih Jenis Pekerjaan',
        sorted(job_categories),
        help="Pilih pekerjaan pelanggan."
    )
    category = st.selectbox(
        'Pilih Kategori Transaksi',
        sorted(category_categories),
        help="Kategori transaksi, seperti elektronik, makanan, atau lainnya."
    )
    city = st.selectbox(
        'Pilih Kota',
        sorted(city_categories),
        help="Kota tempat transaksi dilakukan."
    )
    st.markdown("---")
    return pd.DataFrame(
        {
            'category': [category],
            'amt': [amt],
            'city': [city],
            'job': [job],
        }
    )

# Fungsi untuk memproses input agar sesuai dengan fitur model
def preprocess_input(dataframe):
    _ = dataframe.copy()
    
    # Membuat DataFrame akhir dengan fitur yang sesuai dengan model
    df_dmatrix = pd.DataFrame()
    df_dmatrix['amt'] = _['amt']
    df_dmatrix['job'] = _['job'].astype('category').cat.set_categories(X_train['job'].cat.categories)
    df_dmatrix['category'] = _['category'].astype('category').cat.set_categories(X_train['category'].cat.categories)
    df_dmatrix['city'] = _['city'].astype('category').cat.set_categories(X_train['city'].cat.categories)
    
    return df_dmatrix

# Title dan deskripsi aplikasi
st.title("Fraud Detection Prediction App")
st.markdown("""
### Aplikasi Prediksi Deteksi Penipuan
Aplikasi ini dirancang untuk membantu mendeteksi kemungkinan transaksi mencurigakan
berdasarkan data yang dimasukkan pengguna.
""")
st.markdown("---")

# Mengambil input dari pengguna
input_data = get_input()

# Tombol prediksi
if st.button("Prediksi Transaksi"):
    # Proses input data
    preprocessed_data = preprocess_input(input_data)
    
    # Prediksi menggunakan model
    prediction = model.predict(xgb.DMatrix(preprocessed_data, enable_categorical=True))
    
    # Menampilkan hasil prediksi
    st.markdown("### Hasil Prediksi")
    if prediction[0] > 0.5:
        st.error(f"Transaksi terdeteksi mencurigakan dengan probabilitas: {prediction[0]:.2f}")
    else:
        st.success(f"Transaksi terdeteksi aman dengan probabilitas: {1 - prediction[0]:.2f}")
else:
    st.info("Masukkan data transaksi dan tekan tombol 'Prediksi Transaksi'.")
