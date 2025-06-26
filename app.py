import streamlit as st
import numpy as np
import joblib

# Load model dan scaler
model = joblib.load("model_raisin_nb.pkl")
scaler = joblib.load("scaler_raisin.pkl")

# Judul
st.title("Klasifikasi Jenis Kismis 🍇")
st.markdown("Model prediksi jenis kismis: **Besni** atau **Kecimen**, berdasarkan fitur morfologis buah kismis.")

# Keterangan input
st.sidebar.header("Input Fitur")
area = st.sidebar.number_input("Area", min_value=500.0, max_value=3000.0)
perimeter = st.sidebar.number_input("Perimeter", min_value=50.0, max_value=300.0)
major_axis = st.sidebar.number_input("MajorAxisLength", min_value=10.0, max_value=200.0)
minor_axis = st.sidebar.number_input("MinorAxisLength", min_value=5.0, max_value=150.0)
eccentricity = st.sidebar.number_input("Eccentricity", min_value=0.0, max_value=1.0)
convex_area = st.sidebar.number_input("ConvexArea", min_value=500.0, max_value=3000.0)
extent = st.sidebar.number_input("Extent", min_value=0.0, max_value=1.0)

# Tombol prediksi
if st.sidebar.button("Prediksi"):
    fitur = np.array([[area, perimeter, major_axis, minor_axis, eccentricity, convex_area, extent]])
    fitur_scaled = scaler.transform(fitur)
    prediksi = model.predict(fitur_scaled)[0]

    st.subheader("Hasil Prediksi")
    st.success(f"Jenis Kismis: **{prediksi}**")
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/2/22/Raisins_01.jpg/800px-Raisins_01.jpg", width=300)

# Info model
st.markdown("---")
st.markdown("**Model:** Gaussian Naive Bayes  \n**Dataset:** Raisin (UCI Machine Learning Repository)  \n**Akurasi:** > 90%")
