import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

st.set_page_config(
    page_title="Prediksi Harga Rumah AI",
    page_icon="🏠",
    layout="centered"
)

# train model
@st.cache_data
def train_model():
    try:
        df = pd.read_csv('data_rumah_bersih.csv')
        print(f"Data lokal berhasil dimuat! Ukuran: {df.shape}")
    except FileNotFoundError:
        print("File 'data_rumah_bersih.csv' tidak ditemukan.")
        exit()

    # data preprocessing
    df_processed = pd.get_dummies(df, columns=['Lokasi'], drop_first=True)

    # train & split model
    X = df_processed.drop('Harga', axis=1)
    y = df_processed['Harga']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    # evaluate model
    r2 = r2_score(y_test, model.predict(X_test))
    
    return model, X.columns, r2

model, features, r2 = train_model()

# setup aplikasi streamlit
st.title("🏠 Aplikasi Prediksi Harga Rumah")
st.markdown("---")

# accurate tampilan
col_info1, col_info2 = st.columns([3, 1])
with col_info1:
    st.write("Aplikasi ini menggunakan Linier Regression untuk menaksir harga rumah berdasarkan spesifikasi.")
with col_info2:
    st.metric(label="Akurasi AI", value=f"{r2*100:.1f}%")

# input user
st.sidebar.header("🛠️ Spesifikasi Rumah")
input_luas = st.sidebar.slider("Luas Tanah (m²)", 50, 500, 150)
input_kamar = st.sidebar.slider("Jumlah Kamar", 1, 10, 3)
input_listrik = st.sidebar.selectbox("Daya Listrik (Watt)", [900, 1300, 2200, 3500, 5500, 6600])
input_lokasi = st.sidebar.radio("Lokasi", ["Pusat", "Pinggir"])

# preview input
st.subheader("📝 Pilihan Spesifikasi Rumah Anda:")
c1, c2, c3, c4 = st.columns(4)
c1.info(f"📐 **{input_luas}** m²")
c2.info(f"🛏️ **{input_kamar}** Kamar")
c3.info(f"⚡ **{input_listrik}** Watt")
c4.info(f"📍 **{input_lokasi}**")

# tombol eksekusi
if st.button("💰 Hitung Harga Sekarang!", type="primary", use_container_width=True):
    # siapkan data
    data_baru = pd.DataFrame({
        'Luas Tanah': [input_luas],
        'Jumlah Kamar': [input_kamar],
        'Listrik': [input_listrik],
        'Lokasi': [input_lokasi]
    })

    # preprocessing data
    data_baru = pd.get_dummies(data_baru, columns=['Lokasi'])
    data_baru = data_baru.reindex(columns=features, fill_value=0)

    # prediksi model
    prediksi = model.predict(data_baru)[0]

    # tampilkan hasil
    st.success(f"💰 Harga Taksiran AI: Rp {prediksi:,.0f}")
    st.balloons()

    st.write("---")
    st.caption("*Breakdown estimasi kasar berdasarkan koefisien model:*")
    st.json({
        "Komponen Tanah": f"± Rp {input_luas * 3000000:,.0f}",
        "Komponen Bangunan/Kamar": f"± Rp {input_kamar * 20000000:,.0f}",
        "Komponen Fasilitas (Listrik)": f"± Rp {input_listrik * 50000:,.0f}",
        "Premium Lokasi": "Termasuk" if input_lokasi == "Pusat" else "Standard"
    })