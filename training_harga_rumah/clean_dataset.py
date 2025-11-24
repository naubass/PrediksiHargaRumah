import pandas as pd
import numpy as np

# load data
try:
    df = pd.read_csv('data_rumah_kotor.csv')
    print(f"Data lokal berhasil dimuat! Ukuran: {df.shape}")
except FileNotFoundError:
    print("File 'data_rumah_kotor.csv' tidak ditemukan.")
    exit()

# cek data kosong
print(df.isnull().sum())

# isi median data kosong
median_luas = df['Luas Tanah'].median()
df['Luas Tanah'] = df['Luas Tanah'].fillna(median_luas)

# membersihkan kolom listrik
df['Listrik'] = df['Listrik'].str.replace('Watt', '').astype(int)

# rapikan lokasi (pusat/PUSAT -> Pusat)
df['Lokasi'] = df['Lokasi'].str.title()

# data yang telah dibersihkan
print("\nData Setelah Bersih:\n", df.head())
print("Lokasi Unik:", df['Lokasi'].unique())

# cetak data bersih ke csv
df.to_csv('data_rumah_bersih.csv', index=False)
print("Data telah disimpan ke 'data_rumah_bersih.csv'")

