import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import seaborn as sns

# import warning
import warnings
warnings.filterwarnings('ignore')

# load dataset
try:
    df = pd.read_csv('data_rumah_bersih.csv')
    print(f"Data lokal berhasil dimuat! Ukuran: {df.shape}")
except FileNotFoundError:
    print("File 'data_rumah_bersih.csv' tidak ditemukan.")
    exit()

# choose feature house
print("\nMemproses Data Lokasi...\n")
df = pd.get_dummies(df, columns=['Lokasi'], drop_first=True)
print("Kolom yang tersedia:", df.columns.tolist())

# split data
X = df.drop('Harga', axis=1)
y = df['Harga']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train model
model = LinearRegression()
model.fit(X_train, y_train)

# evaluation
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n" + "="*30)
print("   HASIL EVALUASI")
print("="*30)
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Akurasi (R2 Score)     : {r2:.2f} / 1.00")
print("\n" + "="*30)

# plot visualisasi
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('Harga Asli (Rupiah)')
plt.ylabel('Prediksi Model (Rupiah)')
plt.title('Linear Regression: Harga Asli vs Prediksi')
plt.show()

# cek prediksi rumah impian
print("\n" + "="*30)
print("   CEK PREDIKSI RUMAH IMPIAN")
print("="*30)

# masukkan spesifikasi rumah baru
rumah_baru = pd.DataFrame({
    'Luas Tanah': [150],          
    'Jumlah Kamar': [3],         
    'Listrik': [2200], 
    'Lokasi': ['Pusat']          
})

# preprocessing data baru
rumah_baru = pd.get_dummies(rumah_baru, columns=['Lokasi'])
rumah_baru = rumah_baru.reindex(columns=X_train.columns, fill_value=0)

harga_prediksi = model.predict(rumah_baru)[0]
print(f"💰 Harga Taksiran AI: Rp {harga_prediksi:,.0f}")

# top 5 faktor penentu harga rumah
print("\n" + "="*40)
print("   ANALISIS: KONTRIBUSI RATA-RATA KE HARGA (RUPIAH)")
print("="*40)

# ambil koefisien
koefisien = model.coef_

# hitung rata-rata nilai fitur
rata2_nilai_fitur = X_train.mean(axis=0)

# hitung kontribusi
kontribusi_rupiah = np.abs(koefisien * rata2_nilai_fitur)
nama_fitur = X_train.columns

# buat DataFrame baru untuk plotting
df_imp_fixed = pd.DataFrame({
    'Fitur': nama_fitur,
    'Kontribusi_Rp': kontribusi_rupiah
})

# urutkan dari kontribusi terbesar
df_imp_fixed = df_imp_fixed.sort_values(by='Kontribusi_Rp', ascending=False)

print("Top Faktor Berdasarkan Rata-rata Sumbangan Rupiah:")
pd.options.display.float_format = 'Rp {:,.0f}'.format
print(df_imp_fixed)

# plot bar chart
plt.figure(figsize=(10, 6))
sns.barplot(x='Kontribusi_Rp', y='Fitur', data=df_imp_fixed, palette='magma')

# ubah format sumbu X agar tidak pakai notasi ilmiah (e+07) tapi pakai Juta/Miliar
def format_rupiah(x, pos):
    if x >= 1e9: return f'{x*1e-9:.0f} Miliar'
    elif x >= 1e6: return f'{x*1e-6:.0f} Juta'
    else: return f'{x:.0f}'
plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(format_rupiah))

plt.title('Rata-rata Sumbangan Fitur terhadap Total Harga Rumah')
plt.xlabel('Kontribusi Rata-rata (Rupiah)')
plt.ylabel('Fitur')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()
