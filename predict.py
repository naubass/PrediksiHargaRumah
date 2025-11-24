import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor  
from sklearn.metrics import mean_absolute_error, r2_score

# load dataset
try:
    df = pd.read_csv('train.csv')
    print(f"Data lokal berhasil dimuat! Ukuran: {df.shape}")
except FileNotFoundError:
    print("File 'train.csv' tidak ditemukan.")
    exit()

# choose feature house
features = ['LotArea', 'BedroomAbvGr', 'Neighborhood', 'OverallQual']
target = 'SalePrice'

data = df[features + [target]].copy()

# clean data
data = data.dropna()
data = pd.get_dummies(data, columns=['Neighborhood'], drop_first=True)

# split data
X = data.drop(target, axis=1)
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Sedang melatih Random Forest (Mungkin butuh beberapa detik)...")

model = RandomForestRegressor(n_estimators=100, random_state=42) 
model.fit(X_train, y_train)

# evaluasi
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n" + "="*40)
print("   HASIL RANDOM FOREST (SUPERIOR MODEL)")
print("="*40)
print(f"Rata-rata Meleset (MAE): ${mae:,.0f}")
print(f"Akurasi Model (R2): {r2:.2f} / 1.00")
print("="*40)

# prediksi rumah
print("\n🔍 Simulasi Prediksi Manual:")
rumah_baru = pd.DataFrame({
    'LotArea': [15000],          
    'BedroomAbvGr': [3],         
    'Neighborhood': ['CollgCr'], 
    'OverallQual': [7]           
})

# preprocessing data baru
rumah_baru = pd.get_dummies(rumah_baru, columns=['Neighborhood'])
rumah_baru = rumah_baru.reindex(columns=X_train.columns, fill_value=0)

harga_prediksi = model.predict(rumah_baru)[0]
print(f"Spesifikasi: Luas 15000, 3 Kamar, Lokasi CollgCr, Kualitas 7")
print(f"Harga Taksiran AI: ${harga_prediksi:,.2f}")

# visualisasi hasil
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, color='green') # Warna hijau biar beda
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('Harga Asli (USD)')
plt.ylabel('Prediksi Random Forest (USD)')
plt.title('Random Forest: Real vs Prediction')
plt.show()

print("\n" + "="*30)
print("   CEK PREDIKSI RUMAH IMPIAN")
print("="*30)

# masukkan spesifikasi rumah baru
rumah_baru = pd.DataFrame({
    'LotArea': [15000],         
    'BedroomAbvGr': [3],         
    'Neighborhood': ['CollgCr'], 
    'OverallQual': [7]           
})

print("Spesifikasi Rumah:")
print(rumah_baru)

rumah_baru = pd.get_dummies(rumah_baru, columns=['Neighborhood'])
rumah_baru = rumah_baru.reindex(columns=X_train.columns, fill_value=0)

# harga prediksi
harga_prediksi = model.predict(rumah_baru)[0]

print(f"\nHarga Taksiran AI: ${harga_prediksi:,.2f}")
print("="*30)

print("\n" + "="*40)
print("   ANALISIS: APA YANG PALING PENTING?")
print("="*40)

# mengambil tingkat kepentingan dari model
importances = model.feature_importances_
nama_fitur = X_train.columns

# create dataframe fitur dan tingkat kepentingan
df_imp = pd.DataFrame({'Fitur': nama_fitur, 'Penting': importances})

# urutkan dari yang paling penting
df_imp = df_imp.sort_values(by='Penting', ascending=False).head(10) # Ambil top 10

print("Top 5 Faktor Penentu Harga Rumah:")
print(df_imp.head(5))

# visualisasi bar chart
plt.figure(figsize=(10, 6))
sns.barplot(x='Penting', y='Fitur', data=df_imp, palette='viridis')
plt.title('Apa yang Paling Dilihat AI Saat Menentukan Harga?')
plt.xlabel('Tingkat Kepentingan (0-1)')
plt.ylabel('Fitur')
plt.show()