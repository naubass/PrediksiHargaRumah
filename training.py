import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import seaborn as sns

# create data
jumlah_data = 100
np.random.seed(42)

# generate random data
luas = np.random.randint(100, 500, jumlah_data)
kamar = np.random.randint(2, 6, jumlah_data)
kualitas = np.random.randint(1, 10, jumlah_data)

# tambah noise agar model tidak terlalu sempurna
harga = (luas * 2000000) + (kamar * 10000000) + (kualitas * 5000000) + 50000000
noise = np.random.randint(-20000000, 20000000, jumlah_data)
harga_akhir = harga + noise

df = pd.DataFrame({
    'Luas Tanah': luas,
    'Jumlah Kamar': kamar,
    'Kualitas': kualitas,
    'Harga': harga_akhir
})

# split data
X = df[['Luas Tanah', 'Jumlah Kamar', 'Kualitas']]
y = df['Harga']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train model
model = LinearRegression()
model.fit(X_train, y_train)

# evaluate model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Rata-rata Meleset (MAE): ${mae:,.0f}")
print(f"Akurasi Model (R2): {r2:.2f} / 1.00")

# predict
rumah_baru = pd.DataFrame({
    'Luas Tanah': [300],
    'Jumlah Kamar': [3],
    'Kualitas': [8]
})

rumah_baru = pd.get_dummies(rumah_baru, columns=['Kualitas'])
rumah_baru = rumah_baru.reindex(columns=X_train.columns, fill_value=0)

harga_prediksi = model.predict(rumah_baru)[0]
print(f"💰 Harga Taksiran AI: ${harga_prediksi:,.2f}")

# visualisasi
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, color='green') # Warna hijau biar beda
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('Harga Asli (USD)')
plt.ylabel('Prediksi Random Forest (USD)')
plt.title('Random Forest: Real vs Prediction')
plt.show()

print("\n" + "="*30)
print("   BONGKAR RAHASIA MODEL")
print("="*30)

print(f"Intercept (Harga Dasar): Rp {model.intercept_:,.0f}")
print(f"Koefisien Luas Tanah   : Rp {model.coef_[0]:,.0f} (Seharusnya dekat 2 jt)")
print(f"Koefisien Jumlah Kamar : Rp {model.coef_[1]:,.0f} (Seharusnya dekat 10 jt)")
print(f"Koefisien Kualitas     : Rp {model.coef_[2]:,.0f} (Seharusnya dekat 5 jt)")