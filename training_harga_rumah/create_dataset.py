import pandas as pd
import numpy as np

jumlah_data = 200
np.random.seed(42)

luas = np.random.randint(50, 200, jumlah_data).astype(float)
# buat data kosong
index_kosong_luas = np.random.choice(range(jumlah_data), 10,replace=False)
luas[index_kosong_luas] = np.nan

kamar = np.random.randint(2, 6, jumlah_data)
watt_pilihan = [900, 1300, 2200, 3500]
listrik_raw = np.random.choice(watt_pilihan, jumlah_data)
listrik = [f"{watt} Watt" for watt in listrik_raw]

lokasi_pilihan = ['Pusat', 'Pinggir']
lokasi_raw = np.random.choice(lokasi_pilihan, jumlah_data)

lokasi = []
for l in lokasi_raw:
    acak = np.random.rand()
    if acak < 0.1: lokasi.append(l.lower())
    elif acak < 0.2: lokasi.append(l.upper())
    else: lokasi.append(l)

harga = (pd.Series(luas).fillna(100) * 3000000) + \
        (kamar * 20000000) + \
        (listrik_raw * 50000) + \
        200000000

noise = np.random.randint(-20000000, 20000000, jumlah_data)
harga_akhir = harga + noise

# simpan ke CSV
df_kotor = pd.DataFrame({
    'Luas Tanah': luas,
    'Jumlah Kamar': kamar,
    'Listrik': listrik,
    'Lokasi': lokasi,
    'Harga': harga_akhir
})

df_kotor.to_csv('data_rumah_kotor.csv', index=False)
print("✅ File 'data_rumah_kotor.csv' berhasil dibuat!")
print("Tugasmu: Bersihkan data ini sebelum masuk ke Model AI!")
