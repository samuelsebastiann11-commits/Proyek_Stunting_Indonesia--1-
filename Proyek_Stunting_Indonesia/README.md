# Proyek Streamlit — Monitoring dan Estimasi Tren Data Stunting Indonesia

## Tema yang dipilih
**Aplikasi Prediksi ML.** Aplikasi menampilkan data historis per provinsi dan estimasi tren menggunakan Linear Regression.

## Susunan folder

```text
Proyek_Stunting_Indonesia/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── Data Stunting Indonesia.csv
└── models/
    └── model_stunting.joblib  (dibuat otomatis setelah training)
```

## Isi dataset
File yang diberikan memiliki kolom `Provinsi`, `2020`, `2021`, `2022`, dan `2023`, berisi 38 baris provinsi. Format CSV menggunakan pemisah titik koma dan angka desimal dengan koma. Program sudah menyesuaikan format tersebut serta membuang kolom kosong tambahan.

## Variabel dan metode
- Fitur: `Tahun`.
- Target: `Nilai` pada dataset untuk provinsi yang dipilih.
- Metode: Linear Regression, satu model untuk setiap provinsi.
- Evaluasi: model dilatih dengan tahun sebelum tahun terakhir, lalu diuji pada tahun terakhir menggunakan MAE dan R².
- Model akhir dilatih ulang menggunakan seluruh tahun yang tersedia.

Prediksi adalah estimasi sederhana dari tren empat tahun, bukan angka resmi. Jangan gunakan hasil sebagai diagnosis atau dasar keputusan kebijakan.

## Cara menjalankan di Windows

Buka folder proyek dan klik dua kali `run_app.bat`. Pada penggunaan pertama,
skrip akan membuat virtual environment, memasang dependensi, dan melatih
model jika file model belum tersedia.

Alternatifnya, jalankan perintah berikut di PowerShell pada folder proyek:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe train_model.py
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Setelah perintah terakhir, buka alamat lokal yang ditampilkan Streamlit, biasanya `http://localhost:8501`.

## Fitur aplikasi
1. Ringkasan jumlah provinsi, tahun terakhir, dan rata-rata nilai tahun terakhir.
2. Form pilihan provinsi dan tahun estimasi.
3. Hasil prediksi nilai.
4. Grafik tren historis dan estimasi untuk provinsi terpilih.
5. Grafik 10 provinsi dengan nilai tertinggi pada tahun terakhir.
6. Tabel data dan informasi evaluasi model.

## Sumber dataset
https://raw.githubusercontent.com/samuelsebastiann11-commits/Sistem-Cerdas/refs/heads/master/Data%20Stunting%20Indonesia.csv

Cantumkan sumber asli dan lisensi dataset pada laporan sesuai keterangan pemilik data.

## Contoh pembagian tugas kelompok (3–4 orang)
- Anggota 1: mencari dan memeriksa sumber dataset serta melakukan preprocessing.
- Anggota 2: membuat dan mengevaluasi model Linear Regression pada `train_model.py`.
- Anggota 3: membuat antarmuka Streamlit dan visualisasi pada `app.py`.
- Anggota 4 (jika ada): menguji aplikasi, merapikan README, menyiapkan dokumentasi, dan presentasi.

Jika kelompok terdiri dari tiga orang, tugas dokumentasi dan pengujian dapat dibagi kepada anggota 1–3.
