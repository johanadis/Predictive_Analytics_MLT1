# Laporan Proyek Machine Learning - Johanadi Santoso 

### Analisis Prediktif Harga Emas Menggunakan Model LSTM dan GRU

> **Nama:** Johanadi Santoso  
  **Email:** johanadisantoso1@gmail.com  
  **ID Dicoding:** johanadisantoso  

---
## 📂 **Struktur Direktori:**  
 
```
📂 Predictive_Analytics_MLT1                  # Folder utama proyek prediksi harga emas
│
│   prediksi_harga_emas_lstm_5th.ipynb           # Notebook LSTM untuk eksplorasi, training, dan evaluasi model
│   prediksi_harga_emas_gru_5th.ipynb            # Notebook GRU untuk eksplorasi, training, dan evaluasi model
│   gold_price_dataset_5th.csv                   # Dataset harga emas selama 5 tahun 
│   README.md                                    # Dokumentasi proyek, cara penggunaan, dan penjelasan hasil
│   prediksi_harga_emas_lstm_5th.py              # Versi script `.py` untuk model LSTM (bisa untuk deployment)
│   prediksi_harga_emas_gru_5th.py               # Versi script `.py` untuk model GRU (bisa untuk deployment)
│
├── models/                                   # Folder penyimpanan model dan scaler hasil training
│   │
│   ├── best_model_lstm_5th.h5                   # Model LSTM terbaik dalam format .h5
│   ├── scaler_close_lstm_5th.pkl                # Scaler untuk fitur `Close` model LSTM (pickle)
│   ├── best_model_gru_5th.h5                    # Model GRU terbaik dalam format .h5
│   └── scaler_close_gru_5th.pkl                 # Scaler untuk fitur `Close` model GRU (pickle)
│
└── images/                                   # Folder visualisasi hasil analisis dan prediksi
    │
    ├── distribusi.png                           # Visualisasi distribusi data harga emas
    ├── grafik_harga.png                         # Grafik tren harga emas
    ├── prediksi_gru.png                         # Plot hasil prediksi GRU vs data aktual
    └── prediksi_lstm.png                        # Plot hasil prediksi LSTM vs data aktual
```
---

## 1. Domain Proyek

Harga emas merupakan salah satu indikator ekonomi global yang signifikan dan sering dianggap sebagai aset *safe-haven* oleh investor di tengah ketidakpastian ekonomi. Fluktuasi harga emas dipengaruhi oleh berbagai faktor kompleks, termasuk tingkat inflasi, pergerakan nilai tukar mata uang (khususnya dolar AS), ketidakpastian geopolitik, serta dinamika penawaran dan permintaan di pasar global. Dalam konteks ini, kemampuan untuk memprediksi harga emas secara akurat dapat memberikan keunggulan kompetitif bagi investor, pelaku pasar keuangan, dan institusi yang mengelola portofolio investasi.

Proyek ini bertujuan untuk mengembangkan model prediktif berbasis *deep learning* guna memperkirakan harga penutupan emas harian menggunakan data historis. Dua pendekatan utama yang digunakan adalah **LSTM (*Long Short-Term Memory*)** dan **GRU (*Gated Recurrent Unit*)**, yang keduanya merupakan varian dari *Recurrent Neural Networks* (RNN) yang dirancang khusus untuk menangani data *time series*. Dengan memanfaatkan pola temporal dalam data harga emas, proyek ini berupaya memberikan wawasan yang dapat ditindaklanjuti untuk pengambilan keputusan investasi.

**Referensi Utama:**
- Yahoo Finance. *Historical Gold Prices*. Diakses dari: [Yahoo Finance](https://finance.yahoo.com/quote/GC%3DF/history/)
- Understanding [LSTM](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) Networks
- Understanding [Gated Recurrent Unit (GRU)](https://medium.com/@anishnama20/understanding-gated-recurrent-unit-gru-in-deep-learning-2e54923f3e2) in Deep Learning
---

## 2. Business Understanding

### 2.1 Problem Statements

Proyek ini berfokus pada dua permasalahan utama:
1. **Pernyataan Masalah 1**: Bagaimana cara memanfaatkan data historis harga emas untuk memprediksi harga penutupan di masa depan dengan tingkat akurasi yang tinggi?
2. **Pernyataan Masalah 2**: Di antara dua model *deep learning*, yaitu LSTM dan GRU, manakah yang memberikan performa lebih baik dalam memprediksi harga emas berdasarkan metrik evaluasi standar?

### 2.2 Goals 🎯

Tujuan proyek ini dirancang untuk menjawab permasalahan di atas:
1. **Goal 1**: Membangun model prediktif berbasis *deep learning* yang mampu memberikan estimasi akurat untuk harga penutupan emas harian.
2. **Goal 2**: Melakukan perbandingan mendalam antara model LSTM dan GRU untuk menentukan pendekatan terbaik dalam konteks prediksi *time series* harga emas.

### 2.3 Solution Statements

Solusi yang diusulkan untuk mencapai tujuan tersebut meliputi:
- **Solution 1**: Mengembangkan dan mengoptimalkan model LSTM dan GRU melalui *hyperparameter tuning* untuk memastikan prediksi yang akurat.
- **Solution 2**: Mengevaluasi performa kedua model menggunakan metrik kuantitatif seperti *Root Mean Squared Error* (RMSE), *Mean Absolute Error* (MAE), *Mean Absolute Percentage Error* (MAPE), dan koefisien determinasi (R²) untuk menentukan model yang lebih unggul.

---

## 3. Data Understanding

Data diperoleh dari hasil scraping data [Yahoo Finance](https://finance.yahoo.com/quote/GC=F/) menggunakan simbol *ticker* `GC=F`, yang mencerminkan harga kontrak berjangka emas di pasar internasional. Dataset mencakup 1.258 entri dari tanggal 2 Januari 2020 hingga 31 Desember 2024. Dataset ini berisi informasi harga pembukaan (*Open*), harga tertinggi (*High*), harga terendah (*Low*), harga penutupan (*Close*), dan volume perdagangan (*Volume*).

### 3.1 Sumber Data

- **Tautan Sumber Data**: [Yahoo Finance - Gold Futures](https://finance.yahoo.com/quote/GC%3DF/history/)
- **Nama tiker**: `GC=F`
- **Jumlah Data**: 1258 baris, sesuai dengan jumlah hari perdagangan dalam periode 2020-2025 (dengan asumsi 251 hari perdagangan per tahun).

### 3.2 Kondisi Data Awal
- **Nilai yang Hilang**: Tidak ada nilai yang hilang (*missing values*) dalam dataset setelah pemeriksaan awal.
- **Duplikasi**: Tidak ditemukan baris duplikat berdasarkan kolom *Date*.
- **Outlier**: Analisis awal menunjukkan tidak adanya *outlier* signifikan yang dapat mengganggu model, meskipun fluktuasi tajam terjadi pada beberapa periode.

### 3.3 Deskripsi Fitur

Dataset terdiri dari beberapa kolom yang memberikan gambaran lengkap tentang pergerakan harga emas harian:

| **Fitur** | **Tipe Data** | **Deskripsi** |
|-----------|---------------|----------------|
| Date (index) | DatetimeIndex | Tanggal perdagangan (format: YYYY-MM-DD) |
| Close     | float64       | Harga penutupan emas (target prediksi) |
| High      | float64       | Harga tertinggi emas dalam satu hari |
| Low       | float64       | Harga terendah emas dalam satu hari |
| Open      | float64       | Harga pembukaan emas pada hari tersebut |
| Volume    | object        | Volume perdagangan emas pada hari itu (perlu konversi ke numerik) |


### 3.4 Exploratory Data Analysis (EDA)

Analisis eksplorasi data dilakukan untuk memahami karakteristik dataset sebelum pemodelan.

#### a. Distribusi Harga Penutupan
![Distribusi Harga Penutupan](./images/distribusi.png)

💡 **Insight**:
- Distribusi harga penutupan emas menunjukkan pola yang mendekati distribusi normal dengan sedikit kemiringan positif (*positive skewness*).
- Rentang harga berkisar antara $1,600 hingga $2,500 per troy ounce, mencerminkan volatilitas pasar selama periode tersebut.

#### b. Tren Harga Emas (2020-2025)
![Tren Harga Emas](./images/grafik_harga.png)

💡 **Insight**:
- Terdapat tren kenaikan harga emas secara keseluruhan dari 2020 hingga 2025.
- Fluktuasi signifikan terdeteksi pada tahun 2022 dan 2024, kemungkinan terkait dengan peristiwa ekonomi global seperti kenaikan suku bunga atau konflik geopolitik.
---

## 4. Data Preparation

Persiapan data merupakan langkah kritis untuk memastikan model dapat belajar dengan baik dari dataset.

### 4.1 Pembersihan Data
- Meskipun tidak ada nilai yang hilang atau duplikat, data diperiksa ulang untuk memastikan konsistensi format tanggal dan nilai numerik.

### 4.2 Normalisasi Data
- Data harga penutupan dinormalisasi menggunakan `MinMaxScaler` ke dalam rentang [0,1] untuk mempercepat konvergensi model *deep learning*.
- Rumus normalisasi:  
  $x' = \frac{x - x_{min}}{x_{max} - x_{min}}$

- Normalisasi diterapkan pada fitur *Close*

### 4.3 Pembentukan Data *Time Series*
- Data diubah menjadi format *sequence* menggunakan pendekatan *sliding window*.
- Ukuran jendela (*window size*) ditetapkan pada **60 hari**, yang berarti model akan menggunakan 60 hari sebelumnya untuk memprediksi harga penutupan hari berikutnya.
- Contoh:  
  - Input: Harga penutupan hari ke-1 hingga ke-60  
  - Output: Harga penutupan hari ke-61

### 4.4 Pembagian Dataset
- Dataset dibagi menjadi tiga subset untuk pelatihan dan evaluasi model:
  - **Training**: 70% (sekitar 880 baris)
  - **Validation**: 15% (sekitar 189 baris)
  - **Testing**: 15% (sekitar 189 baris)
- Pembagian dilakukan secara berurutan (tanpa pengacakan) untuk menjaga sifat temporal data.

---

## 5. Modeling

Dua model *deep learning* dikembangkan: **LSTM** dan **GRU**, dengan optimasi *hyperparameter* menggunakan pustaka `Hyperopt`.

### 5.1.  Model 1: LSTM (*Long Short-Term Memory*)

#### ➡️ Pembahasan Cara Kerja
LSTM adalah jenis *Recurrent Neural Network* (RNN) yang dirancang untuk menangkap ketergantungan jangka panjang dalam data *sequence*. Arsitektur ini memiliki tiga *gate* utama: *forget gate*, *input gate*, dan *output gate*. *Forget gate* memutuskan informasi mana dari langkah waktu sebelumnya yang akan dibuang dari memori sel berdasarkan input saat ini dan status sebelumnya. *Input gate* menentukan informasi baru dari input yang akan disimpan ke dalam memori sel, sedangkan *output gate* mengontrol informasi mana yang akan diteruskan ke lapisan berikutnya atau menjadi output. Mekanisme ini memungkinkan LSTM untuk secara selektif menyimpan dan mengakses informasi dari langkah waktu yang jauh di masa lalu, menjadikannya sangat efektif untuk data *time series* seperti harga emas yang memiliki pola temporal kompleks.

> Memproses data time series secara berurutan dengan menyimpan dan memperbarui informasi jangka panjang melalui tiga gate (forget, input, output), sehingga mampu menangkap pola temporal yang kompleks dari langkah waktu sebelumnya.

#### ➡️ Pembahasan Parameter
	
👉 **Parameter Default yang Digunakan (Tidak Disesuaikan):**
- **`activation`**: `'tanh'`
  - Fungsi aktivasi untuk output LSTM. Default adalah hyperbolic tangent (`tanh`), yang umum digunakan karena cocok untuk data time series yang dinormalisasi.
- **`recurrent_activation`**: `'sigmoid'`
  - Fungsi aktivasi untuk gerbang rekuren (forget gate, input gate, output gate). Default adalah sigmoid, yang membatasi output antara 0 dan 1.
- **`use_bias`**: `True`
  - Menentukan apakah lapisan menggunakan vektor bias. Default adalah `True`, sehingga bias diaktifkan untuk semua gerbang.
- **`kernel_initializer`**: `'glorot_uniform'`
  - Inisialisasi bobot untuk matriks kernel (bobot input). Default adalah Glorot Uniform (Xavier Uniform), yang cocok untuk menjaga stabilitas pelatihan.
- **`recurrent_initializer`**: `'orthogonal'`
  - Inisialisasi bobot untuk matriks rekuren (bobot state). Default adalah inisialisasi ortogonal, yang membantu mencegah masalah vanishing/exploding gradients.
- **`bias_initializer`**: `'zeros'`
  - Inisialisasi untuk vektor bias. Default adalah nol, yang merupakan pilihan aman.
- **`unit_forget_bias`**: `True`
  - Jika `True`, menambahkan 1 ke bias forget gate saat inisialisasi untuk mendorong forgetting awal yang lebih lambat. Ini adalah praktik standar untuk LSTM.
- **`kernel_regularizer`**, **`recurrent_regularizer`**, **`bias_regularizer`**: `None`
  - Tidak ada regularisasi (misalnya, L1 atau L2) yang diterapkan pada bobot kernel, rekuren, atau bias secara default.
- **`activity_regularizer`**: `None`
  - Tidak ada regularisasi pada output lapisan.
- **`kernel_constraint`**, **`recurrent_constraint`**, **`bias_constraint`**: `None`
  - Tidak ada batasan (constraints) pada bobot.
- **`recurrent_dropout`**: `0.0`
  - Tidak ada dropout yang diterapkan pada state rekuren.
- **`return_state`**: `False`
  - Tidak mengembalikan state (hidden state dan cell state) secara default.
- **`go_backwards`**: `False`
  - Memproses urutan secara berurutan (bukan mundur).
- **`stateful`**: `False`
  - Tidak mempertahankan state antar batch, yang sesuai untuk pelatihan standar.
- **`time_major`**: `False`
  - Format input adalah `(batch, timesteps, features)`, bukan `(timesteps, batch, features)`.
- **`unroll`**: `False`
  - Tidak meng-unroll jaringan, yang lebih efisien untuk urutan panjang tetapi membutuhkan lebih banyak memori.

👉 **Parameter model LSTM dioptimalkan menggunakan pustaka `Hyperopt` untuk mencapai performa terbaik:**
- **Units**: Jumlah unit LSTM dalam lapisan, diatur pada 96 setelah optimasi (bukan default).
- **Dropout**: Tingkat dropout untuk mencegah *overfitting*, diatur pada 0.295 setelah optimasi (bukan default).
- **Learning Rate**: Kecepatan pembelajaran optimizer, diatur pada 0.00184 setelah optimasi (bukan default, Nilai default pada  optimizer  Adam adalah  0.001).
- **Batch Size**: Ukuran batch untuk pelatihan, diatur pada 16 setelah optimasi (bukan default, default biasanya 32).
- **Epochs**: Jumlah iterasi pelatihan, diatur pada 100 setelah optimasi.

#### ➡️ Kelebihan
- Efektif menangkap pola jangka panjang dalam data *time series*.
- Cocok untuk dataset dengan fluktuasi kompleks.

#### ➡️ Kekurangan
- Proses pelatihan lebih lambat karena kompleksitas arsitektur.
- Memerlukan regularisasi ketat untuk menghindari *overfitting*.

---

### 5.2.  Model 2: GRU (*Gated Recurrent Unit*)

#### ➡️ Pembahasan Cara Kerja
GRU adalah varian dari RNN yang lebih sederhana dibandingkan LSTM, dirancang untuk menangkap dependensi temporal dengan efisiensi komputasi yang lebih tinggi. GRU hanya memiliki dua *gate* utama: *update gate* dan *reset gate*. *Update gate* mengontrol seberapa banyak informasi dari langkah waktu sebelumnya yang akan dibawa ke langkah waktu saat ini, dengan memadukan input baru dan memori sebelumnya. *Reset gate* menentukan seberapa banyak informasi dari memori sebelumnya yang akan dilupakan sebelum menggabungkannya dengan input baru. Dengan struktur yang lebih ringkas ini, GRU dapat menawarkan performa yang kompetitif dibandingkan LSTM sambil mengurangi beban komputasi, menjadikannya pilihan yang baik untuk prediksi *time series* seperti harga emas.

> Mengolah data time series dengan menggabungkan informasi lama dan baru melalui dua gate (update dan reset), memungkinkan pembelajaran pola temporal dengan efisiensi komputasi yang lebih tinggi dibandingkan LSTM.

#### ➡️ Pembahasan Parameter

👉 **Parameter Default yang Digunakan (Tidak Disesuaikan):**


1. **`activation`**  
   - **Nilai Default**: `'tanh'`  
   - **Penjelasan**: Fungsi aktivasi untuk output dari gerbang update dan reset. Karena tidak disebutkan dalam notebook, maka `'tanh'` digunakan.

2. **`recurrent_activation`**  
   - **Nilai Default**: `'sigmoid'`  
   - **Penjelasan**: Fungsi aktivasi untuk gerbang update dan reset dalam perhitungan rekursif. Tidak disebutkan, sehingga `'sigmoid'` digunakan.

3. **`use_bias`**  
   - **Nilai Default**: `True`  
   - **Penjelasan**: Menentukan apakah layer menggunakan vektor bias. Tidak diatur, jadi default `True` berlaku.

4. **`kernel_initializer`**  
   - **Nilai Default**: `'glorot_uniform'`  
   - **Penjelasan**: Inisialisasi untuk matriks bobot kernel (input ke hidden). Tidak disebutkan, sehingga `'glorot_uniform'` digunakan.

5. **`recurrent_initializer`**  
   - **Nilai Default**: `'orthogonal'`  
   - **Penjelasan**: Inisialisasi untuk matriks bobot rekursif (hidden ke hidden). Tidak diatur, jadi `'orthogonal'` digunakan.

6. **`bias_initializer`**  
   - **Nilai Default**: `'zeros'`  
   - **Penjelasan**: Inisialisasi untuk vektor bias. Tidak disebutkan, sehingga `'zeros'` digunakan.

7. **`recurrent_dropout`**  
   - **Nilai Default**: `0.0` (tidak ada dropout)  
   - **Penjelasan**: Fraksi unit yang di-dropout untuk transformasi linear pada state rekursif. Tidak diatur, sehingga default `0.0` digunakan.

8. **`return_state`**  
   - **Nilai Default**: `False`  
   - **Penjelasan**: Apakah layer mengembalikan state terakhir selain output. Tidak disebutkan, jadi `False` digunakan.

9. **`go_backwards`**  
   - **Nilai Default**: `False`  
   - **Penjelasan**: Apakah sequence diproses secara terbalik. Tidak diatur, sehingga `False` digunakan dan akan Memproses data secara berurutan

10. **`stateful`**  
    - **Nilai Default**: `False`  
    - **Penjelasan**: Apakah layer mempertahankan state antar batch. Tidak disebutkan, jadi `False` digunakan.

11. **`unroll`**  
    - **Nilai Default**: `False`  
    - **Penjelasan**: Apakah jaringan di-unroll (berguna untuk sequence pendek). Tidak diatur, sehingga `False` digunakan.




👉 **Parameter model GRU juga dioptimalkan menggunakan `Hyperopt` untuk memastikan performa optimal:**
- **Units**: Jumlah unit GRU dalam lapisan, diatur pada 80 setelah optimasi (bukan default).
- **Dropout**: Tingkat dropout untuk regularisasi, diatur pada 0.4197 setelah optimasi (bukan default).
- **Learning Rate**: Kecepatan pembelajaran optimizer, diatur pada 0.0067 setelah optimasi (bukan default, Nilai default pada  optimizer  Adam adalah  0.001).
- **Batch Size**: Ukuran batch untuk pelatihan, diatur pada 32 setelah optimasi. 
- **Epochs**: Jumlah iterasi pelatihan, diatur pada 150 setelah optimasi.

#### ➡️ Kelebihan
- Proses pelatihan lebih cepat dibandingkan LSTM.
- Performa kompetitif dengan kompleksitas lebih rendah.

#### ➡️ Kekurangan
- Mungkin kurang optimal untuk pola yang sangat rumit atau ketergantungan jangka panjang yang ekstrem.

---

## 6. Evaluasi

Evaluasi dilakukan untuk mengukur performa kedua model pada data *testing* menggunakan metrik standar untuk tugas regresi.

### 6.1 Metrik Evaluasi
- **RMSE**: Mengukur rata-rata kesalahan kuadrat, sensitif terhadap kesalahan besar.  
  **RMSE** = $\sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y_i})^2}$


- **MAE**: Mengukur rata-rata kesalahan absolut, memberikan gambaran kesalahan tanpa mempertimbangkan arah.  
 **MAE** = $\frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y_i}|$

- **MAPE**: Mengukur kesalahan dalam bentuk persentase, memudahkan interpretasi relatif.  
**MAPE** = $\frac{1}{n} \sum_{i=1}^{n} \left| \frac{y_i - \hat{y_i}}{y_i} \right| \times 100\%$

- **R²**: Mengukur seberapa baik model menjelaskan variansi data. Nilai mendekati 1 menunjukkan performa baik.  
**R²** = $1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y_i})^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}$

### 6.2 Hasil Evaluasi

| **Model** | **RMSE** | **MAE** | **MAPE** | **R²** |
|-----------|----------|---------|----------|--------|
| LSTM      | 46.50    | 39.86   | 1.56%    | 0.8960 |
| GRU       | 33.51    | 25.03   | 0.99%    | 0.9460 |

📊 **Analisis**:
- **GRU** secara konsisten mengungguli LSTM pada semua metrik, dengan RMSE dan MAE lebih rendah serta R² lebih tinggi.
- MAPE GRU di bawah 1% menunjukkan tingkat akurasi yang sangat baik dalam konteks prediksi harga emas.

### 6.3 Visualisasi Prediksi

#### ➡️ Prediksi vs Aktual - LSTM
![Prediksi LSTM vs Aktual](./images/prediksi_lstm.png)

💡 **Insight**:
- LSTM mampu menangkap tren umum harga emas, tetapi cenderung kurang akurat pada fluktuasi jangka pendek.
- Beberapa puncak dan lembah tidak terdeteksi dengan presisi tinggi.
- LSTM menunjukkan kesalahan yang lebih besar pada beberapa titik ekstrem.

#### ➡️ Prediksi vs Aktual - GRU
![Prediksi GRU vs Aktual](./images/prediksi_gru.png)

💡 **Insight**:
- GRU menunjukkan kemampuan superior dalam menangkap fluktuasi harga, baik pada tren naik maupun turun.
- Prediksi GRU lebih mendekati nilai aktual, terutama pada periode volatilitas tinggi dari pada LSTM.

---

## 7. Hubungan dengan Problem Statements dan Goals

### ❓ Problem Statement 1:  
_"Bagaimana cara memanfaatkan data historis harga emas untuk memprediksi harga penutupan di masa depan dengan tingkat akurasi yang tinggi?"_

- **Dampak Model**:  
  - Kedua model, LSTM dan GRU, berhasil memprediksi harga penutupan emas dengan akurasi yang memadai, terutama model GRU yang mencapai MAPE di bawah 1%.
  - Namun, **akurasi GRU (MAPE 0.99%)** lebih tinggi dibandingkan LSTM (MAPE 1.56%), menunjukkan bahwa GRU lebih efektif dalam menangkap pola temporal harga emas.
- **Tindakan**:  
  - Model GRU dapat diimplementasikan untuk mendukung keputusan investasi jangka pendek.
  - Perlu eksplorasi lebih lanjut untuk meningkatkan akurasi, misalnya dengan menambahkan fitur eksternal seperti data makroekonomi.

### ❓ Problem Statement 2:  
_"Di antara dua model deep learning, yaitu LSTM dan GRU, manakah yang memberikan performa lebih baik dalam memprediksi harga emas berdasarkan metrik evaluasi standar?"_

- **Dampak Model**:  
  - Berdasarkan metrik evaluasi (RMSE, MAE, MAPE, R²), **GRU unggul** dibandingkan LSTM.
  -  GRU lebih efisien dan akurat untuk tugas ini, kemungkinan karena struktur yang lebih sederhana namun tetap mampu menangkap dependensi temporal.
- **Tindakan**:  
  - Prioritaskan penggunaan GRU untuk prediksi harga emas di masa depan.
  - Pertimbangkan untuk menguji model lain seperti Transformer untuk perbandingan lebih lanjut.

---

### 🎯 Goal 1:  
_"Membangun model prediktif berbasis deep learning yang mampu memberikan estimasi akurat untuk harga penutupan emas harian."_

- **Status**: **Tercapai**.
  - Kedua model memberikan prediksi yang akurat, dengan GRU mencapai R² di atas 0.94.
- **Dampak**:  
  - Model dapat diintegrasikan ke dalam sistem untuk mendukung analisis dan keputusan investasi.
- **Rekomendasi**:  
  - Gunakan model GRU untuk operasional sehari-hari.

### 🎯 Goal 2:  
_"Melakukan perbandingan mendalam antara model LSTM dan GRU untuk menentukan pendekatan terbaik dalam konteks prediksi time series harga emas."_

- **Status**: **Tercapai**.
  - Perbandingan menunjukkan bahwa GRU lebih unggul dalam hal akurasi dan efisiensi.
- **Dampak**:  
  - Memilih GRU sebagai model utama untuk prediksi harga emas.
- **Rekomendasi**:  
  - Fokus pada pengembangan dan optimalisasi model GRU untuk penggunaan jangka panjang.

---

## 8. Kesimpulan dan Rekomendasi 🌟

### 8.1 Kesimpulan 
- **Performa Model**: GRU terbukti lebih unggul dibandingkan LSTM dalam memprediksi harga penutupan emas.
- **Efisiensi**: GRU menawarkan keunggulan dalam hal kecepatan pelatihan dan akurasi, menjadikannya pilihan yang lebih praktis untuk aplikasi prediksi *time series*.
- **Aplikasi**: Kedua model dapat digunakan untuk prediksi jangka pendek, tetapi GRU lebih direkomendasikan karena performanya yang lebih baik.

### 8.2 Rekomendasi Bisnis
- **Investasi Jangka Pendek**: Model GRU dapat diintegrasikan ke dalam sistem untuk mendukung keputusan beli atau jual emas secara real-time.
- **Manajemen Risiko**: Prediksi harga emas membantu investor mengelola risiko volatilitas pasar dalam portofolio mereka.
- **Pengembangan Produk Keuangan**: Institusi keuangan dapat memanfaatkan model ini untuk menciptakan alat analisis harga emas.

### 8.3 Pengembangan Lebih Lanjut
- **Fitur Tambahan**: Menambahkan variabel eksternal seperti tingkat inflasi, nilai tukar dolar AS, fitur teknikal indikator atau sentimen pasar dari berita untuk meningkatkan akurasi.
- **Arsitektur Alternatif**: Mengeksplorasi model seperti *Transformer* atau *Temporal Convolutional Networks* (TCN) untuk menangani pola yang lebih kompleks.
- **Prediksi Jangka Panjang**: Mengadaptasi model untuk prediksi jangka panjang dengan memperluas *window size* atau menggabungkan pendekatan *ensemble*.

---