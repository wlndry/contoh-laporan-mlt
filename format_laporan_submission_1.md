# Laporan Proyek Machine Learning - Wulandari

## Domain Proyek

Penyakit diabetes merupakan salah satu penyakit kronis yang paling banyak diderita di seluruh dunia. Berdasarkan data dari World Health Organization (WHO), pada tahun 2021 terdapat sekitar 422 juta orang di seluruh dunia yang mengidap diabetes, dan angka ini diperkirakan akan terus meningkat. Diabetes tidak hanya memengaruhi kualitas hidup pasien, tetapi juga berdampak besar terhadap sistem kesehatan dan ekonomi.

Pendeteksian dini terhadap penyakit diabetes sangat penting untuk mencegah komplikasi jangka panjang. Oleh karena itu, klasifikasi penyakit diabetes menggunakan machine learning menjadi solusi potensial untuk membantu diagnosa awal berbasis data kesehatan pasien.


**Rubrik/Kriteria Tambahan (Opsional)**:
- Jelaskan mengapa dan bagaimana masalah tersebut harus diselesaikan
- Menyertakan hasil riset terkait atau referensi. Referensi yang diberikan harus berasal dari sumber yang kredibel dan author yang jelas.
- Format Referensi dapat mengacu pada penulisan sitasi [IEEE](https://journals.ieeeauthorcenter.ieee.org/wp-content/uploads/sites/7/IEEE_Reference_Guide.pdf), [APA](https://www.mendeley.com/guides/apa-citation-guide/) atau secara umum seperti [di sini](https://penerbitdeepublish.com/menulis-buku-membuat-sitasi-dengan-mudah/)
- Sumber yang bisa digunakan [Scholar](https://scholar.google.com/)

## Business Understanding

### Problem Statements
- Bagaimana mengklasifikasikan apakah seseorang menderita diabetes berdasarkan data medis yang tersedia?
- Algoritma machine learning mana yang paling akurat untuk klasifikasi penyakit diabetes?

### Goals
- Membangun model klasifikasi untuk mendeteksi penyakit diabetes.
- Membandingkan performa model seperti Logistic Regression dan Random Forest untuk memilih model terbaik.

### Solution Statements
- Menggunakan dua algoritma: Logistic Regression dan Random Forest Classifier.
- Melakukan hyperparameter tuning pada model terbaik.
- Metrik evaluasi: Akurasi, Precision, Recall, F1-score.

## Data Understanding
Dataset diambil dari Kaggle: [Data Penyakit Diabetes](https://www.kaggle.com/datasets/sitirahmahbasri/data-penyakit-diabetes)

### Variabel dalam dataset:
- `Pregnancies`: Jumlah kehamilan
- `Glucose`: Kadar glukosa
- `BloodPressure`: Tekanan darah
- `SkinThickness`: Ketebalan lipatan kulit
- `Insulin`: Kadar insulin
- `BMI`: Indeks massa tubuh
- `DiabetesPedigreeFunction`: Riwayat diabetes keluarga
- `Age`: Umur
- `Outcome`: Label target (1: diabetes, 0: tidak)

### Exploratory Data Analysis (EDA)
Dataset yang digunakan terdiri dari sembilan variabel, yaitu Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, dan Outcome sebagai label target yang menunjukkan apakah seseorang mengidap diabetes atau tidak. Untuk memahami karakteristik data, dilakukan beberapa tahap eksplorasi data sebagai berikut.

Pertama, analisis statistik deskriptif menunjukkan bahwa variabel Glucose memiliki rata-rata kadar glukosa sekitar nilai yang cukup tinggi, yang menjadi indikator penting dalam diagnosis diabetes. Variabel BMI dan Age juga memiliki rentang nilai yang cukup luas, menunjukkan variasi kondisi fisik dan usia responden. Selanjutnya, pemeriksaan nilai kosong menunjukkan bahwa dataset ini relatif bersih tanpa data yang hilang, sehingga tidak diperlukan imputasi data.

![image](https://github.com/user-attachments/assets/85f3a2b8-7de7-45b7-bb41-f1d05ddfd68a)  

Selanjutnya, analisis korelasi antar variabel menggunakan matriks korelasi dan heatmap memperlihatkan hubungan yang signifikan antara variabel Glucose, BMI, dan Age dengan label Outcome. Hal ini menegaskan bahwa ketiga variabel tersebut memiliki peran penting dalam menentukan risiko diabetes. Variabel lain seperti Pregnancies dan DiabetesPedigreeFunction juga menunjukkan korelasi yang moderat terhadap outcome.

![image](https://github.com/user-attachments/assets/7357accd-5951-414b-9ddf-afbc8c73508c)  

Kemudian pada grafik distribusi label diabetes menunjukkan bahwa jumlah individu tanpa diabetes (label 0) jauh lebih banyak dibandingkan dengan individu yang menderita diabetes (label 1), dengan rasio sekitar 2:1. Ketidakseimbangan ini perlu diperhatikan dalam pemodelan karena dapat mempengaruhi performa algoritma klasifikasi, terutama dalam mendeteksi kasus positif diabetes yang lebih sedikit.

## Data Preparation

Pada tahap ini dilakukan beberapa proses persiapan data secara berurutan agar data siap digunakan dalam pemodelan machine learning.

1. **Mengganti Nilai Nol dengan Median Kolom**  
Beberapa kolom seperti `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, dan `BMI` memiliki nilai nol yang tidak valid secara medis. Nilai nol ini digantikan dengan nilai median dari masing-masing kolom untuk menghindari bias dan menjaga distribusi data agar tetap representatif. Median dipilih karena tahan terhadap nilai ekstrem (outlier).

2. **Memisahkan Fitur dan Label**  
Dataset dipisahkan menjadi fitur (semua kolom kecuali `Outcome`) dan label (`Outcome`). Pemisahan ini penting agar model dapat belajar dari fitur dan memprediksi label secara tepat.

3. **Normalisasi Data**  
Fitur numerik dinormalisasi menggunakan metode standar sehingga memiliki rata-rata nol dan standar deviasi satu. Normalisasi ini diperlukan agar semua fitur berada pada skala yang sama, sehingga algoritma machine learning dapat bekerja optimal dan lebih stabil.

4. **Pembagian Data Menjadi Training dan Testing Set**  
Data yang sudah dinormalisasi dibagi menjadi data latih dan data uji dengan proporsi 80:20 menggunakan stratifikasi berdasarkan label `Outcome`. Stratifikasi memastikan proporsi kelas diabetes dan non-diabetes tetap seimbang pada kedua set, sehingga evaluasi model menjadi lebih valid dan menghindari bias akibat ketidakseimbangan kelas.

## Modeling

Pada tahap pemodelan ini, dua algoritma machine learning telah digunakan untuk menyelesaikan permasalahan klasifikasi, yaitu **Logistic Regression** dan **Random Forest Classifier**. Kedua model dilatih menggunakan data pelatihan dan dievaluasi menggunakan data pengujian untuk menentukan performa terbaik.

### 1. Logistic Regression

- **Parameter yang digunakan**:
  - `max_iter=1000`: Menentukan jumlah maksimum iterasi untuk mencapai konvergensi.

- **Kelebihan**:
  - Cepat dan efisien untuk dataset berukuran besar.
  - Mudah diinterpretasikan karena koefisien menunjukkan pengaruh fitur terhadap output.
  - Bekerja baik untuk hubungan linier antara fitur dan target.

- **Kekurangan**:
  - Kurang mampu menangkap hubungan non-linear dalam data.
  - Rentan terhadap multikolinearitas antar fitur.

### 2. Random Forest Classifier

- **Parameter yang digunakan**:
  - `n_estimators=100`: Menentukan jumlah pohon dalam ensemble.
  - `random_state=42`: Untuk memastikan hasil yang reproducible.

- **Kelebihan**:
  - Mampu menangani data yang kompleks dan relasi non-linear antar fitur.
  - Robust terhadap outlier dan overfitting karena menggunakan metode ensemble.
  - Memiliki performa prediksi yang umumnya baik tanpa banyak tuning parameter.

- **Kekurangan**:
  - Waktu pelatihan dan prediksi cenderung lebih lama dibanding model yang lebih sederhana.
  - Interpretabilitas lebih rendah dibandingkan model linear seperti Logistic Regression.

### Pemilihan Model Terbaik

Setelah dilakukan evaluasi menggunakan metrik akurasi, precision, recall, dan F1-score, model **Random Forest Classifier** menunjukkan performa yang lebih baik secara konsisten dibandingkan Logistic Regression. Random Forest dapat menangkap hubungan kompleks dalam data dan lebih tahan terhadap overfitting, sehingga lebih cocok digunakan dalam konteks permasalahan ini.

Dengan mempertimbangkan kelebihan-kelebihan tersebut, **Random Forest Classifier dipilih sebagai model terbaik** dalam menyelesaikan permasalahan klasifikasi ini.

### Pengembangan Model Selanjutnya

Untuk meningkatkan performa model Random Forest, proses tuning hyperparameter seperti `max_depth`, `min_samples_split`, dan `n_estimators` dapat dilakukan dengan menggunakan teknik seperti Grid Search atau Random Search. Hal ini dapat membantu menemukan kombinasi parameter optimal guna menghasilkan model yang lebih akurat dan efisien.

## Evaluation

Evaluasi model dilakukan menggunakan metrik klasifikasi yaitu **akurasi**, **precision**, **recall**, dan **F1-score**. Metrik ini dipilih karena sesuai dengan karakteristik permasalahan klasifikasi yang melibatkan dua kelas dan adanya kebutuhan untuk memahami keseimbangan antara kesalahan tipe I (false positive) dan tipe II (false negative).

### Penjelasan Metrik Evaluasi

- **Akurasi**: Mengukur seberapa sering model memprediksi kelas dengan benar.
  \[
  \text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
  \]

- **Precision**: Mengukur ketepatan prediksi positif yang dilakukan oleh model.
  \[
  \text{Precision} = \frac{TP}{TP + FP}
  \]

- **Recall**: Mengukur seberapa banyak data positif yang berhasil dikenali oleh model.
  \[
  \text{Recall} = \frac{TP}{TP + FN}
  \]

- **F1-score**: Rata-rata harmonis dari precision dan recall, berguna saat terdapat trade-off antara keduanya.
  \[
  \text{F1-score} = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}
  \]

### Hasil Evaluasi

| Model                | Akurasi | Precision | Recall | F1-Score |
|---------------------|---------|-----------|--------|----------|
| Logistic Regression | 0.80    | 0.75–0.81 | 0.61–0.90 | 0.67–0.85 |
| Random Forest       | 0.98    | 0.98–0.99 | 0.98–0.99 | 0.98–0.99 |

**Catatan:**
- Nilai precision, recall, dan f1-score dituliskan dalam rentang karena berbeda untuk tiap kelas (0 dan 1).
- Logistic Regression menunjukkan performa yang baik pada kelas mayoritas (kelas 0), tetapi kurang optimal dalam mendeteksi kelas minoritas (kelas 1).
- Sebaliknya, **Random Forest Classifier** memberikan performa sangat tinggi dan seimbang di kedua kelas, dengan akurasi mencapai 98%.

### Interpretasi Confusion Matrix

- **Logistic Regression**:
  - True Positive (TP): 83
  - True Negative (TN): 236
  - False Positive (FP): 27
  - False Negative (FN): 54

- **Random Forest Classifier**:
  - TP: 134
  - TN: 260
  - FP: 3
  - FN: 3

Random Forest berhasil mengurangi kesalahan klasifikasi secara signifikan pada kedua kelas dibandingkan Logistic Regression.

### Kesimpulan

Berdasarkan evaluasi, **Random Forest Classifier dipilih sebagai model terbaik** karena:

- Memberikan **akurasi sangat tinggi (98%)**.
- Menyeimbangkan precision dan recall dengan sangat baik di kedua kelas.
- Memiliki jumlah kesalahan klasifikasi yang sangat rendah (baik false positive maupun false negative).

