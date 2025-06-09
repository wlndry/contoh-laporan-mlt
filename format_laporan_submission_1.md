# Laporan Proyek Machine Learning - Wulandari

## Domain Proyek

Penyakit diabetes merupakan salah satu penyakit kronis yang paling umum dan berdampak luas secara global. Berdasarkan laporan dari **World Health Organization (WHO)**, pada tahun 2021 tercatat lebih dari **422 juta orang di seluruh dunia** hidup dengan diabetes, dan jumlah ini diprediksi akan terus meningkat secara signifikan dalam beberapa dekade ke depan [1]. Diabetes tidak hanya berdampak pada kualitas hidup individu yang mengidapnya, tetapi juga memberikan beban ekonomi dan sosial yang besar terhadap sistem kesehatan masyarakat, terutama di negara berkembang.

Deteksi dini terhadap penyakit diabetes sangat penting untuk mencegah komplikasi jangka panjang, seperti penyakit kardiovaskular, gagal ginjal, gangguan penglihatan, dan amputasi anggota tubuh. Namun, dalam praktik klinis, proses deteksi dini sering kali masih mengandalkan prosedur manual dan hasil uji laboratorium yang memerlukan waktu serta biaya yang tidak sedikit. Oleh karena itu, diperlukan pendekatan yang lebih efisien, akurat, dan terjangkau dalam mendeteksi penyakit ini pada tahap awal.

Dalam konteks ini, penerapan **machine learning** sebagai alat bantu diagnostik telah menjadi fokus banyak penelitian terbaru. Machine learning memungkinkan pemodelan hubungan kompleks antara fitur-fitur kesehatan pasien (seperti kadar glukosa, tekanan darah, dan indeks massa tubuh) dengan probabilitas terkena diabetes. Algoritma klasifikasi, khususnya, telah terbukti mampu mempelajari pola dalam data medis untuk memberikan prediksi yang akurat terkait status diabetes pasien [2][3].

Dengan adanya model prediktif berbasis machine learning, profesional medis dapat dibantu dalam membuat keputusan lebih cepat dan objektif, serta meningkatkan peluang intervensi medis secara dini. Hal ini bukan hanya dapat menyelamatkan nyawa, tetapi juga mengurangi biaya pengobatan jangka panjang secara signifikan.

### Referensi

[1] World Health Organization, “Diabetes,” *World Health Organization*, 2021. [Online]. Available: https://www.who.int/news-room/fact-sheets/detail/diabetes

[2] P. Kavakiotis, O. Tsave, A. Salifoglou, N. Maglaveras, I. Vlahavas, and I. Chouvarda, "Machine Learning and Data Mining Methods in Diabetes Research," *Computational and Structural Biotechnology Journal*, vol. 15, pp. 104–116, 2017. doi: [10.1016/j.csbj.2016.12.005](https://doi.org/10.1016/j.csbj.2016.12.005)

[3] T. Santhanam and M. Padmavathi, "Application of K-means and Genetic Algorithms for Dimension Reduction by Integrating SVM for Diabetes Diagnosis," *Procedia Computer Science*, vol. 47, pp. 76–83, 2015. doi: [10.1016/j.procs.2015.03.180](https://doi.org/10.1016/j.procs.2015.03.180)


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

**Jumlah data:**  
Dataset terdiri dari 2.000 baris (entri) dan 9 kolom (fitur).

**Kondisi data:**  
- Tidak terdapat missing value (null) pada dataset.  
- Tidak ditemukan data duplikat.  
- Dari 9 kolom, 7 kolom bertipe integer dan 2 kolom bertipe float.

**Uraian variabel dalam dataset:**
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

- **Cara kerja**:  
  Logistic Regression memodelkan probabilitas kelas target dengan menggunakan fungsi logit, yang mengubah kombinasi linier dari fitur menjadi nilai probabilitas antara 0 dan 1 melalui fungsi sigmoid. Model belajar bobot (koefisien) fitur melalui optimisasi untuk meminimalkan loss function.

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

- **Cara kerja**:  
  Random Forest membangun banyak pohon keputusan (decision trees) secara acak dan independen pada subset data dan fitur yang berbeda. Setiap pohon memberikan prediksi, lalu hasil akhir ditentukan dengan metode voting mayoritas untuk klasifikasi. Metode ensemble ini membantu mengurangi overfitting dan meningkatkan akurasi.

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

- **Akurasi**:  
  $$Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$$

- **Precision**:  
  $$Precision = \frac{TP}{TP + FP}$$

- **Recall**:  
  $$Recall = \frac{TP}{TP + FN}$$

- **F1-score**:  
  $$
  F_1\text{-score} = 2 \times \frac{Precision \times Recall}{Precision + Recall}
  $$


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

### Dampak terhadap Business Understanding

- Model ini berhasil menjawab problem statement utama dengan memberikan klasifikasi diabetes yang akurat.
- Goals membandingkan dan memilih model terbaik tercapai dengan Random Forest sebagai pilihan unggulan.
- Solusi ini berdampak positif pada proses diagnosis dan pencegahan dini diabetes, yang dapat mengurangi biaya kesehatan dan meningkatkan kualitas pelayanan.

### Kesimpulan

Berdasarkan evaluasi, **Random Forest Classifier dipilih sebagai model terbaik** karena:

- Memberikan **akurasi sangat tinggi (98%)**.
- Menyeimbangkan precision dan recall dengan sangat baik di kedua kelas.
- Memiliki jumlah kesalahan klasifikasi yang sangat rendah (baik false positive maupun false negative).

