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
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## Modeling
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
- Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.
- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

## Evaluation
Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.

Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**. Jelaskan mengenai beberapa hal berikut:
- Penjelasan mengenai metrik yang digunakan
- Menjelaskan hasil proyek berdasarkan metrik evaluasi

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.

