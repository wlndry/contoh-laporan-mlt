# Laporan Proyek Machine Learning - Wulandari

## Project Overview

Dalam era informasi dan hiburan digital yang sangat berkembang pesat, pengguna layanan streaming seperti Netflix, Disney+, dan Amazon Prime sering kali dihadapkan pada ribuan pilihan film dan serial. Banyaknya pilihan ini justru dapat membuat pengguna kewalahan dalam mengambil keputusan. Berdasarkan studi yang dilakukan oleh Nielsen, lebih dari 60% pengguna merasa frustrasi karena kesulitan memilih konten yang relevan dengan preferensi mereka [1]. Oleh karena itu, sistem rekomendasi menjadi kebutuhan penting untuk membantu pengguna menemukan konten yang tepat dengan lebih cepat dan efisien.

Proyek ini bertujuan untuk membangun **sistem rekomendasi film** yang dapat memberikan rekomendasi film secara otomatis dan personal menggunakan pendekatan **Content-Based Filtering** dan **Collaborative Filtering**. Dataset yang digunakan adalah **TMDB 5000 Movie Metadata** dari Kaggle, yang menyediakan informasi kaya mengenai film, termasuk genre, sinopsis (overview), kata kunci, kru, pemeran, serta skor dan jumlah suara dari pengguna.

### Mengapa Proyek Ini Penting?

- **Meningkatkan Pengalaman Pengguna**: Sistem rekomendasi yang baik membantu pengguna menemukan film yang sesuai dengan selera mereka, sehingga meningkatkan engagement.
- **Efisiensi dalam Penjelajahan Konten**: Pengguna tidak perlu mencari secara manual dalam ribuan film yang tersedia.
- **Nilai Bisnis yang Tinggi**: Rekomendasi yang relevan mendorong retensi pengguna dan meningkatkan konversi serta pendapatan platform [2].

### Relevansi Riset

Banyak penelitian telah membuktikan bahwa sistem rekomendasi memiliki dampak langsung terhadap performa bisnis dan pengalaman pengguna:

- Bobadilla et al. [3] menyebutkan bahwa sistem rekomendasi merupakan bagian integral dari layanan personalisasi modern dan mampu meningkatkan nilai tambah dalam pengambilan keputusan pengguna.
- Ricci et al. [4] menekankan pentingnya menggabungkan berbagai pendekatan, seperti content-based dan collaborative filtering, untuk mengatasi kelemahan masing-masing metode secara individual.

Dengan memanfaatkan data dari TMDB dan menerapkan dua metode rekomendasi yang berbeda, proyek ini menunjukkan bagaimana machine learning dapat diterapkan secara nyata untuk menyelesaikan permasalahan sehari-hari dalam skala industri.

---

**Referensi:**

[1] Nielsen, "The struggle is real: How consumers navigate content overload," Nielsen Insights, 2016. [Online]. Available: https://www.nielsen.com/

[2] Gómez-Uribe, C. A., & Hunt, N. (2016). "The Netflix Recommender System: Algorithms, Business Value, and Innovation." *ACM Transactions on Management Information Systems*, 6(4), 1–19. https://doi.org/10.1145/2843948

[3] Bobadilla, J., Ortega, F., Hernando, A., & Gutiérrez, A. (2013). "Recommender systems survey." *Knowledge-Based Systems*, 46, 109–132. https://doi.org/10.1016/j.knosys.2013.03.012

[4] Ricci, F., Rokach, L., & Shapira, B. (2015). *Recommender Systems Handbook*. Springer. ISBN: 978-1-4899-7636-3

---
## Business Understanding

Di era digital saat ini, platform layanan streaming seperti Netflix, Disney+, dan Amazon Prime menyediakan ribuan judul film dan serial yang dapat diakses kapan saja. Namun, dengan begitu banyaknya pilihan, pengguna kerap merasa kewalahan dan mengalami kesulitan dalam memilih tontonan yang sesuai dengan preferensi mereka. Hal ini menimbulkan kebutuhan mendesak akan sistem rekomendasi yang dapat membantu pengguna menemukan film yang relevan dan menarik secara personal.

Sistem rekomendasi yang efektif tidak hanya meningkatkan kenyamanan pengguna, tetapi juga dapat meningkatkan retensi pelanggan dan engagement di platform. Oleh karena itu, pengembangan sistem rekomendasi film menjadi aspek penting dalam industri hiburan digital.

### Problem Statements

1. **Terlalu banyak pilihan film** yang menyebabkan pengguna kesulitan menemukan film yang sesuai dengan minat atau preferensinya.
2. **Rekomendasi film belum cukup relevan** karena kurangnya pemanfaatan fitur metadata film seperti genre, sinopsis, aktor, dan sutradara.
3. **Sistem rekomendasi berbasis satu pendekatan saja** (misalnya hanya popularitas) cenderung bias dan tidak personal bagi semua jenis pengguna.

### Goals

1. **Membangun sistem rekomendasi film** yang dapat memberikan saran film relevan berdasarkan preferensi pengguna.
2. **Mengintegrasikan metadata film** untuk memahami karakteristik konten dan menyesuaikannya dengan minat pengguna.
3. **Menyediakan dua pendekatan sistem rekomendasi** untuk meningkatkan akurasi dan relevansi hasil rekomendasi.

### Solution Approach

Untuk mencapai tujuan di atas, proyek ini mengusulkan dua pendekatan utama dalam sistem rekomendasi:

1. **Content-Based Filtering**  
   Sistem rekomendasi ini memanfaatkan metadata dari film seperti overview, genre, keywords, pemeran utama, dan sutradara. Dengan menggunakan teknik pemrosesan teks seperti TF-IDF dan cosine similarity, sistem menghitung kemiripan antarfilm untuk merekomendasikan film yang mirip dengan film yang disukai pengguna.

2. **Collaborative Filtering (Berbasis Popularitas)**  
   Sistem ini merekomendasikan film berdasarkan data rating dan jumlah penonton (vote_count dan vote_average) menggunakan formula IMDb. Pendekatan ini membantu mengenali film-film populer dan berkualitas tinggi berdasarkan opini banyak pengguna, bukan hanya konten.

Kedua pendekatan ini saling melengkapi: Content-Based membantu memberikan rekomendasi personal, sementara Collaborative Filtering memastikan kualitas dan popularitas film juga diperhitungkan.


## Data Understanding
Paragraf awal bagian ini menjelaskan informasi mengenai jumlah data, kondisi data, dan informasi mengenai data yang digunakan. Sertakan juga sumber atau tautan untuk mengunduh dataset. Contoh: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Restaurant+%26+consumer+data).

Selanjutnya, uraikanlah seluruh variabel atau fitur pada data. Sebagai contoh:  

Variabel-variabel pada Restaurant UCI dataset adalah sebagai berikut:
- accepts : merupakan jenis pembayaran yang diterima pada restoran tertentu.
- cuisine : merupakan jenis masakan yang disajikan pada restoran.
- dst

**Rubrik/Kriteria Tambahan (Opsional)**:
- Melakukan beberapa tahapan yang diperlukan untuk memahami data, contohnya teknik visualisasi data beserta insight atau exploratory data analysis.

## Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## Modeling
Tahapan ini membahas mengenai model sisten rekomendasi yang Anda buat untuk menyelesaikan permasalahan. Sajikan top-N recommendation sebagai output.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menyajikan dua solusi rekomendasi dengan algoritma yang berbeda.
- Menjelaskan kelebihan dan kekurangan dari solusi/pendekatan yang dipilih.

## Evaluation
Pada bagian ini Anda perlu menyebutkan metrik evaluasi yang digunakan. Kemudian, jelaskan hasil proyek berdasarkan metrik evaluasi tersebut.

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.
