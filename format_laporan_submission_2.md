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

Dataset yang digunakan dalam proyek ini berasal dari [TMDB (The Movie Database)](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata), yang merupakan sumber metadata film berskala global. Dataset ini terdiri dari dua file utama, yaitu:

- `tmdb_5000_movies.csv`: berisi informasi metadata film seperti judul, genre, sinopsis, rating, dan popularitas.
- `tmdb_5000_credits.csv`: berisi informasi mengenai pemeran (cast) dan kru (crew) dalam setiap film.

Kedua file ini memiliki jumlah data yang sama, yaitu **4.803 entri**, namun berbeda dalam jumlah dan jenis kolom. Dataset ini cukup besar dan kaya akan fitur, sehingga sangat cocok digunakan untuk membangun sistem rekomendasi film yang efektif.

Secara umum, kondisi data relatif baik, namun terdapat beberapa **missing values** terutama di kolom non-esensial. Contohnya:
- `homepage`: 3.091 nilai kosong
- `tagline`: 844 nilai kosong
- `overview`: 3 nilai kosong
- `release_date`: 1 nilai kosong
- `runtime`: 2 nilai kosong

Sementara itu, dataset `credits` tidak memiliki nilai kosong, tetapi kolom `cast` dan `crew` tersimpan dalam format string JSON yang memerlukan parsing terlebih dahulu agar bisa digunakan dalam analisis lebih lanjut.

---

### Sumber Dataset:
- [TMDB Movie Metadata Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)

---

### Variabel pada `tmdb_5000_movies.csv`:
- **budget**: Anggaran produksi film (USD).
- **genres**: Daftar genre film (Action, Drama, dll).
- **homepage**: Situs resmi film (jika tersedia).
- **id**: ID unik film dalam database TMDB.
- **keywords**: Kata kunci deskriptif film.
- **original_language**: Bahasa asli film.
- **original_title**: Judul asli sebelum translasi.
- **overview**: Ringkasan atau sinopsis film.
- **popularity**: Skor popularitas berdasarkan metrik TMDB.
- **production_companies**: Nama perusahaan produksi.
- **production_countries**: Negara tempat produksi dilakukan.
- **release_date**: Tanggal rilis film.
- **revenue**: Pendapatan film (USD).
- **runtime**: Durasi film (menit).
- **spoken_languages**: Bahasa yang digunakan dalam film.
- **status**: Status rilis film (misalnya: Released).
- **tagline**: Kalimat promosi singkat.
- **title**: Judul resmi film.
- **vote_average**: Rata-rata rating dari pengguna TMDB.
- **vote_count**: Jumlah total suara/rating yang diberikan.

### Variabel pada `tmdb_5000_credits.csv`:
- **movie_id**: ID film, yang digunakan sebagai kunci penggabungan.
- **title**: Judul film.
- **cast**: Daftar aktor dan aktris dalam film (format JSON string).
- **crew**: Informasi kru film, termasuk direktur, produser, penulis naskah, dsb (format JSON string).

---

### Insight dan Exploratory Data Analysis (EDA)

Untuk memahami karakteristik data lebih lanjut, dilakukan beberapa proses eksplorasi data awal, seperti:

- Visualisasi distribusi genre film dan popularitasnya.
- Analisis jumlah karakter dalam sinopsis (`overview`).
- Korelasi antara fitur numerik seperti `budget`, `revenue`, `vote_average`, dan `popularity`.
- Identifikasi aktor dan sutradara yang paling sering muncul dalam film.
- Deteksi film dengan pendapatan dan rating tertinggi.

Hasil dari EDA ini memberikan insight penting terhadap bagaimana film dikelompokkan berdasarkan genre atau popularitas, dan juga mengungkap variabel apa saja yang berpotensi berpengaruh terhadap sistem rekomendasi film. Data juga menunjukkan adanya bias distribusi pada genre tertentu seperti Drama dan Action yang mendominasi dataset.



## Data Preparation

Tahapan data preparation dalam proyek ini dilakukan secara sistematis untuk memastikan data siap digunakan dalam pemodelan sistem rekomendasi. Proses utama yang dilakukan adalah sebagai berikut:

1. **Penggabungan Dataset**  
   Dataset film (`movies`) digabung dengan dataset `credits` menggunakan kolom `id` pada `movies` dan `movie_id` pada `credits`. Penggabungan ini penting untuk mendapatkan informasi lengkap tentang film, termasuk data pemeran dan kru yang tidak terdapat dalam dataset utama.

2. **Pemilihan Kolom Penting**  
   Setelah penggabungan, hanya kolom-kolom yang relevan untuk pemodelan dipilih, yaitu `id`, `title`, `overview`, `genres`, `keywords`, `cast`, `crew`, `vote_average`, dan `vote_count`. Ini membantu memfokuskan analisis pada fitur yang berkontribusi langsung pada sistem rekomendasi.

3. **Penghapusan Data yang Memiliki Missing Value**  
   Setelah kolom penting dipilih, dilakukan pemeriksaan nilai yang hilang (missing values) pada data. Beberapa baris ditemukan memiliki nilai kosong khususnya pada kolom `overview`, `runtime`, atau `release_date`. Baris-baris yang memiliki nilai kosong tersebut dihapus agar tidak mengganggu proses pemodelan, terutama dalam analisis berbasis teks dan perhitungan skor.

4. **Parsing dan Ekstraksi Data Teks**  
   Beberapa kolom seperti `genres`, `keywords`, `cast`, dan `crew` berisi data dalam format string JSON yang perlu diubah menjadi format teks yang mudah diproses. Fungsi `parse_names` digunakan untuk mengekstrak nama-nama dari list JSON tersebut, misalnya nama genre, kata kunci, atau nama pemeran utama (5 teratas) dan sutradara dari kru.

5. **Penggabungan Fitur Deskriptif**  
   Fitur-fitur seperti `overview`, `genres`, `keywords`, `cast`, dan `crew` digabung menjadi satu kolom `description`. Kolom ini berfungsi sebagai representasi teks gabungan yang akan digunakan dalam metode content-based filtering. Penggabungan ini penting agar model dapat menangkap informasi lengkap dari berbagai aspek film secara terpadu.

6. **Penanganan Missing Values pada Teks**  
   Untuk memastikan proses vektorisasi teks berjalan lancar, nilai kosong pada kolom deskriptif seperti `overview`, `genres`, `keywords`, `cast`, dan `crew` diisi dengan string kosong (`''`), sebagai tindakan tambahan setelah penghapusan data utama yang memiliki missing values.

**Alasan Tahapan Data Preparation**:  
- Penggabungan dan pemilihan kolom penting memastikan hanya data yang relevan diproses sehingga efisien dan fokus pada fitur yang berkontribusi dalam rekomendasi.  
- Penghapusan missing values mencegah error dalam pemrosesan dan menjaga integritas data.  
- Parsing data JSON menjadi teks memudahkan analisis teks dan penggunaan algoritma berbasis teks seperti TF-IDF.  
- Penggabungan fitur deskriptif membuat model dapat memanfaatkan informasi lengkap secara simultan, meningkatkan kualitas rekomendasi content-based.  
- Penanganan nilai kosong dalam teks penting untuk menjaga kompatibilitas dengan metode NLP (Natural Language Processing) seperti TF-IDF Vectorization.


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
