# Klasifikasi Penyakit dan Kondisi Buah Jeruk dengan CNN (TensorFlow/Keras)

Proyek UTS Pembelajaran Mendalam, Kelompok 1, Universitas Mikroskil. Citra buah jeruk diklasifikasikan ke empat kelas
(`blackspot`, `canker`, `fresh`, `grenning`) memakai CNN dasar yang dilatih dari awal (tanpa transfer learning),
lalu dipakai pada aplikasi web Flask.

**Aplikasi (sudah di-deploy):** https://sylvany.pythonanywhere.com/

**Anggota:** 
1. Miranda Nadilla Putri (231112195)
2. Nailah Assyifa (231112334)
3. Rifky Ghifari Siregar (231112918)
4. Selamet Riyadi (231112132)
5. Silvani Chayadi (231112945)

## Isi repository

| Berkas | Fungsi |
|---|---|
| `jeruk_cnn_tensorflow.ipynb` | Eksplorasi data, preprocessing, pembagian data, pipeline `tf.data`, CNN, lima eksperimen, evaluasi, fitur unik |
| `app.py`, `templates/`, `static/` | Aplikasi Flask (memuat `model_cnn.keras`) |
| `model_cnn.keras`, `class_names.json` | Model final dan metadata (urutan kelas, ukuran input, ambang confidence) |
| `model_cnn.tflite`, `gradcam_bobot.npz` | Salinan ringan model yang sama untuk hosting tanpa TensorFlow (dibuat oleh notebook bagian 8) |
| `requirements.txt` | Dependensi untuk pelatihan dan menjalankan lokal |
| `requirements-deploy.txt` | Dependensi ringan untuk hosting (PythonAnywhere), tanpa TensorFlow |

## Dataset

[Orange Diseases Dataset (Kaggle)](https://www.kaggle.com/datasets/jonathansilva2020/orange-diseases-dataset).
Dataset tidak disertakan di repository. Unduh, lalu susun seperti ini (hanya folder `train` yang dipakai, 991 citra):

```
dataset/train/blackspot/
dataset/train/canker/
dataset/train/fresh/
dataset/train/grenning/
```

## Instalasi (Python 3.10 sampai 3.12)

```
python -m venv venv
venv\Scripts\activate          # Windows. Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
```

## Training

1. Jalankan `jupyter notebook` dari folder proyek (sejajar dengan `dataset/`), buka `jeruk_cnn_tensorflow.ipynb`, lalu **Run All**.
   Pelatihan lima eksperimen memakan sekitar 40 menit di CPU (jumlah waktu epoch pada log run kami).
2. Keluaran: `model_cnn.keras`, `class_names.json`, serta gambar dan `hasil_eksperimen.json` di folder `laporan/`.
   Bagian 8 membuat `model_cnn.tflite` dan `gradcam_bobot.npz` untuk hosting. Salin ketiga berkas model ke folder yang sama dengan `app.py`.
3. Bagian 7 notebook (Grad-CAM dan confidence threshold) memakai `model_cnn.keras` yang sudah tersimpan, jadi dapat dijalankan
   tanpa melatih ulang: jalankan sel setup sampai pipeline `tf.data`, lalu bagian 7. Bagian ini juga menulis
   `confidence_threshold` ke `class_names.json`.

## Menjalankan aplikasi (lokal)

```
python app.py
```

Buka http://127.0.0.1:5000. `model_cnn.keras` dan `class_names.json` harus berada di folder yang sama dengan `app.py`.
Fitur: unggah citra, analisis lengkap (langkah 01 sampai 06), deteksi cepat, Grad-CAM, dan pesan "tidak yakin" bila
confidence di bawah ambang.

## Fitur unik

1. **Grad-CAM**: peta panas bagian citra yang paling memengaruhi prediksi (kode di notebook bagian 7.1 dan `gradcam_cnn` pada `app.py`).
2. **Confidence threshold**: prediksi dengan probabilitas tertinggi di bawah ambang (0,80, dipilih dari data validation)
   ditampilkan sebagai "tidak yakin" (notebook bagian 7.2 dan `cek_ambang` pada `app.py`).

## Hasil ringkas (data uji 199 citra, hasil run notebook kami)

| Eksperimen | Akurasi uji | Precision macro | Recall macro |
|---|---|---|---|
| 1. Baseline (BN momentum 0,99) | 30,65% | 13,57% | 28,38% |
| 2. BN momentum 0,9 | 96,98% | 95,94% | 95,91% |
| 3. + augmentasi (model final, dipilih dari validation accuracy) | 94,97% | 93,63% | 93,21% |
| 4. Learning rate awal 3e-4 | 95,98% | 94,78% | 94,48% |
| 5. Dropout 0,3 | 97,49% | 96,58% | 96,58% |

Model final dipilih dari akurasi validation (98,74%), bukan dari data uji. Angka dapat sedikit berbeda bila dijalankan
di komputer lain. Keterbatasan dataset (latar belakang dan format citra berbeda antarkelas) dibahas di laporan.

## Deployment (PythonAnywhere)

Di hosting, aplikasi berjalan tanpa TensorFlow: `app.py` otomatis memakai `model_cnn.tflite` (model yang sama, diekspor
oleh notebook bagian 8) bila TensorFlow tidak terpasang. Di komputer sendiri (ada TensorFlow) aplikasi tetap memakai
`model_cnn.keras`. Alasannya: akun gratis hanya punya disk 512 MB, sedangkan TensorFlow terpasang memakan sekitar 1,8 GB.
Dengan TFLite, dependensi sekitar 330 MB dan RAM aplikasi sekitar 196 MB (hasil pengukuran kami).
