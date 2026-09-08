# 🧵 Simulator Ulos — Klasifikasi Kain Ulos Batak AI

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Django](https://img.shields.io/badge/Django-5.2-092E20?style=for-the-badge&logo=django&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Gemini AI](https://img.shields.io/badge/Gemini_AI-2.5_Flash-4285F4?style=for-the-badge&logo=google&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**Aplikasi web berbasis AI untuk mengklasifikasikan jenis kain Ulos Batak secara otomatis melalui foto.**

[Demo Lokal](#-cara-menjalankan) · [Fitur](#-fitur) · [Arsitektur](#-arsitektur-sistem) · [Dataset](#-dataset--model)

</div>

---

## 📖 Tentang Proyek

**Simulator Ulos** adalah sistem berbasis web interaktif (chatbot) yang dirancang untuk mengidentifikasi dan mengklasifikasikan **14 jenis kain Ulos Batak** secara otomatis berdasarkan foto yang diunggah oleh pengguna. Sistem ini dibuat sebagai bagian dari proyek **Tugas Akhir** untuk pelestarian budaya tradisional Batak dengan memanfaatkan teknologi kecerdasan buatan terbaru.

Sistem ini menerapkan **Double Gate Detection** (Dua Gerbang Deteksi) untuk menjamin akurasi dan pengalaman pengguna yang optimal:
1. **Gate 1 (CNN - Deep Learning)**: Menggunakan arsitektur MobileNetV2 yang di-fine-tuned untuk mendeteksi 14 jenis motif kain Ulos dengan tingkat keyakinan (confidence score) tertentu.
2. **Gate 2 (Google Gemini AI via Vertex AI)**: Berfungsi sebagai *fallback system*. Jika gambar yang dikirim bukan kain Ulos, sistem akan mengidentifikasi jenis objek tersebut secara ramah dan menyarankan pengguna untuk mengunggah foto kain Ulos yang benar.
3. **Chatbot RAG (n8n + Gemini LLM)**: Setelah klasifikasi selesai, pengguna dapat melakukan tanya-jawab interaktif dengan AI mengenai makna filosofis, sejarah, penggunaan adat, dan nilai budaya dari kain Ulos yang teridentifikasi secara real-time.

---

## ✨ Fitur

| Fitur | Keterangan |
|-------|-----------|
| 📷 **Klasifikasi via Kamera** | Potret langsung kain ulos menggunakan kamera perangkat |
| 🖼️ **Upload dari Galeri** | Unggah foto dari penyimpanan lokal |
| 🤖 **Klasifikasi AI (CNN)** | Model MobileNetV2 fine-tuned untuk 14 kelas ulos |
| 🔍 **Fallback Gemini AI** | Identifikasi objek non-ulos menggunakan Google Gemini |
| 💬 **Chatbot RAG** | Tanya-jawab seputar budaya ulos via n8n + Gemini |
| 📊 **Confidence Score** | Tampilkan tingkat keyakinan prediksi dengan progress bar |
| 📱 **Responsif** | Tampilan optimal di desktop maupun HP |
| 🔒 **Privasi** | Gambar dihapus otomatis dari server setelah dianalisis |

---

## 🧵 Kelas Ulos yang Didukung (14 Kelas)

| No | Nama Ulos | No | Nama Ulos |
|----|-----------|-----|-----------|
| 1  | Antak Antak | 8  | Ragi Hidup |
| 2  | Bintang Maratur | 9  | Ragi Hotang |
| 3  | Harungguan | 10  | Runjak |
| 4  | Mangiring | 11  | Sadum |
| 5  | Marinjam Sisi | 12  | Sibolang |
| 6  | Pina Lobu Lobu | 13  | Sitolu Tuho |
| 7  | Pinussaan | 14  | Suri Suri |

---

## 🏗️ Arsitektur Sistem

```
┌─────────────────────────────────────────────────┐
│                  Browser / Client                │
│         (HTML + CSS + JS — ChatGPT-style UI)     │
└────────────────────┬───────────────┬─────────────┘
                     │ Upload Image  │ Text Chat
                     ▼               ▼
┌─────────────────────────────────────────────────┐
│              Django Web Server                   │
│  ┌──────────────────────────────────────────┐   │
│  │              GATE 1: CNN Model           │   │
│  │     MobileNetV2 (ulos_final_model.h5)    │   │
│  │   224×224 px → 14-class softmax output  │   │
│  └──────────┬───────────────────────────────┘   │
│             │                                    │
│    Score ≥ 0.71 → Ulos terdeteksi               │
│    Score 0.41–0.70 → Minta upload ulang          │
│    Score ≤ 0.40 → Bukan Ulos                     │
│             │                                    │
│  ┌──────────▼───────────────────────────────┐   │
│  │           GATE 2: Gemini AI              │   │
│  │    Identifikasi objek non-ulos via       │   │
│  │    Google Gemini 2.5 Flash (fallback)    │   │
│  └──────────────────────────────────────────┘   │
└─────────────────────────┬───────────────────────┘
                          │ n8n Webhook
                          ▼
┌─────────────────────────────────────────────────┐
│           n8n Cloud (RAG Pipeline)               │
│      Gemini AI + Knowledge Base Ulos Batak       │
└─────────────────────────────────────────────────┘
```

---

## 📦 Teknologi yang Digunakan

### Backend
- **Django 5.2** — Web framework Python
- **TensorFlow 2.15 + Keras** — Model inference CNN
- **MobileNetV2** — Arsitektur CNN (fine-tuned)
- **Google Generative AI (Gemini 2.5 Flash)** — Fallback object detection & RAG
- **Pillow** — Image preprocessing
- **NumPy** — Numerical computation
- **Requests** — HTTP ke n8n webhook

### Frontend
- **HTML5 + Vanilla CSS + JavaScript** — Tanpa framework frontend
- **Inter Font (Google Fonts)** — Typography modern
- **Font Awesome 6** — Icon set
- **Marked.js** — Render Markdown dari bot response

### Infrastruktur
- **n8n Cloud** — RAG pipeline orchestration
- **SQLite** — Database (session Django)

---

## 🚀 Cara Menjalankan

### Prasyarat

- Python 3.10 atau lebih baru
- Git
- File model `ulos_final_model.h5`, `label_classes.pkl`, dan `class_names.json` (disediakan dan diletakkan di folder `Sistem Djanggo/`)
- Berkas Service Account Google Cloud JSON (`n8n-api-keys-499101-c0270bfd2f80.json`) untuk autentikasi Google Vertex AI (Gemini)

### 1. Clone Repository

```bash
git clone https://github.com/ReinhardBatubara/TASI-2526-109_SourceCode.git
cd TASI-2526-109_SourceCode
```

### 2. Masuk ke Folder Sistem Django dan Buat Virtual Environment

```bash
cd "Sistem Djanggo"
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3. Install Dependensi

```bash
pip install -r requirements.txt
```

### 4. Letakkan File Model dan Berkas Service Account

Tempatkan berkas-berkas berikut di root folder **Sistem Djanggo/** (sejajar dengan folder `ulos_project/` dan file `requirements.txt`):

```
TASI-2526-109_SourceCode/Sistem Djanggo/
├── ulos_final_model.h5                  ← letakkan di sini
├── label_classes.pkl                    ← letakkan di sini
├── class_names.json                     ← letakkan di sini
├── n8n-api-keys-499101-c0270bfd2f80.json ← letakkan di sini
├── ulos_project/
└── requirements.txt
```

### 5. Konfigurasi Environment (.env)

Salin berkas `.env.example` menjadi `.env` di dalam folder `Sistem Djanggo/`:

```bash
cp .env.example .env
```

Buka file `.env` dan sesuaikan nilai-nilainya:
- `N8N_WEBHOOK_URL`: Isi dengan URL webhook n8n Anda untuk integrasi chatbot RAG.
- `DJANGO_SECRET_KEY`: Isi dengan secret key Django Anda.

### 6. Jalankan Server Django

Untuk menjalankan server lokal Django:

```powershell
# Pastikan venv sudah aktif, lalu pindah ke direktori utama Django project
cd ulos_project

# Jalankan server lokal Django
python manage.py runserver
```

Setelah server aktif, buka browser Anda dan akses:
👉 **[http://127.0.0.1:8000/](http://127.0.0.1:8000/)**

---

## 🎮 Cara Pemakaian

Sistem ini didesain interaktif layaknya aplikasi chat modern. Berikut adalah langkah-langkah pemakaian sistem:

1. **Unggah Foto Kain Ulos**:
   * Klik ikon **Kamera** 📷 untuk mengambil foto langsung melalui webcam/kamera HP Anda, ATAU klik area upload file untuk memilih foto kain Ulos dari penyimpanan komputer/perangkat Anda.
2. **Kirim Teks Pesan (Opsional)**:
   * Anda bisa langsung mengunggah foto kosong, atau mengetikkan pertanyaan tambahan di kolom chat sebelum menekan tombol kirim (contoh: *"Ulos apa ini dan kapan dipakainya?"*).
3. **Analisis Gambar Otomatis**:
   * **Gate 1 (CNN)** akan memproses motif gambar. Jika gambar terlalu polos/hanya satu warna, sistem akan menolak otomatis.
   * Jika sistem yakin gambar adalah Ulos (Confidence $\ge 71\%$), nama Ulos akan terdeteksi.
   * Jika gambar tidak meyakinkan atau bukan Ulos (Confidence $\le 40\%$), **Gate 2 (Gemini Fallback)** akan berjalan untuk mengidentifikasi objek tersebut dan memberikan respon ramah.
4. **Tanya Jawab Lanjutan (Chat RAG)**:
   * Setelah Ulos terdeteksi, Anda bisa melanjutkan bertanya seputar adat, aturan pemakaian, sejarah, atau cara pembuatan Ulos tersebut di kolom chat. Chatbot RAG n8n akan menjawab secara cerdas berdasarkan basis data budaya Ulos.
5. **Reset Percakapan**:
   * Klik tombol **Reset** ↻ di pojok kanan atas untuk menghapus seluruh riwayat chat dan memulai sesi deteksi baru.

---

## 📷 Contoh Input

Berikut adalah kategori skenario input gambar yang didukung oleh sistem beserta responnya:

| Kategori Input | Deskripsi / Karakteristik Gambar | Ekspektasi Respon / Output Sistem |
| :--- | :--- | :--- |
| **Ulos Valid & Jelas** | Foto kain Ulos dengan motif batak yang terlihat jelas dan pencahayaan baik (contoh: Ulos Sadum, Ulos Ragi Hotang). | ✅ Sistem berhasil mengklasifikasikan kelas Ulos (contoh: **Ulos Sadum**) dengan confidence score tinggi (misalnya **95.20%**). |
| **Bukan Ulos (Non-Ulos)** | Foto benda sehari-hari seperti hewan peliharaan, kendaraan, piring makanan, laptop, atau pakaian kaos biasa. | 🔍 Dideteksi oleh Gate 2 (Gemini fallback) yang akan merespon: *"Maaf, sistem mendeteksi gambar ini adalah [Nama Objek]. Harap unggah foto kain Ulos Batak."* |
| **Gambar Polos / Dinding** | Foto satu warna saja tanpa motif (contoh: foto kertas putih kosong, tembok biru polos, penutup lensa hitam). | ⚠️ Ditolak otomatis oleh pra-filter citra sebelum inference: *"Gambar ditolak karena terlalu polos atau hampir satu warna."* |
| **Gambar Buram / Noise** | Foto kain Ulos yang terlalu gelap, kabur (blur), berbayang ekstrem, atau tidak fokus (confidence score 41% - 70%). | 🔄 Sistem meminta foto ulang: *"Gambar yang Anda unggah buram atau kurang jelas. Silakan upload ulang dengan foto yang lebih jelas."* |

---

## 📁 Struktur Folder Proyek

Struktur folder utama repositori saat ini adalah sebagai berikut:

```
TASI-2526-109_SourceCode/ (Repository Root)
│
├── 📄 README.md                          ← Dokumentasi sistem (file ini)
│
├── 📁 Pelatihan model MobileNetV2/       ← Folder kode pelatihan model CNN
│   └── 📄 Pelatihan MobilenetV2.ipynb    ← Jupyter Notebook untuk melatih model
│
├── 📁 Sistem Djanggo/                     ← Aplikasi web Django (Ulos Simulator)
│   ├── 📄 requirements.txt                ← Dependensi Python untuk Django
│   ├── 📄 .env                            ← Konfigurasi environment (di-copy dari .env.example)
│   ├── 📄 .env.example                    ← Template konfigurasi environment
│   ├── 📄 .gitignore                      ← File yang dikecualikan dari Git
│   ├── 📄 ulos_final_model.h5             ← Model CNN MobileNetV2 (fine-tuned)
│   ├── 📄 label_classes.pkl               ← Label encoder kelas ulos
│   ├── 📄 class_names.json                ← Berkas JSON daftar nama kelas ulos
│   ├── 📄 n8n-api-keys-499101-c0270bfd2f80.json ← Service Account JSON untuk Google Vertex AI
│   ├── 📁 media/                          ← Folder media penyimpanan upload sementara
│   │
│   ├── 📁 ulos_project/                   ← Django project root
│   │   ├── 📄 manage.py                   ← Script manajemen Django
│   │   │
│   │   ├── 📁 ulos_project/               ← Folder konfigurasi Django
│   │   │   ├── settings.py
│   │   │   ├── urls.py
│   │   │   └── wsgi.py
│   │   │
│   │   └── 📁 classifier/                 ← Aplikasi utama klasifikasi
│   │       ├── views.py                   ← Logika klasifikasi + API endpoints
│   │       ├── urls.py                    ← URL routing aplikasi
│   │       ├── models.py
│   │       └── 📁 templates/
│   │           └── 📁 classifier/
│   │               └── index.html         ← UI Chat (ChatGPT-style)
│   └── ...
│
└── 📁 Sistem RAG n8n/                     ← Folder integrasi chatbot RAG n8n
    └── 📄 RAG TA 109 Final.json           ← File JSON skema alur kerja (workflow) n8n
```

*Catatan:*
- **Pelatihan model MobileNetV2**: Berisi kode Jupyter Notebook untuk melatih model klasifikasi kain Ulos.
- **Sistem RAG n8n**: Berisi berkas ekspor JSON workflow n8n untuk mengonfigurasi RAG pipeline di platform n8n.

---

## 🔬 Dataset & Model

### Model
- **Arsitektur:** MobileNetV2 (pre-trained ImageNet + fine-tuning)
- **Input Size:** 224 × 224 × 3 (RGB)
- **Output:** 14-class softmax
- **Format:** Keras HDF5 (`.h5`)
- **Preprocessing:** `mobilenet_v2.preprocess_input()` (range [-1, 1])

### Threshold Keputusan
| Rentang Score | Keputusan |
|---------------|-----------|
| ≥ 0.71 | ✅ Ulos terdeteksi (kelas top-1) |
| 0.41 – 0.70 | ⚠️ Gambar buram / belum jelas |
| ≤ 0.40 | ❌ Bukan ulos → Gemini fallback |

### Filter Gambar Polos
Sistem menolak gambar yang terlalu polos (satu warna) sebelum di-infer:
- Pixel std < 8.0
- Unique colors ≤ 10
- Edge strength < 2.0

---

## 🔑 Variabel Konfigurasi Penting

File: `Sistem Djanggo/ulos_project/classifier/views.py` atau di berkas `.env`

| Variabel / Konfigurasi | Sumber | Keterangan | Default / Nilai |
|-------------------------|--------|------------|-----------------|
| `N8N_WEBHOOK_URL` | `.env` / `views.py` | URL webhook n8n untuk chatbot RAG | URL dari `.env` / Cloud |
| `SERVICE_ACCOUNT_PATH` | `views.py` | Path ke berkas JSON Service Account GCP | `../n8n-api-keys-499101-c0270bfd2f80.json` |
| `GEMINI_MODEL_NAME` | `views.py` | Nama model Google Gemini di Vertex AI | `gemini-2.5-flash` |
| `NOT_ULOS_THRESHOLD` | `views.py` | Batas bawah score keyakinan ulos | `0.40` |
| `VALID_ULOS_THRESHOLD` | `views.py` | Batas atas keyakinan valid | `0.71` |
| `TOP_K` | `views.py` | Jumlah prediksi kelas teratas | `3` |

---

## 🛡️ Privasi & Keamanan

- 📌 Gambar yang diunggah **dihapus otomatis** dari server setelah analisis selesai (`try...finally`)
- 📌 Gambar dikembalikan ke client sebagai **Base64 Data URL** (tidak disimpan di server)
- 📌 API key Gemini tidak di-hardcode di frontend
- 📌 CSRF protection aktif di semua form POST

---

## 👨‍💻 Pengembang

**Reinhard Batubara**
- GitHub: [@LennaFebriana](https://github.com/lennafebriana12)

---

## 📄 Lisensi

Proyek ini dibuat untuk keperluan **Tugas Akhir / Skripsi**.  
Penggunaan ulang kode diperbolehkan dengan menyertakan atribusi.

---

<div align="center">
  <sub>Dibuat dengan ❤️ untuk melestarikan budaya kain Ulos Batak</sub>
</div>
