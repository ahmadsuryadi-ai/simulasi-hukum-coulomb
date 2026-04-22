# Coulomb Studio 2D v4

Aplikasi Streamlit untuk pembelajaran fisika materi listrik statis, khususnya Hukum Coulomb, pada level mahasiswa sarjana.

## Fitur
- Multi-muatan 2D, 2 sampai 8 muatan
- Gaya resultan pada setiap muatan
- Medan listrik total di titik uji
- Potensial listrik total di titik uji
- Peta potensial dan arah medan
- Praktikum virtual dan ekspor CSV
- Tugas computational thinking
- Kuis konsep
- Opsi kirim data ke Google Sheet

## Jalankan lokal
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy ke Streamlit Cloud
1. Upload semua file ke GitHub
2. Buka Streamlit Community Cloud
3. Pilih repository dan file `app.py`
4. Jika ingin konek ke Google Sheet, isi secrets sesuai template pada aplikasi

## Struktur repo
```text
simulasi-hukum-coulomb/
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```
