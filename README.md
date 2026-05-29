# Prototype Peer Code Review Feedback Alignment

Aplikasi Streamlit ini menggambarkan prototype integrasi model klasifikasi single-label ke dalam peer code review tool.

## Tujuan

Mahasiswa mengisi komentar pada 6 field rubrik:
1. Variable Names
2. Expressions
3. Control Flow
4. Comments
5. Layout and Formatting
6. Decomposition

Setelah submit, sistem memprediksi label setiap komentar dan mengecek apakah komentar sudah sesuai dengan field rubrik yang diisi.

## File model yang dibutuhkan

Letakkan file berikut dalam folder yang sama dengan `streamlit_app.py`:

```text
RF_Count_Vectors.sav
count_vect_model.sav
encoder.sav
stopword.txt
```

Jika file model belum lengkap, aplikasi tetap dapat berjalan dalam mode prototype/demo, tetapi prediksi bukan berasal dari model ML.

## Menjalankan aplikasi

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```
