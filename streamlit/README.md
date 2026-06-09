# Streamlit Public Statement Extractor

Aplikasi ini mengekstrak pernyataan tokoh publik dari artikel berita bahasa Indonesia menggunakan checkpoint `simplernn_crf_model_state.pt`.

## Cara menjalankan

1. Pastikan file berikut berada dalam folder yang sama:
   - `app.py`
   - `simplernn_crf_model_state.pt`
   - `requirements.txt`
2. Install dependency:

```bash
pip install -r requirements.txt
```

3. Jalankan aplikasi:

```bash
streamlit run app.py
```

Aplikasi juga menyediakan opsi upload file `.pt` melalui sidebar.
