# DocChat AI — Gemini RAG

Streamlit app for document Q&A with Google Gemini, semantic embeddings, source citations, and evidence-only fallback.

## Run locally (Windows CMD)

```cmd
cd /d "C:\New Projects\docchat_rag"
python -m pip install -r requirements.txt
mkdir .streamlit
```

Create `.streamlit/secrets.toml` containing:

```toml
GEMINI_API_KEY = "YOUR_KEY_HERE"
```

Then run `python -m streamlit run app.py`.

Get an API key at https://aistudio.google.com/apikey . API use is subject to Google's current pricing, quotas and regional availability.

## Upgrade your existing Streamlit Community Cloud site

1. Back up the old GitHub repository if desired.
2. Replace existing `app.py`, `requirements.txt`, `README.md`, and `sample_notes.txt` with the files from this ZIP. Add `rag.py`, `.gitignore`, `.streamlit/config.toml`, `.streamlit/secrets.toml.example`, and `tests/test_rag.py`.
3. **Never upload `.streamlit/secrets.toml` or a real API key to GitHub.**
4. Commit to the same GitHub repository and `main` branch. Streamlit Cloud automatically redeploys it.
5. At https://share.streamlit.io open your app → Settings → Secrets and enter:

```toml
GEMINI_API_KEY = "YOUR_REAL_API_KEY"
```

6. Save, reboot the app if needed, upload `sample_notes.txt`, and ask "Where did the candidate work?".

## How it works

1. PDF/DOCX/TXT text extraction; PDFs retain page numbers.
2. Clean and split into overlapping passages.
3. Use `gemini-embedding-001` for semantic document and query vectors. If unavailable, fall back to local TF-IDF.
4. Rank passages by cosine similarity; take the top K.
5. Send only the question and retrieved passages to Gemini for grounded answer generation.
6. Display the answer and inspectable filename/page evidence.

## Limitations and privacy

- Gemini API is a third-party service; uploaded passages and questions are sent to Google when AI is enabled, and document chunks are sent for Gemini embeddings when semantic search is enabled.
- This public demo has no user authentication or per-user quotas. Do not expose a billable key on a public site without usage safeguards; monitor quotas and billing.
- Sessions and indexes are held in server memory, not persistent storage. Cached indexes may remain in the app process until cleared/restarted.
- Scanned PDFs need OCR (not included). DOCX and TXT have no reliable page metadata, so they show page 1.
- Retrieval can miss information; citation references are to retrieved evidence, not a guarantee of factual correctness.
- If Gemini's model identifiers or quotas change, update the model name in `app.py` and/or `rag.py`.

## Test

`python -m pytest -q`
