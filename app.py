import io
import os
import re
import hashlib
import requests
import streamlit as st
from pypdf import PdfReader
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title='DocChat | RAG Q&A', page_icon='📚', layout='wide')
st.title('📚 DocChat — Ask your documents')
st.caption('Upload PDFs, DOCX or TXT files. Answers use retrieved passages from your uploaded files.')


def extract(upload):
    data = upload.getvalue()
    if upload.name.lower().endswith('.pdf'):
        reader = PdfReader(io.BytesIO(data))
        return [(f'{upload.name} · page {i+1}', page.extract_text() or '') for i, page in enumerate(reader.pages)]
    if upload.name.lower().endswith('.docx'):
        doc = Document(io.BytesIO(data))
        return [(upload.name, '\n'.join(p.text for p in doc.paragraphs))]
    return [(upload.name, data.decode('utf-8', errors='replace'))]


def chunk_text(text, size=900, overlap=160):
    text = re.sub(r'\s+', ' ', text).strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            cut = text.rfind(' ', start + size // 2, end)
            if cut > start:
                end = cut
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start = max(start + 1, end - overlap)
    return chunks


def build_index(files):
    passages = []
    for f in files:
        for source, text in extract(f):
            for chunk in chunk_text(text):
                passages.append({'source': source, 'text': chunk})
    if not passages:
        return [], None, None
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), sublinear_tf=True)
    matrix = vectorizer.fit_transform([p['text'] for p in passages])
    return passages, vectorizer, matrix


def retrieve(question, passages, vectorizer, matrix, k=4):
    scores = cosine_similarity(vectorizer.transform([question]), matrix).ravel()
    indices = scores.argsort()[::-1][:k]
    # Always return the best passages, even if lexical similarity is zero.
    # This avoids falsely claiming a document has no answer for pronoun-based questions.
    return [(passages[i], float(scores[i])) for i in indices]


def ollama_answer(question, matches, model, url):
    context = '\n\n'.join(f'[Source {i+1}: {p["source"]}]\n{p["text"]}' for i, (p, _) in enumerate(matches))
    prompt = ('You are a document-grounded assistant. Answer ONLY from the supplied context. '
              'If the context does not contain the answer, say "I could not find that in the uploaded documents." '
              'Do not use outside knowledge. Cite source numbers like [Source 1].\n\n'
              f'CONTEXT:\n{context}\n\nQUESTION: {question}\nANSWER:')
    endpoint = url.rstrip('/') + '/api/generate'
    response = requests.post(endpoint, json={'model': model, 'prompt': prompt, 'stream': False, 'options': {'temperature': 0}}, timeout=180)
    response.raise_for_status()
    return response.json()['response']

with st.sidebar:
    st.header('Settings')
    mode = st.radio('Answer mode', ['Local AI (Ollama)', 'Evidence only (no AI required)'], index=0)
    model = st.text_input('Ollama model', value='llama3.2:3b')
    url = st.text_input('Ollama server', value=os.environ.get('OLLAMA_URL', 'http://localhost:11434'))
    top_k = st.slider('Passages to retrieve', 1, 8, 4)
    st.info('Evidence-only mode works without downloading an AI model. Local AI mode needs Ollama installed and a model pulled.')

uploads = st.file_uploader('Upload one or more documents', type=['pdf', 'docx', 'txt'], accept_multiple_files=True)
if uploads:
    fingerprint = hashlib.sha256(b''.join(f.name.encode() + f.getvalue() for f in uploads)).hexdigest()
    if st.session_state.get('fingerprint') != fingerprint:
        try:
            with st.spinner('Reading and indexing documents...'):
                passages, vectorizer, matrix = build_index(uploads)
            st.session_state.update(fingerprint=fingerprint, passages=passages, vectorizer=vectorizer, matrix=matrix, history=[])
        except Exception as exc:
            st.error(f'Unable to index documents: {exc}')
            st.stop()
    passages = st.session_state['passages']
    if not passages:
        st.warning('No selectable text was found. Scanned/image-only PDFs require OCR, which this version does not include.')
        st.stop()
    st.success(f'Indexed {len(uploads)} file(s) into {len(passages)} searchable passages.')
    with st.expander('Check extracted PDF text'):
        st.text_area('Extracted text preview', '\n\n'.join(p['text'] for p in passages)[:5000], height=220)
        st.caption('If the name or experience is missing here, the PDF extraction needs improvement.')
    for item in st.session_state['history']:
        with st.chat_message('user'):
            st.write(item['question'])
        with st.chat_message('assistant'):
            st.write(item['answer'])
            with st.expander('Evidence and sources'):
                for p, score in item['matches']:
                    st.markdown(f'**{p["source"]}** · similarity {score:.3f}')
                    st.write(p['text'])
    question = st.chat_input('Ask a question about your documents...')
    if question:
        matches = retrieve(question, passages, st.session_state['vectorizer'], st.session_state['matrix'], top_k)
        if not matches:
            answer = 'I could not find that in the uploaded documents.'
        elif mode == 'Evidence only (no AI required)':
            answer = 'Most relevant passages from your documents:\n\n' + '\n\n'.join(
                f'**[{i+1}] {p["source"]}**\n\n{p["text"]}' for i, (p, _) in enumerate(matches))
        else:
            try:
                with st.spinner('Generating a grounded answer...'):
                    answer = ollama_answer(question, matches, model, url)
            except requests.RequestException as exc:
                answer = ('Could not connect to Ollama or generate an answer. Check that Ollama is running '
                          f'and the model `{model}` is installed. You can switch to Evidence-only mode.\n\nDetails: {exc}')
        st.session_state['history'].append({'question': question, 'answer': answer, 'matches': matches})
        st.rerun()
else:
    st.info('Upload a document to start asking questions.')
