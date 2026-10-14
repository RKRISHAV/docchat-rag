import hashlib
import os
import streamlit as st
from rag import extract, Retriever, answer

st.set_page_config(page_title='DocChat AI | Document Intelligence',page_icon='📚',layout='wide',initial_sidebar_state='expanded')
st.markdown('''<style>
.stApp{background:#0b1020;color:#e7edf8}
[data-testid="stSidebar"]{background:#121a2e}
.block-container{padding-top:2rem;max-width:1180px}
.hero{padding:22px 28px;background:linear-gradient(120deg,#18294b,#14213c);border:1px solid #314366;border-radius:18px;margin-bottom:20px}
.hero h1{font-size:2.1rem;margin:0;color:#f5f8ff}.hero p{color:#b9c7e4;margin:9px 0 0}
.note{font-size:.87rem;color:#aab8d5}
[data-testid="stChatMessage"]{border:1px solid #293855;border-radius:15px;background:#121b30}
</style>''',unsafe_allow_html=True)

def get_key():
    key=os.environ.get('GEMINI_API_KEY','')
    try: key=st.secrets.get('GEMINI_API_KEY',key)
    except (FileNotFoundError,KeyError): pass
    return key

@st.cache_resource(show_spinner=False)
def create_client(key):
    from google import genai
    return genai.Client(api_key=key)

@st.cache_resource(show_spinner='Indexing your documents…')
def make_index(signature,files,api_key,semantic):
    passages=[]; failures=[]
    for name,data in files:
        try: passages.extend(extract(name,data))
        except Exception as e: failures.append(f'{name}: {e}')
    if not passages:return None,passages,failures
    client=create_client(api_key) if api_key and semantic else None
    return Retriever(passages,client),passages,failures

st.markdown('<div class="hero"><h1>📚 DocChat AI</h1><p>Ask your documents anything. Grounded answers, semantic retrieval, transparent sources.</p></div>',unsafe_allow_html=True)
with st.sidebar:
    st.header('⚙️ Workspace settings')
    api_key=get_key()
    if api_key: st.success('Gemini API key configured')
    else: st.warning('Gemini API key missing — evidence mode available')
    ai_enabled=st.toggle('AI-generated answers',value=bool(api_key),disabled=not bool(api_key))
    semantic=st.toggle('Gemini semantic search',value=bool(api_key),disabled=not bool(api_key))
    model=st.selectbox('Gemini answer model',['gemini-2.5-flash','gemini-2.5-flash-lite'],index=0)
    top_k=st.slider('Source passages',1,8,4)
    st.divider()
    st.caption('Your documents are processed in this session. Gemini mode sends retrieved passages and questions to Google. Do not upload sensitive documents without permission.')
    if st.button('🗑️ Clear chat',use_container_width=True):
        st.session_state.messages=[]
        st.rerun()

files=st.file_uploader('Upload PDF, DOCX or TXT documents',type=['pdf','docx','txt'],accept_multiple_files=True)
if 'messages' not in st.session_state:st.session_state.messages=[]
if files:
    data=[(f.name,f.getvalue()) for f in files]
    sig=hashlib.sha256(b''.join(n.encode()+hashlib.sha256(b).digest() for n,b in data)).hexdigest()
    if st.session_state.get('doc_signature')!=sig:
        st.session_state.doc_signature=sig
        st.session_state.messages=[]
    retriever,passages,failures=make_index(sig,tuple(data),api_key if semantic else '',semantic)
    if failures:
        for failure in failures: st.warning(f'Could not read {failure}')
    if retriever:
        a,b,c=st.columns(3)
        a.metric('Documents',len(files));b.metric('Searchable passages',len(passages));c.metric('Retrieval',retriever.mode)
        if getattr(retriever,'embedding_warning',None):st.warning('Semantic embeddings unavailable; using local TF-IDF instead. '+retriever.embedding_warning[:250])
        with st.expander('📄 View extracted text'):
            for p in passages[:12]:st.markdown(f'**{p.filename} · page {p.page}**');st.write(p.text)
    else: st.error('No readable text found. Scanned/image-only PDFs require OCR, which is not included.')
else:
    retriever=None
    st.info('👆 Upload one or more documents to start asking questions.')

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])
        if message.get('hits'):
            with st.expander('🔎 Evidence and page citations'):
                for i,(p,score) in enumerate(message['hits'],1):
                    st.markdown(f'**[{i}] {p.filename} — page {p.page}** · relevance {score:.2f}')
                    st.caption(p.text)

question=st.chat_input('Ask a question about your documents…',disabled=retriever is None)
if question and retriever:
    st.session_state.messages.append({'role':'user','content':question})
    with st.chat_message('user'):st.markdown(question)
    with st.chat_message('assistant'):
        with st.spinner('Finding supporting passages…'):
            hits=retriever.search(question,top_k)
            if ai_enabled:
                try: reply=answer(create_client(api_key),model,question,hits)
                except Exception as e:reply=f'⚠️ Gemini request failed: {e}. You can disable AI answers to inspect the evidence.'
            else:
                reply='**Relevant evidence from your documents:**\n\n'+'\n\n'.join(f'**[{i}] {p.filename}, page {p.page}:** {p.text}' for i,(p,_) in enumerate(hits,1))
        st.markdown(reply)
        with st.expander('🔎 Evidence and page citations'):
            for i,(p,score) in enumerate(hits,1):
                st.markdown(f'**[{i}] {p.filename} — page {p.page}** · relevance {score:.2f}')
                st.caption(p.text)
    st.session_state.messages.append({'role':'assistant','content':reply,'hits':hits})
