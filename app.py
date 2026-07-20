import os
import sqlite3
import tempfile
import numpy as np
import pymupdf
import faiss
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
from gtts import gTTS
from duckduckgo_search import DDGS

load_dotenv()

st.set_page_config(
    page_title="DocChat AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed"
)

api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")
client = Groq(api_key=api_key)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
* { font-family: 'Inter', sans-serif; }
.stApp { background: #0f0f1a; color: #ffffff; }
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.hero { text-align: center; padding: 2rem 0 1rem 0; }
.hero-title {
    font-size: 3rem; font-weight: 800;
    background: linear-gradient(90deg, #00d2ff, #7b2ff7, #ff6b6b);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin-bottom: 0.3rem;
}
.hero-sub { color: #666; font-size: 0.9rem; letter-spacing: 2px; text-transform: uppercase; }
.feature-card {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border: 1px solid #ffffff11; border-radius: 16px;
    padding: 1.5rem; text-align: center; margin: 0.5rem 0;
}
.feature-icon { font-size: 2rem; margin-bottom: 0.5rem; }
.feature-label { font-size: 0.85rem; color: #aaa; font-weight: 600; letter-spacing: 1px; }
.stat-box {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border: 1px solid #00d2ff33; border-radius: 12px;
    padding: 1rem; text-align: center; margin: 0.3rem 0;
}
.stat-num { font-size: 1.8rem; font-weight: 800; color: #00d2ff; }
.stat-lbl { font-size: 0.75rem; color: #666; text-transform: uppercase; letter-spacing: 1px; }
.summary-card {
    background: linear-gradient(135deg, #0d1a0d, #1a2a1a);
    border: 1px solid #00ff4433; border-radius: 16px;
    padding: 1.5rem; margin: 1rem 0; color: #90ee90;
}
.summary-head {
    color: #00ff88; font-weight: 700; font-size: 0.85rem;
    text-transform: uppercase; letter-spacing: 2px; margin-bottom: 1rem;
}
.user-bubble {
    background: linear-gradient(135deg, #7b2ff7, #5a1fd1);
    border-radius: 20px 20px 4px 20px; padding: 1rem 1.2rem;
    margin: 0.5rem 0; margin-left: 20%; color: white;
    box-shadow: 0 4px 20px rgba(123,47,247,0.3);
}
.ai-bubble {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border: 1px solid #00d2ff22; border-radius: 20px 20px 20px 4px;
    padding: 1rem 1.2rem; margin: 0.5rem 0; margin-right: 20%;
    color: #e0e0e0; box-shadow: 0 4px 20px rgba(0,210,255,0.1);
}
.cite-tag {
    display: inline-block; background: #00d2ff22;
    border: 1px solid #00d2ff44; border-radius: 20px;
    padding: 0.2rem 0.8rem; font-size: 0.72rem; color: #00d2ff; margin: 0.2rem;
}
.web-tag {
    display: inline-block; background: #ff6b6b22;
    border: 1px solid #ff6b6b44; border-radius: 20px;
    padding: 0.2rem 0.8rem; font-size: 0.72rem; color: #ff6b6b; margin: 0.2rem;
}
.web-source {
    background: linear-gradient(135deg, #1a1020, #2a1a2e);
    border: 1px solid #ff6b6b22; border-radius: 12px;
    padding: 0.8rem 1rem; margin: 0.3rem 0; font-size: 0.82rem;
}
.web-source-title { color: #ff6b6b; font-weight: 600; margin-bottom: 0.2rem; }
.web-source-url { color: #888; font-size: 0.75rem; }
.source-section {
    border-top: 1px solid #ffffff11; margin-top: 0.8rem; padding-top: 0.8rem;
}
.upload-zone {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border: 2px dashed #00d2ff44; border-radius: 16px;
    padding: 2rem; text-align: center; margin: 1rem 0;
}
div[data-testid="stTextInput"] input {
    background: #1a1a2e !important; border: 1px solid #00d2ff44 !important;
    border-radius: 30px !important; color: white !important;
    padding: 0.8rem 1.5rem !important; font-size: 0.95rem !important;
}
div[data-testid="stButton"] button {
    background: linear-gradient(90deg, #00d2ff, #7b2ff7) !important;
    border: none !important; border-radius: 30px !important;
    color: white !important; font-weight: 700 !important;
    padding: 0.7rem 1.5rem !important; width: 100% !important;
}
.stTabs [data-baseweb="tab-list"] {
    background: #1a1a2e; border-radius: 12px; padding: 4px; gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent; border-radius: 8px; color: #888; font-weight: 600;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg, #00d2ff22, #7b2ff722) !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)


# ── Database ─────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect("chat_history.db")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT, document_name TEXT,
            question TEXT, answer TEXT,
            doc_citations TEXT, web_sources TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_chat(doc_name, question, answer, doc_cites, web_sources):
    conn = sqlite3.connect("chat_history.db")
    conn.execute(
        "INSERT INTO chat_history VALUES (NULL,?,?,?,?,?,?)",
        (datetime.now().isoformat(), doc_name, question, answer,
         ", ".join(doc_cites), str(web_sources))
    )
    conn.commit()
    conn.close()

def get_history(doc_name):
    conn = sqlite3.connect("chat_history.db")
    rows = conn.execute(
        """SELECT question, answer, doc_citations, web_sources, timestamp
           FROM chat_history WHERE document_name=?
           ORDER BY id DESC LIMIT 30""",
        (doc_name,)
    ).fetchall()
    conn.close()
    return rows


# ── RAG Functions ─────────────────────────────────────────
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

def extract_text(pdf_path):
    doc = pymupdf.open(pdf_path)
    pages, count = [], len(doc)
    for i, page in enumerate(doc):
        t = page.get_text()
        if t.strip():
            pages.append({"page": i+1, "text": t})
    doc.close()
    return pages, count

def chunk_pages(pages, size=200):
    chunks = []
    for p in pages:
        words = p["text"].split()
        for i in range(0, len(words), size):
            c = " ".join(words[i:i+size])
            if c.strip():
                chunks.append({"text": c, "page": p["page"]})
    return chunks

def build_index(chunks, embedder):
    embs = np.array(embedder.encode([c["text"] for c in chunks])).astype("float32")
    idx = faiss.IndexFlatL2(embs.shape[1])
    idx.add(embs)
    return idx

def search_docs(query, idx, chunks, embedder, k=3):
    q = np.array(embedder.encode([query])).astype("float32")
    _, ids = idx.search(q, k)
    return [chunks[i] for i in ids[0] if i < len(chunks)]

def summarize_doc(pages):
    text = " ".join(p["text"] for p in pages)[:3000]
    r = client.chat.completions.create(
        model="llama-3.3-70b-versatile", temperature=0.3,
        messages=[
            {"role": "system", "content": "Summarize this document in 5 bullet points. Be concise."},
            {"role": "user", "content": text}
        ]
    )
    return r.choices[0].message.content


# ── Web Search ────────────────────────────────────────────
def web_search(query, max_results=3):
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        return results
    except Exception as e:
        return []


# ── Hybrid Answer ─────────────────────────────────────────
def ask_hybrid(question, idx, chunks, embedder, use_web=True):
    # 1. Search document
    doc_hits = search_docs(question, idx, chunks, embedder)
    doc_context = "\n".join(
        f"[Document - Page {h['page']}]: {h['text']}" for h in doc_hits
    )
    doc_citations = list(set(f"Page {h['page']}" for h in doc_hits))

    # 2. Search web
    web_results = []
    web_context = ""
    if use_web:
        web_results = web_search(question)
        if web_results:
            web_context = "\n".join(
                f"[Web Source - {r.get('title', 'Unknown')}]: {r.get('body', '')}"
                for r in web_results
            )

    # 3. Build combined prompt
    combined_context = doc_context
    if web_context:
        combined_context += f"\n\n--- WEB SEARCH RESULTS ---\n{web_context}"

    system_prompt = """You are an expert research assistant with access to both a PDF document and live web search results.

Your job is to give the most comprehensive, accurate answer possible by combining BOTH sources.

Structure your answer as follows:
1. Start with what the document says (cite the page number)
2. Then expand with additional context from web search results
3. End with a brief synthesis combining both sources

Always be clear about which information comes from the document vs the web.
If the document doesn't cover something, say so and rely on web results.
If web results don't add anything new, say the document is sufficient."""

    r = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=0.4,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{combined_context}\n\nQuestion: {question}"}
        ]
    )

    return r.choices[0].message.content, doc_citations, web_results


def tts(text):
    g = gTTS(text[:400], lang="en")
    f = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    g.save(f.name)
    return f.name


# ── Main App ──────────────────────────────────────────────
def main():
    init_db()
    embedder = load_embedder()

    for k, v in [("messages", []), ("idx", None), ("chunks", None),
                  ("doc_name", None), ("summary", None), ("stats", {})]:
        if k not in st.session_state:
            st.session_state[k] = v

    # Hero
    st.markdown("""
    <div class="hero">
        <div class="hero-title">📚 DocChat AI</div>
        <div class="hero-sub">
            Document RAG + Live Web Search &nbsp;·&nbsp; Built during MTN AI Internship
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    tab1, tab2, tab3 = st.tabs(["📤  Upload & Process", "💬  Chat", "🕘  History"])

    # ═══ TAB 1 — UPLOAD ══════════════════════════════════
    with tab1:
        c1, c2 = st.columns([3, 2], gap="large")

        with c1:
            st.markdown("### Upload Your PDFs")
            st.markdown('<div class="upload-zone">', unsafe_allow_html=True)
            files = st.file_uploader(
                "Drop PDFs here",
                type=["pdf"],
                accept_multiple_files=True,
                label_visibility="collapsed"
            )
            st.markdown('</div>', unsafe_allow_html=True)

            if files:
                st.markdown(f"**{len(files)} file(s) selected:**")
                for f in files:
                    st.markdown(f"✅ {f.name}")

                if st.button("🚀 Process & Summarize Documents"):
                    all_pages, all_chunks, names = [], [], []
                    bar = st.progress(0, text="Starting...")

                    for i, f in enumerate(files):
                        bar.progress(i / len(files), text=f"Reading {f.name}...")
                        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                        tmp.write(f.getbuffer())
                        tmp.close()
                        pages, _ = extract_text(tmp.name)
                        chunks = chunk_pages(pages)
                        all_pages.extend(pages)
                        all_chunks.extend(chunks)
                        names.append(f.name)
                        os.unlink(tmp.name)

                    bar.progress(0.8, text="Building vector database...")
                    idx = build_index(all_chunks, embedder)

                    bar.progress(0.9, text="Generating AI summary...")
                    summary = summarize_doc(all_pages)

                    st.session_state.idx = idx
                    st.session_state.chunks = all_chunks
                    st.session_state.doc_name = ", ".join(names)
                    st.session_state.summary = summary
                    st.session_state.messages = []
                    st.session_state.stats = {
                        "docs": len(files), "pages": len(all_pages),
                        "chunks": len(all_chunks), "vectors": idx.ntotal
                    }
                    bar.progress(1.0, text="Done!")
                    st.success("✅ Documents ready! Go to the Chat tab.")

        with c2:
            st.markdown("### Capabilities")
            for icon, label in [
                ("🔍", "Semantic Document Search"),
                ("🌐", "Live Web Search"),
                ("📖", "Page Citations"),
                ("🔗", "Web References & URLs"),
                ("🎤", "Voice Responses"),
                ("💾", "Chat History"),
            ]:
                st.markdown(f"""
                <div class="feature-card">
                    <div class="feature-icon">{icon}</div>
                    <div class="feature-label">{label}</div>
                </div>
                """, unsafe_allow_html=True)

            if st.session_state.stats:
                st.markdown("### Document Stats")
                s = st.session_state.stats
                ca, cb = st.columns(2)
                for col, num, lbl in [
                    (ca, s["docs"], "Docs"), (cb, s["pages"], "Pages"),
                    (ca, s["chunks"], "Chunks"), (cb, s["vectors"], "Vectors")
                ]:
                    with col:
                        st.markdown(f'<div class="stat-box"><div class="stat-num">{num}</div><div class="stat-lbl">{lbl}</div></div>', unsafe_allow_html=True)

    # ═══ TAB 2 — CHAT ════════════════════════════════════
    with tab2:
        if st.session_state.idx is None:
            st.info("Upload and process your PDFs in the Upload tab first.")
        else:
            if st.session_state.summary:
                st.markdown(f"""
                <div class="summary-card">
                    <div class="summary-head">📋 Document Summary</div>
                    {st.session_state.summary}
                </div>
                """, unsafe_allow_html=True)

            st.markdown(f"*Chatting with:* **{st.session_state.doc_name}**")
            st.divider()

            opt1, opt2, opt3 = st.columns(3)
            with opt1:
                voice = st.toggle("🎤 Voice Responses", value=False)
            with opt2:
                use_web = st.toggle("🌐 Web Search", value=True)
            with opt3:
                if use_web:
                    st.markdown('<p style="color:#ff6b6b; font-size:0.8rem; margin-top:0.6rem;">✅ AI will search the internet for deeper answers</p>', unsafe_allow_html=True)
                else:
                    st.markdown('<p style="color:#888; font-size:0.8rem; margin-top:0.6rem;">Document only mode</p>', unsafe_allow_html=True)

            col_clr, col_new = st.columns(2)
            with col_clr:
                if st.button("🗑️ Clear Chat"):
                    st.session_state.messages = []
                    st.rerun()
            with col_new:
                if st.button("📂 New Document"):
                    for k in ["idx", "chunks", "doc_name", "summary", "stats"]:
                        st.session_state[k] = None if k != "stats" else {}
                    st.session_state.messages = []
                    st.rerun()

            st.divider()

            # Display messages
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    st.markdown(
                        f'<div class="user-bubble">👤 &nbsp;{msg["content"]}</div>',
                        unsafe_allow_html=True
                    )
                else:
                    body = msg["content"].replace("\n", "<br>")

                    # Document citations
                    doc_tags = "".join(
                        f'<span class="cite-tag">📄 {c}</span>'
                        for c in msg.get("doc_citations", [])
                    )

                    # Web source tags
                    web_tags = "".join(
                        f'<span class="web-tag">🌐 {r.get("title", "Web")[:30]}...</span>'
                        for r in msg.get("web_results", [])
                    )

                    sources_html = ""
                    if doc_tags or web_tags:
                        sources_html = f'<div class="source-section"><strong style="color:#aaa; font-size:0.75rem;">SOURCES:</strong><br>{doc_tags}{web_tags}</div>'

                    st.markdown(
                        f'<div class="ai-bubble">🤖 &nbsp;{body}{sources_html}</div>',
                        unsafe_allow_html=True
                    )

                    # Show web sources as expandable cards
                    if msg.get("web_results"):
                        with st.expander(f"🌐 {len(msg['web_results'])} Web References"):
                            for r in msg["web_results"]:
                                st.markdown(f"""
                                <div class="web-source">
                                    <div class="web-source-title">{r.get('title', 'Unknown')}</div>
                                    <div style="color:#ccc; font-size:0.82rem; margin: 0.3rem 0;">{r.get('body', '')[:200]}...</div>
                                    <div class="web-source-url">🔗 {r.get('href', '')}</div>
                                </div>
                                """, unsafe_allow_html=True)

                    if voice and msg.get("audio"):
                        st.audio(msg["audio"], format="audio/mp3")

            st.divider()

            # Input area
            q_col, btn_col = st.columns([5, 1])
            with q_col:
                question = st.text_input(
                    "q", label_visibility="collapsed",
                    placeholder="Ask anything — I'll check your document AND search the web...",
                    key="q_input"
                )
            with btn_col:
                send = st.button("Send")

            if send and question.strip():
                st.session_state.messages.append({"role": "user", "content": question})

                search_msg = "Searching document and web..." if use_web else "Searching document..."
                with st.spinner(search_msg):
                    answer, doc_cites, web_results = ask_hybrid(
                        question,
                        st.session_state.idx,
                        st.session_state.chunks,
                        embedder,
                        use_web=use_web
                    )

                audio = tts(answer) if voice else None
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "doc_citations": doc_cites,
                    "web_results": web_results,
                    "audio": audio
                })
                save_chat(
                    st.session_state.doc_name, question,
                    answer, doc_cites, web_results
                )
                st.rerun()

    # ═══ TAB 3 — HISTORY ═════════════════════════════════
    with tab3:
        if not st.session_state.doc_name:
            st.info("No document loaded yet.")
        else:
            st.markdown(f"### Chat History — *{st.session_state.doc_name}*")
            rows = get_history(st.session_state.doc_name)
            if not rows:
                st.info("No history saved yet.")
            else:
                for q, a, doc_c, web_s, t in rows:
                    with st.expander(f"🕘 {t[:19]}  —  {q[:60]}..."):
                        st.markdown(f"**Question:** {q}")
                        st.markdown(f"**Answer:** {a[:300]}...")
                        st.markdown(f"**Document Sources:** {doc_c}")

    st.divider()
    st.markdown(
        '<p style="text-align:center; color:#444; font-size:0.8rem;">'
        'Built by <strong style="color:#00d2ff;">Muhammed Adetunji Ibraheem</strong> '
        '· MTN AI Internship · '
        '<a href="https://github.com/Aim2443456" style="color:#7b2ff7;">GitHub</a> · '
        '<a href="https://www.linkedin.com/in/muhammed-ibraheem-ab340a239" style="color:#00d2ff;">LinkedIn</a>'
        '</p>',
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()