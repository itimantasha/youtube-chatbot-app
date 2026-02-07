import os
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="YouTube RAG Bot", layout="wide")
st.title("🎥 YouTube Video Q&A Bot (RAG)")
st.write("Ask questions directly from any YouTube video's transcript.")

# -----------------------------
# API Key from Render ENV
# -----------------------------
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OPENAI_API_KEY not found! Set it in Render Environment Variables.")
    st.stop()
os.environ["OPENAI_API_KEY"] = api_key

# -----------------------------
# Inputs
# -----------------------------
video_id = st.text_input("YouTube Video ID", placeholder="Gfr50f6ZBvo")
question = st.text_input("Ask your question")

# -----------------------------
# Core Pipeline Functions
# -----------------------------
@st.cache_data(show_spinner=False)
def fetch_transcript(video_id):
    api = YouTubeTranscriptApi()
    transcript_list = api.fetch(video_id, languages=["en"])
    transcript = " ".join(chunk.text for chunk in transcript_list)
    return transcript

@st.cache_resource(show_spinner=False)
def build_vector_store(transcript):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.create_documents([transcript])
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    store = FAISS.from_documents(docs, embeddings)
    return store

def answer_question(store, question):
    retriever = store.as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke(question)
    context = "\n\n".join(d.page_content for d in docs)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    prompt = f"""
Answer ONLY from the context.
If answer not present say 'Not found in video'.

Context:
{context}

Question: {question}
"""
    result = llm.invoke(prompt)
    return result.content

# -----------------------------
# Run Button
# -----------------------------
if st.button("Run"):
    if not video_id or not question:
        st.error("Enter video ID and question")
        st.stop()

    try:
        with st.spinner("Fetching transcript..."):
            transcript = fetch_transcript(video_id)

        with st.spinner("Creating embeddings..."):
            store = build_vector_store(transcript)

        with st.spinner("Thinking..."):
            answer = answer_question(store, question)

        st.success("Answer:")
        st.write(answer)

    except TranscriptsDisabled:
        st.error("Transcripts disabled for this video")
    except Exception as e:
        st.error(str(e))
