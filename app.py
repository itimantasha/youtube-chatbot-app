import os
import streamlit as st

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

# -----------------------
# Page Config
# -----------------------
st.set_page_config(page_title="YouTube RAG App", layout="wide")
st.title("🎥 YouTube Video Q&A (RAG App)")

st.write("Ask questions from YouTube videos. If YouTube is blocked, upload a transcript file instead.")

# -----------------------
# API Key
# -----------------------
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("OPENAI_API_KEY not found. Set it in Render Environment Variables.")
    st.stop()

# -----------------------
# Inputs
# -----------------------
url = st.text_input("Enter YouTube URL")
uploaded_file = st.file_uploader("Or upload transcript file (.txt)", type=["txt"])
question = st.text_input("Ask a question about this video or transcript")

def extract_video_id(url):
    if "youtu.be/" in url:
        return url.split("/")[-1].split("?")[0]
    if "v=" in url:
        return url.split("v=")[1].split("&")[0]
    return url

# -----------------------
# Core Pipeline Functions
# -----------------------
@st.cache_data(show_spinner=False)
def fetch_transcript(video_id):
    """
    Fetch transcript from YouTube using the new API
    """
    try:
        api = YouTubeTranscriptApi()
        transcript_list = api.list(video_id)
        # get English transcript, fallback to generated
        transcript_obj = transcript_list.find_transcript(['en'])
        transcript = " ".join(chunk.text for chunk in transcript_obj.fetch())
        return transcript
    except TranscriptsDisabled:
        st.warning("Transcripts are disabled for this video")
        return None
    except Exception as e:
        st.warning(f"Could not fetch from YouTube: {e}")
        return None

@st.cache_resource(show_spinner=False)
def build_vector_store(transcript):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
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

# -----------------------
# Run App
# -----------------------
if st.button("Run"):

    if not api_key:
        st.error("Set OPENAI_API_KEY in Render Environment Variables.")
        st.stop()

    if not question:
        st.error("Enter a question")
        st.stop()

    transcript = None

    # Priority 1: uploaded file
    if uploaded_file:
        transcript = uploaded_file.read().decode("utf-8")
    
    # Priority 2: YouTube URL
    elif url:
        video_id = extract_video_id(url)
        with st.spinner("Fetching transcript from YouTube..."):
            transcript = fetch_transcript(video_id)

    if not transcript:
        st.error("No transcript available. Upload a file or try a different video.")
        st.stop()

    # Build vector store and get answer
    with st.spinner("Processing embeddings..."):
        store = build_vector_store(transcript)
    
    with st.spinner("Thinking..."):
        answer = answer_question(store, question)
    
    st.success("Answer:")
    st.write(answer)
