import os
import streamlit as st

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

# -----------------------
# Page config
# -----------------------
st.set_page_config(page_title="YouTube RAG App", layout="wide")
st.title("🎥 YouTube Video Q&A (RAG App)")

# -----------------------
# Get API Key from Render ENV
# -----------------------
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("OPENAI_API_KEY not found. Set it in Render Environment Variables.")
    st.stop()

# -----------------------
# Input
# -----------------------
url = st.text_input("Enter YouTube URL")

def extract_video_id(url):
    if "youtu.be/" in url:
        return url.split("/")[-1].split("?")[0]
    if "v=" in url:
        return url.split("v=")[1].split("&")[0]
    return url

# -----------------------
# Build Vector DB
# -----------------------
@st.cache_resource
def build_vector_store(video_id):
    try:
        api = YouTubeTranscriptApi()
        transcript_list = api.fetch(video_id, languages=["en"])
    except TranscriptsDisabled:
        st.error("Transcript is disabled for this video.")
        return None
    except Exception as e:
        st.error(f"Error fetching transcript: {e}")
        return None

    transcript = " ".join(chunk.text for chunk in transcript_list)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = splitter.create_documents([transcript])

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    return FAISS.from_documents(docs, embeddings)

# -----------------------
# Ask question
# -----------------------
if url:
    video_id = extract_video_id(url)

    with st.spinner("Processing transcript..."):
        vector_store = build_vector_store(video_id)
        if not vector_store:
            st.stop()
        retriever = vector_store.as_retriever()

    question = st.text_input("Ask a question about this video")

    if question:
        docs = retriever.invoke(question)
        context = "\n\n".join(doc.page_content for doc in docs)

        prompt = f"""
Answer ONLY from the context below.
If not found, say 'Not in video'.

Context:
{context}

Question:
{question}
"""

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

        answer = llm.invoke(prompt)

        st.success(answer.content)
