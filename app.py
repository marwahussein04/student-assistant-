# =========================================================================
# START OF student_assistant.py - LANGCHAIN 1.0+ COMPATIBLE
# =========================================================================

import os
from dotenv import load_dotenv

# يحمّل المتغيرات من ملف .env
load_dotenv() 

import streamlit as st

# UPDATED IMPORTS FOR LANGCHAIN 1.0+
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper
from googleapiclient.discovery import build

# Text splitting and embeddings - Compatible with 1.0+
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# File processing
from pypdf import PdfReader
import docx

@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# --- 1. Define Output Schema ---
class Recommendation(BaseModel):
    title: str = Field(description="The title of the suggested source.")
    url: str = Field(description="The URL (link) to the suggested source.")
    source: str = Field(description="The type of source (e.g., YouTube or Arxiv).")

class AssistantOutput(BaseModel):
    explanation_text: str = Field(description="A simplified and detailed explanation.")
    key_takeaways: list[str] = Field(description="3-5 key points.")
    video_recommendations: list[Recommendation] = Field(description="Top 5 YouTube videos.")
    paper_recommendations: list[Recommendation] = Field(description="Top 3 arXiv papers.")

# --- 2. File Processing ---
def extract_text_from_pdf(file):
    try:
        pdf_reader = PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        return text
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

def extract_text_from_docx(file):
    try:
        doc = docx.Document(file)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text
    except Exception as e:
        return f"Error reading DOCX: {str(e)}"

def extract_text_from_txt(file):
    try:
        return file.read().decode('utf-8')
    except Exception as e:
        return f"Error reading TXT: {str(e)}"

def process_uploaded_file(file):
    file_type = file.name.split('.')[-1].lower()
    if file_type == 'pdf':
        return extract_text_from_pdf(file)
    elif file_type in ['docx', 'doc']:
        return extract_text_from_docx(file)
    elif file_type == 'txt':
        return extract_text_from_txt(file)
    else:
        return "Unsupported file format."

def create_rag_chain(text):
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(text)
        embeddings = get_embedding_model()
        vectorstore = FAISS.from_texts(chunks, embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"Error creating RAG: {str(e)}")
        return None

# --- 3. Search Tools ---
def search_youtube(query, max_results=5):
    try:
        youtube_api_key = os.getenv("YOUTUBE_API_KEY")
        if not youtube_api_key:
             return "YouTube search error: YOUTUBE_API_KEY not found."
             
        youtube = build('youtube', 'v3', developerKey=youtube_api_key)
        request = youtube.search().list(q=query, part='snippet', type='video', maxResults=max_results)
        response = request.execute()
        results = []
        for item in response.get('items', []):
            if 'videoId' in item['id']:
                video_id = item['id']['videoId']
                title = item['snippet']['title']
                url = f"https://www.youtube.com/watch?v={video_id}"
                results.append(f"Title: {title}\nURL: {url}\n")
        return "\n".join(results) if results else "No videos found."
    except Exception as e:
        return f"YouTube search error: {str(e)}"

try:
    arxiv_wrapper = ArxivAPIWrapper(top_k_results=3, doc_content_chars_max=2000, load_max_docs=3)
    arxiv_tool_run = arxiv_wrapper.run
except Exception:
    def arxiv_tool_run(query):
        return "ArXiv unavailable."

def safe_arxiv_search(query: str):
    import time
    clean_query = query.replace("what is", "").replace("?", "").strip()
    try:
        time.sleep(1)
        result = arxiv_tool_run(clean_query)
        if result and len(result) > 50:
            return result
        return "No papers found."
    except:
        return "ArXiv temporarily unavailable."

# --- 4. Document Functions ---
def query_document(vectorstore, query):
    try:
        docs = vectorstore.similarity_search(query, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer based on context. Context:\n\n{context}"),
            ("human", "{query}")
        ])
        chain = prompt | llm
        response = chain.invoke({"context": context, "query": query})
        return response.content
    except Exception as e:
        return f"Error: {str(e)}"

def summarize_document(text):
    try:
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)
        if len(text) > 8000:
            text = text[:8000] + "..."
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Summarize this academic content clearly."),
            ("human", "Summarize:\n\n{text}")
        ])
        chain = prompt | llm
        response = chain.invoke({"text": text})
        return response.content
    except Exception as e:
        return f"Error: {str(e)}"

def explain_document(text, topic=None):
    try:
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.5)
        if len(text) > 8000:
            text = text[:8000] + "..."
        topic_instruction = f" focusing on {topic}" if topic else ""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Explain this content clearly and simply."),
            ("human", f"Explain{topic_instruction}:\n\n{{text}}")
        ])
        chain = prompt | llm
        response = chain.invoke({"text": text})
        return response.content
    except Exception as e:
        return f"Error: {str(e)}"

def get_student_assistant_response(query: str):
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7) 
    parser = PydanticOutputParser(pydantic_object=AssistantOutput)
    
    youtube_results_text = search_youtube(query, max_results=5)
    arxiv_results_text = safe_arxiv_search(query)
    
    # FIX 2: Modify LLM prompt to be extremely explicit about extracting the URL
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are an expert student assistant. Provide: "
         "1. A clear, simplified, and detailed explanation in 'explanation_text'. "
         "2. 3-5 key points in 'key_takeaways'. "
         "3. Parse the following raw YouTube search results into a list of 5 valid 'Recommendation' objects in 'video_recommendations'. RAW YOUTUBE RESULTS:\n{youtube_results_text}\n"
         "4. Parse the following raw arXiv search results into a list of 3 valid 'Recommendation' objects in 'paper_recommendations'. RAW ARXIV RESULTS:\n{arxiv_results_text}\n"
         "IMPORTANT: For arXiv papers, the raw results contain the full URL. YOU MUST extract the full URL and place it in the 'url' field. DO NOT leave the 'url' field empty or use an invalid value. "
         "NEVER return empty lists. The source for YouTube must be 'YouTube' and for Arxiv must be 'Arxiv'. Use the exact title and URL provided in the raw results. JSON format:\n{format_instructions}"),
        ("human", "Explain and provide resources for: {query}"),
    ])
    
    chain = prompt | llm | parser
    response = chain.invoke({
        "query": query,
        "youtube_results_text": youtube_results_text,
        "arxiv_results_text": arxiv_results_text,
        "format_instructions": parser.get_format_instructions()
    })
    return response

# --- 5. Beautiful UI ---
def main():
    if 'qa_mode' not in st.session_state:
        st.session_state.qa_mode = False
    
    st.set_page_config(
        page_title="Smart Student Assistant 🎓", 
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # FIX 1: Implement a Dark Theme for maximum readability and contrast
    st.markdown("""
        <style>
        /* General App Background: Dark and professional */
        .stApp {
            background-color: #1e1e1e; /* Dark Gray */
            color: #ffffff; /* White text */
        }
        
        /* Main Header: Contrasting and visible */
        .main-header {
            background: #007bff; /* Primary Blue */
            padding: 2rem;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 8px 32px rgba(0,0,0,0.5);
            margin-bottom: 2rem;
        }
        .main-header h1 {
            color: white;
            font-size: 3rem;
            font-weight: 800;
            margin: 0;
        }
        .main-header p {
            color: #e9ecef; /* Light text for subtitle */
        }
        
        /* Text Input and General Text */
        .stTextInput > div > div > input {
            color: #ffffff; /* White text in input */
            background-color: #2d2d2d; /* Slightly lighter dark background */
        }
        
        /* Buttons: Clear Call to Action */
        .stButton>button {
            background-color: #28a745; /* Success Green */
            color: white;
            border-radius: 10px;
            padding: 0.75rem 2rem;
            border: none;
            cursor: pointer;
            transition: background-color 0.3s;
        }
        .stButton>button:hover {
            background-color: #218838; /* Darker Green on hover */
        }
        
        /* Info Boxes (Explanation): High contrast on dark background */
        .stAlert.info {
            background-color: #004085; /* Dark blue background */
            color: #ffffff; /* White text */
            border-left: 5px solid #007bff; /* Primary blue border */
        }
        
        /* Links: Ensure visibility */
        a {
            color: #4da6ff; /* Light blue for links */
            text-decoration: none;
        }
        a:hover {
            text-decoration: underline;
        }
        
        /* Sidebar text color */
        .sidebar .stRadio > label > div {
            color: #ffffff;
        }
        
        /* General text color for Streamlit components */
        h1, h2, h3, h4, h5, h6, p, label, .stMarkdown {
            color: #ffffff;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="main-header">
            <h1>🎓 Smart Student Assistant</h1>
            <p>Your AI-Powered Learning Companion</p>
        </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### 🎯 Choose Your Mode")
        mode = st.radio("", ["🔍 Search & Learn", "📄 Analyze Document"])

    if mode == "🔍 Search & Learn":
        query = st.text_input("🔎 What do you want to learn today?", key="search_query")
        
        if st.button("🚀 Search", key="run_search"):
            if query:
                if 'vectorstore' in st.session_state:
                    del st.session_state.vectorstore
                st.session_state.qa_mode = False
                
                with st.spinner('🔍 Searching and explaining...'):
                    try:
                        result = get_student_assistant_response(query)
                        
                        st.success(f"✅ Found results for: **{query}**")
                        
                        st.markdown("### 💡 Explanation")
                        st.info(result.explanation_text)
                        
                        st.markdown("### 📌 Key Takeaways")
                        for idx, point in enumerate(result.key_takeaways):
                            st.markdown(f"**{idx+1}.** {point}")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("### 📺 Videos")
                            for rec in result.video_recommendations:
                                st.markdown(f"🎥 [{rec.title}]({rec.url})")
                        
                        with col2:
                            st.markdown("### 📄 Papers")
                            for rec in result.paper_recommendations:
                                # FIX 3: Re-implement robust PDF link conversion and new tab opening
                                arxiv_url = rec.url
                                
                                if "arxiv.org/abs/" in arxiv_url:
                                    # Convert /abs/ link to /pdf/ link for direct PDF access
                                    pdf_url = arxiv_url.replace("http://", "https://").replace("/abs/", "/pdf/") + ".pdf"
                                else:
                                    # Fallback for non-standard or already-PDF links
                                    pdf_url = arxiv_url
                                    
                                # Use st.markdown with unsafe_allow_html=True to force opening in a new tab
                                st.markdown(f'📖 <a href="{pdf_url}" target="_blank" style="text-decoration: none;">{rec.title}</a>', unsafe_allow_html=True)
                                
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
                        st.exception(e)

    elif mode == "📄 Analyze Document":
        st.markdown("### 📄 Upload Document")
        uploaded_file = st.file_uploader("Upload a PDF, DOCX, or TXT file", type=["pdf", "docx", "txt"])
        
        if uploaded_file:
            file_content = process_uploaded_file(uploaded_file)
            
            if file_content.startswith("Error") or file_content.startswith("Unsupported"):
                st.error(file_content)
            else:
                st.success("File uploaded and text extracted successfully!")
                
                # Store text in session state for re-use
                st.session_state.document_text = file_content
                
                # Display summary
                with st.spinner("Summarizing document..."):
                    summary = summarize_document(file_content)
                    st.markdown("### 📝 Document Summary")
                    st.info(summary)
                
                # Create RAG chain for Q&A
                if 'vectorstore' not in st.session_state:
                    with st.spinner("Preparing document for Q&A..."):
                        st.session_state.vectorstore = create_rag_chain(file_content)
                        st.session_state.qa_mode = True
                        st.success("Document ready for Q&A!")
                else:
                    st.session_state.qa_mode = True
                    st.success("Document ready for Q&A!")
                
                # Explanation section
                st.markdown("### 💡 Simple Explanation")
                with st.expander("Click to see a simplified explanation of the document's main topic"):
                    with st.spinner("Generating explanation..."):
                        explanation = explain_document(file_content)
                        st.write(explanation)
                
    if st.session_state.qa_mode and 'vectorstore' in st.session_state:
        st.markdown("---")
        st.markdown("### ❓ Ask a Question about the Document")
        qa_query = st.text_input("Enter your question:", key="qa_query")
        
        if st.button("Ask Document", key="run_qa"):
            if qa_query:
                with st.spinner("Searching document for answer..."):
                    answer = query_document(st.session_state.vectorstore, qa_query)
                    st.markdown("### 💬 Answer")
                    st.success(answer)

if __name__ == "__main__":
    main()
# =========================================================================
# END OF student_assistant.py
# =========================================================================