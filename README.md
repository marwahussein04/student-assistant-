
🎓 Student Assistant AI App
This is an AI-powered academic assistant built using Streamlit, LangChain, and Groq. It's designed to help students quickly understand complex topics, analyze documents (PDF, DOCX, TXT), and find relevant learning resources (YouTube videos and ArXiv papers).

✨ Features
Quick Search & Explanation: Get simplified explanations, key takeaways, and external resources for any academic topic.

Document Analysis: Upload and analyze PDF, DOCX, or TXT files.

Summarize: Generates a concise summary of the document.

Explain: Provides a clear, simplified explanation of the document's content.

Q&A (RAG): Ask specific questions about the document using Retrieval-Augmented Generation (RAG).

Resource Recommendation: Automatically fetches and recommends relevant YouTube videos and scholarly ArXiv papers.

High Performance: Leverages the Groq API for extremely fast LLM inference.

🚀 Setup and Installation
Follow these steps to set up and run the application locally on your machine.

1. Prerequisites
You need Python 3.8+ installed on your system.

2. Clone the Repository
Clone the GitHub repository to your local machine:

Bash

git clone https://github.com/marwahussein04/student-assistant-
cd student-assistant-
3. Install Dependencies
Install all required Python packages using the requirements.txt file (ensure this file is present in your repo):

Bash

pip install -r requirements.txt
4. Configure API Keys (Crucial Step)
The application requires API keys for the Language Model (Groq) and the search tool (YouTube).

Create a file named .env in the root directory of the project (next to app.py).

Add your keys to the .env file in the following format (replace the placeholders with your actual keys):

Code snippet

# .env file
GROQ_API_KEY="YOUR_GROQ_API_KEY"
YOUTUBE_API_KEY="YOUR_YOUTUBE_API_KEY"
▶️ Running the Application
Execute the following command in your terminal:

Bash

streamlit run app.py
The application will automatically open in your default web browser (usually at http://localhost:8501).

⚙️ Core Technologies
Frontend/App Framework: Streamlit

LLM Integration: LangChain

LLM Provider: Groq (using llama-3.3-70b-versatile model)

Vector Store: FAISS

Embeddings: HuggingFace Embeddings (sentence-transformers/all-MiniLM-L6-v2)

Search Tools: YouTube Data API, ArXiv API

File Handling: PyPDF2, python-docx

🤝 Contribution
Feel free to open issues or submit pull requests if you have suggestions or improvements!
