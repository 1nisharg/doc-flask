from flask import Flask, render_template, request, session, url_for, redirect
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import requests
from bs4 import BeautifulSoup
import json
from youtube_transcript_api import YouTubeTranscriptApi
import os
from dotenv import load_dotenv
import gc
from functools import lru_cache

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', "6fK9P6WcfpBz7bWJ9qV2eP2Qv5dA8D8z")

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Configuration constants
MAX_CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
MAX_PDF_SIZE = 10 * 1024 * 1024  
MAX_VIDEOS = 6

@app.after_request
def cleanup(response):
    gc.collect()
    return response

@lru_cache(maxsize=1)
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': "cpu"}
    )

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            if pdf.content_length and pdf.content_length > MAX_PDF_SIZE:
                continue
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
        except Exception:
            continue
    return text

def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=MAX_CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = get_embeddings()
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

@lru_cache(maxsize=1)
def get_qa_chain():
    prompt_template = """ 
    Answer the question as detailed as possible from the provided context keeping the tone professional and 
    acting like an expert. If you don't know the answer, just say "Answer is not there within the context", 
    don't provide a wrong answer.\n\n
    Context: \n{context}?\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGroq(model="mixtral-8x7b-32768", groq_api_key=GROQ_API_KEY)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def user_ip(user_question):
    try:
        embeddings = get_embeddings()
        new_db = FAISS.load_local("faiss_index", embeddings=embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question, k=3)
        
        chain = get_qa_chain()
        response = chain(
            {'input_documents': docs, "question": user_question},
            return_only_outputs=True
        )
        return response["output_text"], docs
    except Exception as e:
        return f"Error: {str(e)}", []

def get_video_recommendations(query):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(
            f'https://www.youtube.com/results?search_query={"+".join(query.split())}',
            headers=headers,
            timeout=10
        )
        soup = BeautifulSoup(response.text, 'html.parser')
        scripts = soup.find_all('script')
        
        for script in scripts:
            if 'var ytInitialData' in str(script.string or ''):
                data = json.loads(script.string[script.string.find('{'):script.string.rfind('}')+1])
                contents = data.get('contents', {}).get('twoColumnSearchResultsRenderer', {}).get('primaryContents', {}).get('sectionListRenderer', {}).get('contents', [{}])[0].get('itemSectionRenderer', {}).get('contents', [])
                video_ids = [i.get('videoRenderer', {}).get('videoId') for i in contents if 'videoRenderer' in i][:MAX_VIDEOS]
                return [
                    {
                        "video_id": vid,
                        "thumbnail_url": f"https://img.youtube.com/vi/{vid}/hqdefault.jpg"
                    }
                    for vid in video_ids if vid
                ]
        return []
    except Exception:
        return []

@app.route('/start_over', methods=['POST'])
def start_over():
    session.clear()
    if os.path.exists("faiss_index"):
        import shutil
        shutil.rmtree("faiss_index")
    return redirect(url_for('index'))

@app.route('/', methods=['GET', 'POST'])
def index():
    response = None
    recommendations = []
    uploaded_filenames = []

    if request.method == 'POST':
        user_question = request.form.get('question', '').strip()
        pdf_files = request.files.getlist('pdf_docs')

        if pdf_files and any(pdf.filename for pdf in pdf_files):
            raw_text = get_pdf_text(pdf_files)
            if raw_text:
                text_chunks = get_chunks(raw_text)
                get_vector_store(text_chunks)
                uploaded_filenames = [pdf.filename for pdf in pdf_files if pdf.filename]

        if user_question:
            response, docs = user_ip(user_question)
            if response and docs:
                context_text = " ".join(doc.page_content for doc in docs)
                video_query = f"{response} {context_text}".strip()
                recommendations = get_video_recommendations(video_query)

    return render_template('index.html',
                         response=response,
                         recommendations=recommendations,
                         uploaded_filenames=uploaded_filenames)

if __name__ == '__main__':
    app.run(debug=False, threaded=True)