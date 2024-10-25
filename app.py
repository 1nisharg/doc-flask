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
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from io import BytesIO
from flask import send_file
import tempfile


app = Flask(__name__)
app.secret_key = "6fK9P6WcfpBz7bWJ9qV2eP2Qv5dA8D8z"


load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

@app.route('/start_over', methods=['POST'])
def start_over():
    # Clear any session data or reset state as necessary
    session.clear()  # Clears all session data
    return redirect(url_for('index')) 

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""  # Avoid NoneType issues
        except Exception as e:
            print(f"Error reading {pdf.filename}: {e}")
    return text

def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': "cpu"})
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def conv_chain():
    prompt_template = """ 
Answer the question as detailed as possible from the provided context keeping the tone professional and 
acting like an expert. If you don't know the answer, just say "Answer is not there within the context", don't provide a wrong answer.\n\n
Context: \n{context}?\n
Question: \n{question}\n

Answer:
    """
    model = ChatGroq(model="mixtral-8x7b-32768", groq_api_key=GROQ_API_KEY)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    # Create a QA chain with the prompt
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_ip(user_question):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': "cpu"})

    try:
        # Load the vector store securely
        new_db = FAISS.load_local("faiss_index", embeddings=embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        print(f"Error loading FAISS index: {e}")
        return "Error loading vector store.", []  # Return an error message

    # Perform similarity search
    try:
        docs = new_db.similarity_search(user_question, k=5)
    except Exception as e:
        print(f"Error during similarity search: {e}")
        return "Error during similarity search.", []  # Return an error message

    # Prepare the QA chain
    chain = conv_chain()

    # Generate response using the QA chain
    try:
        response = chain(
            {'input_documents': docs, "question": user_question},
            return_only_outputs=True
        )
        return response["output_text"], docs  # Return response and docs
    except Exception as e:
        print(f"Error generating response: {e}")
        return "Error generating response.", []  # Return an error message

def get_video_ids(user_query):
    headers = {"User-Agent": "Guest"}
    video_res = requests.get(f'https://www.youtube.com/results?search_query={"+".join(user_query.split(" "))}', headers=headers)
    soup = BeautifulSoup(video_res.text, 'html.parser')
    arr_video = soup.find_all('script')

    arr_main = []
    for i in arr_video:
        if 'var ytInitialData' in str(i.get_text()):
            arr_main.append(i)
            break
    main_script = arr_main[0].get_text()[arr_main[0].get_text().find('{'):arr_main[0].get_text().rfind('}') + 1]
    data = json.loads(main_script)
    video_data = data.get('contents').get('twoColumnSearchResultsRenderer').get('primaryContents').get('sectionListRenderer').get('contents')[0].get('itemSectionRenderer').get('contents')
    video_json = [i for i in video_data if 'videoRenderer' in str(i)]

    video_ids = [i.get('videoRenderer').get('videoId') for i in video_json if i.get('videoRenderer')]
    return video_ids

def get_transcript(video_ids):
    yt_data = []
    for i in video_ids:
        txt = ""
        try:
            transcript = YouTubeTranscriptApi.get_transcript(i, languages=['en'])
            for j in transcript:
                txt += j['text'] + " "
            yt_data.append({"video_id": i, "transcript": txt})
        except Exception as e:
            print(f"Error fetching transcript {i}: {e}")
            continue
    return yt_data

def video_recommendation(user_question):
    video_ids = get_video_ids(user_question)
    transcripts = get_transcript(video_ids)

    # Sort videos based on the length of transcript as a proxy for relevance
    sorted_videos = sorted(transcripts, key=lambda x: len(x['transcript']), reverse=True)

    video_info = [
        {
            "video_id": video["video_id"],
            "thumbnail_url": f"https://img.youtube.com/vi/{video['video_id']}/hqdefault.jpg"
        }
        for video in sorted_videos[:8]  # Return top 8 videos
    ]
    
    return video_info  # Return list of video info

@app.route('/', methods=['GET', 'POST'])
def index():
    response = None  # Initialize response
    recommendations = []  # Initialize recommendations
    uploaded_filenames = []

    if request.method == 'POST':
        user_question = request.form.get('question', '').strip()
        pdf_files = request.files.getlist('pdf_docs')

        if pdf_files:  # Ensure pdf_files is checked before use
            raw_text = get_pdf_text(pdf_files)
            text_chunks = get_chunks(raw_text)
            get_vector_store(text_chunks)
            uploaded_filenames = [pdf.filename for pdf in pdf_files]
        
        if user_question:
            try:
                response, docs = user_ip(user_question)
                context_text = " ".join([doc.page_content for doc in docs]) if docs else ""
                video_query = f"{response} {context_text}".strip()
                recommendations = video_recommendation(video_query) if response and context_text else []
            except Exception as e:
                print(f"Error processing user question: {e}")
                response = "Error processing the question."

    return render_template('index.html', response=response, recommendations=recommendations,
                           uploaded_filenames=uploaded_filenames)

if __name__ == '__main__':
    app.run(debug=False, threaded=True)
