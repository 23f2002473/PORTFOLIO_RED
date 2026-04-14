import os
import json
import uuid
import sqlite3
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
from openai import OpenAI
import PyPDF2

# Load environment variables
load_dotenv()

app = Flask(__name__, static_folder='.')
CORS(app)

# Configuration
CONFIG = {
    'EMBEDDING_MODEL': 'text_embedding_3_small',
    'CHAT_MODEL': 'gpt-4o-mini',
    'CHUNK_SIZE': 500,
    'CHUNK_OVERLAP': 50,
    'TOP_K': 3,
    'PDF_PATH': 'rohit.pdf'
}

DB_PATH = 'portfolio.db'

# ── DATABASE ──────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS feedback (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        email       TEXT NOT NULL,
        message     TEXT NOT NULL,
        emoji       TEXT,
        emoji_label TEXT,
        timestamp   TEXT NOT NULL
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS chat_logs (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        role       TEXT NOT NULL,
        message    TEXT NOT NULL,
        timestamp  TEXT NOT NULL
    )''')
    conn.commit()
    conn.close()

def db():
    return sqlite3.connect(DB_PATH)

# Global state
resume_chunks = []
client = None

def init_openai():
    global client
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key and api_key != "your_api_key_here":
        client = OpenAI(api_key=api_key)
        return True
    return False

def load_pdf(path):
    try:
        text = ""
        with open(path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n\n"
        return text
    except Exception as e:
        print(f"Error loading PDF: {e}")
        return ""

def create_chunks(text):
    chunks = []
    chunk_size = CONFIG['CHUNK_SIZE']
    overlap = CONFIG['CHUNK_OVERLAP']
    text = " ".join(text.split())
    if not text:
        return []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        if len(chunk) > 50:
            chunks.append({'text': chunk, 'index': len(chunks)})
    return chunks

def generate_embeddings(chunks):
    if not client:
        return []
    try:
        texts = [c['text'] for c in chunks]
        batch_size = 20
        embedded_chunks = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=batch
            )
            for j, data in enumerate(response.data):
                embedded_chunks.append({
                    'text': batch[j],
                    'embedding': data.embedding,
                    'index': chunks[i+j]['index']
                })
        return embedded_chunks
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return []

def cosine_similarity(vec_a, vec_b):
    return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))

def retrieve_relevant_chunks(query, top_k=3):
    if not client or not resume_chunks:
        return []
    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=query
        )
        query_embedding = response.data[0].embedding
        similarities = []
        for chunk in resume_chunks:
            sim = cosine_similarity(query_embedding, chunk['embedding'])
            similarities.append((chunk, sim))
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [item[0]['text'] for item in similarities[:top_k]]
    except Exception as e:
        print(f"Error retrieving chunks: {e}")
        return []

# ── INIT ──────────────────────────────────────────────────────
print("Initializing backend...")
init_db()
print("Database ready.")
if init_openai():
    print("OpenAI client initialized.")
    if os.path.exists(CONFIG['PDF_PATH']):
        print(f"Loading PDF from {CONFIG['PDF_PATH']}...")
        pdf_text = load_pdf(CONFIG['PDF_PATH'])
        raw_chunks = create_chunks(pdf_text)
        print(f"Generated {len(raw_chunks)} chunks. Generating embeddings...")
        resume_chunks = generate_embeddings(raw_chunks)
        print(f"Backend ready with {len(resume_chunks)} embedded chunks.")
    else:
        print(f"Warning: PDF file not found at {CONFIG['PDF_PATH']}")
else:
    print("Warning: OPENAI_API_KEY not found or invalid. Chat will return an error.")

# ── ROUTES ────────────────────────────────────────────────────
@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('.', path)

@app.route('/api/feedback', methods=['POST'])
def save_feedback():
    data = request.json or {}
    email       = data.get('email', '').strip()
    message     = data.get('message', '').strip()
    emoji       = data.get('emoji', '')
    emoji_label = data.get('emoji_label', '')

    if not email or not message:
        return jsonify({'error': 'Email and message are required'}), 400

    conn = db()
    conn.execute(
        'INSERT INTO feedback (email, message, emoji, emoji_label, timestamp) VALUES (?, ?, ?, ?, ?)',
        (email, message, emoji, emoji_label, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()
    return jsonify({'success': True})

@app.route('/api/chat', methods=['POST'])
def chat():
    if not client:
        return jsonify({'error': 'OpenAI API key not configured on server'}), 503

    data         = request.json or {}
    user_message = data.get('message', '')
    session_id   = data.get('session_id') or str(uuid.uuid4())

    if not user_message:
        return jsonify({'error': 'Message is required'}), 400

    # Log user message
    conn = db()
    conn.execute(
        'INSERT INTO chat_logs (session_id, role, message, timestamp) VALUES (?, ?, ?, ?)',
        (session_id, 'user', user_message, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()

    try:
        relevant_texts = retrieve_relevant_chunks(user_message, CONFIG['TOP_K'])
        context = "\n\n---\n\n".join(relevant_texts)

        system_prompt = f"""You are an AI assistant helping answer questions about a resume/CV.
You will be provided with relevant excerpts from the resume to help answer the user's question.

IMPORTANT: Base your answers ONLY on the information provided in the context below. If the information
is not available in the context, say so clearly.

Context from resume:
{context}
"""
        completion = client.chat.completions.create(
            model=CONFIG['CHAT_MODEL'],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message}
            ]
        )
        reply = completion.choices[0].message.content

        # Log assistant reply
        conn = db()
        conn.execute(
            'INSERT INTO chat_logs (session_id, role, message, timestamp) VALUES (?, ?, ?, ?)',
            (session_id, 'assistant', reply, datetime.now().isoformat())
        )
        conn.commit()
        conn.close()

        return jsonify({'reply': reply, 'session_id': session_id})

    except Exception as e:
        print(f"Error processing chat: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
