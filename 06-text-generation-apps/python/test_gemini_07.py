import os
import json
import numpy as np
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from flask import Flask, request, render_template, session, redirect, url_for
import google.generativeai as genai
import faiss

load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Needed for sessions

# Get website content
def get_website_content(url):
    response = requests.get(url)
    if response.status_code != 200:
        return ""
    soup = BeautifulSoup(response.text, 'html.parser')
    paragraphs = soup.find_all('p')
    return "\n".join(p.get_text() for p in paragraphs).strip()

# Chunk text
def chunk_text(text, max_words=100):
    words = text.split()
    return [' '.join(words[i:i+max_words]) for i in range(0, len(words), max_words)]

# Load and embed content
def embed_and_index(content):
    chunks = chunk_text(content)
    embeddings = [
        genai.embed_content(
            model="models/embedding-001",
            content=chunk,
            task_type="retrieval_document"
        )['embedding'] for chunk in chunks
    ]
    matrix = np.array(embeddings).astype("float32")
    idx = faiss.IndexFlatL2(matrix.shape[1])
    idx.add(matrix)
    return idx, chunks

# Generate quiz from context
def generate_quiz(context):
    prompt = f"""
Context:
{context}

Generate a quiz with 5 multiple-choice questions. Each question must have 4 options (A, B, C, D), a correct answer, and a short explanation.
Format your response strictly as JSON:
{{
  "questions": [...],
  "options": [...],
  "answers": [...],
  "explanations": [...]
}}
    """
    model = genai.GenerativeModel("models/gemini-1.5-pro-latest")
    response = model.generate_content(prompt)
    content = response.text.strip().replace('```json', '').replace('```', '')
    return json.loads(content)

# Route: Home for URL input
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        url = request.form['url']
        raw_content = get_website_content(url)
        if not raw_content:
            return "Failed to retrieve content."
        index, chunks = embed_and_index(raw_content)

        # Simple relevance search
        query = "create a quiz based on this content"
        query_embedding = genai.embed_content(
            model="models/embedding-001",
            content=query,
            task_type="retrieval_query"
        )['embedding']
        D, I = index.search(np.array([query_embedding]).astype("float32"), k=5)
        context = "\n\n".join([chunks[i] for i in I[0]])

        quiz = generate_quiz(context)
        session['quiz'] = quiz
        session['current'] = 0
        session['score'] = 0
        return redirect(url_for('quiz'))

    return '''
    <h2>Enter a URL to generate a quiz:</h2>
    <form method="post">
        <input name="url" size="80">
        <input type="submit">
    </form>
    '''

# Route: Quiz step-by-step
@app.route('/quiz', methods=['GET', 'POST'])
def quiz():
    quiz = session.get('quiz')
    current = session.get('current', 0)
    score = session.get('score', 0)

    if request.method == 'POST':
        selected = request.form.get('answer')
        correct = quiz['answers'][current]
        if selected == correct:
            session['score'] = score + 1
        session['current'] = current + 1
        return redirect(url_for('quiz'))

    if current >= len(quiz['questions']):
        return render_template('result.html', score=score, quiz=quiz)

    return render_template('quiz.html',
        question=quiz['questions'][current],
        options=quiz['options'][current],
        q_num=current + 1
    )

# Route: Reset quiz
@app.route('/reset')
def reset():
    session.clear()
    return redirect(url_for('index'))

# Run
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
