import os
import json
import re
import requests
import numpy as np
import faiss
from bs4 import BeautifulSoup
from flask import Flask, request, render_template_string
import anthropic
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load env variables
load_dotenv()

# Initialise clients
claude = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
embed_model = SentenceTransformer("all-MiniLM-L6-v2")  # downloads once ~80MB, then cached

app = Flask(__name__)

# -------------------------------
# Helpers
# -------------------------------
def get_website_content(url):
    try:
        resp = requests.get(url)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        paragraphs = soup.find_all("p")
        text = "\n".join([p.get_text() for p in paragraphs])
        return text
    except Exception as e:
        return f"Error fetching website: {e}"

def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def embed_text(text):
    return np.array(embed_model.encode(text), dtype="float32")

def get_claude_response(system_instruction, user_prompt):
    response = claude.messages.create(
        model="claude-haiku-4-5",
        max_tokens=1500,
        system=system_instruction,
        messages=[{"role": "user", "content": user_prompt}]
    )
    return response.content[0].text.strip()

def clean_json_text(text):
    text = text.strip()
    text = re.sub(r"^```json", "", text)
    text = re.sub(r"^```", "", text)
    text = re.sub(r"```$", "", text)
    return text.strip()

# -------------------------------
# HTML Template
# -------------------------------
QUIZ_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Quiz</title>
</head>
<body>
    <h2>Quiz</h2>
    <form method="post" action="/quiz">
        {% for q in quiz.questions %}
            <p><b>Q{{ loop.index }}: {{ q }}</b></p>
            {% for opt in quiz.options[loop.index0] %}
                <input type="radio" name="q{{ loop.index0 }}" value="{{ opt[0] }}"> {{ opt }}<br>
            {% endfor %}
            <br>
        {% endfor %}
        <input type="hidden" name="quiz_data" value='{{ quiz|tojson }}'>
        <input type="submit" value="Submit">
    </form>

    {% if score is not none %}
        <h3>Your Score: {{ score }} / {{ total }}</h3>
        <ul>
        {% for i in range(total) %}
            <li><b>Q{{ i+1 }}:</b> Correct Answer: {{ quiz.answers[i] }}<br>
                Explanation: {{ quiz.explanations[i] }}</li>
        {% endfor %}
        </ul>
    {% endif %}
</body>
</html>
"""

# -------------------------------
# Routes
# -------------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        url = request.form.get("url", "").strip()
        topic = request.form.get("topic", "").strip()

        if not url and not topic:
            return "<h3>Please enter a URL or a Topic</h3>"

        # Step 1: Get content
        if url:
            content = get_website_content(url)
        else:
            content = get_website_content(f"https://en.wikipedia.org/wiki/{topic}")

        chunks = chunk_text(content)
        embeddings = [embed_text(chunk) for chunk in chunks]
        embedding_matrix = np.array(embeddings).astype("float32")

        dimension = embedding_matrix.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embedding_matrix)

        # Step 2: Build Quiz
        query_embedding = embed_text(topic or "study material")
        D, I = index.search(np.array([query_embedding]).astype("float32"), k=5)
        relevant_chunks = [chunks[i] for i in I[0]]
        context = "\n\n".join(relevant_chunks)

        system_instruction = "You are a helpful teacher. Always respond with valid JSON only — no markdown, no explanation, just the raw JSON object."
        quiz_prompt = f"""
        Create a quiz with 5 multiple-choice questions (A–D options) from the following study material:

        {context}

        Format strictly as JSON:
        {{
          "questions": ["Q1 text", "Q2 text", ...],
          "options": [
            ["A. ...", "B. ...", "C. ...", "D. ..."],
            ...
          ],
          "answers": ["A","C",...],
          "explanations": ["explanation1","explanation2",...]
        }}
        """
        quiz_json = get_claude_response(system_instruction, quiz_prompt)
        quiz_json = clean_json_text(quiz_json)

        try:
            quiz = json.loads(quiz_json)
        except Exception:
            return f"<h3>Quiz generation failed:</h3><pre>{quiz_json}</pre>"

        return render_template_string(QUIZ_TEMPLATE, quiz=quiz, score=None, total=len(quiz["questions"]))

    return """
    <h2>Generate a Quiz</h2>
    <form method="post">
      <label>Enter a Website URL:</label><br>
      <input type="text" name="url" style="width:400px"><br><br>
      <label>Or enter a Topic (Wikipedia will be used):</label><br>
      <input type="text" name="topic" style="width:400px"><br><br>
      <input type="submit" value="Generate Quiz">
    </form>
    """

@app.route("/quiz", methods=["POST"])
def quiz():
    quiz_data = json.loads(request.form["quiz_data"])
    score = 0
    total = len(quiz_data["questions"])
    for i in range(total):
        selected = request.form.get(f"q{i}")
        if selected and selected == quiz_data["answers"][i]:
            score += 1
    return render_template_string(QUIZ_TEMPLATE, quiz=quiz_data, score=score, total=total)

# -------------------------------
# Run
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True, port=5000)