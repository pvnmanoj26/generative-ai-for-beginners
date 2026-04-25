import os
import json
import re
import requests
import numpy as np
import faiss
from bs4 import BeautifulSoup
from flask import Flask, request, render_template_string
import google.generativeai as genai
from dotenv import load_dotenv

# --------------------------------
# Config
# --------------------------------
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found in environment")

genai.configure(api_key=API_KEY)

MODEL_TEXT = "models/gemini-1.0-pro"
MODEL_EMBED = "models/embedding-001"

app = Flask(__name__)

# --------------------------------
# Helpers
# --------------------------------
def get_website_content(url: str) -> str:
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        paragraphs = soup.find_all("p")
        return "\n".join(p.get_text() for p in paragraphs)
    except Exception as e:
        return f"Error fetching website: {e}"

def chunk_text(text: str, chunk_size: int = 500):
    words = text.split()
    return [
        " ".join(words[i:i + chunk_size])
        for i in range(0, len(words), chunk_size)
    ]

def embed_text(text: str, task_type="retrieval_document"):
    emb = genai.embed_content(
        model=MODEL_EMBED,
        content=text,
        task_type=task_type,
    )
    return np.array(emb["embedding"], dtype="float32")

def clean_json_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```json", "", text)
    text = re.sub(r"^```", "", text)
    text = re.sub(r"```$", "", text)
    return text.strip()

def get_gemini_response(system_instruction: str, user_prompt: str) -> str:
    model = genai.GenerativeModel(
        MODEL_TEXT,
        system_instruction=system_instruction,
    )
    response = model.generate_content(user_prompt)
    return response.text.strip()

# --------------------------------
# HTML Template
# --------------------------------
QUIZ_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Quiz Generator</title>
</head>
<body>
    <h2>Quiz</h2>

    <form method="post" action="/quiz">
        {% for q in quiz.questions %}
            <p><b>Q{{ loop.index }}: {{ q }}</b></p>
            {% for opt in quiz.options[loop.index0] %}
                <input type="radio"
                       name="q{{ loop.index0 }}"
                       value="{{ opt[0] }}">
                {{ opt }}<br>
            {% endfor %}
            <br>
        {% endfor %}

        <input type="hidden"
               name="quiz_data"
               value='{{ quiz | tojson }}'>

        <input type="submit" value="Submit">
    </form>

    {% if score is not none %}
        <h3>Your Score: {{ score }} / {{ total }}</h3>
        <ul>
        {% for i in range(total) %}
            <li>
                <b>Q{{ i + 1 }}:</b>
                Correct Answer: {{ quiz.answers[i] }}<br>
                Explanation: {{ quiz.explanations[i] }}
            </li>
        {% endfor %}
        </ul>
    {% endif %}
</body>
</html>
"""

# --------------------------------
# Routes
# --------------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        url = request.form.get("url", "").strip()
        topic = request.form.get("topic", "").strip()

        if not url and not topic:
            return "<h3>Please enter a URL or a topic</h3>"

        # 1. Fetch content
        if url:
            content = get_website_content(url)
        else:
            content = get_website_content(
                f"https://en.wikipedia.org/wiki/{topic}"
            )

        # 2. Chunk + embed
        chunks = chunk_text(content)
        if not chunks:
            return "<h3>No content found</h3>"

        embeddings = [embed_text(c) for c in chunks]
        embedding_matrix = np.array(embeddings, dtype="float32")

        index = faiss.IndexFlatL2(embedding_matrix.shape[1])
        index.add(embedding_matrix)

        # 3. Retrieve relevant chunks
        query_embedding = embed_text(
            topic or "study material",
            task_type="retrieval_query",
        )
        _, I = index.search(
            np.array([query_embedding], dtype="float32"),
            k=min(5, len(chunks)),
        )

        context = "\n\n".join(chunks[i] for i in I[0])

        # 4. Generate quiz
        system_instruction = (
            "You are a helpful teacher. "
            "Always respond with valid JSON only."
        )

        quiz_prompt = f"""
Create a quiz with 5 multiple-choice questions (A–D options)
from the following study material:

{context}

Format strictly as JSON:
{{
  "questions": ["Q1", "Q2", "Q3", "Q4", "Q5"],
  "options": [
    ["A. ...", "B. ...", "C. ...", "D. ..."],
    ...
  ],
  "answers": ["A", "B", "C", "D", "A"],
  "explanations": ["...", "...", "...", "...", "..."]
}}
"""

        quiz_text = get_gemini_response(system_instruction, quiz_prompt)
        quiz_text = clean_json_text(quiz_text)

        try:
            quiz = json.loads(quiz_text)
        except Exception:
            return f"<h3>Quiz generation failed</h3><pre>{quiz_text}</pre>"

        return render_template_string(
            QUIZ_TEMPLATE,
            quiz=quiz,
            score=None,
            total=len(quiz["questions"]),
        )

    return """
    <h2>Generate a Quiz</h2>
    <form method="post">
      <label>Website URL:</label><br>
      <input type="text" name="url" style="width:400px"><br><br>

      <label>OR Topic (Wikipedia):</label><br>
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
        if selected == quiz_data["answers"][i]:
            score += 1

    return render_template_string(
        QUIZ_TEMPLATE,
        quiz=quiz_data,
        score=score,
        total=total,
    )

# --------------------------------
# Run
# --------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
