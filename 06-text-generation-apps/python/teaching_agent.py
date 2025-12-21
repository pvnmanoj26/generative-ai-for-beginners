# teaching_agent.py  (agentic upgrade)

import os
import re
import json
import uuid
import requests
import numpy as np
from flask import Flask, request, render_template_string, session, redirect, url_for,jsonify
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import faiss
import google.generativeai as genai

# ==============================
# CONFIG
# ==============================
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set. Add it to your environment or .env file.")
genai.configure(api_key=API_KEY)

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key")  # replace in production

# In-memory store for large objects (FAISS index, chunks) per session
MEM = {}  # sid -> {"index": faiss.IndexFlatL2, "chunks": [str]}

# Preferred model names
TEXT_MODEL = "models/gemini-1.5-flash"
EMBED_CANDIDATES = ("models/text-embedding-004", "models/embedding-001")


# ==============================
# SCRAPE / PREP
# ==============================
def fetch_content(topic_or_url: str, max_chars: int = 20000) -> str:
    """
    If input looks like a URL -> fetch that page; else fetch Wikipedia article for the topic.
    Returns concatenated paragraph text, trimmed to max_chars.
    """
    if topic_or_url.lower().startswith(("http://", "https://")):
        url = topic_or_url
    else:
        # Wikipedia page for the topic
        slug = topic_or_url.strip().replace(" ", "_")
        url = f"https://en.wikipedia.org/wiki/{slug}"

    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    # Prefer the main content area on Wikipedia; otherwise fallback to all <p>
    content_div = soup.find("div", {"id": "mw-content-text"}) or soup
    paragraphs = content_div.find_all("p")
    text = " ".join(p.get_text(separator=" ", strip=True) for p in paragraphs if p.get_text(strip=True))

    # clean citations like [1], [a], etc.
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\[[a-zA-Z]+\]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]


def chunk_text(text: str, max_words: int = 120):
    """Split text into overlapping word chunks for better retrieval."""
    words = text.split()
    chunks = []
    step = max_words - 20 if max_words > 40 else max_words  # small overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + max_words])
        if len(chunk.split()) >= max_words * 0.5:  # ignore tiny tails
            chunks.append(chunk)
    return chunks


# ==============================
# EMBEDDINGS / FAISS
# ==============================
def _try_embed(model_name: str, text: str):
    """Helper to try embedding with a specific model name."""
    return genai.embed_content(model=model_name, content=text)["embedding"]

def choose_embedding_model():
    for candidate in EMBED_CANDIDATES:
        try:
            _ = _try_embed(candidate, "ping")
            return candidate
        except Exception:
            continue
    raise RuntimeError("No available embedding model (tried text-embedding-004 and embedding-001).")

EMBED_MODEL = None  # lazy-init after first request

def embed_texts(texts):
    """
    Embed a list of strings using the preferred model with fallback.
    """
    global EMBED_MODEL
    if EMBED_MODEL is None:
        EMBED_MODEL = choose_embedding_model()
    vectors = []
    for t in texts:
        vec = _try_embed(EMBED_MODEL, t)
        vectors.append(vec)
    return np.array(vectors, dtype="float32")


def build_faiss_index(chunks):
    if not chunks:
        raise ValueError("No chunks to index.")
    emb = embed_texts(chunks)
    dim = emb.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(emb)
    return index, emb


def retrieve_context(index, chunks, query, k=5):
    """Get top-k relevant chunks for the query."""
    q_emb = embed_texts([query])
    D, I = index.search(q_emb, k=min(k, len(chunks)))
    selected = [chunks[i] for i in I[0] if 0 <= i < len(chunks)]
    return "\n\n".join(selected)


# ==============================
# QUIZ GENERATION + REPAIR (STRICT JSON)
# ==============================
def clean_json_fences(text: str) -> str:
    """Remove code fences if the model returns ```json ... ```."""
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def generate_quiz_json(context: str, total: int = 5, retries: int = 2):
    """
    Ask Gemini to generate a quiz in strict JSON format.
    Retries + repairs if the JSON is invalid.
    """
    model = genai.GenerativeModel(
        TEXT_MODEL,
        generation_config={"temperature": 0.7, "max_output_tokens": 1400},
        system_instruction=(
            "You are a helpful teacher. Always return ONLY valid JSON that strictly matches the requested schema. "
            "Never include extra keys or any prose outside JSON. Keep options labeled A–D."
        ),
    )

    base_prompt = f"""
Create a multiple-choice quiz with exactly {total} questions from the following study material.
Each question must have four options labeled "A.", "B.", "C.", and "D.".
Provide the correct answer as "A", "B", "C", or "D".
Also include a 1–2 sentence explanation per question.

Return ONLY valid JSON in this exact schema:

{{
  "questions": ["Question 1?", "Question 2?", "..."],
  "options": [
    ["A. ...", "B. ...", "C. ...", "D. ..."],
    ...
  ],
  "answers": ["A", "C", "..."],
  "explanations": ["short explanation 1", "short explanation 2", "..."]
}}

Study material:
{context}
""".strip()

    # Try/retry base generation
    last_raw = ""
    for attempt in range(retries + 1):
        resp = model.generate_content(base_prompt)
        raw = resp.text or ""
        last_raw = raw
        cleaned = clean_json_fences(raw)
        try:
            json.loads(cleaned)
            return cleaned
        except Exception:
            if attempt < retries:
                continue

    # If still invalid, attempt a targeted repair
    repair_prompt = f"""
Fix the following text so that it becomes EXACTLY valid JSON matching this schema:

Schema:
{{
  "questions": ["Question 1?", "..."],
  "options": [["A. ...", "B. ...", "C. ...", "D. ..."], ...],
  "answers": ["A","B","C","D", ...],
  "explanations": ["...", "...", ...]
}}

Text to fix (may contain prose/errors):
{last_raw}

Return ONLY the corrected JSON. No prose.
""".strip()

    repair_resp = model.generate_content(repair_prompt)
    repaired = clean_json_fences(repair_resp.text or "")
    # Final parse (or raise)
    json.loads(repaired)
    return repaired


def parse_quiz(quiz_json: str):
    data = json.loads(quiz_json)
    # Basic validation
    q = data.get("questions", [])
    o = data.get("options", [])
    a = data.get("answers", [])
    e = data.get("explanations", [])
    if not (len(q) == len(o) == len(a) == len(e) and len(q) > 0):
        raise ValueError("Quiz JSON failed validation: lengths mismatch or empty.")
    # normalize answer letters (just in case)
    data["answers"] = [str(ans).strip().upper()[:1] for ans in a]
    return data


# ==============================
# HINTS & EXPLANATIONS (AGENT ACTIONS)
# ==============================
def generate_hint(question: str, context: str) -> str:
    """Summarize the retrieved context into a concise hint."""
    model = genai.GenerativeModel(
        TEXT_MODEL,
        generation_config={"temperature": 0.3, "max_output_tokens": 200},
        system_instruction="Provide short, spoiler-minimized hints from the context.",
    )
    prompt = f"""
Given the question:

{question}

Use ONLY the following context to craft a 1–2 sentence hint that helps recall the answer without giving it away:

Context:
{context}

Return just the hint sentence(s).
"""
    resp = model.generate_content(prompt)
    return (resp.text or "").strip()


def generate_followup_quiz(chunks, index, wrong_questions, total=3):
    """Create a smaller practice quiz focused on the wrong items."""
    focused_contexts = []
    for q in wrong_questions:
        focused_contexts.append(retrieve_context(index, chunks, q, k=3))
    combined = "\n\n".join(focused_contexts)
    repaired_json = generate_quiz_json(combined, total=min(total, len(wrong_questions)) or 3)
    return parse_quiz(repaired_json)


# ==============================
# FLASK TEMPLATES (INLINE)
# ==============================
HOME_TEMPLATE = """
<!doctype html>
<html>
<head>
  <title>Agentic Quiz Generator</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 40px; }
    input[type=text] { width: 520px; padding: 8px; }
    input[type=number] { width: 90px; padding: 6px; }
    button, input[type=submit] { padding: 10px 16px; border-radius: 10px; border: 1px solid #ccc; cursor: pointer; }
    .note { color: #666; font-size: 0.95em; }
  </style>
</head>
<body>
  <h1>Agentic Quiz Generator</h1>
  <form method="post" action="{{ url_for('generate') }}">
    <label>Topic or URL:</label><br>
    <input type="text" name="topic" placeholder="e.g., Telugu cinema OR https://en.wikipedia.org/wiki/Telugu_cinema" required>
    <br><br>
    <label>Number of questions:</label>
    <input type="number" name="nq" min="3" max="10" value="5">
    <br><br>
    <input type="submit" value="Generate Quiz">
  </form>
  <p class="note">Tip: Paste any Wikipedia URL or type a topic name. The agent builds a retriever, generates a quiz, and adapts follow-ups.</p>
</body>
</html>
"""

QUIZ_TEMPLATE = """
<!doctype html>
<html>
<head>
  <title>Take the Quiz</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 40px; }
    .q { margin-bottom: 20px; padding: 16px; border: 1px solid #eee; border-radius: 12px; }
    .title { font-weight: 600; margin-bottom: 10px; }
    .actions { margin-top: 10px; }
    button, input[type=submit] { padding: 10px 16px; border-radius: 10px; border: 1px solid #ccc; cursor: pointer; }
  </style>
</head>
<body>
  <h2>Quiz on: {{ topic }}</h2>
  <div class="q">
    <div class="title">Q{{ idx+1 }} of {{ total }}:</div>
    <div style="margin-bottom:10px;">{{ question }}</div>
    <form method="post" action="{{ url_for('answer') }}">
      {% for opt in options %}
        <label>
          <input type="radio" name="answer" value="{{ opt[0] }}" required> {{ opt }}
        </label><br>
      {% endfor %}
      <input type="hidden" name="idx" value="{{ idx }}">
      <div class="actions">
        <input type="submit" value="Submit">
        <a href="{{ url_for('hint', idx=idx) }}"><button type="button">Hint</button></a>
      </div>
    </form>
  </div>
  {% if hint %}
    <div class="q" style="background:#f9fafb;">
      <div class="title">Hint</div>
      <div>{{ hint }}</div>
    </div>
  {% endif %}
  <p><a href="{{ url_for('home') }}">Start over</a></p>
</body>
</html>
"""

RESULTS_TEMPLATE = """
<!doctype html>
<html>
<head>
  <title>Your Results</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 40px; }
    .card { margin-bottom: 16px; padding: 16px; border: 1px solid #eee; border-radius: 12px; }
    .correct { color: #0a7a2f; font-weight: 600; }
    .wrong { color: #b40022; font-weight: 600; }
    .ex { color: #444; }
    button, input[type=submit] { padding: 10px 16px; border-radius: 10px; border: 1px solid #ccc; cursor: pointer; }
  </style>
</head>
<body>
  <h2>Score: {{ score }} / {{ total }}</h2>
  {% for i in range(total) %}
    <div class="card">
      <div><b>Q{{ i+1 }}:</b> {{ quiz["questions"][i] }}</div>
      <div>
        Your answer:
        {% if user_answers[i] %}
          <span class="{{ 'correct' if user_answers[i]==quiz['answers'][i] else 'wrong' }}">{{ user_answers[i] }}</span>
        {% else %}
          <em>not answered</em>
        {% endif %}
      </div>
      <div>Correct answer: <b>{{ quiz["answers"][i] }}</b></div>
      <div class="ex">Explanation: {{ quiz["explanations"][i] }}</div>
    </div>
  {% endfor %}
  <form method="post" action="{{ url_for('practice') }}">
    <input type="submit" value="Practice only the ones I missed">
  </form>
  <p><a href="{{ url_for('home') }}">Create another quiz</a></p>
</body>
</html>
"""


# ==============================
# HELPERS
# ==============================
def get_sid():
    if "sid" not in session:
        session["sid"] = str(uuid.uuid4())
    return session["sid"]


def ensure_store(sid):
    if sid not in MEM:
        MEM[sid] = {}
    return MEM[sid]


# ==============================
# ROUTES
# ==============================
@app.route("/", methods=["GET"])
def home():
    return render_template_string(HOME_TEMPLATE)


@app.route("/generate", methods=["POST"])
def generate():
    sid = get_sid()
    store = ensure_store(sid)

    topic = request.form.get("topic", "").strip()
    total = int(request.form.get("nq", 5) or 5)
    total = max(3, min(10, total))  # clamp 3..10

    # 1) Fetch & prep content
    try:
        text = fetch_content(topic)
    except Exception as e:
        return f"<h3>Failed to fetch content:</h3><pre>{e}</pre>"

    # 2) Chunk + FAISS
    chunks = chunk_text(text, max_words=120) if len(text) >= 200 else [text]
    try:
        index, _ = build_faiss_index(chunks)
    except Exception:
        # fallback: minimal index over raw text
        chunks = [text]
        index, _ = build_faiss_index(chunks)

    # save large objects in server memory
    store["index"] = index
    store["chunks"] = chunks

    # 3) Build focused context (broad seed from topic)
    context = retrieve_context(index, chunks, topic, k=6)

    # 4) Ask Gemini to produce strict JSON (with repair loop)
    try:
        quiz_json = generate_quiz_json(context, total=total)
        quiz = parse_quiz(quiz_json)
    except Exception as e:
        return f"<h3>Quiz generation failed:</h3><pre>{e}</pre>"

    # Initialize quiz session state
    session["quiz"] = quiz
    session["topic"] = topic
    session["idx"] = 0
    session["answers"] = []
    session["wrong_qs"] = []
    session.modified = True

    return redirect(url_for("quiz"))


@app.route("/quiz", methods=["GET"])
def quiz():
    sid = get_sid()
    store = ensure_store(sid)

    quiz = session.get("quiz")
    topic = session.get("topic", "your topic")
    idx = session.get("idx", 0)

    if not quiz:
        return redirect(url_for("home"))

    total = len(quiz["questions"])
    if idx >= total:
        return redirect(url_for("results"))

    question = quiz["questions"][idx]
    options = quiz["options"][idx]
    hint_text = session.pop("hint", None)  # show once

    return render_template_string(
        QUIZ_TEMPLATE,
        topic=topic,
        idx=idx,
        total=total,
        question=question,
        options=options,
        hint=hint_text
    )

@app.route("/debug")
def debug():
    debug_info = {}
    for sid in MEM:
        docs = MEM[sid]["data"]
        debug_info[sid] = {
            "count": len(docs),
            "samples": [doc[:120] + "..." for doc in docs]  # preview text
        }
    print("DEBUG - Current memory:", debug_info)  # also logs to console
    return jsonify(debug_info)


@app.route("/hint", methods=["GET"])
def hint():
    sid = get_sid()
    store = ensure_store(sid)
    quiz = session.get("quiz")
    idx = int(request.args.get("idx", session.get("idx", 0)))

    if not quiz or "index" not in store or "chunks" not in store:
        return redirect(url_for("home"))

    question = quiz["questions"][idx]
    # Per-question retrieval for hint
    ctx = retrieve_context(store["index"], store["chunks"], question, k=3)
    hint_text = generate_hint(question, ctx)
    session["hint"] = hint_text
    session.modified = True
    return redirect(url_for("quiz"))


@app.route("/answer", methods=["POST"])
def answer():
    sid = get_sid()
    store = ensure_store(sid)

    quiz = session.get("quiz")
    if not quiz:
        return redirect(url_for("home"))

    idx = int(request.form.get("idx", session.get("idx", 0)))
    user_ans = request.form.get("answer", "").strip().upper()[:1]

    # Record answer
    answers = session.get("answers", [])
    # extend list up to current idx
    while len(answers) < idx:
        answers.append("")  # missing previous (shouldn't happen)
    if len(answers) == idx:
        answers.append(user_ans)
    else:
        answers[idx] = user_ans
    session["answers"] = answers

    # Track wrong questions for follow-up
    correct = quiz["answers"][idx]
    wrong_qs = session.get("wrong_qs", [])
    if user_ans != correct:
        wrong_qs.append(quiz["questions"][idx])
    session["wrong_qs"] = wrong_qs

    # Next question
    session["idx"] = idx + 1
    session.modified = True

    total = len(quiz["questions"])
    if session["idx"] >= total:
        return redirect(url_for("results"))
    return redirect(url_for("quiz"))


@app.route("/results", methods=["GET"])
def results():
    quiz = session.get("quiz")
    if not quiz:
        return redirect(url_for("home"))

    total = len(quiz["questions"])
    answers = session.get("answers", [])
    # pad answers if needed
    if len(answers) < total:
        answers += [""] * (total - len(answers))

    score = sum(1 for i in range(total) if i < len(answers) and answers[i] == quiz["answers"][i])

    return render_template_string(
        RESULTS_TEMPLATE,
        score=score,
        total=total,
        quiz=quiz,
        user_answers=answers,
    )


@app.route("/practice", methods=["POST"])
def practice():
    """Generate a smaller follow-up quiz focusing on wrong questions."""
    sid = get_sid()
    store = ensure_store(sid)

    quiz = session.get("quiz")
    wrong_qs = session.get("wrong_qs", [])
    if not quiz or not wrong_qs or "index" not in store or "chunks" not in store:
        return redirect(url_for("home"))

    try:
        follow_quiz = generate_followup_quiz(
            chunks=store["chunks"],
            index=store["index"],
            wrong_questions=wrong_qs,
            total=min(5, len(wrong_qs)) or 3
        )
    except Exception as e:
        return f"<h3>Follow-up generation failed:</h3><pre>{e}</pre>"

    # Reset session for new quiz flow
    session["quiz"] = follow_quiz
    session["idx"] = 0
    session["answers"] = []
    session["wrong_qs"] = []
    session.modified = True

    return redirect(url_for("quiz"))


# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    # Run: python teaching_agent.py
    # Then open http://127.0.0.1:5000
    app.run(debug=True)
