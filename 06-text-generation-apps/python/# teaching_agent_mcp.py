import os
import requests
import sqlite3
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify, render_template_string
import google.generativeai as genai
import faiss
import numpy as np
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

app = Flask(__name__)

# --- FAISS Setup ---
dimension = 768
index = faiss.IndexFlatL2(dimension)
docstore = []

# --- SQLite Setup for scores ---
DB_FILE = "quiz_scores.db"
conn = sqlite3.connect(DB_FILE, check_same_thread=False)
c = conn.cursor()
c.execute("""
CREATE TABLE IF NOT EXISTS scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user TEXT,
    topic TEXT,
    score INTEGER
)
""")
conn.commit()

# ----------------------------
# MCP TOOL REGISTRY
# ----------------------------
TOOLS = {
    "generate_quiz": {
        "description": "Generate a multiple-choice quiz from a topic or URL",
        "args": {"topic": "string", "num_questions": "integer"}
    },
    "save_score": {
        "description": "Save a user's quiz score into the database",
        "args": {"user": "string", "topic": "string", "score": "integer"}
    }
}

# --- TOOL 1: GENERATE QUIZ ---
@app.route("/mcp/generate_quiz", methods=["POST"])
def generate_quiz():
    data = request.json
    topic = data.get("topic")
    num_q = int(data.get("num_questions", 5))

    # If topic looks like URL → scrape
    if topic.startswith("http"):
        html = requests.get(topic).text
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text()
    else:
        text = topic

    # Store in FAISS
    embedding = genai.embed_content(
        model="models/embedding-001",
        content=text
    )["embedding"]
    index.add(np.array([embedding], dtype="float32"))
    docstore.append(text)

    # Generate quiz
    prompt = f"Create a {num_q}-question MCQ quiz from:\n{text[:2000]}"
    response = genai.GenerativeModel("gemini-pro").generate_content(prompt)

    return jsonify({"quiz": response.text})

# --- TOOL 2: SAVE SCORE ---
@app.route("/mcp/save_score", methods=["POST"])
def save_score():
    data = request.json
    user = data.get("user", "guest")
    topic = data.get("topic", "unknown")
    score = int(data.get("score", 0))

    c.execute("INSERT INTO scores (user, topic, score) VALUES (?, ?, ?)", (user, topic, score))
    conn.commit()

    return jsonify({"message": f"Score saved for {user} on {topic}: {score} points"})

# --- MCP: LIST TOOLS ---
@app.route("/mcp/list_tools", methods=["GET"])
def list_tools():
    return jsonify(TOOLS)

# --- MCP: AGENT DISPATCHER ---
@app.route("/mcp/agent", methods=["POST"])
def agent():
    user_prompt = request.json.get("prompt")

    tool_prompt = f"""
    You are an MCP agent. Available tools:

    {TOOLS}

    User request: {user_prompt}

    Decide the tool and arguments in JSON:
    {{
      "tool": "...",
      "arguments": {{...}}
    }}
    """

    decision = genai.GenerativeModel("gemini-pro").generate_content(tool_prompt).text

    import json
    try:
        decision_json = json.loads(decision)
    except:
        return jsonify({"error": "LLM returned invalid JSON", "raw": decision})

    tool = decision_json.get("tool")
    args = decision_json.get("arguments", {})

    # Route request
    if tool == "generate_quiz":
        with app.test_client() as client:
            resp = client.post("/mcp/generate_quiz", json=args)
            return resp.get_data(as_text=True)

    elif tool == "save_score":
        with app.test_client() as client:
            resp = client.post("/mcp/save_score", json=args)
            return resp.get_data(as_text=True)

    return jsonify({"error": f"Unknown tool {tool}"})


# ----------------------------
# BASIC HTML FRONTEND
# ----------------------------
HOME_TEMPLATE = """
<!doctype html>
<html>
<head>
  <title>MCP Quiz Agent</title>
</head>
<body>
  <h1>MCP Quiz Agent</h1>
  <form method="post" action="/chat">
    <input type="text" name="prompt" placeholder="Ask me anything..." size="60">
    <button type="submit">Send</button>
  </form>
  {% if result %}
    <h3>Result:</h3>
    <pre>{{ result }}</pre>
  {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    return render_template_string(HOME_TEMPLATE)

@app.route("/chat", methods=["POST"])
def chat():
    prompt = request.form.get("prompt")
    resp = requests.post("http://localhost:5000/mcp/agent", json={"prompt": prompt})
    return render_template_string(HOME_TEMPLATE, result=resp.text)

if __name__ == "__main__":
    app.run(debug=True)
