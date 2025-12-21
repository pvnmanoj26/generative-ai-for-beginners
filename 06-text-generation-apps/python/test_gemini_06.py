import os
import requests
import numpy as np
from flask import Flask, request, render_template_string
from dotenv import load_dotenv
import google.generativeai as genai
import faiss
from bs4 import BeautifulSoup

# Load environment variable
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Configure Gemini
genai.configure(api_key=gemini_api_key)

# -------- Utility Functions -------- #
def get_website_content(url):
    """Fetch all <p> text content from a webpage."""
    response = requests.get(url)
    if response.status_code != 200:
        return "Failed to retrieve content."
    soup = BeautifulSoup(response.text, 'html.parser')
    paragraphs = soup.find_all('p')
    return "\n".join(p.get_text() for p in paragraphs).strip()

def chunk_text(text, max_words=100):
    """Split text into word-based chunks."""
    words = text.split()
    return [' '.join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

def get_gemini_response(system_instruction, prompt, temperature=0.9):
    """Use Gemini 2.0 to generate a response."""
    try:
        model = genai.GenerativeModel(
            "models/gemini-1.5-flash",
            system_instruction=system_instruction,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": 800
            }
        )
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print("Gemini API Error:", e)
        return "Sorry, I couldn't generate a response."

# -------- Document Setup -------- #
url = "https://byjus.com/biology/human-heart/"
content = get_website_content(url)
chunks = chunk_text(content)

# Embed the chunks
embeddings = [
    genai.embed_content(
        model="models/embedding-001",
        content=chunk,
        task_type="retrieval_document"
    )['embedding'] for chunk in chunks
]
embedding_matrix = np.array(embeddings).astype("float32")

# Create FAISS index
dimension = embedding_matrix.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embedding_matrix)

# -------- Flask Setup -------- #
app = Flask(__name__)

TEMPLATE = """
<!doctype html>
<title>Biology Q&A</title>
<h2>Ask a Biology Question</h2>
<form method="post">
  <input name="query" size="80">
  <input type="submit">
</form>
{% if answer %}
  <h3>Answer:</h3>
  <p>{{ answer }}</p>
{% endif %}
"""

@app.route("/", methods=["GET", "POST"])
def home():
    answer = None
    if request.method == "POST":
        user_input = request.form["query"]

        # Step 1: Embed the user question
        query_embedding = genai.embed_content(
            model="models/embedding-001",
            content=user_input,
            task_type="retrieval_query"
        )['embedding']

        # Step 2: Get top-k similar chunks from FAISS
        D, I = index.search(np.array([query_embedding]).astype("float32"), k=3)
        relevant_chunks = [chunks[i] for i in I[0]]
        context = "\n\n".join(relevant_chunks)

        # Step 3: Prompt the Gemini LLM with context
        prompt = f"""Use the following context to answer the user's question:

Context:
{context}

User question: {user_input}
"""
        system_instruction = "You are a helpful AI biology tutor. Use only the given context to answer user questions factually and clearly."
        answer = get_gemini_response(system_instruction, prompt)

    return render_template_string(TEMPLATE, answer=answer)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
