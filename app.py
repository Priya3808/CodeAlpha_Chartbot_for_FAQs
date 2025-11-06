from flask import Flask, render_template, request, jsonify
import math

# Try to import sklearn for TF-IDF + cosine similarity. If not available, fallback to difflib.
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False
    from difflib import SequenceMatcher

app = Flask(__name__)

# --- SAMPLE FAQ DATA (you can expand this)
FAQS = [
    ("What is CodeAlpha internship?", "CodeAlpha internship is a program that offers hands-on AI and software projects for students."),
    ("How do I submit my project?", "Upload your project to GitHub, post a short demo video on LinkedIn, and submit the repo link using the submission form."),
    ("Which languages are supported?", "Our sample tool supports English, Hindi, Kannada, Tamil, Telugu, French. You can add more languages in the code."),
    ("How to get a certificate?", "Certificates are given after successful completion — check the internship perks in your task sheet."),
    ("How do I run the projects?", "Use Python 3.8+, create a virtual environment, install requirements, then run python app.py and open http://127.0.0.1:5000"),
    ("Can I get a recommendation letter?", "A letter of recommendation is provided based on performance. Speak to your mentor after completion."),
    ("How can I contact support?", "You can contact CodeAlpha support via the email or Slack channel provided when you joined the internship.")
]

# Precompute structures for fast lookup
QUESTIONS = [q for q, a in FAQS]
ANSWERS = [a for q, a in FAQS]

if SKLEARN_AVAILABLE:
    vectorizer = TfidfVectorizer().fit(QUESTIONS)
    question_vectors = vectorizer.transform(QUESTIONS)
else:
    # No preprocessing needed for difflib fallback
    vectorizer = None
    question_vectors = None

def find_best_answer(user_text):
    user_text = (user_text or "").strip()
    if not user_text:
        return {"answer": "Please ask a question.", "score": 0.0, "index": None}

    # Use sklearn TF-IDF + cosine similarity if available
    if SKLEARN_AVAILABLE:
        user_vec = vectorizer.transform([user_text])
        sims = cosine_similarity(user_vec, question_vectors).flatten()
        best_idx = int(sims.argmax())
        score = float(sims[best_idx])
        return {"answer": ANSWERS[best_idx], "score": score, "index": best_idx}
    else:
        # Fallback: use difflib SequenceMatcher ratio on raw text
        best_idx = None
        best_score = -1.0
        for i, q in enumerate(QUESTIONS):
            ratio = SequenceMatcher(None, user_text.lower(), q.lower()).ratio()
            if ratio > best_score:
                best_score = ratio
                best_idx = i
        return {"answer": ANSWERS[best_idx], "score": float(best_score), "index": best_idx}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json or {}
    text = data.get('text', '').strip()
    try:
        res = find_best_answer(text)
        # Round score to 3 decimals for nice display
        res['score'] = round(res.get('score', 0.0), 3)
        return jsonify({"ok": True, "question": text, "answer": res['answer'], "score": res['score']})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)