from flask import Flask, render_template, request
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def compute_similarity(text1, text2):
    emb1 = model.encode([text1])
    emb2 = model.encode([text2])
    return cosine_similarity(emb1, emb2)[0][0]

def get_feedback(score):
    if score > 0.75:
        return f"✅ Strong match ({score:.2f}): Your resume aligns well!"
    elif score > 0.5:
        return f"🟡 Moderate match ({score:.2f}): Add more relevant skills."
    else:
        return f"🔴 Weak match ({score:.2f}): Consider tailoring your resume."

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        resume = request.files['resume']
        jd_text = request.form['jd']

        resume_text = extract_text_from_pdf(resume)
        score = compute_similarity(resume_text, jd_text)
        feedback = get_feedback(score)

        return render_template("index.html", feedback=feedback, score=score, jd_text=jd_text)

    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)
