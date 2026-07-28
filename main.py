# main.py
import os
import uuid
from pathlib import Path
from io import BytesIO
from tempfile import NamedTemporaryFile

from flask import Flask, render_template_string, request, send_file
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from typing import TypedDict
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import pdfplumber

load_dotenv()

# ── Flask setup ────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = "resume-improver-secret-key"

# ── LLMs with stricter evaluator ───────────────────────────────────────
evaluator_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
optimizer_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)

class ResumeEvaluation(BaseModel):
    score: int = Field(description="Strict overall resume score out of 10")
    feedback: str = Field(description="Actionable feedback for improvement")

structured_evaluator_llm = evaluator_llm.with_structured_output(ResumeEvaluation)

class ResumeState(TypedDict):
    resume: str
    score: int
    feedback: str
    iteration: int
    max_iteration: int

def evaluate_resume(state: ResumeState):
    prompt = f"""
    You are a senior recruiter at a top tech company. Critically evaluate the following resume using this exact 10‑point scale:

    - 1‑3: Major issues – missing sections, poor formatting, spelling/grammar errors, no measurable achievements.
    - 4‑5: Below average – content exists but weak, lacks keywords, no quantification.
    - 6‑7: Average – decent structure, some keywords, a few measurable results.
    - 8‑9: Strong – excellent formatting, clear achievements, impact quantified, relevant skills strongly highlighted.
    - 10: Exceptional – flawless, every section optimized, outstanding impact statements, highly tailored to role.

    Evaluate:
    - Formatting & readability
    - Presence of keywords (skills, tools, methodologies)
    - Quantified accomplishments (%, $, numbers)
    - Relevance to typical software/tech roles
    - Overall impact

    Resume:
    {state["resume"]}

    Return:
    1. Score (1-10) - be precise and critical.
    2. Feedback - 2-3 bullet points on what to improve.
    """
    response = structured_evaluator_llm.invoke(prompt)
    return {"score": response.score, "feedback": response.feedback}

def route_resume(state: ResumeState):
    if state["score"] >= 8 or state["iteration"] >= state["max_iteration"]:
        return "approved"
    return "improve"

def improve_resume(state: ResumeState):
    prompt = f"""Act as a professional resume writer. Improve the resume based on this feedback:
{state["feedback"]}

Current resume:
{state['resume']}

Return only the improved resume text, with no additional commentary."""
    improved = optimizer_llm.invoke(prompt).content
    return {"resume": improved, "iteration": state["iteration"] + 1}

# ── Workflow graph ─────────────────────────────────────────────────────
workflow = StateGraph(ResumeState)
workflow.add_node("evaluate_resume", evaluate_resume)
workflow.add_node("improve_resume", improve_resume)
workflow.add_edge(START, "evaluate_resume")
workflow.add_conditional_edges("evaluate_resume", route_resume, {"approved": END, "improve": "improve_resume"})
workflow.add_edge("improve_resume", "evaluate_resume")
workflow = workflow.compile()

def run_resume_workflow(initial_resume: str, max_iterations: int = 3):
    state = {
        "resume": initial_resume,
        "score": 0,
        "feedback": "",
        "iteration": 0,
        "max_iteration": max_iterations
    }
    steps = []
    current = state.copy()
    for i in range(1, max_iterations + 1):
        eval_out = evaluate_resume(current)
        current.update(eval_out)
        steps.append({
            "iteration": i,
            "score": current["score"],
            "feedback": current["feedback"],
            "resume_before": current["resume"]
        })
        if current["score"] >= 8 or i >= max_iterations:
            break
        improved = improve_resume(current)
        current.update(improved)
        steps[-1]["resume_after"] = current["resume"]
    return current, steps

def extract_text_from_pdf(file_storage) -> str:
    with pdfplumber.open(BytesIO(file_storage.read())) as pdf:
        text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    return text.strip()

# ── Apple Design HTML Template ─────────────────────────────────────────
HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Resume Improver</title>
<style>
  :root {
    --primary: #0066cc;
    --ink: #1d1d1f;
    --body-on-dark: #ffffff;
    --ink-muted-48: #7a7a7a;
    --hairline: #e0e0e0;
    --canvas: #ffffff;
    --canvas-parchment: #f5f5f7;
    --surface-tile-1: #272729;
    --surface-black: #000000;
    --shadow-product: 0 5px 30px rgba(0,0,0,0.22);
    --rounded-pill: 9999px;
    --rounded-lg: 18px;
    --spacing-lg: 24px;
    --spacing-xl: 32px;
    --spacing-section: 80px;
  }
  *, *::before, *::after { margin:0; padding:0; box-sizing:border-box; }
  body {
    font-family: 'SF Pro Text', system-ui, -apple-system, sans-serif;
    color: var(--ink);
    background: var(--canvas-parchment);
    -webkit-font-smoothing: antialiased;
  }
  .global-nav {
    background: var(--surface-black);
    height: 44px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--body-on-dark);
    font-size: 12px;
    letter-spacing: -0.12px;
  }
  .sub-nav {
    background: rgba(245,245,247,0.8);
    backdrop-filter: blur(20px);
    height: 52px;
    display: flex;
    align-items: center;
    padding: 0 var(--spacing-lg);
    border-bottom: 1px solid var(--hairline);
    justify-content: space-between;
    font-size: 21px;
    font-weight: 600;
    letter-spacing: 0.231px;
  }
  .content {
    max-width: 980px;
    margin: 0 auto;
    padding: var(--spacing-section) var(--spacing-lg);
  }
  .product-tile-light {
    background: var(--canvas);
    padding: var(--spacing-section) var(--spacing-xl);
    margin-bottom: var(--spacing-xl);
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
  }
  h1 {
    font-family: 'SF Pro Display', system-ui, -apple-system, sans-serif;
    font-size: 40px;
    font-weight: 600;
    line-height: 1.1;
    margin-bottom: var(--spacing-sm);
  }
  .lead {
    font-size: 28px;
    font-weight: 400;
    line-height: 1.14;
    color: var(--ink-muted-48);
    margin-bottom: var(--spacing-xl);
  }
  .btn-primary, .btn-secondary-pill {
    display: inline-block;
    padding: 11px 22px;
    border-radius: var(--rounded-pill);
    font-size: 17px;
    cursor: pointer;
    text-decoration: none;
    transition: all 0.3s;
  }
  .btn-primary {
    background: var(--primary);
    color: white;
    border: none;
  }
  .btn-secondary-pill {
    background: transparent;
    color: var(--primary);
    border: 1px solid var(--primary);
  }
  .btn-primary:hover, .btn-secondary-pill:hover { transform: scale(0.97); }
  textarea, input[type="file"] {
    width: 100%;
    padding: 12px 16px;
    font-family: inherit;
    font-size: 17px;
    border: 1px solid var(--hairline);
    border-radius: 8px;
    background: var(--canvas);
  }
  .form-group { margin-bottom: var(--spacing-lg); }
  /* PDF Preview */
  .pdf-preview-container {
    margin-top: var(--spacing-lg);
    border-radius: var(--rounded-lg);
    overflow: hidden;
    box-shadow: var(--shadow-product);
    background: white;
  }
  .pdf-preview-container iframe {
    width: 100%;
    height: 400px;
    border: none;
  }
  /* Loading Animation */
  .loading-overlay {
    position: fixed;
    top: 0; left: 0; width: 100%; height: 100%;
    background: rgba(255,255,255,0.9);
    backdrop-filter: blur(8px);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    z-index: 1000;
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.4s ease;
  }
  .loading-overlay.active {
    opacity: 1;
    pointer-events: auto;
  }
  .pulsing-pill {
    width: 120px;
    height: 48px;
    background: var(--primary);
    border-radius: var(--rounded-pill);
    box-shadow: 0 0 20px rgba(0,102,204,0.4);
    animation: pulse 1.5s infinite ease-in-out;
  }
  @keyframes pulse {
    0% { transform: scale(0.95); opacity: 0.7; }
    50% { transform: scale(1.05); opacity: 1; }
    100% { transform: scale(0.95); opacity: 0.7; }
  }
  .progress-bar {
    width: 300px;
    height: 6px;
    background: #eee;
    border-radius: 3px;
    margin: 24px 0;
  }
  .progress-fill {
    height: 100%;
    width: 0%;
    background: var(--primary);
    border-radius: 3px;
    animation: progress 2s infinite linear;
  }
  @keyframes progress {
    0% { width: 0%; }
    50% { width: 70%; }
    100% { width: 100%; }
  }
  .loading-text {
    font-size: 21px;
    font-weight: 600;
    color: var(--ink);
    opacity: 0;
    animation: fadeText 2s forwards;
  }
  @keyframes fadeText {
    0% { opacity: 0; transform: translateY(8px); }
    100% { opacity: 1; transform: translateY(0); }
  }
  /* Results tiles */
  .resume-tile {
    padding: var(--spacing-xl);
    border-radius: var(--rounded-lg);
    margin-bottom: var(--spacing-lg);
    white-space: pre-wrap;
    word-break: break-word;
  }
  .tile-light { background: var(--canvas); color: var(--ink); }
  .tile-dark  { background: var(--surface-tile-1); color: var(--body-on-dark); }
  .score-card {
    background: var(--canvas);
    border-radius: var(--rounded-lg);
    padding: var(--spacing-lg);
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    margin-bottom: var(--spacing-lg);
  }
  .score-num {
    font-size: 34px;
    font-weight: 600;
    color: var(--primary);
    margin-right: 8px;
  }
</style>
</head>
<body>

<div class="global-nav"> Resume Improver </div>
<div class="sub-nav">
  <span>Resume Optimizer</span>
  <span style="font-size:14px; font-weight:400; color:var(--ink-muted-48);">Score ≥ 8 → stop</span>
</div>

<!-- Loading Overlay -->
<div class="loading-overlay" id="loadingOverlay">
  <div class="pulsing-pill"></div>
  <div class="progress-bar"><div class="progress-fill"></div></div>
  <div class="loading-text">Analyzing your resume...</div>
</div>

<div class="content">

{% if not result %}
  <!-- Upload Form -->
  <div class="product-tile-light">
    <h1>Improve your resume.</h1>
    <p class="lead">Upload a PDF or paste text. The AI will score and optimize it.</p>
    <form id="resumeForm" method="POST" enctype="multipart/form-data" onsubmit="showLoading()">
      <div class="form-group">
        <label style="font-weight:600; display:block; margin-bottom:4px;">Upload PDF</label>
        <input type="file" id="pdfFile" name="pdf_file" accept=".pdf" onchange="previewPDF()">
      </div>
      <div class="form-group">
        <label style="font-weight:600; display:block; margin-bottom:4px;">… or paste text</label>
        <textarea name="resume_text" rows="10" placeholder="Paste your resume here..."></textarea>
      </div>
      <div style="display:flex; gap:12px;">
        <button type="submit" class="btn-primary">Analyze & Improve</button>
        <button type="button" class="btn-secondary-pill" onclick="document.getElementById('pdfFile').value=''; document.getElementById('pdfPreview').innerHTML='';">Clear</button>
      </div>
    </form>
    <!-- PDF Preview will appear here -->
    <div id="pdfPreview" class="pdf-preview-container" style="display:none;"></div>
  </div>

{% else %}
  <!-- Results Page -->
  <a href="/" class="btn-secondary-pill" style="margin-bottom:16px;">← New Resume</a>
  <h1 style="font-size:34px;">Final Improved Resume</h1>
  
  <!-- Final resume dark tile for contrast -->
  <div class="resume-tile tile-dark">{{ result.resume | replace('\n', '<br>') | safe }}</div>

  <!-- Score & Feedback -->
  <div class="score-card">
    <span class="score-num">{{ result.score }}/10</span><span style="color:var(--ink-muted-48);">after {{ result.iteration }} iteration(s)</span>
    <p style="margin-top:12px; line-height:1.5;">{{ result.feedback }}</p>
  </div>

  <!-- Iteration Steps -->
  <h2 style="font-family:'SF Pro Display',sans-serif; font-size:28px; font-weight:600; margin:32px 0 16px;">Progress</h2>
  {% for step in steps %}
  <div class="score-card">
    <span class="score-num" style="font-size:21px;">Iteration {{ step.iteration }} – Score {{ step.score }}/10</span>
    <p style="margin:12px 0;">{{ step.feedback }}</p>
    <div style="font-weight:600; margin:12px 0 8px;">After improvement:</div>
    <div class="resume-tile {% if step.iteration % 2 == 0 %}tile-dark{% else %}tile-light{% endif %}" style="max-height:200px; overflow-y:auto;">
      {{ step.resume_after | replace('\n', '<br>') | safe }}
    </div>
  </div>
  {% endfor %}
{% endif %}
</div>

<script>
  // PDF preview
  function previewPDF() {
    const fileInput = document.getElementById('pdfFile');
    const previewDiv = document.getElementById('pdfPreview');
    if (!fileInput.files || !fileInput.files[0]) {
      previewDiv.style.display = 'none';
      return;
    }
    const file = fileInput.files[0];
    if (file.type !== 'application/pdf') return;
    const blobUrl = URL.createObjectURL(file);
    previewDiv.innerHTML = `<iframe src="${blobUrl}" title="PDF Preview"></iframe>`;
    previewDiv.style.display = 'block';
    // Clean up blob after iframe loads to avoid memory leaks
    const iframe = previewDiv.querySelector('iframe');
    iframe.onload = () => URL.revokeObjectURL(blobUrl);
  }

  // Loading state
  function showLoading() {
    document.getElementById('loadingOverlay').classList.add('active');
    // Optionally disable form submission if already submitted
    const btn = document.querySelector('.btn-primary');
    if (btn) btn.disabled = true;
  }
</script>

</body>
</html>
"""

# ── Routes ─────────────────────────────────────────────────────────────
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        resume_text = None
        if "pdf_file" in request.files and request.files["pdf_file"].filename:
            try:
                resume_text = extract_text_from_pdf(request.files["pdf_file"])
            except Exception as e:
                return render_template_string(HTML_TEMPLATE, result=None, error=f"PDF error: {e}")
        elif request.form.get("resume_text", "").strip():
            resume_text = request.form["resume_text"].strip()
        
        if not resume_text:
            return render_template_string(HTML_TEMPLATE, result=None, error="No resume provided.")
        
        final_state, steps = run_resume_workflow(resume_text)
        result = {
            "resume": final_state["resume"],
            "score": final_state["score"],
            "feedback": final_state["feedback"],
            "iteration": final_state["iteration"]
        }
        return render_template_string(HTML_TEMPLATE, result=result, steps=steps, error=None)
    
    return render_template_string(HTML_TEMPLATE, result=None, steps=None, error=None)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
