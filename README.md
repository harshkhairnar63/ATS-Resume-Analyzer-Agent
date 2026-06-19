# ATS-Resume-Analyzer-Agent
Built an Agentic AI system using LangGraph and LangChain that analyzes uploaded resumes, generates ATS scores, provides improvement suggestions, and iteratively rewrites resumes using LLM-based evaluation workflows.

🚀 Project Overview

The ATS Resume Analyzer Agent is built using LangGraph, LangChain, and Large Language Models (LLMs) to simulate how modern recruiters and ATS systems evaluate resumes.

Unlike traditional resume checkers that provide static feedback, this project uses an agent-based workflow where multiple AI agents collaborate to:

Analyze resume content
Compare it against job requirements
Generate ATS scores
Identify missing keywords
Suggest improvements
Rewrite weak sections
Re-evaluate the updated resume
Produce a final optimized version

This creates a continuous feedback loop that improves resume quality automatically.

🎯 Problem Statement

Many candidates get rejected before a recruiter even reads their resume because ATS systems filter resumes based on:

Keyword matching
Skill relevance
Resume structure
Experience alignment
Education requirements

Most job seekers do not know why their resumes are rejected.

This project helps candidates understand ATS behavior and improve their resumes using AI-powered recommendations.

🏗️ System Architecture
1. Resume Parsing Agent

Extracts information from uploaded resumes.

Extracted Details:

Name
Contact Information
Skills
Education
Projects
Experience
Certifications
2. ATS Evaluation Agent

Analyzes:

Resume formatting
Keyword density
Technical skills
Soft skills
Experience relevance
Education relevance

Outputs:

ATS Score (0–100)
Strengths
Weaknesses
Missing Keywords
3. Feedback Agent

Generates actionable recommendations such as:

Add measurable achievements
Include missing technologies
Improve project descriptions
Quantify impact using metrics

Example:

❌ Developed a chatbot using Python.

✅ Built an AI-powered chatbot using Python and LangChain, reducing response time by 40%.

4. Resume Rewriter Agent

Automatically rewrites:

Summary section
Project descriptions
Experience bullet points
Skills section

while preserving factual information.

5. Quality Review Agent

Reviews the rewritten resume and checks:

ATS compatibility
Readability
Professional tone
Keyword optimization
6. Decision Agent

Determines whether:

Resume quality is acceptable
Another improvement cycle is needed

This creates a self-improving loop using LangGraph.

🔄 Agent Workflow
Resume Upload
       │
       ▼
 Resume Parser
       │
       ▼
 ATS Evaluator
       │
       ▼
 Feedback Generator
       │
       ▼
 Resume Rewriter
       │
       ▼
 Quality Reviewer
       │
       ▼
  Good Enough?
    │      │
   No      Yes
    │       │
    └──────►Final Resume
🛠️ Tech Stack
AI Frameworks
LangGraph
LangChain
OpenAI / Groq LLMs
Python
NLP & Processing
PyPDF2
PDFPlumber
Regex
Text Processing
Backend
FastAPI
Flask
Frontend
Streamlit
Storage
SQLite
PostgreSQL (Optional)
✨ Key Features
ATS Score Generation

Produces a detailed ATS compatibility score.

Example:

Overall ATS Score: 82/100

Keyword Match: 88%
Skills Match: 80%
Experience Relevance: 75%
Formatting: 90%
Keyword Gap Analysis

Identifies missing skills from job descriptions.

Example:

Missing Keywords:
- LangGraph
- Docker
- Kubernetes
- REST API
AI Resume Enhancement

Transforms weak resume content into recruiter-friendly descriptions.

Example:

Before:
Created a website using React.

After:
Developed a responsive React-based web application with dynamic state management, improving user engagement and performance.
Multi-Agent Collaboration

Each AI agent has a specific responsibility:

Agent	Responsibility
Parser Agent	Extract resume information
ATS Agent	Calculate ATS score
Feedback Agent	Suggest improvements
Rewriter Agent	Improve content
Reviewer Agent	Evaluate quality
Decision Agent	Control workflow
📊 Example Use Case
Input

A student uploads a resume for an AI Engineer role.

System Analysis
ATS Score: 68/100

Missing Skills:
- LangGraph
- Vector Databases
- RAG
- FastAPI

Weak Areas:
- Generic project descriptions
- No quantified achievements
Output
ATS Score After Optimization: 89/100

Improvements:
✓ Added AI-related keywords
✓ Enhanced project descriptions
✓ Improved professional summary
✓ Optimized skill categorization
📈 Future Enhancements
Multi-job comparison
Resume ranking system
Cover letter generation
LinkedIn profile optimization
Voice-based resume review
RAG-powered industry-specific recommendations
Resume-to-Portfolio website generator
🎓 Learning Outcomes

Through this project, I gained practical experience in:

Agentic AI Development
LangGraph Workflow Design
LangChain Integration
Prompt Engineering
LLM Evaluation Systems
Multi-Agent Architectures
Resume Parsing & NLP
FastAPI Development
State Management in AI Agents
Human-in-the-Loop AI Systems
