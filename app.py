"""
CareerCompass — single-file Flask application.
Includes all career roadmaps, career/resource libraries, AI helpers,
DB models, and routes in one self-contained app.py.
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import re
import json
import os
import urllib.parse
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.dialects import mysql
from datetime import datetime

import pymysql
pymysql.install_as_MySQLdb()  # lets SQLAlchemy use PyMySQL as if it were MySQLdb

# ─── App & DB setup ────────────────────────────────────────────────────────────

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "careercompass-dev-secret-2026")

# DB_ENGINE controls whether we connect to MySQL or fall back to local SQLite.
# The app now defaults to SQLite so it runs out of the box in local development.
db_engine = os.environ.get("DB_ENGINE", "sqlite").strip().lower()

if db_engine == "mysql":
    db_user     = os.environ.get("DB_USER", "root")
    db_password = urllib.parse.quote_plus(os.environ.get("DB_PASSWORD", ""))
    db_host     = os.environ.get("DB_HOST", "localhost")
    db_port     = os.environ.get("DB_PORT", "3306")
    db_name     = os.environ.get("DB_NAME", "careercompass")

    database_url = os.environ.get("DATABASE_URL") or (
        f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}?charset=utf8mb4"
    )
else:
    database_url = os.environ.get("DATABASE_URL") or "sqlite:///careercompass.db"

app.config["SQLALCHEMY_DATABASE_URI"] = database_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 280,    # avoids MySQL's "server has gone away" on idle connections
    "pool_pre_ping": True,  # checks connection health before using it
}

db = SQLAlchemy(app)


# ─── Models ────────────────────────────────────────────────────────────────────

class Student(db.Model):
    __tablename__ = "students"
    id             = db.Column(db.Integer, primary_key=True, autoincrement=True)
    full_name      = db.Column(db.String(120), nullable=False)
    email          = db.Column(db.String(180), nullable=False)
    age            = db.Column(db.SmallInteger, nullable=False)
    semester       = db.Column(db.String(30), nullable=False)
    subject        = db.Column(db.String(80), nullable=False)
    skill_level    = db.Column(db.String(20), nullable=False, default="None")
    career_interest= db.Column(db.String(80), nullable=False)
    user_skills    = db.Column(db.Text, nullable=True)
    created_at     = db.Column(db.DateTime, default=datetime.utcnow)
    analyses       = db.relationship("Analysis", backref="student", lazy=True, cascade="all, delete")


class Analysis(db.Model):
    __tablename__ = "analyses"
    id               = db.Column(db.Integer, primary_key=True, autoincrement=True)
    student_id       = db.Column(db.Integer, db.ForeignKey("students.id"), nullable=False)
    ai_provider      = db.Column(db.String(20), nullable=False, default="anthropic")
    ai_model         = db.Column(db.String(80), nullable=False)
    prompt_text      = db.Column(db.Text().with_variant(mysql.LONGTEXT(), "mysql"), nullable=False)
    response_json    = db.Column(db.Text().with_variant(mysql.LONGTEXT(), "mysql"), nullable=False)
    summary          = db.Column(db.Text, nullable=True)
    motivational_tip = db.Column(db.Text, nullable=True)
    status           = db.Column(db.String(10), nullable=False, default="success")
    error_message    = db.Column(db.Text().with_variant(mysql.LONGTEXT(), "mysql"), nullable=True)
    created_at       = db.Column(db.DateTime, default=datetime.utcnow)


# ─── Validation ────────────────────────────────────────────────────────────────

def validate_form(data):
    errors = []
    name = data.get("name", "").strip()
    if not name:
        errors.append("Full name is required.")
    elif len(name) < 2:
        errors.append("Full name must be at least 2 characters.")
    email = data.get("email", "").strip()
    if not email:
        errors.append("Email is required.")
    elif not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
        errors.append("Please enter a valid email address.")
    age = data.get("age", "").strip()
    if not age:
        errors.append("Age is required.")
    else:
        try:
            age_int = int(age)
            if age_int < 10 or age_int > 60:
                errors.append("Age must be between 10 and 60.")
        except ValueError:
            errors.append("Age must be a valid number.")
    semester = data.get("semester", "").strip()
    if not semester:
        errors.append("Class / Semester is required.")
    return errors


# ─── AI helpers ────────────────────────────────────────────────────────────────

def build_ai_prompt(form):
    return f"""You are CareerCompass, an expert student career guidance counselor.
Analyze the following student profile and provide detailed, personalized career guidance.

STUDENT PROFILE:
- Name: {form.get("name")}
- Age: {form.get("age")}
- Class/Semester: {form.get("semester")}
- Favorite Subject: {form.get("subject")}
- Programming Skill Level: {form.get("skill")}
- Career Interest: {form.get("interest")}
- Skills: {form.get("user_skills")}

Respond ONLY with a valid JSON object (no markdown, no explanation) in this exact structure:
{{
  "summary": "2-3 sentence personalized overview of the student's profile and potential",
  "top_careers": [
    {{
      "title": "Career Title",
      "field": "Field or Subject Area",
      "description": "Why this is a great fit for this student",
      "demand": "High Demand | Trending | Stable | Emerging",
      "match_percent": 92
    }}
  ],
  "skill_scores": {{
    "Technical Skills": 80,
    "Communication": 70,
    "Problem Solving": 85,
    "Creativity": 65,
    "Leadership": 60
  }},
  "motivational_tip": "A personalized motivational message for this student"
}}

Provide career recommendations ONLY related to the student's selected subject and career interest.
Each career should include a field or subject-area label.
All match_percent values must be integers between 50 and 99.
All skill_scores values must be integers between 40 and 99."""


def get_ai_provider():
    requested = os.environ.get("AI_PROVIDER", "").strip().lower()
    if requested in ("anthropic", "openai", "local"):
        return requested
    has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY", "").strip())
    has_openai    = bool(os.environ.get("OPENAI_API_KEY", "").strip())
    if has_openai and not has_anthropic:
        return "openai"
    if has_anthropic:
        return "anthropic"
    return "local"


def get_ai_model(provider):
    if provider == "openai":
        return os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    return os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")


def call_anthropic_api(prompt):
    import urllib.request
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not configured. Set ANTHROPIC_API_KEY or switch to "
            "OpenAI using AI_PROVIDER=openai and OPENAI_API_KEY."
        )
    payload = json.dumps({
        "model": get_ai_model("anthropic"),
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1500,
    }).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read())
    if "content" in result:
        content = result["content"]
        if isinstance(content, list):
            return "".join(b.get("text", "") for b in content if b.get("type") == "text")
        return str(content)
    if "choices" in result and result["choices"]:
        c = result["choices"][0]
        return c.get("message", {}).get("content", "") if isinstance(c, dict) else c.get("text", "")
    raise RuntimeError("Unable to parse Anthropic response")


def call_openai_api(prompt):
    import urllib.request
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not configured.")
    payload = json.dumps({
        "model": get_ai_model("openai"),
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1500,
        "temperature": 0.7,
    }).encode()
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read())
    choices = result.get("choices", [])
    if choices:
        c = choices[0]
        if isinstance(c, dict):
            return c.get("message", {}).get("content") or c.get("text", "")
    raise RuntimeError("Unable to parse OpenAI response")


# ─── Career roadmaps (from career_roadmaps.py) ─────────────────────────────────

CAREER_ROADMAPS = {
    # ── Technology & Software ──────────────────────────────────────────────────
    "Software Developer": [
        {"step": 1, "title": "Master a Programming Language", "description": "Start with Python or JavaScript. Build small programs daily — calculators, to-do apps, and games to solidify fundamentals."},
        {"step": 2, "title": "Learn Data Structures & Algorithms", "description": "Study arrays, linked lists, trees, and sorting algorithms. Practice on LeetCode and HackerRank to sharpen problem-solving skills."},
        {"step": 3, "title": "Build Real Projects", "description": "Create a portfolio with 3–5 projects: a web app, a REST API, and an open-source contribution on GitHub to showcase production thinking."},
        {"step": 4, "title": "Learn Version Control & Deployment", "description": "Master Git, GitHub, and basic DevOps. Deploy your apps using Heroku, Vercel, or AWS to demonstrate production-ready skills."},
    ],
    "Full-Stack Developer": [
        {"step": 1, "title": "Learn HTML, CSS & JavaScript", "description": "Master the building blocks of the web. Build static websites and progressively enhance them with JavaScript interactivity."},
        {"step": 2, "title": "Pick a Frontend Framework", "description": "Learn React or Vue.js. Build dynamic SPAs and understand component-based architecture, state management, and API calls."},
        {"step": 3, "title": "Master Backend & Databases", "description": "Learn Node.js (Express) or Python (Django/Flask). Understand REST APIs, SQL/NoSQL databases, and user authentication flows."},
        {"step": 4, "title": "Deploy & Build a Portfolio", "description": "Deploy full-stack apps on Railway or Render. Build a portfolio showcasing both frontend design and backend engineering skills."},
    ],
    "AI Engineer": [
        {"step": 1, "title": "Learn Python & Math Foundations", "description": "Master Python, linear algebra, calculus, and probability — the three pillars every AI engineer must know deeply before anything else."},
        {"step": 2, "title": "Study Machine Learning", "description": "Learn supervised and unsupervised algorithms. Complete Andrew Ng's ML course and implement models from scratch using scikit-learn."},
        {"step": 3, "title": "Deep Dive into Deep Learning", "description": "Study neural networks, CNNs, and transformers using PyTorch or TensorFlow. Experiment with pre-trained models on Hugging Face."},
        {"step": 4, "title": "Build & Deploy AI Projects", "description": "Build end-to-end AI applications — an image classifier, a chatbot, or a recommendation engine — and deploy them as live APIs."},
    ],
    "Data Scientist": [
        {"step": 1, "title": "Learn Python & Statistics", "description": "Master Python (NumPy, Pandas), descriptive statistics, probability distributions, and hypothesis testing — your core daily toolkit."},
        {"step": 2, "title": "Study Machine Learning", "description": "Learn regression, classification, clustering, and model evaluation using scikit-learn. Work on Kaggle datasets to build real experience."},
        {"step": 3, "title": "Data Visualization & SQL", "description": "Master Matplotlib, Seaborn, and Tableau for storytelling with data. Learn SQL to query and manipulate large datasets in databases."},
        {"step": 4, "title": "Complete End-to-End Projects", "description": "Build 3 full data science projects covering data cleaning, EDA, model building, and insights reporting. Share them on GitHub and Kaggle."},
    ],
    "Machine Learning Engineer": [
        {"step": 1, "title": "Strengthen Python & Math", "description": "Deeply understand Python, linear algebra, statistics, and calculus. These are non-negotiable foundations for ML engineering work."},
        {"step": 2, "title": "Master Core ML Algorithms", "description": "Implement gradient descent, decision trees, SVMs, and neural networks from scratch before relying on libraries like scikit-learn."},
        {"step": 3, "title": "Learn MLOps & Deployment", "description": "Study Docker, FastAPI, and ML pipelines. Deploy and monitor models using tools like MLflow, BentoML, or AWS SageMaker."},
        {"step": 4, "title": "Specialize & Publish Work", "description": "Choose a niche (NLP, Computer Vision, RL). Reproduce a research paper, enter a Kaggle competition, and publish your findings publicly."},
    ],
    "Cybersecurity Analyst": [
        {"step": 1, "title": "Learn Networking & OS Fundamentals", "description": "Understand TCP/IP, DNS, HTTP, Linux commands, and Windows administration — the infrastructure cybersecurity professionals protect daily."},
        {"step": 2, "title": "Study Security Concepts", "description": "Learn firewalls, encryption, vulnerability assessment, and common attacks (SQL injection, phishing, MITM) through TryHackMe and Cybrary."},
        {"step": 3, "title": "Get Certified", "description": "Earn CompTIA Security+, CEH, or Google Cybersecurity Certificate. These validate your skills and unlock entry-level security analyst roles."},
        {"step": 4, "title": "Practice in Labs & CTFs", "description": "Solve Capture The Flag challenges on HackTheBox and PicoCTF. Build a home lab to practice penetration testing safely and ethically."},
    ],
    "Cloud Engineer": [
        {"step": 1, "title": "Learn Networking & Linux Basics", "description": "Understand IP addressing, DNS, load balancers, and Linux command-line skills — the foundational knowledge cloud engineering builds on."},
        {"step": 2, "title": "Master a Cloud Platform", "description": "Pick AWS, Azure, or GCP. Learn core services: compute (EC2), storage (S3), databases (RDS), and serverless functions (Lambda)."},
        {"step": 3, "title": "Study Infrastructure as Code", "description": "Learn Terraform and Docker to provision cloud infrastructure programmatically and containerize applications for scale and repeatability."},
        {"step": 4, "title": "Get Certified & Build Projects", "description": "Earn AWS Solutions Architect Associate or Azure Fundamentals. Deploy a multi-tier web app on the cloud as your capstone project."},
    ],
    "DevOps Engineer": [
        {"step": 1, "title": "Learn Linux & Scripting", "description": "Master Linux administration and write Bash/Python scripts to automate repetitive tasks. Shell scripting is the connective tissue of DevOps work."},
        {"step": 2, "title": "Study CI/CD Pipelines", "description": "Learn Git, Jenkins, and GitHub Actions. Build automated pipelines that test, build, and deploy code on every single commit automatically."},
        {"step": 3, "title": "Master Containers & Orchestration", "description": "Learn Docker deeply, then Kubernetes. Understand how to deploy, scale, and monitor containerized microservices in production environments."},
        {"step": 4, "title": "Cloud & Monitoring", "description": "Learn cloud fundamentals (AWS/GCP) and monitoring tools like Prometheus and Grafana. Aim for CKA or AWS DevOps Professional certification."},
    ],
    "Mobile App Developer": [
        {"step": 1, "title": "Choose Your Platform", "description": "Decide between React Native (cross-platform), Flutter (cross-platform), or native Swift (iOS) / Kotlin (Android) based on your target audience."},
        {"step": 2, "title": "Learn UI/UX for Mobile", "description": "Understand mobile design principles, gesture interactions, and platform guidelines (Material Design / Human Interface Guidelines). Always design first."},
        {"step": 3, "title": "Integrate APIs & Storage", "description": "Learn REST API integration, local storage, Firebase real-time database, and push notifications to build feature-rich real-world mobile apps."},
        {"step": 4, "title": "Publish Your App", "description": "Launch an app on Google Play Store or Apple App Store. Gather user feedback, iterate on it, and feature it prominently in your portfolio."},
    ],
    "Game Developer": [
        {"step": 1, "title": "Learn a Game Engine", "description": "Start with Unity (C#) or Godot (GDScript). Build simple games like Pong, Flappy Bird, or a platformer to understand core game development concepts."},
        {"step": 2, "title": "Study Game Design Principles", "description": "Learn game loops, physics, collision detection, UI systems, and level design. Read The Art of Game Design by Jesse Schell for deep insights."},
        {"step": 3, "title": "Add Audio, Art & Polish", "description": "Integrate free assets from itch.io and OpenGameArt. Learn basic animations, particle effects, and sound design to bring your games to life."},
        {"step": 4, "title": "Publish & Build a Portfolio", "description": "Publish your games on itch.io or the Play Store. Participate in game jams (Ludum Dare, Global Game Jam) to build reputation and meet collaborators."},
    ],
    "Blockchain Developer": [
        {"step": 1, "title": "Understand Blockchain Fundamentals", "description": "Study how Bitcoin, Ethereum, and distributed ledgers work. Understand consensus mechanisms, cryptographic hashing, and digital wallets thoroughly."},
        {"step": 2, "title": "Learn Solidity & Smart Contracts", "description": "Write and deploy smart contracts on Ethereum using Solidity. Use Remix IDE and Hardhat for development, testing, and local blockchain simulation."},
        {"step": 3, "title": "Build DApps", "description": "Connect smart contracts to a React front end using ethers.js or web3.js. Build a token, NFT marketplace, or DeFi application as a portfolio project."},
        {"step": 4, "title": "Audit & Specialize", "description": "Learn smart contract security and common exploits (reentrancy, overflow attacks). Specialize in DeFi protocols, NFTs, or Layer-2 scaling solutions."},
    ],
    "Robotics Engineer": [
        {"step": 1, "title": "Learn Programming & Electronics", "description": "Master Python and C++, then basic electronics fundamentals. Build circuits and experiment with Arduino or Raspberry Pi for hands-on hardware experience."},
        {"step": 2, "title": "Study Mechanics & Control Systems", "description": "Understand kinematics, dynamics, PID controllers, and sensors like encoders, IMUs, and LiDAR. These form the physical backbone of every robot."},
        {"step": 3, "title": "Learn ROS (Robot Operating System)", "description": "Get hands-on with ROS2 for robot software development. Simulate robots in Gazebo before building and testing on physical prototypes."},
        {"step": 4, "title": "Build & Compete", "description": "Build a robot for a real task — a line follower, robotic arm, or autonomous vehicle. Compete in RoboSumo, FIRST Robotics, or regional competitions."},
    ],
    # ── Healthcare ─────────────────────────────────────────────────────────────
    "Doctor": [
        {"step": 1, "title": "Excel in Science Subjects", "description": "Focus intensely on Biology, Chemistry, and Physics in school. These subjects form the entire foundation of medical education and the NEET exam."},
        {"step": 2, "title": "Clear NEET / Medical Entrance", "description": "Prepare rigorously for NEET. Consistent practice with previous year papers, NCERT mastery, and regular mock tests are the keys to success."},
        {"step": 3, "title": "Complete MBBS & Internship", "description": "Pursue your MBBS degree (5.5 years with internship). Clinical rotations across specialties give you exposure to real patient care from Year 1."},
        {"step": 4, "title": "Specialize & Register", "description": "Choose a specialization (Cardiology, Paediatrics, Surgery, etc.) and pursue MD/MS. Register with the Medical Council of India to begin licensed practice."},
    ],
    "Nurse": [
        {"step": 1, "title": "Study Biology & Health Sciences", "description": "Develop a strong foundation in Biology and Chemistry. Good grades in science are key for nursing program admissions across India."},
        {"step": 2, "title": "Pursue B.Sc Nursing or GNM", "description": "Enroll in a 3–4 year nursing degree. Study anatomy, pharmacology, medical-surgical nursing, and community health nursing with clinical practice."},
        {"step": 3, "title": "Complete Clinical Placements", "description": "Gain hands-on experience in hospital wards during rotations. Practice patient assessment, IV administration, wound care, and emergency response."},
        {"step": 4, "title": "Get Licensed & Specialize", "description": "Register with the State Nursing Council. Specialize in ICU nursing, Oncology, or Paediatric nursing for significantly better career prospects and pay."},
    ],
    "Pharmacist": [
        {"step": 1, "title": "Build a Strong Chemistry Base", "description": "Excel in Chemistry and Biology in school. Understanding drug chemistry and biology is fundamental to all aspects of pharmaceutical sciences."},
        {"step": 2, "title": "Complete B.Pharm / D.Pharm", "description": "Enroll in a pharmacy degree program. Study pharmacology, pharmaceutics, pharmaceutical chemistry, and pharmacognosy across the curriculum."},
        {"step": 3, "title": "Complete Internship & Registration", "description": "Complete your mandatory internship in a hospital or retail pharmacy. Register with the Pharmacy Council of India to practice legally."},
        {"step": 4, "title": "Specialize or Explore Industry", "description": "Choose clinical pharmacy, regulatory affairs, pharmaceutical research, or retail management. Pursue M.Pharm for advanced research and industry roles."},
    ],
    "Physiotherapist": [
        {"step": 1, "title": "Excel in Biology & Human Anatomy", "description": "Build a strong understanding of biology, human anatomy, and physiology — the science behind movement, posture, and physical rehabilitation."},
        {"step": 2, "title": "Complete BPT Degree", "description": "Pursue a Bachelor of Physiotherapy (BPT) program (4.5 years). Study musculoskeletal, neurological, and cardiopulmonary physiotherapy with clinical practice."},
        {"step": 3, "title": "Clinical Internship", "description": "Complete 6 months of mandatory clinical internship across orthopaedic wards, neurology departments, and sports rehabilitation centres."},
        {"step": 4, "title": "Specialize & Register", "description": "Specialize in sports physiotherapy, neurological rehabilitation, or paediatric physiotherapy. Pursue MPT for senior hospital and research positions."},
    ],
    "Psychologist": [
        {"step": 1, "title": "Study Psychology & Social Sciences", "description": "Develop curiosity about human behavior. Take psychology as a subject early and read foundational books like Thinking, Fast and Slow by Kahneman."},
        {"step": 2, "title": "Complete BA/B.Sc in Psychology", "description": "Pursue a psychology undergraduate degree covering cognitive psychology, developmental psychology, abnormal psychology, and research methods."},
        {"step": 3, "title": "Earn an MA/M.Sc in Psychology", "description": "Specialize in Clinical, Counselling, or Educational Psychology. A master's degree is essential for most professional-level psychology practice roles."},
        {"step": 4, "title": "Get Licensed & Gain Experience", "description": "Complete supervised clinical internship hours. Register with the Rehabilitation Council of India (RCI) for the legal right to clinical practice."},
    ],
    "Biomedical Engineer": [
        {"step": 1, "title": "Excel in Biology, Physics & Math", "description": "These three subjects underpin biomedical engineering. Aim for excellence especially in maths and physics for engineering entrance exams."},
        {"step": 2, "title": "Pursue B.Tech in Biomedical Engineering", "description": "Study biomechanics, medical imaging, bioelectronics, and physiology. Many BTech programs offer hospital visits and interdisciplinary lab projects."},
        {"step": 3, "title": "Learn Medical Device Development", "description": "Work on projects involving prosthetics, diagnostic equipment, or wearable health sensors. Collaborate with medical teams to understand clinical needs."},
        {"step": 4, "title": "Specialize & Get Certified", "description": "Specialize in medical imaging, neural engineering, or regulatory affairs. ISO and FDA training program certifications add significant professional value."},
    ],
    # ── Engineering ────────────────────────────────────────────────────────────
    "Mechanical Engineer": [
        {"step": 1, "title": "Build Strong Physics & Maths Skills", "description": "Master mechanics, thermodynamics, and calculus in school. These directly translate into your core engineering subjects at the BTech level."},
        {"step": 2, "title": "Pursue B.Tech in Mechanical Engineering", "description": "Study fluid mechanics, machine design, manufacturing processes, and CAD. Use SolidWorks and AutoCAD for hands-on design project work."},
        {"step": 3, "title": "Get Hands-On with Projects & Internships", "description": "Build a mini project — a robotic arm, engine model, or hydraulic system. Intern at manufacturing, automotive, or aerospace companies for industry exposure."},
        {"step": 4, "title": "Specialize & Get Certified", "description": "Choose Thermal, Design, Manufacturing, or Automobile engineering as your specialty. Certifications in Six Sigma, AutoCAD, or ANSYS add strong career value."},
    ],
    "Civil Engineer": [
        {"step": 1, "title": "Strengthen Physics & Mathematics", "description": "Excel in Physics and Maths at school level. Structural analysis and fluid mechanics build directly on these core scientific foundations."},
        {"step": 2, "title": "Pursue B.Tech in Civil Engineering", "description": "Study structural engineering, soil mechanics, hydraulics, and construction management. Learn AutoCAD and STAAD Pro for technical design work."},
        {"step": 3, "title": "Gain Site Experience", "description": "Intern at construction companies or government infrastructure projects. Understanding real site challenges is as important as classroom learning."},
        {"step": 4, "title": "Get Licensed & Specialize", "description": "Register as a licensed engineer with CIDC. Specialize in structural, transportation, environmental, or geotechnical engineering for focused career growth."},
    ],
    "Electrical Engineer": [
        {"step": 1, "title": "Master Physics & Mathematics", "description": "Focus on electricity, magnetism, and calculus in school. These form the theoretical backbone for circuits, signals, and control systems."},
        {"step": 2, "title": "Pursue B.Tech in Electrical Engineering", "description": "Study circuit analysis, power systems, control theory, and electronics. Learn MATLAB and simulation tools for system analysis and design."},
        {"step": 3, "title": "Work on Electronics Projects", "description": "Build Arduino-based projects, design PCBs, and experiment with renewable energy systems. Practical project experience accelerates learning enormously."},
        {"step": 4, "title": "Get Licensed & Choose a Specialty", "description": "Pursue professional engineer licensure and specialize in power systems, automation, renewable energy, or embedded systems engineering."},
    ],
    "Aerospace Engineer": [
        {"step": 1, "title": "Excel in Physics & Mathematics", "description": "Aerodynamics and spacecraft design demand deep understanding of mechanics, thermodynamics, and advanced math from the very beginning."},
        {"step": 2, "title": "Pursue B.Tech in Aerospace Engineering", "description": "Study aerodynamics, propulsion systems, aircraft structures, and orbital mechanics. Learn CFD tools like ANSYS Fluent and MATLAB for simulations."},
        {"step": 3, "title": "Participate in Competitions & Research", "description": "Join SAE Aero Design, rocketry clubs, or ISRO internship programs. Building actual aerospace models gives you a critical edge in job applications."},
        {"step": 4, "title": "Specialize & Pursue M.Tech/Research", "description": "Specialize in propulsion, avionics, or satellite design. Pursue M.Tech or research at ISRO, DRDO, HAL, or international aerospace organizations."},
    ],
    "Automobile Engineer": [
        {"step": 1, "title": "Build Physics & Math Foundation", "description": "Understand mechanics, thermodynamics, and material science deeply. These subjects directly apply to vehicle design and engine development work."},
        {"step": 2, "title": "Pursue B.Tech in Automobile/Mechanical Engineering", "description": "Study vehicle dynamics, IC engines, transmission systems, and chassis design. Learn CAD tools like SolidWorks and CATIA for vehicle design."},
        {"step": 3, "title": "Intern at Automotive Companies", "description": "Get hands-on experience at OEMs like Tata, Mahindra, or Maruti Suzuki. Participate in BAJA SAE or Formula Student competitions to stand out."},
        {"step": 4, "title": "Specialize in EVs or Motorsport", "description": "Specialize in Electric Vehicles, ADAS systems, or motorsport engineering. EV powertrain and battery technology is an exceptionally high-growth area."},
    ],
    "Chemical Engineer": [
        {"step": 1, "title": "Excel in Chemistry, Physics & Maths", "description": "Strong fundamentals in all three sciences are essential. Thermodynamics and reaction kinetics run through every chemical engineering course."},
        {"step": 2, "title": "Pursue B.Tech in Chemical Engineering", "description": "Study mass transfer, heat transfer, process design, and reaction engineering. Work in labs on distillation, extraction, and reactor design projects."},
        {"step": 3, "title": "Intern in Process Industries", "description": "Gain experience in oil & gas, pharmaceuticals, or specialty chemicals. Real plant exposure teaches process control, safety protocols, and efficiency optimization."},
        {"step": 4, "title": "Get Certified & Specialize", "description": "Specialize in petroleum, polymer, or pharmaceutical engineering. Certifications in HAZOP analysis, Process Safety Management, or Six Sigma strengthen your profile."},
    ],
    "Environmental Engineer": [
        {"step": 1, "title": "Study Environmental Science & Chemistry", "description": "Develop passion for sustainability. Study environmental science, chemistry, and biology to understand ecosystems, pollution dynamics, and climate systems."},
        {"step": 2, "title": "Pursue B.Tech in Environmental Engineering", "description": "Study water treatment, air quality management, solid waste management, and environmental impact assessment. Learn GIS and AutoCAD for field work."},
        {"step": 3, "title": "Get Field Experience", "description": "Intern with environmental consulting firms, water treatment plants, or government pollution control boards for critical practical field exposure."},
        {"step": 4, "title": "Get Certified & Contribute to Policy", "description": "Pursue ISO 14001 Environmental Management Systems certification. Work with NGOs or government policy teams to create measurable environmental impact."},
    ],
    # ── Design & Creative ──────────────────────────────────────────────────────
    "Graphic Designer": [
        {"step": 1, "title": "Learn Design Fundamentals", "description": "Study color theory, typography, composition, and visual hierarchy. These principles are the foundation of every great design, digital or print."},
        {"step": 2, "title": "Master Design Tools", "description": "Get proficient in Adobe Illustrator, Photoshop, and Canva. Practice daily — recreate logos, posters, and social media graphics to build muscle memory."},
        {"step": 3, "title": "Build a Portfolio", "description": "Create 10–15 varied projects: brand identities, posters, infographics, and packaging designs. Host them on Behance or a custom portfolio website."},
        {"step": 4, "title": "Freelance & Specialize", "description": "Take freelance projects on Fiverr or Upwork to build real client experience. Specialize in branding, editorial, or motion graphics to stand out in the market."},
    ],
    "UI/UX Designer": [
        {"step": 1, "title": "Learn UX Research Principles", "description": "Understand user psychology, empathy mapping, and user interview techniques. Great design always starts with deeply understanding the people you're designing for."},
        {"step": 2, "title": "Master Figma & Prototyping", "description": "Learn Figma for wireframing and high-fidelity prototyping. Practice designing mobile apps and web dashboards from scratch using real product briefs."},
        {"step": 3, "title": "Study Accessibility & Design Systems", "description": "Learn WCAG accessibility guidelines and how to build scalable design systems. Study Google Material Design and Apple Human Interface Guidelines."},
        {"step": 4, "title": "Build a Case Study Portfolio", "description": "Document 3–4 end-to-end UX case studies showing your research process, design decisions, and final solution. Your portfolio is your strongest hiring asset."},
    ],
    "Animator": [
        {"step": 1, "title": "Learn Animation Principles", "description": "Study Disney's 12 principles of animation — squash & stretch, anticipation, timing, follow-through. These govern all great animation regardless of style or medium."},
        {"step": 2, "title": "Master Animation Software", "description": "Learn Adobe Animate for 2D, or Blender and Maya for 3D animation. Start with simple bouncing ball and character walk cycle exercises to build fundamentals."},
        {"step": 3, "title": "Build a Demo Reel", "description": "Create a 60–90 second demo reel showcasing your best animation work across different styles. Your reel is your resume in the animation industry."},
        {"step": 4, "title": "Specialize & Join the Industry", "description": "Specialize in character animation, motion graphics, or VFX. Contribute to animated shorts, game studios, or advertising agencies to build professional credits."},
    ],
    "Fashion Designer": [
        {"step": 1, "title": "Develop Your Aesthetic & Sketch", "description": "Study fashion history, current trends, and textile science. Learn fashion illustration — sketching garments is a core communication skill for every designer."},
        {"step": 2, "title": "Pursue Fashion Design Education", "description": "Enroll at NIFT, NID, or a recognized fashion institute. Study pattern making, draping, garment construction techniques, and fashion business management."},
        {"step": 3, "title": "Build Collections & Intern", "description": "Create 2–3 original collections as portfolio pieces. Intern with established designers or major fashion brands to understand the full industry production workflow."},
        {"step": 4, "title": "Find Your Niche & Launch", "description": "Specialize in bridal, streetwear, sustainable fashion, or accessories design. Start your own label or join established fashion houses to grow your creative career."},
    ],
    "Interior Designer": [
        {"step": 1, "title": "Study Design & Architecture Basics", "description": "Learn drawing, art history, and spatial thinking. Understanding architecture and human ergonomics is foundational to professional interior design practice."},
        {"step": 2, "title": "Pursue a Degree in Interior Design", "description": "Study space planning, color theory, materials science, lighting design, and building systems. Learn AutoCAD and SketchUp for producing technical drawings."},
        {"step": 3, "title": "Build a Physical & Digital Portfolio", "description": "Design 3–5 interior projects — even personal or mock projects. Photograph and present them professionally to attract your first real paying clients."},
        {"step": 4, "title": "Specialize & Network", "description": "Specialize in residential, commercial, hospitality, or healthcare interiors. Join interior design associations and attend trade shows to build your professional network."},
    ],
    "Product Designer": [
        {"step": 1, "title": "Learn Design Thinking", "description": "Master the design thinking process: empathize, define, ideate, prototype, test. This human-centered framework is the heart of all product design work."},
        {"step": 2, "title": "Study Industrial Design or UX", "description": "Learn form, function, and usability deeply. For physical products, master SolidWorks or Rhino. For digital products, master Figma and user research methods."},
        {"step": 3, "title": "Build Prototypes & Test", "description": "Create physical or digital prototypes and test them with real users. Iterating based on genuine user feedback is what separates average products from great ones."},
        {"step": 4, "title": "Develop a Case Study Portfolio", "description": "Document your entire design process in detail — problem definition, research, concepts, iterations, final design outcomes. Share it on Behance and LinkedIn."},
    ],
    # ── Business & Finance ─────────────────────────────────────────────────────
    "Chartered Accountant": [
        {"step": 1, "title": "Clear CA Foundation", "description": "After 12th Commerce, register with ICAI and clear the CA Foundation exam covering Mathematics, Economics, Accounting, and Business Laws."},
        {"step": 2, "title": "Clear CA Intermediate", "description": "Clear both groups of CA Intermediate covering direct and indirect taxation, auditing, financial management, and accounting standards."},
        {"step": 3, "title": "Complete 3-Year Articleship", "description": "Work under a practicing CA for 3 years. This hands-on training is the most valuable part of your CA journey — learn everything you possibly can."},
        {"step": 4, "title": "Clear CA Final & Get Membership", "description": "Clear the CA Final exam covering SFM, Advanced Auditing, and Strategic Management. Become an ICAI member and begin your professional practice or join a firm."},
    ],
    "Investment Banker": [
        {"step": 1, "title": "Build Finance & Math Foundations", "description": "Excel in Mathematics, Economics, and Commerce. Develop analytical thinking and genuine curiosity for financial markets from an early stage in your studies."},
        {"step": 2, "title": "Pursue Finance/Economics Degree", "description": "Complete B.Com, BBA, or B.Sc Economics from a reputed institution. Summer internships at banks or financial firms during college are highly valuable."},
        {"step": 3, "title": "Learn Financial Modeling & Valuation", "description": "Master Excel, DCF modeling, comparable company analysis, and LBO models. CFI and Wall Street Prep offer excellent courses specifically for this work."},
        {"step": 4, "title": "Pursue MBA & Land an Internship", "description": "Target top MBA programs or pursue CFA certification alongside your career. Internships at bulge-bracket banks are the most common entry point into IB."},
    ],
    "Financial Analyst": [
        {"step": 1, "title": "Build Strong Quantitative Skills", "description": "Excel in Mathematics, Statistics, and Economics. Financial analysts work heavily with numbers, models, and data analysis every single working day."},
        {"step": 2, "title": "Pursue Finance or Economics Degree", "description": "Complete a degree in Finance, Economics, or Commerce. Learn Excel, PowerPoint, and basic financial statement analysis as early in your studies as possible."},
        {"step": 3, "title": "Get Certified", "description": "Pursue CFA Level 1 or FRM certification. These globally recognized credentials significantly improve hiring prospects at investment firms and banks."},
        {"step": 4, "title": "Specialize in a Sector", "description": "Specialize in equity research, credit analysis, or corporate finance. Learn Bloomberg Terminal and develop deep expertise in sector-specific financial modeling."},
    ],
    "Entrepreneur": [
        {"step": 1, "title": "Identify a Real Problem to Solve", "description": "Observe markets, talk to potential customers, and find a genuine pain point. The best startups are built by founders who solve problems they personally experienced."},
        {"step": 2, "title": "Build a Minimum Viable Product (MVP)", "description": "Launch the simplest version of your idea quickly. Don't wait for perfection — get real user feedback as early as possible to validate or pivot your concept."},
        {"step": 3, "title": "Learn Business Fundamentals", "description": "Understand unit economics, customer acquisition cost, cash flow, and marketing funnels. Read Zero to One and The Lean Startup for proven mental frameworks."},
        {"step": 4, "title": "Scale, Fund & Build a Team", "description": "Apply to startup accelerators like Y Combinator or Startup India programs. Build a complementary founding team and seek seed funding from angel investors."},
    ],
    "Business Analyst": [
        {"step": 1, "title": "Build Analytical & Communication Skills", "description": "Practice data analysis, problem structuring, and clear written communication. BAs must bridge the gap between business stakeholders and technical development teams."},
        {"step": 2, "title": "Learn Excel, SQL & Data Visualization", "description": "Master Excel for analysis, SQL for querying databases, and Power BI or Tableau for creating insightful business dashboards that drive decision-making."},
        {"step": 3, "title": "Understand Business Processes", "description": "Study process mapping (BPMN), requirement gathering, and Agile/Scrum methodologies. Internships in consulting or IT services firms provide excellent real exposure."},
        {"step": 4, "title": "Get Certified & Build Domain Expertise", "description": "Earn CBAP or PMI-PBA certification. Develop domain expertise in Finance, Healthcare, or E-commerce to become a highly sought-after specialist in your sector."},
    ],
    "Product Manager": [
        {"step": 1, "title": "Develop Curiosity About Products", "description": "Use products critically every day. Ask why every design decision was made. Read widely about product strategy and build the habit of thinking in user problems."},
        {"step": 2, "title": "Learn Product Management Fundamentals", "description": "Study roadmapping, user story writing, prioritization frameworks (RICE, MoSCoW), and product metrics. Take PM courses on Coursera, Reforge, or Pragmatic Institute."},
        {"step": 3, "title": "Build Something & Work Cross-Functionally", "description": "Launch a side project or contribute to a startup. PM roles require constant collaboration with engineers, designers, data analysts, and business stakeholders."},
        {"step": 4, "title": "Get APM Role & Iterate", "description": "Target Associate PM programs at technology companies. Build a track record of shipped features, measurable user impact, and strong cross-functional collaboration."},
    ],
    "Digital Marketer": [
        {"step": 1, "title": "Learn Marketing Fundamentals", "description": "Understand the 4Ps of marketing, buyer personas, and the customer journey funnel. Study successful campaigns from brands like Apple, Nike, Zomato, and Swiggy."},
        {"step": 2, "title": "Master Core Digital Channels", "description": "Learn SEO, Google Ads, Meta Ads, email marketing, and social media management. Run small test campaigns with minimal budgets to gain real practical experience."},
        {"step": 3, "title": "Learn Analytics & Data", "description": "Master Google Analytics 4, Meta Ads Manager, and Excel analysis. Data-driven marketers who can prove ROI with numbers are the most sought-after professionals."},
        {"step": 4, "title": "Get Certified & Build a Portfolio", "description": "Earn Google Ads, HubSpot Content, and Meta Blueprint certifications. Build case studies documenting campaigns you ran, results achieved, and key learnings."},
    ],
    "Human Resource Manager": [
        {"step": 1, "title": "Study Human Behavior & Communication", "description": "Develop strong interpersonal skills, deep empathy, and conflict resolution abilities. HR is fundamentally about working with and for people every single day."},
        {"step": 2, "title": "Pursue BBA/MBA in HR", "description": "Study organizational behavior, labor law, talent acquisition, compensation & benefits design, and performance management in a well-regarded business program."},
        {"step": 3, "title": "Intern in HR Departments", "description": "Gain hands-on experience in recruitment, onboarding, payroll processing, and employee engagement programs. Real HR experience is essential before senior roles."},
        {"step": 4, "title": "Get Certified & Specialize", "description": "Earn SHRM-CP or XLRI HR certifications. Specialize in talent acquisition, learning & development, HR analytics, or strategic HRBP roles based on your strengths."},
    ],
    # ── Law & Government ───────────────────────────────────────────────────────
    "Lawyer": [
        {"step": 1, "title": "Develop Critical Thinking & Communication", "description": "Law is about constructing and presenting compelling arguments. Practice debate, essay writing, and critical reading from an early stage in your education."},
        {"step": 2, "title": "Pursue LLB (3-Year or 5-Year Integrated)", "description": "Study Constitutional Law, Criminal Law, Contract Law, Civil Procedure, and Evidence Act. CLAT is the gateway for top National Law Universities across India."},
        {"step": 3, "title": "Gain Court Experience", "description": "Complete internships at law firms, High Courts, or with senior advocates. Moot court competitions and legal aid clinics during college are invaluable preparation."},
        {"step": 4, "title": "Specialize & Build a Practice", "description": "Enroll with the Bar Council after completing LLB. Specialize in Corporate Law, Criminal Law, Intellectual Property, or Family Law based on your passion and aptitude."},
    ],
    "IAS Officer": [
        {"step": 1, "title": "Build a Strong Academic Foundation", "description": "Excel in your graduation across any subject. Read newspapers daily, follow current affairs closely, and develop broad understanding of governance and Indian history."},
        {"step": 2, "title": "Complete Graduation & Start UPSC Prep", "description": "Enroll in a graduation program and begin UPSC CSE preparation simultaneously. Choose an optional subject that aligns well with your academic background."},
        {"step": 3, "title": "Clear UPSC Prelims & Mains", "description": "Study NCERTs, Laxmikanth's Indian Polity, Spectrum for Modern History, and current affairs thoroughly. Prelims tests breadth; Mains tests analytical depth."},
        {"step": 4, "title": "Ace the UPSC Personality Test (Interview)", "description": "Prepare for the 275-mark interview by developing balanced opinions on national issues, staying confident under pressure, and practising with mock interview panels."},
    ],
    "Army Officer": [
        {"step": 1, "title": "Build Physical & Academic Fitness", "description": "Maintain excellent physical fitness (running, strength, endurance) alongside strong academic performance. Leadership qualities and character matter as much as marks."},
        {"step": 2, "title": "Apply Through NDA / CDS / TES", "description": "After 12th, apply for the National Defence Academy (NDA) examination. Post-graduation, apply via CDS or Technical Entry Scheme (TES) for direct officer entries."},
        {"step": 3, "title": "Clear the SSB Interview", "description": "Prepare for the 5-day Services Selection Board (SSB) which evaluates personality, intelligence, and leadership through psychological tests, group tasks, and interviews."},
        {"step": 4, "title": "Complete Officer Training Academy", "description": "After selection, undergo intensive training at IMA, OTA, or through NDA commissioning. Graduate as a commissioned officer and serve the nation with pride."},
    ],
    "Police Officer": [
        {"step": 1, "title": "Maintain Physical & Academic Standards", "description": "Stay physically fit and score well academically. Police recruitment exams test both written aptitude and physical endurance through multiple rigorous stages."},
        {"step": 2, "title": "Clear State Police or UPSC IPS Exam", "description": "Apply for state-level police constable or Sub-Inspector exams, or aim for IPS through UPSC CSE. Study general knowledge, regional law, and current affairs."},
        {"step": 3, "title": "Clear Physical & Medical Tests", "description": "Pass fitness tests (running, height, chest measurement) and comprehensive medical examinations. Maintain a clean background for mandatory character verification."},
        {"step": 4, "title": "Complete Police Training Academy", "description": "Undergo training at state or national police academies covering law enforcement, criminal procedure, investigation skills, first aid, and physical combat techniques."},
    ],
    "Criminal Investigator": [
        {"step": 1, "title": "Study Law & Forensic Science", "description": "Understand criminal law, evidence law, and the Indian Penal Code. Pair this with forensic science knowledge for a strong investigative career foundation."},
        {"step": 2, "title": "Pursue B.Sc Forensic Science or LLB", "description": "Study crime scene investigation, digital forensics, toxicology, fingerprint analysis, and criminal psychology fundamentals in a forensic or legal program."},
        {"step": 3, "title": "Gain Investigative Experience", "description": "Intern with law firms handling criminal cases or shadow active investigators. Apply for trainee roles in government investigation agencies to build case experience."},
        {"step": 4, "title": "Join CBI / State Investigation Departments", "description": "Apply to CBI, NIA, or state crime branches through competitive examinations. Specialize in financial crimes, cybercrime, or organized crime investigation."},
    ],
    # ── Media & Communication ──────────────────────────────────────────────────
    "Journalist": [
        {"step": 1, "title": "Cultivate Curiosity & Writing Skills", "description": "Read widely — newspapers, books, and long-form journalism. Practice writing every day. Journalists must write quickly, clearly, and compellingly under deadline pressure."},
        {"step": 2, "title": "Pursue BA in Journalism / Mass Communication", "description": "Study reporting, media law, ethics, photography, and broadcast journalism. Start writing for your college newspaper or online publication from your very first year."},
        {"step": 3, "title": "Build a Portfolio of Published Bylines", "description": "Write for local newspapers, online publications, or start your own blog. Your portfolio of published work is the single most critical tool for getting your first media job."},
        {"step": 4, "title": "Specialize & Network", "description": "Choose investigative, business, political, or digital journalism as your focus. Join Press Clubs and journalistic associations to build the industry network that sustains careers."},
    ],
    "Content Creator": [
        {"step": 1, "title": "Find Your Niche & Start Creating", "description": "Choose a specific topic you genuinely love — tech, finance, education, or lifestyle. Start creating consistently, even before your content quality is perfect."},
        {"step": 2, "title": "Master Content Platforms & Tools", "description": "Learn video editing (CapCut, Premiere Pro), thumbnail design in Photoshop, YouTube SEO techniques, and caption writing for Instagram and LinkedIn audiences."},
        {"step": 3, "title": "Grow an Audience Systematically", "description": "Post on a consistent schedule, engage authentically with comments, collaborate with other creators in your niche, and study your analytics to optimize performance."},
        {"step": 4, "title": "Monetize & Build a Personal Brand", "description": "Diversify revenue through AdSense, brand sponsorships, digital products, memberships, and paid courses. Build a personal brand that stands distinctly apart in your niche."},
    ],
    "Public Relations Manager": [
        {"step": 1, "title": "Develop Communication & Writing Skills", "description": "Write press releases, media pitches, and thought leadership articles daily. Strong writing and verbal communication are the bedrock of every successful PR career."},
        {"step": 2, "title": "Pursue Mass Communication or BA in PR", "description": "Study media relations, crisis communication, event management, and brand storytelling. Intern at PR agencies during your degree for essential real-world client exposure."},
        {"step": 3, "title": "Build a Media Network", "description": "Cultivate genuine relationships with journalists, editors, and influencers. A strong, trusted media network is a PR professional's most valuable long-term career asset."},
        {"step": 4, "title": "Specialize & Handle Real Campaigns", "description": "Specialize in corporate PR, celebrity management, crisis communications, or digital PR. Every major campaign you manage builds your professional reputation."},
    ],
    "Video Producer": [
        {"step": 1, "title": "Learn Filmmaking Fundamentals", "description": "Study cinematography, lighting, sound design, and visual storytelling principles. Even with just a smartphone, practice composition and visual grammar every day."},
        {"step": 2, "title": "Master Video Editing Software", "description": "Learn DaVinci Resolve or Adobe Premiere Pro for editing, and After Effects for motion graphics. These tools are the industry standard for professional video production."},
        {"step": 3, "title": "Build a Portfolio Showreel", "description": "Create short films, music videos, event videos, or YouTube content as portfolio pieces. A strong showreel is your most critical asset for attracting clients and employers."},
        {"step": 4, "title": "Work on Real Productions", "description": "Start as a video editor or production assistant on professional sets. Progress through ad films, corporate videos, documentaries, and eventually independent productions."},
    ],
    # ── Education ──────────────────────────────────────────────────────────────
    "Teacher": [
        {"step": 1, "title": "Excel in Your Subject Area", "description": "Develop deep mastery of the subject you wish to teach. Genuine passion for your subject is immediately visible to students and makes learning genuinely come alive."},
        {"step": 2, "title": "Pursue B.Ed After Graduation", "description": "Complete a Bachelor of Education (B.Ed). Study pedagogy, educational psychology, curriculum design, and classroom management across a 2-year professional program."},
        {"step": 3, "title": "Clear CTET / State TET", "description": "Pass the Central or State Teacher Eligibility Test required for government school teaching positions. Prepare for both Paper 1 (Primary) and Paper 2 (Upper Primary) thoroughly."},
        {"step": 4, "title": "Gain Experience & Specialize", "description": "Start as a substitute or junior teacher and build expertise in your grade level or subject. Pursue M.Ed for leadership roles like principal or curriculum development coordinator."},
    ],
    "Professor": [
        {"step": 1, "title": "Excel Academically in Your Subject", "description": "Achieve top grades in your Bachelor's degree. A strong academic record is the essential foundation for gaining entry into quality Master's and PhD research programs."},
        {"step": 2, "title": "Complete Master's Degree", "description": "Pursue M.A., M.Sc., or M.Tech in your chosen field. During your master's program, identify specific research problems that deeply excite you for doctoral work."},
        {"step": 3, "title": "Clear NET/SET & Pursue PhD", "description": "Clear the UGC-NET exam for assistant professorship eligibility. Simultaneously pursue a PhD which deepens expertise and opens research university faculty positions."},
        {"step": 4, "title": "Publish Research & Join a University", "description": "Publish papers in peer-reviewed journals. Your research publication record is the primary selection criterion for faculty positions at colleges and universities."},
    ],
    # ── Aviation & Hospitality ─────────────────────────────────────────────────
    "Pilot": [
        {"step": 1, "title": "Excel in Physics & Mathematics", "description": "Aeronautics, navigation, and meteorology all rely on strong physics and math foundations. These subjects are tested at every selection stage in pilot training."},
        {"step": 2, "title": "Get a Student Pilot License (SPL)", "description": "Join a DGCA-approved flying school. Begin with ground school covering aviation theory, meteorology, air traffic procedures, and regulations before your first flight."},
        {"step": 3, "title": "Build 200+ Flying Hours for CPL", "description": "Log required flight hours — including solo navigation flights, cross-country routes, and instrument flying — to qualify for your Commercial Pilot License (CPL)."},
        {"step": 4, "title": "Clear DGCA Exams & Get Type Rated", "description": "Pass all DGCA written examinations and flight skill tests. Complete type rating on a commercial aircraft (A320, Boeing 737) and apply to regional or major airlines."},
    ],
    "Chef": [
        {"step": 1, "title": "Develop Your Culinary Passion & Palate", "description": "Cook every single day. Experiment with flavors, cuisines, and techniques at home. A refined palate and genuine love for food are a chef's most irreplaceable tools."},
        {"step": 2, "title": "Pursue Culinary Arts Education", "description": "Enroll in IHM or a recognized culinary institute. Study knife techniques, classical French cooking methods, baking, pastry arts, menu planning, and food safety standards."},
        {"step": 3, "title": "Stage & Work in Professional Kitchens", "description": "Work in professional restaurant kitchens as a commis chef. Every station — hot, cold, pastry, garde-manger — builds invaluable skills through high-pressure real service."},
        {"step": 4, "title": "Specialize & Build Your Reputation", "description": "Specialize in a cuisine (French, Japanese, Indian) or cooking style (pastry, molecular gastronomy, plant-based). Build your reputation and eventually open your own establishment."},
    ],
    "Hotel Manager": [
        {"step": 1, "title": "Study Hospitality & Customer Service", "description": "Develop a genuine service mindset. Great hotel managers love creating memorable guest experiences — this fundamental attitude should be cultivated from an early age."},
        {"step": 2, "title": "Pursue BHM or B.Sc Hotel Management", "description": "Study front office management, housekeeping operations, F&B service, and revenue management. IHMs across India provide excellent hands-on hospitality training."},
        {"step": 3, "title": "Work Across Hotel Departments", "description": "Gain direct experience in housekeeping, front desk, banquets, and restaurant operations. Hotel managers who have worked every department earn far more team respect."},
        {"step": 4, "title": "Pursue MBA in Hospitality & Get Certified", "description": "Pursue an MBA in Hospitality Management for senior leadership roles. AHLEI or Cornell's eCornell online certifications are globally recognized hospitality credentials."},
    ],
    "Event Manager": [
        {"step": 1, "title": "Develop Organizational & Creative Skills", "description": "Practice organizing college fests, seminars, and social events. Event management demands exceptional multitasking, vendor coordination, and creative problem-solving under pressure."},
        {"step": 2, "title": "Pursue Hospitality or Event Management Degree", "description": "Study event planning, logistics management, marketing, financial budgeting, and vendor coordination. Volunteer for every college event to gain early real production experience."},
        {"step": 3, "title": "Intern with Event Companies", "description": "Work with event management companies during weddings, corporate conferences, and large concerts. Real high-pressure production experience builds the instincts your career needs."},
        {"step": 4, "title": "Build Your Network & Portfolio", "description": "Document every event you manage with professional photos and client testimonials. Build a strong vendor network — caterers, AV teams, decorators — for your future independent practice."},
    ],
    # ── Science ────────────────────────────────────────────────────────────────
    "Research Scientist": [
        {"step": 1, "title": "Excel in Your Science Subjects", "description": "Achieve top marks in Physics, Chemistry, Biology, or Mathematics. Science olympiad participation and inter-school competitions add significant credibility to your profile."},
        {"step": 2, "title": "Pursue B.Sc or B.Tech in Your Field", "description": "Complete an undergraduate degree with active research exposure. Join faculty labs, attend academic seminars, and contribute to any available research projects during your studies."},
        {"step": 3, "title": "Complete a Research Master's & PhD", "description": "Pursue M.Sc followed by a doctoral PhD program. Your dissertation is your primary contribution to scientific knowledge — choose a problem you find genuinely fascinating."},
        {"step": 4, "title": "Publish, Collaborate & Apply for Grants", "description": "Publish research papers in peer-reviewed journals, attend international conferences, and collaborate globally. Apply for DST, CSIR, or DBT research grants for funding independence."},
    ],
    "Biotechnologist": [
        {"step": 1, "title": "Excel in Biology & Chemistry", "description": "Develop deep understanding of cell biology, molecular genetics, and biochemistry. These subjects are the foundational pillars of all modern biotechnology research."},
        {"step": 2, "title": "Pursue B.Sc / B.Tech in Biotechnology", "description": "Study molecular biology, genetic engineering, fermentation technology, and bioinformatics. Critical lab skills include PCR, gel electrophoresis, ELISA, and cell culture."},
        {"step": 3, "title": "Gain Lab Research Experience", "description": "Work in university or industry laboratories during your degree. Research internships at CSIR, DBT institutes, or pharma companies provide excellent foundational experience."},
        {"step": 4, "title": "Pursue M.Sc / PhD & Specialize", "description": "Specialize in bioinformatics, medical biotechnology, agricultural biotech, or industrial microbiology. A PhD unlocks high-level research leadership positions in industry and academia."},
    ],
    "Environmental Scientist": [
        {"step": 1, "title": "Develop Passion for Environment & Science", "description": "Engage deeply with environmental issues, climate data, and ecological research. Genuine passion for the planet is the core motivation that sustains long careers in this field."},
        {"step": 2, "title": "Pursue B.Sc in Environmental Science", "description": "Study ecology, environmental chemistry, geographic information systems (GIS), climatology, and environmental policy. Field work and practical project experience are especially valuable."},
        {"step": 3, "title": "Gain Field & Research Experience", "description": "Work on environmental impact assessments, water quality studies, or biodiversity surveys. NGO internships and government project contributions build a strong professional profile."},
        {"step": 4, "title": "Pursue M.Sc & Specialize", "description": "Specialize in climate change policy, wildlife conservation, pollution control, or environmental law. Work with CPCB, NGT, or international organizations like UNEP for global impact."},
    ],
    "Astrophysicist": [
        {"step": 1, "title": "Excel in Physics & Mathematics", "description": "Astrophysics is among the most mathematically demanding of all sciences. Build complete mastery in calculus, differential equations, and classical mechanics at the school level."},
        {"step": 2, "title": "Pursue B.Sc in Physics or Astronomy", "description": "Study classical mechanics, quantum physics, electromagnetism, special relativity, and astrophysics. Target institutions like IISc, IISER, or Tata Institute of Fundamental Research."},
        {"step": 3, "title": "Pursue M.Sc & Research Internships", "description": "Complete an M.Sc with a research thesis component. Apply for internship projects at ISRO, IUCAA, or international observatories to work with real astronomical datasets."},
        {"step": 4, "title": "Complete PhD & Publish Research", "description": "A PhD in astrophysics is the standard entry to research careers. Publish papers, present at international conferences, and apply for competitive postdoctoral positions globally."},
    ],
    "Forensic Scientist": [
        {"step": 1, "title": "Study Chemistry, Biology & Physics", "description": "Forensic science draws on all three core sciences. Excel especially in Chemistry and Biology, and develop deep curiosity about how physical evidence reveals hidden truths."},
        {"step": 2, "title": "Pursue B.Sc in Forensic Science", "description": "Study DNA profiling, toxicology, fingerprint analysis, ballistics, document examination, and crime scene investigation in a dedicated forensic science program."},
        {"step": 3, "title": "Complete Lab Internships", "description": "Intern at state Forensic Science Laboratories (FSLs), police forensic units, or private forensic consulting agencies to gain hands-on analytical practice with real evidence."},
        {"step": 4, "title": "Specialize & Join a Forensic Agency", "description": "Specialize in DNA forensics, cyber forensics, toxicological analysis, or questioned document examination. Apply to CBI, CFSL, state FSLs, or private investigation firms."},
    ],
    # ── Aviation & Defense ─────────────────────────────────────────────────────
    "Air Traffic Controller": [
        {"step": 1, "title": "Excel in Physics & Mathematics", "description": "ATC involves radar technology, navigation systems, and meteorology — all grounded in physics and math. Quick mental calculations and spatial awareness are essential daily skills."},
        {"step": 2, "title": "Pursue a Relevant Degree", "description": "Complete B.Sc in Aviation or Physics, or an engineering degree in Electronics. Alternatively, apply directly to AAI's ATC recruitment after 12th with PCM."},
        {"step": 3, "title": "Clear AAI Recruitment Exam", "description": "Apply to the Airports Authority of India (AAI) ATC recruitment examination. Clear written tests covering English communication, general knowledge, reasoning, and technical subjects."},
        {"step": 4, "title": "Complete ATC Training", "description": "Undergo intensive training at AAI's Civil Aviation Training College (CATC) in Allahabad. Earn your ATC license through radar simulator training and live tower practical experience."},
    ],
    # ── Default fallback ───────────────────────────────────────────────────────
    "default": [
        {"step": 1, "title": "Explore Your Interest Area", "description": "Research your chosen career deeply — talk to working professionals, watch industry documentaries, and read sector reports to understand what the daily reality truly looks like."},
        {"step": 2, "title": "Build Core Skills", "description": "Identify the 3–5 skills most valued in your target career and dedicate focused time each week to developing them through structured courses, projects, and deliberate practice."},
        {"step": 3, "title": "Gain Practical Experience", "description": "Seek internships, volunteer positions, or freelance projects in your chosen field. Practical hands-on experience always accelerates career growth far more than classroom study alone."},
        {"step": 4, "title": "Network & Apply Strategically", "description": "Connect with industry professionals on LinkedIn, attend relevant events and webinars, and apply to entry-level roles. A strong referral network dramatically shortens any job search."},
    ],
}


def get_career_roadmap(career_interest):
    """Return a career-specific 4-step roadmap, with fuzzy matching for career names."""
    if career_interest in CAREER_ROADMAPS:
        return CAREER_ROADMAPS[career_interest]

    career_lower = career_interest.lower()
    for key, roadmap in CAREER_ROADMAPS.items():
        if key.lower() in career_lower or career_lower in key.lower():
            return roadmap

    keyword_map = {
        "software": "Software Developer", "developer": "Software Developer",
        "full stack": "Full-Stack Developer", "fullstack": "Full-Stack Developer",
        "ai ": "AI Engineer", "artificial intelligence": "AI Engineer",
        "machine learning": "Machine Learning Engineer",
        "data scientist": "Data Scientist", "data science": "Data Scientist",
        "cyber": "Cybersecurity Analyst", "cloud": "Cloud Engineer",
        "devops": "DevOps Engineer", "mobile": "Mobile App Developer",
        "android": "Mobile App Developer", "ios": "Mobile App Developer",
        "game": "Game Developer", "blockchain": "Blockchain Developer",
        "robot": "Robotics Engineer", "doctor": "Doctor", "medical": "Doctor",
        "nurse": "Nurse", "pharma": "Pharmacist", "physio": "Physiotherapist",
        "psycholog": "Psychologist", "biomedical": "Biomedical Engineer",
        "mechanical": "Mechanical Engineer", "civil": "Civil Engineer",
        "electrical": "Electrical Engineer", "aerospace": "Aerospace Engineer",
        "automobile": "Automobile Engineer", "chemical engineer": "Chemical Engineer",
        "environmental engineer": "Environmental Engineer",
        "graphic": "Graphic Designer", "ui": "UI/UX Designer", "ux": "UI/UX Designer",
        "animat": "Animator", "fashion": "Fashion Designer",
        "interior": "Interior Designer", "product design": "Product Designer",
        "chartered": "Chartered Accountant", "investment bank": "Investment Banker",
        "financial analyst": "Financial Analyst", "entrepreneur": "Entrepreneur",
        "business analyst": "Business Analyst", "product manager": "Product Manager",
        "digital market": "Digital Marketer", "hr ": "Human Resource Manager",
        "human resource": "Human Resource Manager", "lawyer": "Lawyer",
        "ias": "IAS Officer", "army": "Army Officer", "police": "Police Officer",
        "journalist": "Journalist", "content creator": "Content Creator",
        "public relation": "Public Relations Manager", "video": "Video Producer",
        "teacher": "Teacher", "professor": "Professor", "pilot": "Pilot",
        "chef": "Chef", "hotel": "Hotel Manager", "event": "Event Manager",
        "research scientist": "Research Scientist", "biotechnolog": "Biotechnologist",
        "environmental scientist": "Environmental Scientist",
        "astrophys": "Astrophysicist", "forensic": "Forensic Scientist",
        "air traffic": "Air Traffic Controller", "criminal invest": "Criminal Investigator",
    }
    for keyword, mapped_career in keyword_map.items():
        if keyword in career_lower:
            return CAREER_ROADMAPS.get(mapped_career, CAREER_ROADMAPS["default"])

    return CAREER_ROADMAPS["default"]


# ─── Career library ────────────────────────────────────────────────────────────

ALL_CAREERS = [
    {"title": "Software Developer",        "field": "Technology",        "description": "Builds software applications and systems.",                              "demand": "High Demand", "match_percent": 88},
    {"title": "AI Engineer",               "field": "Data Science & AI", "description": "Creates machine learning and AI systems.",                            "demand": "Trending",    "match_percent": 91},
    {"title": "Data Scientist",            "field": "Data Science & AI", "description": "Analyzes data for insights and predictions.",                         "demand": "High Demand", "match_percent": 89},
    {"title": "Cybersecurity Analyst",     "field": "Technology",        "description": "Protects systems from cyber threats.",                                "demand": "High Demand", "match_percent": 86},
    {"title": "Cloud Engineer",            "field": "Technology",        "description": "Manages cloud infrastructure and services.",                           "demand": "Trending",    "match_percent": 84},
    {"title": "Game Developer",            "field": "Technology",        "description": "Designs and develops video games.",                                   "demand": "Emerging",    "match_percent": 82},
    {"title": "Mobile App Developer",      "field": "Technology",        "description": "Builds Android and iOS applications.",                                "demand": "High Demand", "match_percent": 87},
    {"title": "Blockchain Developer",      "field": "Technology",        "description": "Creates decentralized applications.",                                 "demand": "Emerging",    "match_percent": 78},
    {"title": "Robotics Engineer",         "field": "Technology",        "description": "Designs intelligent robotic systems.",                                "demand": "Trending",    "match_percent": 85},
    {"title": "DevOps Engineer",           "field": "Technology",        "description": "Automates deployment and server systems.",                            "demand": "High Demand", "match_percent": 83},
    {"title": "Full-Stack Developer",      "field": "Technology",        "description": "Builds both front-end and back-end web applications.",                "demand": "High Demand", "match_percent": 90},
    {"title": "Machine Learning Engineer", "field": "Data Science & AI", "description": "Develops and deploys ML models at scale.",                           "demand": "Trending",    "match_percent": 92},
    {"title": "Doctor",                    "field": "Healthcare",        "description": "Treats and diagnoses patients.",                                      "demand": "Stable",      "match_percent": 90},
    {"title": "Nurse",                     "field": "Healthcare",        "description": "Provides patient care and support.",                                  "demand": "High Demand", "match_percent": 84},
    {"title": "Pharmacist",                "field": "Healthcare",        "description": "Dispenses medicines and healthcare advice.",                           "demand": "Stable",      "match_percent": 81},
    {"title": "Physiotherapist",           "field": "Healthcare",        "description": "Helps patients recover physical movement.",                           "demand": "Trending",    "match_percent": 80},
    {"title": "Psychologist",              "field": "Healthcare",        "description": "Studies behavior and mental health.",                                 "demand": "Trending",    "match_percent": 83},
    {"title": "Radiologist",               "field": "Healthcare",        "description": "Interprets medical imaging for diagnosis.",                           "demand": "Stable",      "match_percent": 82},
    {"title": "Biomedical Engineer",       "field": "Healthcare",        "description": "Develops medical devices and equipment.",                             "demand": "Emerging",    "match_percent": 85},
    {"title": "Mechanical Engineer",       "field": "Engineering",       "description": "Designs machines and mechanical systems.",                            "demand": "Stable",      "match_percent": 86},
    {"title": "Civil Engineer",            "field": "Engineering",       "description": "Builds roads, bridges, and infrastructure.",                          "demand": "Stable",      "match_percent": 84},
    {"title": "Electrical Engineer",       "field": "Engineering",       "description": "Works on electrical systems and devices.",                            "demand": "High Demand", "match_percent": 85},
    {"title": "Aerospace Engineer",        "field": "Engineering",       "description": "Designs aircraft and spacecraft.",                                    "demand": "Emerging",    "match_percent": 82},
    {"title": "Automobile Engineer",       "field": "Engineering",       "description": "Develops vehicles and transport systems.",                            "demand": "Trending",    "match_percent": 79},
    {"title": "Chemical Engineer",         "field": "Engineering",       "description": "Applies chemistry to industrial processes.",                          "demand": "Stable",      "match_percent": 80},
    {"title": "Environmental Engineer",    "field": "Engineering",       "description": "Solves environmental problems using engineering principles.",          "demand": "Trending",    "match_percent": 78},
    {"title": "Research Scientist",        "field": "Science",           "description": "Conducts experiments to advance knowledge.",                          "demand": "Stable",      "match_percent": 83},
    {"title": "Astrophysicist",            "field": "Science",           "description": "Studies the physics of stars and the universe.",                      "demand": "Emerging",    "match_percent": 76},
    {"title": "Biotechnologist",           "field": "Science",           "description": "Uses biology and technology for research and production.",            "demand": "Trending",    "match_percent": 84},
    {"title": "Forensic Scientist",        "field": "Science",           "description": "Applies science to criminal investigations.",                         "demand": "Trending",    "match_percent": 80},
    {"title": "Environmental Scientist",   "field": "Science",           "description": "Studies environmental systems and climate.",                          "demand": "Trending",    "match_percent": 78},
    {"title": "Teacher",                   "field": "Education",         "description": "Educates students in schools or colleges.",                           "demand": "Stable",      "match_percent": 80},
    {"title": "Professor",                 "field": "Education",         "description": "Teaches and researches academic subjects.",                           "demand": "Stable",      "match_percent": 78},
    {"title": "Educational Technologist",  "field": "Education",         "description": "Improves learning using technology.",                                 "demand": "Trending",    "match_percent": 76},
    {"title": "School Counselor",          "field": "Education",         "description": "Guides students in academic and personal development.",               "demand": "Stable",      "match_percent": 74},
    {"title": "Graphic Designer",          "field": "Design",            "description": "Creates visual graphics and branding.",                               "demand": "Trending",    "match_percent": 81},
    {"title": "UI/UX Designer",            "field": "Design",            "description": "Designs user-friendly digital interfaces.",                           "demand": "High Demand", "match_percent": 87},
    {"title": "Animator",                  "field": "Design",            "description": "Creates animated movies and content.",                                "demand": "Emerging",    "match_percent": 79},
    {"title": "Fashion Designer",          "field": "Design",            "description": "Designs clothing and accessories.",                                   "demand": "Stable",      "match_percent": 75},
    {"title": "Interior Designer",         "field": "Design",            "description": "Designs attractive interior spaces.",                                 "demand": "Trending",    "match_percent": 77},
    {"title": "Product Designer",          "field": "Design",            "description": "Creates physical and digital product experiences.",                   "demand": "High Demand", "match_percent": 86},
    {"title": "Lawyer",                    "field": "Law",               "description": "Provides legal advice and representation.",                           "demand": "Stable",      "match_percent": 84},
    {"title": "Judge",                     "field": "Law",               "description": "Oversees court proceedings and justice.",                             "demand": "Stable",      "match_percent": 76},
    {"title": "Criminal Investigator",     "field": "Law",               "description": "Investigates criminal cases.",                                        "demand": "Trending",    "match_percent": 80},
    {"title": "IAS Officer",               "field": "Government",        "description": "Works in civil services administration.",                             "demand": "Prestigious", "match_percent": 86},
    {"title": "Police Officer",            "field": "Government",        "description": "Maintains law and order.",                                            "demand": "Stable",      "match_percent": 79},
    {"title": "Army Officer",              "field": "Defense",           "description": "Serves in national defense forces.",                                  "demand": "Stable",      "match_percent": 82},
    {"title": "Digital Marketer",          "field": "Marketing",         "description": "Promotes brands online.",                                             "demand": "High Demand", "match_percent": 85},
    {"title": "Content Creator",           "field": "Marketing",         "description": "Creates engaging online content.",                                    "demand": "Trending",    "match_percent": 79},
    {"title": "Journalist",                "field": "Journalism",        "description": "Reports news and current events.",                                    "demand": "Stable",      "match_percent": 74},
    {"title": "Public Relations Manager",  "field": "Marketing",         "description": "Manages brand image and media.",                                      "demand": "Trending",    "match_percent": 78},
    {"title": "Copywriter",                "field": "Journalism",        "description": "Writes persuasive content for brands and advertising.",               "demand": "Trending",    "match_percent": 76},
    {"title": "Video Producer",            "field": "Arts & Media",      "description": "Plans, shoots, and edits video content.",                            "demand": "Trending",    "match_percent": 80},
    {"title": "Chartered Accountant",      "field": "Finance",           "description": "Handles financial audits and taxation.",                              "demand": "Stable",      "match_percent": 88},
    {"title": "Investment Banker",         "field": "Finance",           "description": "Manages investments and financial deals.",                            "demand": "High Demand", "match_percent": 83},
    {"title": "Financial Analyst",         "field": "Finance",           "description": "Analyzes financial market trends.",                                   "demand": "Trending",    "match_percent": 84},
    {"title": "Bank Manager",              "field": "Finance",           "description": "Oversees banking operations.",                                        "demand": "Stable",      "match_percent": 77},
    {"title": "Entrepreneur",              "field": "Business",          "description": "Starts and manages businesses.",                                      "demand": "Emerging",    "match_percent": 89},
    {"title": "Business Analyst",          "field": "Business",          "description": "Improves business processes.",                                        "demand": "High Demand", "match_percent": 82},
    {"title": "Product Manager",           "field": "Business",          "description": "Leads product planning and strategy.",                                "demand": "Trending",    "match_percent": 84},
    {"title": "Human Resource Manager",    "field": "Business",          "description": "Handles employee relations and hiring.",                              "demand": "Stable",      "match_percent": 78},
    {"title": "Pilot",                     "field": "Aviation",          "description": "Operates aircraft and flights.",                                      "demand": "Trending",    "match_percent": 85},
    {"title": "Air Traffic Controller",    "field": "Aviation",          "description": "Manages aircraft movement safely.",                                   "demand": "High Demand", "match_percent": 80},
    {"title": "Chef",                      "field": "Hospitality",       "description": "Prepares food professionally.",                                       "demand": "Trending",    "match_percent": 74},
    {"title": "Hotel Manager",             "field": "Hospitality",       "description": "Manages hotel operations.",                                           "demand": "Stable",      "match_percent": 73},
    {"title": "Event Manager",             "field": "Hospitality",       "description": "Organizes events and programs.",                                      "demand": "Trending",    "match_percent": 77},
]

CAREER_FIELD_KEYWORD_MAP = {
    "technology": ["Technology", "Data Science & AI"],
    "computer": ["Technology", "Data Science & AI"],
    "programming": ["Technology", "Data Science & AI"],
    "software": ["Technology", "Data Science & AI"],
    "it": ["Technology", "Data Science & AI"],
    "web": ["Technology", "Data Science & AI"],
    "coding": ["Technology", "Data Science & AI"],
    "data": ["Data Science & AI", "Technology"],
    "ai": ["Data Science & AI", "Technology"],
    "machine learning": ["Data Science & AI", "Technology"],
    "artificial intelligence": ["Data Science & AI", "Technology"],
    "design": ["Design"],
    "art": ["Design", "Arts & Media"],
    "animation": ["Design", "Arts & Media"],
    "fashion": ["Design"],
    "creative": ["Design", "Arts & Media"],
    "business": ["Business", "Finance"],
    "management": ["Business", "Finance"],
    "entrepreneurship": ["Business"],
    "commerce": ["Business", "Finance"],
    "finance": ["Finance", "Business"],
    "banking": ["Finance"],
    "accounting": ["Finance"],
    "economics": ["Finance", "Business"],
    "medical": ["Healthcare"],
    "health": ["Healthcare"],
    "biology": ["Healthcare", "Science"],
    "medicine": ["Healthcare"],
    "pharmacy": ["Healthcare"],
    "nursing": ["Healthcare"],
    "doctor": ["Healthcare"],
    "law": ["Law", "Government"],
    "legal": ["Law"],
    "upsc": ["Government"],
    "civil services": ["Government"],
    "government": ["Government", "Law"],
    "defense": ["Defense", "Government"],
    "army": ["Defense"],
    "science": ["Science", "Engineering"],
    "physics": ["Science", "Engineering"],
    "chemistry": ["Science", "Healthcare"],
    "engineering": ["Engineering", "Science"],
    "mechanical": ["Engineering"],
    "civil": ["Engineering"],
    "electrical": ["Engineering"],
    "maths": ["Science", "Technology"],
    "mathematics": ["Science", "Technology"],
    "media": ["Arts & Media", "Journalism", "Marketing"],
    "journalism": ["Journalism", "Arts & Media"],
    "communication": ["Marketing", "Journalism"],
    "marketing": ["Marketing", "Business"],
    "digital marketing": ["Marketing"],
    "education": ["Education"],
    "teaching": ["Education"],
    "teacher": ["Education"],
    "aviation": ["Aviation"],
    "pilot": ["Aviation"],
    "hospitality": ["Hospitality"],
    "hotel": ["Hospitality"],
    "event": ["Hospitality"],
}


def get_careers_for_profile(subject, interest):
    combined = f"{subject} {interest}".lower()
    interest_lower = interest.lower()
    if "teacher" in interest_lower or "professor" in interest_lower:
        matched_fields = {"Education"}
    else:
        matched_fields = set()
        for kw, fields in CAREER_FIELD_KEYWORD_MAP.items():
            if kw in combined:
                matched_fields.update(fields)
    filtered = [c for c in ALL_CAREERS if c["field"] in matched_fields]
    if len(filtered) < 4:
        if matched_fields == {"Education"}:
            filtered = [c for c in ALL_CAREERS if c["field"] == "Education"]
        else:
            filtered = [c for c in ALL_CAREERS if c["field"] in ("Technology", "Business", "Science")]
    filtered.sort(key=lambda c: c["match_percent"], reverse=True)
    return filtered[:8]


# ─── Resource library ──────────────────────────────────────────────────────────

ALL_RESOURCES = [
    {"name": "Codecademy",                              "type": "Platform",  "field": "Technology",        "url": "https://www.codecademy.com"},
    {"name": "freeCodeCamp",                            "type": "Platform",  "field": "Technology",        "url": "https://www.freecodecamp.org"},
    {"name": "The Odin Project",                        "type": "Platform",  "field": "Technology",        "url": "https://www.theodinproject.com"},
    {"name": "LeetCode",                                "type": "Platform",  "field": "Technology",        "url": "https://www.leetcode.com"},
    {"name": "GitHub",                                  "type": "Platform",  "field": "Technology",        "url": "https://www.github.com"},
    {"name": "Stack Overflow",                          "type": "Community", "field": "Technology",        "url": "https://stackoverflow.com"},
    {"name": "CS50 by Harvard",                         "type": "Course",    "field": "Technology",        "url": "https://cs50.harvard.edu"},
    {"name": "GeeksforGeeks",                           "type": "Platform",  "field": "Technology",        "url": "https://www.geeksforgeeks.org"},
    {"name": "HackerRank",                              "type": "Platform",  "field": "Technology",        "url": "https://www.hackerrank.com"},
    {"name": "W3Schools",                               "type": "Platform",  "field": "Technology",        "url": "https://www.w3schools.com"},
    {"name": "Kaggle",                                  "type": "Platform",  "field": "Data Science & AI", "url": "https://www.kaggle.com"},
    {"name": "fast.ai",                                 "type": "Course",    "field": "Data Science & AI", "url": "https://www.fast.ai"},
    {"name": "Google ML Crash Course",                  "type": "Course",    "field": "Data Science & AI", "url": "https://developers.google.com/machine-learning/crash-course"},
    {"name": "Towards Data Science",                    "type": "Community", "field": "Data Science & AI", "url": "https://towardsdatascience.com"},
    {"name": "DataCamp",                                "type": "Platform",  "field": "Data Science & AI", "url": "https://www.datacamp.com"},
    {"name": "Hugging Face",                            "type": "Platform",  "field": "Data Science & AI", "url": "https://huggingface.co"},
    {"name": "Khan Academy",                            "type": "Platform",  "field": "General",           "url": "https://www.khanacademy.org"},
    {"name": "Coursera",                                "type": "Platform",  "field": "General",           "url": "https://www.coursera.org"},
    {"name": "edX",                                     "type": "Platform",  "field": "General",           "url": "https://www.edx.org"},
    {"name": "Udemy",                                   "type": "Platform",  "field": "General",           "url": "https://www.udemy.com"},
    {"name": "MIT OpenCourseWare",                      "type": "Course",    "field": "General",           "url": "https://ocw.mit.edu"},
    {"name": "NPTEL (India)",                           "type": "Course",    "field": "General",           "url": "https://nptel.ac.in"},
    {"name": "Swayam",                                  "type": "Platform",  "field": "General",           "url": "https://swayam.gov.in"},
    {"name": "Canva Design School",                     "type": "Course",    "field": "Design",            "url": "https://www.canva.com/learn/design"},
    {"name": "Adobe Creative Cloud Tutorials",          "type": "Course",    "field": "Design",            "url": "https://helpx.adobe.com/creative-cloud/tutorials-explore.html"},
    {"name": "Dribbble",                                "type": "Community", "field": "Design",            "url": "https://dribbble.com"},
    {"name": "Behance",                                 "type": "Community", "field": "Design",            "url": "https://www.behance.net"},
    {"name": "Figma Community",                         "type": "Platform",  "field": "Design",            "url": "https://www.figma.com/community"},
    {"name": "Interaction Design Foundation",           "type": "Course",    "field": "Design",            "url": "https://www.interaction-design.org"},
    {"name": "Harvard Business Review",                 "type": "Book",      "field": "Business",          "url": "https://hbr.org"},
    {"name": "SCORE Mentoring",                         "type": "Community", "field": "Business",          "url": "https://www.score.org"},
    {"name": "Startup India",                           "type": "Platform",  "field": "Business",          "url": "https://www.startupindia.gov.in"},
    {"name": "Y Combinator Startup Library",            "type": "Book",      "field": "Business",          "url": "https://www.ycombinator.com/library"},
    {"name": "Google Garage",                           "type": "Course",    "field": "Business",          "url": "https://learndigital.withgoogle.com/digitalgarage"},
    {"name": "Investopedia",                            "type": "Platform",  "field": "Finance",           "url": "https://www.investopedia.com"},
    {"name": "CFA Institute",                           "type": "Platform",  "field": "Finance",           "url": "https://www.cfainstitute.org"},
    {"name": "Zerodha Varsity",                         "type": "Course",    "field": "Finance",           "url": "https://zerodha.com/varsity"},
    {"name": "NSE India Learning",                      "type": "Course",    "field": "Finance",           "url": "https://www.nseindia.com/invest/nse-pathashaala"},
    {"name": "MedlinePlus",                             "type": "Platform",  "field": "Healthcare",        "url": "https://medlineplus.gov"},
    {"name": "Osmosis",                                 "type": "Course",    "field": "Healthcare",        "url": "https://www.osmosis.org"},
    {"name": "Amboss",                                  "type": "Platform",  "field": "Healthcare",        "url": "https://www.amboss.com"},
    {"name": "Medscape",                                "type": "Platform",  "field": "Healthcare",        "url": "https://www.medscape.com"},
    {"name": "WHO Learning Hub",                        "type": "Course",    "field": "Healthcare",        "url": "https://openwho.org"},
    {"name": "Indian Kanoon",                           "type": "Platform",  "field": "Law",               "url": "https://indiankanoon.org"},
    {"name": "UPSC Official Portal",                    "type": "Platform",  "field": "Government",        "url": "https://upsc.gov.in"},
    {"name": "Drishti IAS",                             "type": "Platform",  "field": "Government",        "url": "https://www.drishtiias.com"},
    {"name": "Vision IAS",                              "type": "Platform",  "field": "Government",        "url": "https://www.visionias.in"},
    {"name": "Bar & Bench",                             "type": "Community", "field": "Law",               "url": "https://www.barandbench.com"},
    {"name": "Brilliant.org",                           "type": "Platform",  "field": "Science",           "url": "https://brilliant.org"},
    {"name": "NASA STEM Engagement",                    "type": "Platform",  "field": "Science",           "url": "https://www.nasa.gov/stem"},
    {"name": "PhET Simulations",                        "type": "Platform",  "field": "Science",           "url": "https://phet.colorado.edu"},
    {"name": "ISRO Student Programs",                   "type": "Platform",  "field": "Science",           "url": "https://www.isro.gov.in/students.html"},
    {"name": "Khan Academy – Science",                  "type": "Course",    "field": "Science",           "url": "https://www.khanacademy.org/science"},
    {"name": "Skillshare",                              "type": "Platform",  "field": "Arts & Media",      "url": "https://www.skillshare.com"},
    {"name": "Poynter Institute",                       "type": "Course",    "field": "Journalism",        "url": "https://www.poynter.org"},
    {"name": "HubSpot Academy",                         "type": "Course",    "field": "Marketing",         "url": "https://academy.hubspot.com"},
    {"name": "Google Digital Garage",                   "type": "Course",    "field": "Marketing",         "url": "https://learndigital.withgoogle.com/digitalgarage"},
    {"name": "Semrush Academy",                         "type": "Course",    "field": "Marketing",         "url": "https://www.semrush.com/academy"},
    {"name": "Teach For India",                         "type": "Community", "field": "Education",         "url": "https://teachforindia.org"},
    {"name": "Teachers Pay Teachers",                   "type": "Platform",  "field": "Education",         "url": "https://www.teacherspayteachers.com"},
    {"name": "Edutopia",                                "type": "Community", "field": "Education",         "url": "https://www.edutopia.org"},
    {"name": "AHLEI",                                   "type": "Course",    "field": "Hospitality",       "url": "https://www.ahlei.org"},
    {"name": "Eventbrite Resources",                    "type": "Platform",  "field": "Hospitality",       "url": "https://www.eventbrite.com/blog"},
    {"name": "DGCA India",                              "type": "Platform",  "field": "Aviation",          "url": "https://dgca.gov.in"},
    {"name": "Join Indian Army",                        "type": "Platform",  "field": "Defense",           "url": "https://joinindianarmy.nic.in"},
    {"name": "SSB Crack",                               "type": "Platform",  "field": "Defense",           "url": "https://ssbcrack.com"},
    {"name": "LinkedIn Learning",                       "type": "Platform",  "field": "Career",            "url": "https://www.linkedin.com/learning"},
    {"name": "Internshala",                             "type": "Platform",  "field": "Career",            "url": "https://internshala.com"},
    {"name": "Indeed Career Guide",                     "type": "Platform",  "field": "Career",            "url": "https://www.indeed.com/career-advice"},
    {"name": "Glassdoor",                               "type": "Platform",  "field": "Career",            "url": "https://www.glassdoor.com"},
]

FIELD_KEYWORD_MAP = {
    "technology": ["Technology", "Data Science & AI", "General"],
    "computer": ["Technology", "Data Science & AI", "General"],
    "programming": ["Technology", "Data Science & AI", "General"],
    "software": ["Technology", "Data Science & AI", "General"],
    "it": ["Technology", "Data Science & AI", "General"],
    "data": ["Data Science & AI", "Technology", "General"],
    "ai": ["Data Science & AI", "Technology", "General"],
    "machine learning": ["Data Science & AI", "Technology"],
    "design": ["Design", "Arts & Media", "General"],
    "art": ["Design", "Arts & Media", "General"],
    "animation": ["Design", "Arts & Media"],
    "fashion": ["Design", "Arts & Media"],
    "business": ["Business", "Finance", "General"],
    "management": ["Business", "Finance", "General"],
    "entrepreneurship": ["Business", "General"],
    "commerce": ["Business", "Finance", "General"],
    "finance": ["Finance", "Business", "General"],
    "banking": ["Finance", "Business"],
    "accounting": ["Finance", "Business"],
    "medical": ["Healthcare", "Science", "General"],
    "health": ["Healthcare", "Science", "General"],
    "biology": ["Healthcare", "Science", "General"],
    "medicine": ["Healthcare", "Science"],
    "pharmacy": ["Healthcare", "Science"],
    "law": ["Law", "Government", "General"],
    "legal": ["Law", "Government"],
    "upsc": ["Government", "Law"],
    "civil services": ["Government", "Law"],
    "government": ["Government", "Law"],
    "science": ["Science", "Technology", "General"],
    "physics": ["Science", "Technology"],
    "chemistry": ["Science", "Healthcare"],
    "engineering": ["Science", "Technology", "General"],
    "maths": ["Science", "Technology", "General"],
    "mathematics": ["Science", "Technology", "General"],
    "media": ["Arts & Media", "Journalism", "Marketing"],
    "journalism": ["Journalism", "Arts & Media"],
    "communication": ["Arts & Media", "Journalism", "Marketing"],
    "marketing": ["Marketing", "Business", "General"],
    "digital marketing": ["Marketing", "Business"],
    "education": ["Education", "General"],
    "teaching": ["Education", "General"],
    "teacher": ["Education"],
    "aviation": ["Aviation", "Defense"],
    "pilot": ["Aviation"],
    "defense": ["Defense", "Government"],
    "army": ["Defense", "Government"],
    "hospitality": ["Hospitality", "Business"],
    "hotel": ["Hospitality"],
    "event": ["Hospitality"],
}


def get_resources_for_profile(subject, interest):
    combined = f"{subject} {interest}".lower()
    interest_lower = interest.lower()
    if "teacher" in interest_lower or "professor" in interest_lower:
        matched_fields = {"Education"}
    else:
        matched_fields = set()
        for kw, fields in FIELD_KEYWORD_MAP.items():
            if kw in combined:
                matched_fields.update(fields)
    matched_fields.add("Career")
    matched_fields.add("General")
    filtered = [r for r in ALL_RESOURCES if r["field"] in matched_fields]
    if len(filtered) < 6:
        filtered = [r for r in ALL_RESOURCES if r["field"] in ("General", "Career", "Technology")]
    return filtered


def call_local_api(prompt):
    name_m     = re.search(r"^- Name:\s*(.+)$",             prompt, re.MULTILINE)
    subject_m  = re.search(r"^- Favorite Subject:\s*(.+)$", prompt, re.MULTILINE)
    interest_m = re.search(r"^- Career Interest:\s*(.+)$",  prompt, re.MULTILINE)
    name     = name_m.group(1).strip()     if name_m     else "Student"
    subject  = subject_m.group(1).strip()  if subject_m  else ""
    interest = interest_m.group(1).strip() if interest_m else ""
    careers   = get_careers_for_profile(subject, interest)
    resources = get_resources_for_profile(subject, interest)
    data = {
        "summary": f"{name} has a strong foundation and is ready to explore career paths aligned with their skills and interests.",
        "top_careers": careers,
        "skill_scores": {
            "Technical Skills": 78, "Communication": 72,
            "Problem Solving": 82, "Creativity": 75, "Leadership": 68,
        },
        "resources": resources,
        "motivational_tip": "Keep learning every day and turn small wins into your long-term career plan.",
    }
    return json.dumps(data)


# ─── Routes ────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    form_data = session.get("form_data", {})
    return render_template("index.html", errors=[], form=form_data)


@app.route("/analyze", methods=["POST"])
def analyze():
    form_data = {
        "name":        request.form.get("name", "").strip(),
        "email":       request.form.get("email", "").strip(),
        "age":         request.form.get("age", "").strip(),
        "semester":    request.form.get("semester", "").strip(),
        "subject":     request.form.get("subject", ""),
        "skill":       request.form.get("skill", ""),
        "interest":    request.form.get("interest", ""),
        "user_skills": request.form.get("user_skills", "").strip(),
    }
    errors = validate_form(form_data)
    if errors:
        return render_template("index.html", errors=errors, form=form_data)
    session["form_data"] = form_data
    session["ai_prompt"] = build_ai_prompt(form_data)
    return redirect(url_for("results"))


@app.route("/results")
def results():
    form_data = session.get("form_data")
    if not form_data:
        return redirect(url_for("index"))
    prompt = session.get("ai_prompt", "")
    return render_template("results.html", form=form_data, prompt=prompt)


@app.route("/reset")
def reset():
    session.pop("form_data", None)
    session.pop("ai_prompt", None)
    return redirect(url_for("index"))


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    data   = request.get_json(silent=True) or {}
    prompt = data.get("prompt", "")
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    provider = get_ai_provider()
    start    = __import__("time").time()

    try:
        if provider == "openai":
            text = call_openai_api(prompt)
        elif provider == "anthropic":
            text = call_anthropic_api(prompt)
        else:
            text = call_local_api(prompt)

        text     = re.sub(r"```json|```", "", text).strip()
        analysis = json.loads(text)

        # Inject curated career-specific roadmap & resources
        form_data       = session.get("form_data", {})
        career_interest = form_data.get("interest", "")
        subject         = form_data.get("subject", "")

        analysis["roadmap"]   = get_career_roadmap(career_interest)
        curated_resources     = get_resources_for_profile(subject, career_interest)
        if curated_resources:
            analysis["resources"] = curated_resources

        if not analysis.get("top_careers"):
            analysis["top_careers"] = get_careers_for_profile(subject, career_interest)

        # Save to DB
        student = Student.query.filter_by(email=form_data.get("email", "")).first()
        if not student:
            student = Student(
                full_name       = form_data.get("name", ""),
                email           = form_data.get("email", ""),
                age             = int(form_data.get("age", 0)),
                semester        = form_data.get("semester", ""),
                subject         = form_data.get("subject", ""),
                skill_level     = form_data.get("skill", "None"),
                career_interest = form_data.get("interest", ""),
                user_skills     = form_data.get("user_skills", ""),
            )
            db.session.add(student)
            db.session.flush()

        analysis_row = Analysis(
            student_id       = student.id,
            ai_provider      = provider,
            ai_model         = get_ai_model(provider),
            prompt_text      = prompt,
            response_json    = json.dumps(analysis),
            summary          = analysis.get("summary", ""),
            motivational_tip = analysis.get("motivational_tip", ""),
            status           = "success",
        )
        db.session.add(analysis_row)
        db.session.commit()

        return jsonify({"success": True, "analysis": analysis})

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500


# ─── Bootstrap ─────────────────────────────────────────────────────────────────
with app.app_context():
    db.create_all()

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)