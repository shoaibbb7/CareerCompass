"""
CareerCompass AI Model Inference & Manager Engine
Provides inference, model loading/caching, metadata retrieval,
and training triggers for the CareerCompass Flask application.
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_PATH = os.path.join(BASE_DIR, "data", "career_dataset.csv")

# In-memory cache for models
_PREPROCESSOR = None
_CAREER_CLASSIFIER = None
_SKILL_REGRESSOR = None
_METADATA = None

CAREER_FIELD_LOOKUP = {
    "Software Developer": "Technology",
    "Full-Stack Developer": "Technology",
    "AI Engineer": "Technology",
    "Data Scientist": "Data Science & AI",
    "Cybersecurity Analyst": "Technology",
    "Cloud Engineer": "Technology",
    "Mobile App Developer": "Technology",
    "DevOps Engineer": "Technology",
    "Doctor": "Healthcare",
    "Dentist": "Healthcare",
    "Pharmacist": "Healthcare",
    "Psychologist": "Healthcare",
    "Biomedical Engineer": "Healthcare",
    "Research Scientist": "Science",
    "Astrophysicist": "Science",
    "Mechanical Engineer": "Engineering",
    "Civil Engineer": "Engineering",
    "Electrical Engineer": "Engineering",
    "Aerospace Engineer": "Engineering",
    "Robotics Engineer": "Engineering",
    "UI/UX Designer": "Design",
    "Graphic Designer": "Design",
    "Animator": "Design",
    "Product Designer": "Design",
    "Chartered Accountant": "Finance",
    "Investment Banker": "Finance",
    "Financial Analyst": "Finance",
    "Business Analyst": "Business",
    "Product Manager": "Business",
    "Entrepreneur": "Business",
    "Digital Marketer": "Marketing",
    "Lawyer": "Law",
    "IAS Officer": "Government",
    "Journalist": "Marketing",
    "Professor": "Education",
    "Teacher": "Education",
    "Pilot": "Aviation",
    "Chef": "Hospitality",
}

CAREER_DEMAND_LOOKUP = {
    "Software Developer": "High Demand",
    "Full-Stack Developer": "High Demand",
    "AI Engineer": "High Demand",
    "Data Scientist": "High Demand",
    "Cybersecurity Analyst": "High Demand",
    "Cloud Engineer": "High Demand",
    "Mobile App Developer": "Trending",
    "DevOps Engineer": "High Demand",
    "Doctor": "High Demand",
    "Dentist": "Stable",
    "Pharmacist": "Stable",
    "Psychologist": "High Demand",
    "Biomedical Engineer": "Trending",
    "Research Scientist": "Stable",
    "Astrophysicist": "Emerging",
    "Mechanical Engineer": "Stable",
    "Civil Engineer": "Stable",
    "Electrical Engineer": "High Demand",
    "Aerospace Engineer": "Emerging",
    "Robotics Engineer": "High Demand",
    "UI/UX Designer": "High Demand",
    "Graphic Designer": "Trending",
    "Animator": "Emerging",
    "Product Designer": "High Demand",
    "Chartered Accountant": "Stable",
    "Investment Banker": "High Demand",
    "Financial Analyst": "Trending",
    "Business Analyst": "High Demand",
    "Product Manager": "Trending",
    "Entrepreneur": "Emerging",
    "Digital Marketer": "High Demand",
    "Lawyer": "Stable",
    "IAS Officer": "Prestigious",
    "Journalist": "Stable",
    "Professor": "Stable",
    "Teacher": "Stable",
    "Pilot": "Trending",
    "Chef": "Trending",
}

CAREER_DESCRIPTIONS = {
    "Software Developer": "Develops core computer applications and software architecture.",
    "Full-Stack Developer": "Builds complete web and cloud applications from frontend to backend.",
    "AI Engineer": "Designs and deploys artificial intelligence and deep learning models.",
    "Data Scientist": "Extracts predictive insights from complex datasets using machine learning.",
    "Cybersecurity Analyst": "Safeguards systems, networks, and data against security threats.",
    "Cloud Engineer": "Architects scalable, reliable cloud infrastructure and services.",
    "Mobile App Developer": "Creates native and cross-platform mobile apps for iOS and Android.",
    "DevOps Engineer": "Automates deployment pipelines, cloud environments, and CI/CD workflows.",
    "Doctor": "Diagnoses, treats, and helps prevent human illnesses and diseases.",
    "Dentist": "Specializes in oral healthcare, teeth hygiene, and surgical care.",
    "Pharmacist": "Prepares and dispenses medications while advising on optimal usage.",
    "Psychologist": "Studies human behavior, mental processes, and provides clinical counseling.",
    "Biomedical Engineer": "Develops medical devices, prosthetics, and healthcare instrumentation.",
    "Research Scientist": "Conducts controlled scientific investigations and discovers new principles.",
    "Astrophysicist": "Explores the physical laws governing astronomical phenomena and the cosmos.",
    "Mechanical Engineer": "Designs mechanical systems, thermal equipment, and automated machines.",
    "Civil Engineer": "Designs and supervises construction of bridges, buildings, and infrastructure.",
    "Electrical Engineer": "Develops electrical components, circuits, and energy systems.",
    "Aerospace Engineer": "Designs spacecraft, aircraft, satellites, and aeronautical hardware.",
    "Robotics Engineer": "Builds autonomous robotic systems, actuators, and computer vision systems.",
    "UI/UX Designer": "Creates user-centered digital interfaces, design systems, and product flows.",
    "Graphic Designer": "Communicates visual ideas through typography, branding, and layout.",
    "Animator": "Produces 2D and 3D animations, visual effects, and CGI.",
    "Product Designer": "Crafts digital and physical product experiences with high usability.",
    "Chartered Accountant": "Manages financial audits, tax strategies, and compliance.",
    "Investment Banker": "Advises on capital raising, corporate mergers, and investments.",
    "Financial Analyst": "Evaluates financial health, market trends, and investment opportunities.",
    "Business Analyst": "Aligns business requirements with technology solutions to optimize workflows.",
    "Product Manager": "Defines product vision, strategy, and roadmap from conception to launch.",
    "Entrepreneur": "Builds innovative ventures, identifies market opportunities, and scales businesses.",
    "Digital Marketer": "Leads multichannel online growth, SEO campaigns, and content strategies.",
    "Lawyer": "Provides legal counsel, drafts contracts, and represents clients in proceedings.",
    "IAS Officer": "Directs civil administration, public policy implementation, and governance.",
    "Journalist": "Investigates, writes, and broadcasts timely news and factual stories.",
    "Professor": "Instructs higher education students and leads academic research initiatives.",
    "Teacher": "Educates and inspires foundational learning and developmental growth.",
    "Pilot": "Navigates and operates civil and commercial aircraft safely.",
    "Chef": "Leads culinary creations, recipe design, and kitchen operations.",
}

def is_model_trained():
    """Returns True if trained model artifacts exist."""
    required = [
        os.path.join(MODELS_DIR, "preprocessor.joblib"),
        os.path.join(MODELS_DIR, "career_classifier.joblib"),
        os.path.join(MODELS_DIR, "skill_regressor.joblib"),
        os.path.join(MODELS_DIR, "model_metadata.json"),
    ]
    return all(os.path.exists(p) for p in required)

def get_model_metadata():
    """Loads and returns model training metadata and metrics."""
    meta_path = os.path.join(MODELS_DIR, "model_metadata.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "status": "not_trained",
        "algorithm": "Random Forest Ensemble (Pending Training)",
        "accuracy": 0,
        "top3_accuracy": 0,
        "num_samples": 0,
        "num_classes": 0,
        "trained_at": "Never",
        "top_features": [],
    }

def load_models(force_reload=False):
    """Loads preprocessor, classifier, and regressor into memory."""
    global _PREPROCESSOR, _CAREER_CLASSIFIER, _SKILL_REGRESSOR, _METADATA

    if not force_reload and _PREPROCESSOR is not None and _CAREER_CLASSIFIER is not None and _SKILL_REGRESSOR is not None:
        return _PREPROCESSOR, _CAREER_CLASSIFIER, _SKILL_REGRESSOR

    if not is_model_trained():
        return None, None, None

    try:
        _PREPROCESSOR = joblib.load(os.path.join(MODELS_DIR, "preprocessor.joblib"))
        _CAREER_CLASSIFIER = joblib.load(os.path.join(MODELS_DIR, "career_classifier.joblib"))
        _SKILL_REGRESSOR = joblib.load(os.path.join(MODELS_DIR, "skill_regressor.joblib"))
        _METADATA = get_model_metadata()
        return _PREPROCESSOR, _CAREER_CLASSIFIER, _SKILL_REGRESSOR
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None, None

def train_model_pipeline():
    """Triggers retraining of the model and refreshes the in-memory cache."""
    import train_model
    meta = train_model.train_and_evaluate(dataset_path=DATA_PATH, save_artifacts=True)
    load_models(force_reload=True)
    return meta

def predict_career_and_skills(profile, real_skill_scores=None):
    """
    Given student profile dict:
    - age, semester, subject, skill, interest, user_skills
    Returns AI-model-predicted top careers with match percentages.

    Skill scores: if `real_skill_scores` is provided (a dict of the 5 skill
    labels -> 0-100 scores computed from the student's own self-assessment
    answers), those are used as-is. Otherwise this falls back to the
    regression model's estimate, which is a rougher guess based on profile
    fields rather than an actual measurement.
    """
    preprocessor, clf, reg = load_models()
    
    # If model is not trained yet, trigger training now
    if preprocessor is None or clf is None or reg is None:
        train_model_pipeline()
        preprocessor, clf, reg = load_models()

    # Format input DataFrame
    input_data = pd.DataFrame([{
        "age": float(profile.get("age", 20) or 20),
        "semester": str(profile.get("semester", "1st Semester")),
        "favorite_subject": str(profile.get("subject", "Computer Science")),
        "programming_skill": str(profile.get("skill", "None")),
        "career_interest": str(profile.get("interest", "Software Developer")),
        "skills_text": str(profile.get("user_skills", "")),
    }])

    # Preprocess
    X_trans = preprocessor.transform(input_data)

    # 1. Career Probability Predictions
    probs = clf.predict_proba(X_trans)[0]
    classes = clf.classes_
    top_indices = np.argsort(probs)[::-1]

    # Normalize match percentages into a realistic 60% - 98% scale
    max_p = probs[top_indices[0]]
    ranked_careers = []
    
    for rank, idx in enumerate(top_indices[:8]):
        c_name = classes[idx]
        raw_prob = probs[idx]
        
        # Scale match percentage
        if max_p > 0:
            rel_score = raw_prob / max_p
        else:
            rel_score = 0.5
        
        match_pct = int(round(62 + (rel_score * 35) - (rank * 2)))
        match_pct = max(55, min(98, match_pct))

        field = CAREER_FIELD_LOOKUP.get(c_name, "General")
        demand = CAREER_DEMAND_LOOKUP.get(c_name, "High Demand")
        desc = CAREER_DESCRIPTIONS.get(c_name, f"Excelling in {field} through focused skill development.")

        ranked_careers.append({
            "title": c_name,
            "field": field,
            "demand": demand,
            "match_percent": match_pct,
            "description": desc,
            "raw_confidence": round(float(raw_prob) * 100, 2),
        })

    # 2. Skill Scores
    if real_skill_scores:
        # Grounded in the student's own answers — use directly, no model involved.
        skill_scores = dict(real_skill_scores)
    else:
        # Fallback only: no self-assessment was supplied, so estimate from the
        # regression model. This is a rougher, less trustworthy signal than
        # real_skill_scores and should be avoided when real answers exist.
        skill_preds = reg.predict(X_trans)[0]
        skill_names = [
            "Technical Skills",
            "Communication",
            "Problem Solving",
            "Creativity",
            "Leadership"
        ]
        skill_scores = {}
        for name, val in zip(skill_names, skill_preds):
            score = int(round(val))
            skill_scores[name] = max(45, min(98, score))

    # Dynamic AI summary derived from model insights
    top_career = ranked_careers[0]["title"]
    top_pct = ranked_careers[0]["match_percent"]
    student_name = profile.get("name", "Student") or "Student"
    subject = profile.get("subject", "their academic field")

    summary = (
        f"{student_name} demonstrates strong alignment with {top_career} "
        f"(confidence match: {top_pct}%) based on their background in {subject} and practical skills. "
        f"The AI model predicts peak capability in {max(skill_scores, key=skill_scores.get)}."
    )

    motivational_tip = (
        f"Consistently practice your core strengths in {subject}. "
        f"With your current trajectory toward {top_career}, focus on project-based learning and portfolio development."
    )

    return {
        "summary": summary,
        "top_careers": ranked_careers,
        "skill_scores": skill_scores,
        "motivational_tip": motivational_tip,
        "ai_model_name": "CareerCompass Random Forest Ensemble v1.0",
        "ai_accuracy": get_model_metadata().get("accuracy", 95.0),
    }

def add_training_sample(sample):
    """Appends a new training sample to career_dataset.csv and returns row count."""
    row = {
        "age": int(sample.get("age", 20)),
        "semester": sample.get("semester", "1st Semester"),
        "favorite_subject": sample.get("subject", "Computer Science"),
        "programming_skill": sample.get("skill", "None"),
        "career_interest": sample.get("interest", "Software Developer"),
        "skills_text": sample.get("user_skills", ""),
        "target_career": sample.get("target_career", sample.get("interest", "Software Developer")),
        "career_demand": sample.get("career_demand", "High Demand"),
        "technical_score": int(sample.get("technical_score", 75)),
        "communication_score": int(sample.get("communication_score", 75)),
        "problem_solving_score": int(sample.get("problem_solving_score", 75)),
        "creativity_score": int(sample.get("creativity_score", 75)),
        "leadership_score": int(sample.get("leadership_score", 75)),
    }
    df = pd.DataFrame([row])
    df.to_csv(DATA_PATH, mode="a", header=not os.path.exists(DATA_PATH), index=False)
    
    total = len(pd.read_csv(DATA_PATH))
    return total