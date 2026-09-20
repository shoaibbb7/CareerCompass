"""
CareerCompass AI Model Training Pipeline
Trains a multi-modal Scikit-Learn Machine Learning pipeline:
- Career Recommendation Engine (Classification with Probability Calibration)
- Multi-dimensional Skill Assessment Engine (Multi-Output Regression)
Exports serialized artifacts and training metadata.
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import accuracy_score, top_k_accuracy_score, mean_absolute_error, classification_report

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "career_dataset.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

SKILL_TARGET_COLS = [
    "technical_score",
    "communication_score",
    "problem_solving_score",
    "creativity_score",
    "leadership_score"
]

def build_preprocessor():
    """Builds the feature extraction pipeline combining text, categorical, and numerical features."""
    categorical_features = ["favorite_subject", "programming_skill", "semester", "career_interest"]
    numeric_features = ["age"]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("skills_tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=350, stop_words="english"), "skills_text"),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
            ("num", StandardScaler(), numeric_features),
        ],
        remainder="drop"
    )
    return preprocessor

def train_and_evaluate(dataset_path=DATA_PATH, save_artifacts=True):
    """Executes the full training and evaluation workflow."""
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Training dataset not found at: {dataset_path}")

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading dataset from {dataset_path}...")
    df = pd.read_csv(dataset_path)
    
    # Fill missing values if any
    df["skills_text"] = df["skills_text"].fillna("")
    df["career_interest"] = df["career_interest"].fillna("General")
    df["favorite_subject"] = df["favorite_subject"].fillna("General")
    df["programming_skill"] = df["programming_skill"].fillna("None")
    df["semester"] = df["semester"].fillna("1st Semester")
    df["age"] = pd.to_numeric(df["age"], errors="coerce").fillna(20)

    features = ["skills_text", "favorite_subject", "programming_skill", "semester", "career_interest", "age"]
    X = df[features]
    y_career = df["target_career"]
    y_skills = df[SKILL_TARGET_COLS]

    # Train / Test split
    X_train, X_test, y_c_train, y_c_test, y_s_train, y_s_test = train_test_split(
        X, y_career, y_skills, test_size=0.20, random_state=42, stratify=y_career
    )

    print(f"Training set: {len(X_train)} samples | Test set: {len(X_test)} samples")
    print(f"Unique career classes: {y_career.nunique()}")

    # ── 1. Fit Preprocessor ────────────────────────────────────────────────────
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Fitting feature preprocessor...")
    preprocessor = build_preprocessor()
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    # Extract feature names
    feature_names = []
    try:
        feature_names = preprocessor.get_feature_names_out().tolist()
    except Exception:
        feature_names = [f"f_{i}" for i in range(X_train_trans.shape[1])]

    # ── 2. Train Career Classifier ─────────────────────────────────────────────
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Training Random Forest Career Classifier...")
    clf = RandomForestClassifier(
        n_estimators=140,
        max_depth=25,
        min_samples_split=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train_trans, y_c_train)

    # Career predictions
    y_c_pred = clf.predict(X_test_trans)
    y_c_probs = clf.predict_proba(X_test_trans)

    accuracy = float(accuracy_score(y_c_test, y_c_pred))
    
    # Top-3 Accuracy
    try:
        top3_acc = float(top_k_accuracy_score(y_c_test, y_c_probs, k=3, labels=clf.classes_))
    except Exception:
        top3_acc = accuracy

    print(f"Career Classifier Accuracy: {accuracy * 100:.2f}% | Top-3 Accuracy: {top3_acc * 100:.2f}%")

    # ── 3. Train Skill Assessor (Multi-Output Regressor) ───────────────────────
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Training Multi-Output Skill Regressor...")
    reg = MultiOutputRegressor(
        RandomForestRegressor(n_estimators=70, max_depth=16, random_state=42, n_jobs=-1)
    )
    reg.fit(X_train_trans, y_s_train)

    # Skill predictions
    y_s_pred = reg.predict(X_test_trans)
    skill_maes = {}
    for i, col in enumerate(SKILL_TARGET_COLS):
        mae = float(mean_absolute_error(y_s_test.iloc[:, i], y_s_pred[:, i]))
        skill_maes[col] = round(mae, 2)

    print("Skill Regression MAE:", skill_maes)

    # ── 4. Feature Importances ────────────────────────────────────────────────
    importances = clf.feature_importances_
    top_indices = np.argsort(importances)[::-1][:20]
    top_features = []
    for idx in top_indices:
        raw_name = feature_names[idx]
        clean_name = raw_name.replace("skills_tfidf__", "Skill: ").replace("cat__", "").replace("num__", "Metric: ")
        top_features.append({
            "feature": clean_name,
            "importance": round(float(importances[idx]), 4)
        })

    # Summary Report
    meta = {
        "status": "trained",
        "algorithm": "Random Forest Ensemble (Classifier & Multi-Output Regressor)",
        "accuracy": round(accuracy * 100, 2),
        "top3_accuracy": round(top3_acc * 100, 2),
        "num_samples": len(df),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "num_classes": int(y_career.nunique()),
        "classes": clf.classes_.tolist(),
        "skill_metrics_mae": skill_maes,
        "top_features": top_features,
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Save artifacts
    if save_artifacts:
        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(preprocessor, os.path.join(MODELS_DIR, "preprocessor.joblib"))
        joblib.dump(clf, os.path.join(MODELS_DIR, "career_classifier.joblib"))
        joblib.dump(reg, os.path.join(MODELS_DIR, "skill_regressor.joblib"))
        
        with open(os.path.join(MODELS_DIR, "model_metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)
            
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Saved all trained models and metadata to: {MODELS_DIR}")

    return meta

if __name__ == "__main__":
    train_and_evaluate()

