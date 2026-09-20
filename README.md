# CareerCompass 🧭
### Student Career Guidance System powered by a locally trained Machine Learning model

CareerCompass is a full-stack Flask web application that helps students explore suitable career paths. A student fills in a short profile (subject, skill level, career interest, skills) and completes a 15-question self-assessment. The app then shows career recommendations with match scores, skill scores, a step-by-step roadmap, and curated learning resources.

Recommendations come from a **scikit-learn model that is trained and run locally**. **No external AI service is used** — there are no calls to OpenAI, Anthropic, Google, Hugging Face or any other AI API, and no API keys are needed.

---

## Features

- **Career recommendations** — a Random Forest classifier ranks careers for the student and returns match percentages.
- **Skill self-assessment** — 15 Likert-scale questions (1–5), 3 per category, scored 0–100 for Technical, Communication, Problem Solving, Creativity and Leadership. These are the skill scores shown to the student.
- **Career roadmap** — a curated, career-specific step-by-step action plan.
- **Learning resources** — curated courses, platforms and communities matched to the student's subject and interest.
- **Rule-based fallback** — if the ML model files are missing, a deterministic rule-based engine produces the recommendations instead (no network access needed).
- **Result actions** — download a readable `.txt` report, copy a summary to the clipboard, or save results locally (all handled in the browser).
- **Database storage** — each student profile and analysis result is saved via SQLAlchemy (SQLite by default, MySQL optional).
- **Training tools** — scripts and JSON endpoints to regenerate the dataset, retrain the model and check model status.

---

## How the "AI" works

| Component | What it is | File |
|---|---|---|
| Career classifier | `RandomForestClassifier` (140 trees, balanced class weights) | `train_model.py`, `ml_model.py` |
| Skill regressor | `MultiOutputRegressor(RandomForestRegressor)` predicting 5 skill scores | `train_model.py`, `ml_model.py` |
| Feature pipeline | `ColumnTransformer`: TF-IDF on skills text, One-Hot on categorical fields, Standard Scaling on age | `train_model.py` |
| Fallback engine | Deterministic rule-based logic (`call_local_api`) | `app.py` |

The engine is chosen automatically by `get_ai_provider()` in `app.py`:

- `ml` — used when the trained model files exist in `models/`.
- `local` — the rule-based fallback, used when they don't.

You can force one with the `AI_PROVIDER` environment variable (`ml` or `local`). Those are the only two options.

### Model performance

Metrics from the current `models/model_metadata.json`:

| Metric | Value |
|---|---|
| Career classes | 38 |
| Dataset size | 1,801 samples (1,440 train / 361 test) |
| Top-1 accuracy | 96.12% |
| Top-3 accuracy | 99.17% |
| Skill regressor MAE | Technical 3.48 · Communication 4.80 · Problem Solving 3.72 · Creativity 5.14 · Leadership 5.53 |

> **Note:** The training data in `data/career_dataset.csv` is **synthetically generated** by `data/generate_dataset.py`, not collected from real students. The accuracy figures therefore show how well the model learned the generated patterns, not how accurate it would be on real students' career outcomes. Treat the output as guidance, not a prediction.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3, Flask |
| Machine learning | scikit-learn, pandas, NumPy, joblib |
| Database / ORM | SQLite (default) or MySQL, via Flask-SQLAlchemy + PyMySQL |
| Frontend | HTML, CSS, vanilla JavaScript |
| Frontend libraries (CDN) | [Tom Select](https://tom-select.js.org/) (searchable dropdown), Google Fonts |

The browser loads Google Fonts, Tom Select (jsDelivr) and an Unsplash background image. None of these are AI services.

---

## Project Structure

```
Career compass1/
├── app.py                       # Flask app: routes, DB models, roadmaps, career/resource libraries, fallback engine
├── ml_model.py                  # Model loading, inference, retraining and adding training samples
├── train_model.py               # Training pipeline (run from the command line)
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── data/
│   ├── career_dataset.csv       # Training data (1,801 synthetic student records)
│   └── generate_dataset.py      # Generates the synthetic dataset (38 career archetypes)
├── models/                      # Trained artifacts
│   ├── career_classifier.joblib
│   ├── skill_regressor.joblib
│   ├── preprocessor.joblib
│   └── model_metadata.json      # Accuracy, MAE, class list, training timestamp
├── instance/
│   └── careercompass.db         # SQLite database (created automatically)
└── templates/
    ├── index.html               # Profile form + self-assessment
    └── results.html             # Career results, skills, roadmap, resources
```

---

## Installation & Setup

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **(Optional) Regenerate the dataset and retrain the model**
   The repository already includes a trained model. Only do this if you change the data or want a fresh model.
   ```bash
   python data/generate_dataset.py
   python train_model.py
   ```

3. **Run the app**
   ```bash
   python app.py
   ```

4. **Open it in your browser:** <http://127.0.0.1:5000>

---

## Configuration (environment variables)

All settings are optional; the defaults work for local development.

| Variable | Default | Purpose |
|---|---|---|
| `SECRET_KEY` | built-in dev key | Flask session secret. **Set your own value in production.** |
| `DB_ENGINE` | `sqlite` | Set to `mysql` to use MySQL. |
| `DATABASE_URL` | `sqlite:///careercompass.db` | Full SQLAlchemy connection URL (overrides the settings below). |
| `DB_USER` / `DB_PASSWORD` / `DB_HOST` / `DB_PORT` / `DB_NAME` | `root` / *(empty)* / `localhost` / `3306` / `careercompass` | MySQL connection details (used when `DB_ENGINE=mysql`). |
| `AI_PROVIDER` | auto | `ml` or `local`. Forces the ML model or the rule-based fallback. |

Example with MySQL:

```bash
export DB_ENGINE=mysql
export DB_USER=root
export DB_PASSWORD=yourpassword
export DB_NAME=careercompass
python app.py
```

---

## Routes & API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Profile form and self-assessment |
| `POST` | `/analyze` | Validates the form, computes skill scores and stores them in the session |
| `GET` | `/results` | Results page (calls `/api/analyze` when it loads) |
| `GET` | `/reset` | Clears the session and returns to the form |
| `POST` | `/api/analyze` | Generates the analysis using the ML model (or the fallback) and saves it to the database |
| `POST` | `/api/train` | Retrains the model and returns the updated metrics |
| `GET` | `/api/model-status` | Returns the current model metadata (accuracy, MAE, timestamp) |
| `POST` | `/api/add-training-data` | Appends a new training sample to `data/career_dataset.csv` |

There is no web page for training. Use `train_model.py` or the JSON endpoints above.

> **Security note:** `/api/train` and `/api/add-training-data` have no authentication. Do not expose them publicly without adding access control.

---

## Database

Two tables are created automatically on first run:

- `students` — name, email, age, semester, subject, skill level, career interest, skills.
- `analyses` — the linked student, engine used (`ai_provider` = `ml` or `local`), model name, prompt text, result JSON, summary and motivational tip.

---

## Privacy

Student data is stored only in your own database. Nothing is sent to any third-party AI provider. The only outbound requests are the browser loading fonts, Tom Select and one image from the CDNs listed above.
