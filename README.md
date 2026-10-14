# CareerCompass

CareerCompass is a Flask web application that helps students explore career paths. Students enter a profile and complete a short skills assessment to receive career recommendations, skill scores, a career roadmap, and learning resources. An in-app advisor chat can answer career-planning questions.

Recommendations are generated locally with a scikit-learn model when its trained artifacts are available. The application also includes a deterministic rule-based fallback. It does not require an external AI service or provider API key.

## Features

- Career recommendations with match scores.
- A 15-question self-assessment across technical, communication, problem-solving, creativity, and leadership skills.
- Career-specific roadmaps and curated learning resources.
- Advisor chat through the application's local guidance logic.
- SQLite database storage by default, with optional MySQL configuration.
- Optional scripts and protected API endpoints for regenerating training data and retraining the model.
- A browser-friendly results page with report actions.

## Requirements

- Python 3.10 or newer.
- `pip`.

## Setup and run

From this directory, create and activate a virtual environment, install dependencies, and start the Flask application.

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

Open <http://127.0.0.1:5000> in a browser. The SQLite database is created automatically when the application initializes.

## Model and training data

The `models/` directory contains the preprocessor, career classifier, skill regressor, and model metadata used for local inference. If the model artifacts are unavailable, the application falls back to its rule-based recommendation engine.

The checked-in dataset in `data/career_dataset.csv` is synthetic and is produced by `data/generate_dataset.py`. Its records are not real student outcomes. The current metadata reports 1,801 samples across 38 career classes, with 96.12% top-1 and 99.17% top-3 accuracy on a held-out split. These metrics measure performance on synthetic data and should not be interpreted as real-world career prediction accuracy.

To regenerate the dataset and retrain the model:

```bash
python data/generate_dataset.py
python train_model.py
```

Retraining replaces the artifacts in `models/`.

## Configuration

Configuration is supplied through environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `APP_ENV` | `development` | Set to `production` to enable secure session cookies and require `SECRET_KEY`. |
| `SECRET_KEY` | Local development fallback | Flask session signing key. Set a unique, high-entropy value for production. |
| `AI_PROVIDER` | Automatic | Set to `ml` to use the trained model or `local` to force the rule-based fallback. |
| `DATABASE_URL` | `sqlite:///careercompass.db` | Full SQLAlchemy database URL. When set, this takes precedence over the MySQL settings. |
| `DB_ENGINE` | `sqlite` | Set to `mysql` to build a MySQL connection URL from the `DB_*` settings. |
| `DB_USER` | `root` | MySQL username. |
| `DB_PASSWORD` | Empty | MySQL password. |
| `DB_HOST` | `localhost` | MySQL server host. |
| `DB_PORT` | `3306` | MySQL server port. |
| `DB_NAME` | `careercompass` | MySQL database name. |
| `ADMIN_API_KEY` | Unset | Enables protected model-training and training-data API endpoints when configured. |
| `HOST` | `127.0.0.1` | Address for the built-in development server. |
| `PORT` | `5000` | Port for the built-in development server. |
| `FLASK_DEBUG` | Disabled | Set to `1` to enable the local debugger outside production. Do not enable in production. |

For production, set `APP_ENV=production` and provide a secure `SECRET_KEY`. Configure `ADMIN_API_KEY` only if the protected training endpoints are needed, and do not expose the built-in Flask development server to the public internet.

## Routes

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/` | Student profile and self-assessment form. |
| `POST` | `/analyze` | Validates the submitted form and prepares the results session. |
| `GET` | `/results` | Displays recommendations and assessment results. |
| `GET` | `/reset` | Clears the current session and returns to the form. |
| `POST` | `/api/analyze` | Generates and stores a career analysis. |
| `POST` | `/api/chat` | Sends a message to the career advisor. |
| `POST` | `/api/train` | Retrains the model; requires the `X-Admin-API-Key` header. |
| `GET` | `/api/model-status` | Returns model availability and metadata. |
| `POST` | `/api/add-training-data` | Adds a training sample; requires the `X-Admin-API-Key` header. |

The training endpoints return an error when `ADMIN_API_KEY` is not configured. When enabled, send its value in the `X-Admin-API-Key` request header.

## Project layout

```text
.
├── app.py
├── ml_model.py
├── train_model.py
├── requirements.txt
├── data/
│   ├── career_dataset.csv
│   └── generate_dataset.py
├── models/
│   ├── career_classifier.joblib
│   ├── model_metadata.json
│   ├── preprocessor.joblib
│   └── skill_regressor.joblib
├── templates/
│   ├── chatbot.html
│   ├── index.html
│   └── results.html
└── instance/  # Local SQLite database files; ignored by Git
```

## Data and privacy

Student profiles and generated analyses are stored in the configured database. Recommendation generation and advisor responses use application code and the local model; student data is not sent to an external AI provider. The frontend loads some fonts and the searchable dropdown library from third-party CDNs, so those browser assets require an internet connection.
