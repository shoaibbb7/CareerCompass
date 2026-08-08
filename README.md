# CareerCompass 🧭
### AI-Powered Student Career Guidance System

CareerCompass is a full-stack Flask web application that helps students discover their ideal career path using AI. Students fill in a profile — subject, skill level, and career goals — and receive personalized career recommendations, a skill assessment, a step-by-step roadmap, curated learning resources, and a full course recommendation engine.

---

## Features

- **AI Career Analysis** — Powered by Anthropic Claude or OpenAI GPT
- **Career Recommendations** — Matches students to 68+ careers by field, demand, and fit percentage
- **Skill Assessment** — Visual breakdown of Technical, Communication, Problem Solving, Creativity, and Leadership scores
- **Personalized Career Roadmap** — 4-step action plan tailored to each student (40+ career-specific roadmaps built in)
- **Course Recommendation Engine** — Detailed topic-by-topic learning paths with resource links for each career
- **Learning Resources** — Curated courses, platforms, and communities per career field
- **MySQL or SQLite Database** — Saves every student profile and analysis result (SQLite works out of the box)
- **Search & Filter** — Browse career recommendations by profession group or keyword
- **Download Report** — Export results as a readable `.txt` file
- **Fallback Mode** — Works fully without any API key using the built-in local career library

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3, Flask |
| Frontend | HTML, CSS, Vanilla JS |
| Database | MySQL 8.0+ or SQLite (auto-detected) |
| ORM | Flask-SQLAlchemy + PyMySQL |
| AI Provider | Anthropic Claude / OpenAI GPT (configurable) |
| Fonts | Google Fonts — DM Serif Display, DM Sans |
| UI Components | Tom Select (searchable dropdowns) |

---

## Project Structure

```
careercompass/
├── app.py                  # Main Flask application (all-in-one)
└── templates/
    ├── index.html          # Home page + career analysis form
    └── results.html        # AI results page with tabs and course panel
```

> **Note:** `app.py` is fully self-contained — career roadmaps, resource library, career library, AI helpers, database models, and all routes are in one file.

---

## Prerequisites

Make sure you have these installed:

- [Python 3.8+](https://www.python.org/downloads/)
- [MySQL 8.0+](https://dev.mysql.com/downloads/mysql/) *(optional — SQLite works by default)*
- [VS Code](https://code.visualstudio.com/) (recommended editor)
- [MySQL Workbench](https://dev.mysql.com/downloads/workbench/) *(optional — for viewing data)*

---

## Installation & Setup

### 1. Open the project in VS Code

Go to **File → Open Folder** and select your project folder.

Open the terminal with `` Ctrl + ` ``

---

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

```bash
# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

You will see `(venv)` in the terminal — this means it is active.

---

### 3. Install dependencies

```bash
pip install flask flask-sqlalchemy pymysql
```

If `pip` doesn't work, try:

```bash
python -m pip install flask flask-sqlalchemy pymysql
```

---

### 4. Configure environment variables *(optional)*

Create a `.env` file in the root of your project, or just set variables directly in the terminal.

**To use SQLite (no setup needed — default):**

No configuration required. The app creates `careercompass.db` automatically on first run.

**To use MySQL:**

```env
DB_ENGINE=mysql
DB_USER=root
DB_PASSWORD=yourpassword
DB_HOST=localhost
DB_PORT=3306
DB_NAME=careercompass
```

**To enable AI analysis:**

```env
# Anthropic Claude (recommended)
ANTHROPIC_API_KEY=your-anthropic-key-here

# OR OpenAI GPT
OPENAI_API_KEY=your-openai-key-here
```

Set variables in your terminal:

```bash
# Windows
set ANTHROPIC_API_KEY=your-key-here
set DB_ENGINE=sqlite

# Mac / Linux
export ANTHROPIC_API_KEY=your-key-here
export DB_ENGINE=sqlite
```

> **No API key?** The app runs in local fallback mode automatically — all career data, roadmaps, and resources still work, just without AI-generated summaries.

---

### 5. Set up MySQL database *(only if using MySQL)*

Run the schema file once:

```bash
mysql -u root -p < careercompass.sql
```

This creates the `careercompass` database, all 9 tables, and seeds 68 careers and 64 resources.

---

### 6. Run the app

```bash
python app.py
```

You will see:

```
* Running on http://127.0.0.1:5000
* Press CTRL+C to quit
```

Open your browser and visit:

```
http://127.0.0.1:5000
```

---

## How to Use

1. Open `http://127.0.0.1:5000`
2. Fill in the **Personal Information** card — name, email, age, semester
3. Fill in the **Academic Information** card — favourite subject, skill level, career interest
4. Enter your **skills** in the text area (e.g. Python, Leadership, Public Speaking)
5. Click **Analyse My Career**
6. Wait for the AI to process your profile (~5–10 seconds)
7. View your results across 5 tabs:
