# Player Churn Prediction

## Summary
A Streamlit app to predict player churn using machine learning and provide interactive risk analysis.

## Overview
This project lets you upload a player dataset, train churn models, evaluate model performance, and inspect player-level risk through dashboard sections like Data Overview, Model Training, Model Evaluation, Player Risk Analysis, and Decision Tree Explorer.

## Milestone 2 (Agentic Engagement Assistant)
This repo now includes an **Agentic AI Game Engagement Optimization Assistant** that:
- Interprets churn risk predictions for a selected player
- Summarizes observable engagement/progression/monetization signals
- Retrieves relevant retention strategies from a local knowledge base
- Produces a **structured engagement optimization report** with recommendations, references, and ethical disclaimers

The assistant uses an explicit state/audit trail and only promotes recommendations that are supported by player signals or retrieved strategies.

## Tech Stack
- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

## Dependencies
Install dependencies with:

```bash
pip install -r requirements.txt
```

## Setup Steps
1. Clone the repository:
```bash
git clone https://github.com/atharvgit2005/player_churn_prediction.git
cd player_churn_prediction
```

2. (Recommended) Create and activate a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the app:
```bash
streamlit run app.py
```

5. Open the local URL shown in terminal (usually `http://localhost:8501`).

### Using The Assistant
1. Upload a CSV (or use the included sample dataset).
2. Train a model in **Model Training**.
3. Open **Engagement Optimization Assistant** in the sidebar.
4. Select a player row and generate the report.

## Milestone 2

### Run the updated app
1. Create and activate a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```
2. Install the dependencies:
```bash
pip install -r requirements.txt
```
3. Launch the Streamlit app:
```bash
streamlit run app.py
```

### Trigger PDF export
1. Open **Engagement Optimization Assistant** from the sidebar.
2. Select a trained model and a player row.
3. Click **Generate Engagement Optimization Report**.
4. Use the **Download as PDF** button shown below the hero summary.

The selected player needs enough data to build the assistant report. At minimum, the workflow expects a trained model prediction plus player-level gameplay features such as sessions, session duration, progression, genre, or engagement-level fields. If any mandatory report section is empty, PDF export is blocked and the UI lists the missing sections.

### Run the new tests
Run the full test suite with:
```bash
pytest
```

### New dependencies
Install the new packages with:
```bash
pip install reportlab pypdf pytest
```

These are also included in `requirements.txt`.
