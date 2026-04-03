# Credit Risk Predictor

An end-to-end machine learning project that predicts the probability of loan default and serves the result through a FastAPI backend and a clean web interface.

## Live Demo
- Frontend: `https://ephemeral-puffpuff-c2b819.netlify.app`
- Backend API: `https://creditrisk-api-09ps.onrender.com`

## Overview
- Trained an XGBoost-based credit risk model on Lending Club style loan data.
- Built a FastAPI service for real-time risk prediction.
- Created a simple, professional frontend for applicant input and model results.
- Deployed the backend with Docker on Render and the frontend on Netlify.

## Project Structure
```text
CreditRisk/
  api.py
  Dockerfile
  render.yaml
  frontend/
    index.html
    styles.css
    app.js
  models/
    credit_risk_model.pkl
  src/
    config.py
    logger.py
    predict.py
    preprocess.py
    template.csv
    template.py
    train_model.py
  tests/
    test_api.py
```

## Tech Stack
- Python
- FastAPI
- scikit-learn
- XGBoost
- HTML, CSS, JavaScript
- Docker
- Render
- Netlify

## Local Run
```bash
uv sync
uv run pytest
uv run python api.py
```

Then open `frontend/index.html` in a browser.

## Docker Run
```bash
docker build -t credit-risk-api .
docker run --rm -p 8000:8000 -e CORS_ORIGINS=http://localhost,http://127.0.0.1 credit-risk-api
```

## Model Performance
- ROC-AUC: `0.7289`
- Accuracy: `0.5537`
- Precision: `0.2361`
- Recall: `0.8179`
- F1-score: `0.3664`
- Decision threshold: `0.40`

Confusion matrix:
```text
[[6638, 6527],
 [ 449, 2017]]
```

## Notes
- The large training dataset is kept local and is not committed to GitHub.
- Runtime settings are defined in `src/config.py` and can be overridden with environment variables.
- The frontend automatically uses the deployed backend URL in production.
