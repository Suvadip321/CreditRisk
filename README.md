# Credit Risk Predictor

Beginner-friendly end-to-end ML project to predict whether a loan applicant is likely to default.

## Demo
- Live App: add your deployment URL here (Render/Railway/Hugging Face Spaces).
- Video Walkthrough: add your Loom/YouTube link here.

## Deployment (Docker: Render + Netlify)
### 1. Deploy backend API on Render
1. Push this repository to GitHub.
2. In Render, create a new Web Service from the repo.
3. Render can use `render.yaml` automatically (Docker build from `Dockerfile`).
4. In Render environment variables, set:
- `CORS_ORIGINS=https://<your-frontend-domain>`
5. Ensure your model and template files are present in repo paths:
- `models/credit_risk_model.pkl`
- `src/template.csv`

### 2. Deploy frontend on Netlify (or Vercel)
1. Deploy `frontend/` as a static site.
2. In deployed app UI, set API Base URL to your Render URL, for example:
- `https://credit-risk-api.onrender.com`
3. Verify:
- `GET /health` returns `200`
- Prediction form can submit successfully

## What This Project Includes
- `src/train_model.py`: trains an XGBoost pipeline with preprocessing and saves model artifacts.
- `api.py`: FastAPI service with `/health` and `/predict_risk`.
- `frontend/`: simple HTML/CSS/JS form to call the API.
- `src/template.csv`: schema template used to align prediction input columns.
- `src/template.py`: utility to regenerate `src/template.csv` from the dataset.

## Project Structure
```text
CreditRisk/
  api.py
  pyproject.toml
  .env.example
  data/
  models/
  src/
    config.py
    logger.py
    preprocess.py
    train_model.py
    predict.py
    template.py
    template.csv
  frontend/
    index.html
    styles.css
    app.js
```

## Quick Start
1. Install dependencies:
```bash
uv sync
```

2. (Optional) Create `.env`:
```bash
copy .env.example .env
```

3. If model file does not exist, train it:
```bash
uv run train-model
```

4. Start the API:
```bash
uv run python api.py
```

5. Run tests:
```bash
uv run pytest
```

6. Open frontend:
- Open `frontend/index.html` in a browser.
- Keep API base URL as `http://127.0.0.1:8000`.

## Run With Docker (Local)
1. Build image:
```bash
docker build -t credit-risk-api .
```

2. Run container:
```bash
docker run --rm -p 8000:8000 -e CORS_ORIGINS=http://localhost,http://127.0.0.1 credit-risk-api
```

3. Verify:
- Open `http://127.0.0.1:8000/health`

## API Example
```bash
curl -X POST http://127.0.0.1:8000/predict_risk ^
  -H "Content-Type: application/json" ^
  -d "{\"loan_amnt\":15000,\"term\":\"36 months\",\"installment\":450,\"purpose\":\"debt_consolidation\",\"issue_d\":\"Dec-2023\",\"emp_length\":\"10+ years\",\"home_ownership\":\"MORTGAGE\",\"annual_inc\":85000,\"verification_status\":\"Source Verified\",\"zip_code\":\"940xx\",\"addr_state\":\"CA\",\"dti\":18.5,\"revol_bal\":14000,\"revol_util\":55,\"earliest_cr_line\":\"Jan-2005\",\"fico_range_low\":700,\"fico_range_high\":704,\"inq_last_6mths\":1,\"open_acc\":12,\"total_acc\":24,\"mort_acc\":1,\"delinq_2yrs\":0,\"pub_rec\":0,\"pub_rec_bankruptcies\":0,\"mths_since_last_delinq\":999}"
```

## Notes
- Decision threshold is configured in `src/config.py` (`PREDICTION_THRESHOLD`, default `0.40`).
- CORS allowed domains are configured with `CORS_ORIGINS` (comma-separated).
- `logs/` and `__pycache__/` are runtime artifacts and are ignored by git.
- The dataset file is large; keep it local for training, and avoid committing new large files.
- If XGBoost prints a model-version warning while loading, retrain once using `uv run train-model` to regenerate artifacts with your local package versions.

## Model Results
Evaluation snapshot (generated on April 3, 2026 from `models/credit_risk_model.pkl`, test split `issue_d >= 2018-01-01`):

- Test rows: `15,631`
- ROC-AUC: `0.7289`
- Threshold used for decision: `0.40`
- Accuracy: `0.5537`
- Precision: `0.2361`
- Recall: `0.8179`
- F1-score: `0.3664`

Confusion matrix at threshold `0.40` (`[[TN, FP], [FN, TP]]`):
```text
[[6638, 6527],
 [ 449, 2017]]
```

Why `0.40` instead of `0.50`:
- At `0.40`, recall is much higher (`0.8179`) than `0.50` (`0.6484`), so the model catches more risky loans.
- This is a deliberate tradeoff: lower threshold increases false positives but reduces missed defaults.
