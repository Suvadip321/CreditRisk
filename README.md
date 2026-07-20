# Credit Risk Predictor

Predicts whether a loan applicant is likely to default. Fill in the loan
details, and the model returns a probability of default along with an
APPROVE or REJECT decision. Built with XGBoost on Lending Club loan data,
served through a FastAPI backend and a clean web interface.

## Live Demo
- Frontend: `https://credit-risk-livid.vercel.app`
- Backend API: `https://creditrisk-api-09ps.onrender.com`

## How it works

The model is an XGBoost classifier inside a sklearn Pipeline. The pipeline
handles all preprocessing - parsing string fields, engineering domain
features like credit age and revolving utilization, and dropping
post-approval columns that would leak the outcome.

Training uses a time-based split at 2018-01-01 to prevent data leakage,
with TimeSeriesSplit cross-validation. Hyperparameters are tuned with
RandomizedSearchCV over 20 iterations, optimizing for AUC.

At inference time, if the predicted probability of default exceeds 0.40,
the decision is REJECT, otherwise APPROVE.

## Tech Stack
- Python, FastAPI, scikit-learn, XGBoost
- HTML, CSS, JavaScript
- Docker, Render (API), Vercel (frontend)

## Running locally

```bash
uv sync
python -m src.train_model   # train the model first
uv run python api.py        # then start the API
```

Then open `frontend/index.html` in a browser. The training dataset is not
committed - put it at `data/lending_club_subset.csv` before training.

## Docker

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
- The training dataset is not committed to the repo.
- Runtime settings are in `src/config.py` and can be overridden with env vars.
- The frontend automatically uses the deployed backend URL in production.
