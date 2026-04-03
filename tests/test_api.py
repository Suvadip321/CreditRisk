import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api import app


client = TestClient(app)


def sample_payload() -> dict:
    return {
        "loan_amnt": 15000.0,
        "term": "36 months",
        "installment": 450.00,
        "purpose": "debt_consolidation",
        "issue_d": "Dec-2023",
        "emp_length": "10+ years",
        "home_ownership": "MORTGAGE",
        "annual_inc": 85000.0,
        "verification_status": "Source Verified",
        "zip_code": "940xx",
        "addr_state": "CA",
        "dti": 18.5,
        "revol_bal": 14000.0,
        "revol_util": 55.0,
        "earliest_cr_line": "Jan-2005",
        "fico_range_low": 700.0,
        "fico_range_high": 704.0,
        "inq_last_6mths": 1.0,
        "open_acc": 12.0,
        "total_acc": 24.0,
        "mort_acc": 1.0,
        "delinq_2yrs": 0.0,
        "pub_rec": 0.0,
        "pub_rec_bankruptcies": 0.0,
        "mths_since_last_delinq": 999.0,
    }


def test_health_endpoint_returns_service_status():
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert "status" in body
    assert "model_loaded" in body
    assert "version" in body
    assert body["api"] == "credit_risk_predictor"


def test_predict_risk_happy_path_returns_contract_fields():
    response = client.post("/predict_risk", json=sample_payload())

    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {"risk_class", "probability_of_default", "decision"}
    assert body["risk_class"] in [0, 1]
    assert 0.0 <= body["probability_of_default"] <= 1.0
    assert body["decision"] in ["APPROVE", "REJECT"]


def test_predict_risk_missing_required_field_returns_422():
    payload = sample_payload()
    payload.pop("loan_amnt")

    response = client.post("/predict_risk", json=payload)
    assert response.status_code == 422
