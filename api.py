from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
import uvicorn

from src.config import settings
from src.logger import setup_logger
from src.predict import predict, is_model_loaded

logger = setup_logger("api", settings.LOG_LEVEL, settings.full_log_path)

app = FastAPI(
    title="Credit Risk Predictor",
    version=settings.API_VERSION,
    description="API for predicting credit default risk"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Applicant(BaseModel):
    loan_amnt: float
    term: str
    installment: float              
    purpose: str
    issue_d: str             
    emp_length: str
    home_ownership: str
    annual_inc: float
    verification_status: str
    zip_code: str      
    addr_state: str      
    dti: float
    revol_bal: float
    revol_util: float
    earliest_cr_line: str
    fico_range_low: float
    fico_range_high: float
    inq_last_6mths: float
    open_acc: float
    total_acc: float
    mort_acc: float
    delinq_2yrs: float
    pub_rec: float
    pub_rec_bankruptcies: float
    mths_since_last_delinq: Optional[float] = None


class PredictionResponse(BaseModel):
    risk_class: int
    probability_of_default: float
    decision: str


@app.get("/health")
def health_check():
    """Health check endpoint for orchestrators and load balancers."""
    model_status = is_model_loaded()
    
    health_response = {
        "status": "healthy" if model_status else "degraded",
        "model_loaded": model_status,
        "version": settings.API_VERSION,
        "api": "credit_risk_predictor"
    }
    
    logger.debug(f"Health check: {health_response}")
    return health_response


@app.get("/")
def read_root():
    return RedirectResponse(url="/docs")


@app.post("/predict_risk", response_model=PredictionResponse)
def get_prediction(applicant: Applicant):
    """Predict credit default risk for a loan applicant."""
    try:
        data_dict = applicant.model_dump()
        logger.info(f"Prediction request received for loan_amnt={data_dict.get('loan_amnt')}")
        
        prediction_class, probability = predict(data_dict)
        
        result = {
            "risk_class": int(prediction_class),
            "probability_of_default": float(probability),
            "decision": "REJECT" if prediction_class == 1 else "APPROVE"
        }
        
        logger.info(f"Prediction result: {result['decision']} (prob={probability:.4f})")
        return result
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Prediction failed due to an internal server error.")


if __name__ == "__main__":
    logger.info(f"Starting API server on {settings.API_HOST}:{settings.API_PORT}")
    uvicorn.run(app, host=settings.API_HOST, port=settings.API_PORT)
