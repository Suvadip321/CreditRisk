from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    # base paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    
    # data paths
    DATA_PATH: str = "data/lending_club_subset.csv"
    TEMPLATE_PATH: str = "src/template.csv"
    
    # model paths
    MODEL_DIR: str = "models"
    MODEL_NAME: str = "credit_risk_model.pkl"
    
    # training configs
    TARGET_COL: str = "loan_status"
    DATE_COL: str = "issue_d"
    RANDOM_STATE: int = 42
    TRAIN_TEST_SPLIT_DATE: str = "2018-01-01"
    
    # logging configs
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/app.log"
    
    # API configs
    API_HOST: str = "127.0.0.1"
    API_PORT: int = 8000
    API_VERSION: str = "1.0.0"
    PREDICTION_THRESHOLD: float = 0.40
    CORS_ORIGINS: str = "http://localhost,http://127.0.0.1"
    
    # computed paths
    @property
    def full_data_path(self) -> Path:
        return self.BASE_DIR / self.DATA_PATH
    
    @property
    def full_template_path(self) -> Path:
        return self.BASE_DIR / self.TEMPLATE_PATH
    
    @property
    def full_model_dir(self) -> Path:
        return self.BASE_DIR / self.MODEL_DIR
    
    @property
    def full_model_path(self) -> Path:
        return self.full_model_dir / self.MODEL_NAME
    
    @property
    def full_log_path(self) -> Path:
        return self.BASE_DIR / self.LOG_FILE

    @property
    def cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()

settings = get_settings()
