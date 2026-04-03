import numpy as np
import pandas as pd

from src.config import settings
from src.logger import setup_logger

logger = setup_logger("data sampling", settings.LOG_LEVEL, settings.full_log_path)

def main():
    """Create a blank template CSV that matches the training data columns."""
    df = pd.read_csv(settings.full_data_path, nrows=1)
    for col in df.columns:
        df[col] = np.nan
    df.to_csv(settings.full_template_path, index=False)
    logger.info(f"Data template created at {settings.full_template_path}")

if __name__ == "__main__":
    main()
