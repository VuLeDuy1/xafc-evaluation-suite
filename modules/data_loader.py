# modules/data_loader.py
import pandas as pd
import json
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _parse_json_string(json_string: str) -> Optional[dict]:
    """Safely parses a JSON string into a dictionary, returning None on failure."""
    if isinstance(json_string, dict):
        return json_string
    try:
        return json.loads(json_string)
    except (json.JSONDecodeError, TypeError):
        return None


def _normalize_data_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize legacy and current payload formats into a DataFrame with parsed data columns."""
    if 'llm_output_data' not in df.columns:
        if 'LLM_output' in df.columns:
            df['llm_output_data'] = df['LLM_output'].apply(_parse_json_string)
        else:
            df['llm_output_data'] = None

    if 'compliance_data' not in df.columns:
        if 'compliance' in df.columns:
            df['compliance_data'] = df['compliance'].apply(_parse_json_string)
        else:
            df['compliance_data'] = None

    df['llm_output_data'] = df['llm_output_data'].apply(
        lambda value: value if isinstance(value, dict) else _parse_json_string(value)
    )

    df['compliance_data'] = df['compliance_data'].apply(
        lambda value: value if isinstance(value, dict) else _parse_json_string(value)
    )

    return df


def load_and_prepare_data_from_csv(file_path: str) -> Optional[pd.DataFrame]:
    """
    Loads data from a CSV file, parses JSON columns, and cleans invalid rows.
    """
    logging.info(f"Attempting to load data from: {file_path}")
    try:
        df = pd.read_csv(file_path)
        df = _normalize_data_frame(df)

        initial_rows = len(df)
        df.dropna(subset=['llm_output_data'], inplace=True)
        cleaned_rows = len(df)

        if initial_rows > cleaned_rows:
            logging.warning(f"Dropped {initial_rows - cleaned_rows} rows because they lacked valid llm output data.")

        logging.info(f"Successfully loaded and prepared {cleaned_rows} rows.")
        return df

    except FileNotFoundError:
        logging.error(f"Data file not found at the specified path: {file_path}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred during data loading: {e}")
        return None


def load_and_prepare_data_from_json(file_path: str) -> Optional[pd.DataFrame]:
    """
    Loads data from a JSON file, parses JSON columns, and cleans invalid rows.
    """
    logging.info(f"Attempting to load data from: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as handle:
            payload = json.load(handle)

        if isinstance(payload, list):
            df = pd.DataFrame(payload)
        elif isinstance(payload, dict):
            if isinstance(payload.get('results'), list):
                df = pd.DataFrame(payload['results'])
            elif isinstance(payload.get('data'), list):
                df = pd.DataFrame(payload['data'])
            else:
                df = pd.DataFrame([payload])
        else:
            raise ValueError("Unsupported JSON payload type")

        df = _normalize_data_frame(df)

        initial_rows = len(df)
        df.dropna(subset=['llm_output_data'], inplace=True)
        cleaned_rows = len(df)

        if initial_rows > cleaned_rows:
            logging.warning(f"Dropped {initial_rows - cleaned_rows} rows because they lacked valid llm output data.")

        logging.info(f"Successfully loaded and prepared {cleaned_rows} rows.")
        return df

    except FileNotFoundError:
        logging.error(f"Data file not found at the specified path: {file_path}")
        return None
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON in data file: {e}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred during data loading: {e}")
        return None