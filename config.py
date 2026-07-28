# config.py
import os
from dotenv import load_dotenv

# Load environment variables from a .env file for local development
load_dotenv()

# --- Core File & Directory Paths ---
# Resolve paths relative to the project root so the suite finds its data and prompt files.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
CSV_FILE_NAME = "Llama-3.3-70B-Instruct-Turbo-Free (2).csv"
JSON_FILE_NAME = "Qwen2.5-7B-Instruct-Turbo_20260728_142309_output.json"
CSV_FILE_PATH = os.path.join(DATA_DIR, CSV_FILE_NAME)
JSON_FILE_PATH = os.path.join(DATA_DIR, JSON_FILE_NAME)
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
PROMPT_FILE_PATH = os.path.join(PROJECT_ROOT, 'prompts', 'fides_score_judge.txt')

# --- Evaluation Parameters ---
# Set to a small integer for testing, or None to process all rows.
# WARNING: Processing all rows can be time-consuming and costly.
MAX_ROWS_TO_PROCESS: int | None = 5
CONFIDENCE_THRESHOLD: float = 6.0  # Scores < 6.0 are considered "non-compliant" (label 1)

# --- LLM-as-a-Judge Configuration ---
OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
LLM_JUDGE_MODEL: str = "gpt-4o"
API_REQUEST_TIMEOUT_SECONDS: int = 60
API_CALL_DELAY_SECONDS: float = 1.0  # Delay between API calls to respect rate limits
