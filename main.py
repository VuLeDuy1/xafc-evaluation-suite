# main.py
import sys
import logging
import config
from modules.data_loader import load_and_prepare_data_from_csv,load_and_prepare_data_from_json
from modules.classification_analyzer import analyze_classification_performance
from modules.explanation_evaluator import evaluate_explanations

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """Main entry point for the XAFC evaluation suite."""
    logging.info("🚀 LAUNCHING XAFC EVALUATION SUITE 🚀")

    # --- 1. Load and Prepare Data ---
    df = load_and_prepare_data_from_json(config.JSON_FILE_PATH)
    if df is None or df.empty:
        logging.error("Data loading failed or returned an empty DataFrame. Terminating.")
        sys.exit(1)

    # # --- 2. Classification Performance Analysis ---
    # analyze_classification_performance(df, config)

    # --- 3. Explanation Quality Analysis (LLM-as-a-Judge) ---
    if not config.OPENAI_API_KEY:
        logging.warning("OPENAI_API_KEY is not configured. Skipping FIDES-Score evaluation.")
    else:
        evaluate_explanations(df, config)
    
    logging.info("🎉 EVALUATION SUITE FINISHED SUCCESSFULLY 🎉")

if __name__ == "__main__":
    main()
