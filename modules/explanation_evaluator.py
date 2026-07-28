# modules/explanation_evaluator.py
import json
import re
import time
import logging
import pandas as pd
from openai import OpenAI, APIError
from typing import Dict, Any, List, Optional

class LLMJudge:
    """A class to encapsulate the LLM-as-a-Judge functionality."""

    def __init__(self, client: OpenAI, model: str, prompt_template: str, timeout: int):
        self.client = client
        self.model = model
        self.prompt_template = prompt_template
        self.timeout = timeout

    def _parse_json_response(self, content: Any) -> Dict[str, Any]:
        """Parse a JSON payload from the model while tolerating code fences and extra text."""
        if isinstance(content, dict):
            return content
        if not isinstance(content, str):
            raise ValueError("Expected a string response from the LLM judge")

        text = content.strip()
        if not text:
            raise ValueError("Received an empty response from the LLM judge")

        fence_pattern = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.IGNORECASE | re.DOTALL)
        match = fence_pattern.match(text)
        if match:
            text = match.group(1).strip()

        if text.startswith("{") and text.endswith("}"):
            return json.loads(text)

        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start:end + 1])

        raise ValueError(f"Could not locate a JSON object in response: {text}")

    def evaluate(self, llm_output_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Sends a single evaluation request to the OpenAI API."""
        try:
            llm_output_json = json.dumps(llm_output_data, indent=2, ensure_ascii=False)
            prompt = self.prompt_template.replace("{llm_output_json}", llm_output_json)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"},
                timeout=self.timeout
            )
            return self._parse_json_response(response.choices[0].message.content)
        except APIError as e:
            logging.error(f"OpenAI API error: {e}")
        except (json.JSONDecodeError, ValueError) as e:
            logging.error(f"Failed to decode JSON response from LLM Judge: {e}")
        except Exception as e:
            logging.error(f"An unexpected error occurred during LLM evaluation: {e}")
        return None

def evaluate_explanations(df: pd.DataFrame, config):
    """Orchestrates the LLM-as-a-Judge evaluation process."""
    logging.info("Starting explanation quality evaluation (FIDES-Score).")

    try:
        with open(config.PROMPT_FILE_PATH, 'r', encoding='utf-8') as f:
            prompt_template = f.read()
    except FileNotFoundError:
        logging.error(f"Prompt file not found at: {config.PROMPT_FILE_PATH}")
        return

    client = OpenAI(api_key=config.OPENAI_API_KEY)
    judge = LLMJudge(client, config.LLM_JUDGE_MODEs_to_procL, prompt_template, config.API_REQUEST_TIMEOUT_SECONDS)
    
    all_evaluations: List[Dict[str, Any]] = []
    
    rows_to_process = df if config.MAX_ROWS_TO_PROCESS is None else df.head(config.MAX_ROWS_TO_PROCESS)
    logging.info(f"Processing a maximum of {len(rowess)} rows with the LLM Judge.")

    for index, row in rows_to_process.iterrows():
        logging.info(f"Judging row {index + 1}...")
        evaluation = judge.evaluate(row['llm_output_data'])
        
        if evaluation:
            all_evaluations.append(evaluation)
        
        time.sleep(config.API_CALL_DELAY_SECONDS)

    if not all_evaluations:
        logging.warning("No evaluations were successfully completed by the LLM Judge.")
        return

    num_instances = len(all_evaluations)
    avg_fidelity = sum(e.get('fidelity_accuracy', {}).get('score', 0) for e in all_evaluations) / num_instances
    avg_soundness = sum(e.get('justification_soundness', {}).get('score', 0) for e in all_evaluations) / num_instances
    avg_clarity = sum(e.get('clarity_coherence', {}).get('score', 0) for e in all_evaluations) / num_instances
    avg_fides = (avg_fidelity + avg_soundness + avg_clarity) / 3

    print("\n" + "="*50)
    print("⚖️ EXPLANATION QUALITY ANALYSIS (FIDES-SCORE)")
    print("="*50)
    print(f"  Samples Evaluated: {num_instances}")
    print(f"  Avg. Fidelity & Accuracy:      {avg_fidelity:.2f} / 5.0")
    print(f"  Avg. Justification Soundness:  {avg_soundness:.2f} / 5.0")
    print(f"  Avg. Clarity & Coherence:      {avg_clarity:.2f} / 5.0")
    print("-" * 50)
    print(f"  Overall Average FIDES-Score:   {avg_fides:.2f} / 5.0")
