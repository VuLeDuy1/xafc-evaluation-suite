import os
import pandas as pd
from modules.data_loader import load_and_prepare_data_from_json


def test_load_and_prepare_data_from_json_handles_nested_payload(tmp_path):
    data_path = tmp_path / "sample.json"
    payload = [
        {
            "llm_output_data": {
                "claim_text": "Example claim",
                "surrounding_context": "Example context",
                "predicted_classification": "SUPPORTED",
                "reasoning": "Example reasoning",
                "evidences": ["a"],
            }
        }
    ]
    data_path.write_text(pd.Series(payload).to_json(orient="records"), encoding="utf-8")

    df = load_and_prepare_data_from_json(str(data_path))

    assert df is not None
    assert not df.empty
    assert "compliance_data" in df.columns
    assert "llm_output_data" in df.columns
    assert df.iloc[0]["llm_output_data"]["claim_text"] == "Example claim"
