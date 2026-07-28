from types import SimpleNamespace

from modules.explanation_evaluator import LLMJudge


def test_evaluate_handles_literal_braces_in_prompt_template():
    captured = {}

    class DummyClient:
        def __init__(self):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        def _create(self, **kwargs):
            captured["messages"] = kwargs["messages"]
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content='{"fidelity_accuracy": {"score": 4.0}}')
                    )
                ]
            )

    prompt_template = """You must return JSON.

Example output:
{
  \"fidelity_accuracy\": {
    \"score\": <float>
  }
}

Input: {llm_output_json}"""

    judge = LLMJudge(DummyClient(), "gpt-test", prompt_template, 10)
    result = judge.evaluate({"claim_text": "Example"})

    assert result == {"fidelity_accuracy": {"score": 4.0}}
    assert "Input:" in captured["messages"][0]["content"]
    assert "{llm_output_json}" not in captured["messages"][0]["content"]
