"""llm_automation_evals: a small eval harness for LLM-powered automations.

A task is a prompt plus an output schema; a suite is that task and a list of
golden cases. The runner collects raw model outputs from any OpenAI-compatible
endpoint, the scorers grade them offline, and the report is what you attach
to a pull request.
"""

__version__ = "0.1.0"
