"""Tests never call a model: whatever .env turns on locally (a System One server, the injection model), the suite runs
deterministic and offline. Environment variables win over .env in Settings, and they are set before it is read."""
import os

os.environ.update({"INJECTION_MODEL": "false", "DECIDER": "llm"})
