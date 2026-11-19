"""Tests must never spend money: a dummy key set here wins over the real one in .env, and no base URL is inherited."""
import os

os.environ["OPENAI_API_KEY"] = "test-dummy-key"
os.environ.pop("OPENAI_BASE_URL", None)
