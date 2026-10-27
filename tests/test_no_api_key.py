"""The default test run must work on a fresh clone: no .env file and no API key."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CODE = """
import os, dotenv
dotenv.load_dotenv = lambda *args, **kwargs: False   # pretend there is no .env file
import app.extractor, app.main, app.rag.embeddings, app.rag.coverage, app.rag.index, app.rag.retriever
print("imported")
"""


def test_every_module_imports_without_an_api_key():
    env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
    result = subprocess.run([sys.executable, "-c", CODE], cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr[-500:]
    assert "imported" in result.stdout
