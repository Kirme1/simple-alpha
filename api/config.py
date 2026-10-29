import os

from dotenv import load_dotenv

load_dotenv()


def _required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is not set. Add it to .env (see DEVELOPMENT.md).")
    return value

def _optional(name: str) -> str:
    return os.environ.get(name, "").strip()



CHECKPOINT_PATH = _required("CHECKPOINT_PATH")
FRED_API_KEY = _optional("FRED_API_KEY")