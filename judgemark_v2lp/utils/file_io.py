import json
from loguru import logger

def load_json_file(file_path: str) -> dict:
    """Loads a JSON file (returns empty if not found)."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"File {file_path} not found, returning empty dict.")
        return {}

def save_json_file(data: dict, file_path: str):
    """Saves a dict to disk as JSON."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    logger.debug(f"Saved JSON data to {file_path}")
