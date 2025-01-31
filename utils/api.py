import os
import time
import logging
import requests
from typing import List, Dict
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

BASE_URL = os.getenv("OPENAI_API_URL", "https://openrouter.ai/api/v1/chat/completions")
API_KEY = os.getenv("OPENAI_API_KEY")
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
RETRY_DELAY = int(os.getenv("RETRY_DELAY", "5"))

def send_to_judge_model(messages: List[Dict], judge_model: str, max_retries: int = MAX_RETRIES) -> str:
    """
    Sends user messages to the judge model with basic retry logic.
    Expects an OpenAI-compatible endpoint.
    """
    for attempt in range(1, max_retries + 1):
        try:
            # temp and top_k are set to produce diversity in judge outputs between runs,
            # but constrained to be near the model's best answer (since we are doing numerical scoring).
            data = {
                "model": judge_model,
                "messages": messages,
                "temperature": 0.5,
                "top_k": 3,
                "max_tokens": 8096,
                #"provider": {
                #    "order": [
                #        "DeepSeek"
                #   ]
                #}
            }
            response = requests.post(BASE_URL, headers=HEADERS, json=data)
            response.raise_for_status()
            res_json = response.json()
            return res_json['choices'][0]['message']['content']
        except Exception as e:
            logging.error(f"Error on attempt {attempt} for judge model {judge_model}: {e}")
            if attempt == max_retries:
                logging.critical(f"Max retries reached for judge model {judge_model}")
                raise
            time.sleep(RETRY_DELAY)
    return ""