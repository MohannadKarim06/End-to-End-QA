import requests
import os

API_URL = os.getenv("API_URL")
API_TOKEN = os.getenv("API_TOKEN")

if not API_URL or not API_TOKEN:
    raise EnvironmentError("Missing API_URL or API_TOKEN environment variables.")

headers = {
    "Authorization": f"Bearer {API_TOKEN}"
}

def get_answer(question: str, context: str):
    payload = {
        "inputs": {
            "question": question,
            "context": context
        }
    }

    response = requests.post(API_URL, headers=headers, json=payload)

    if response.status_code == 200:
        data = response.json()
        answer = data.get("answer", "")
        score = data.get("score", 0.0)

        return {
            "answer": answer.strip(),
            "confidence": score
        }

    return {
        "answer": "API error or quota exceeded.",
        "confidence": 0.0
    }


