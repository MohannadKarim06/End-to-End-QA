import requests
import os

API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
API_TOKEN = "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"

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
        score = round(data.get("score", 0.0) * 100, 2)

        return {
            "answer": answer.strip(),
            "confidence": score
        }

    return {
        "answer": "API error or quota exceeded.",
        "confidence": 0.0
    }
