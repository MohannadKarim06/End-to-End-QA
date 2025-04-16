import requests
import os

API_URL = "https://api-inference.huggingface.co/models/deepset/roberta-base-squad2"
API_TOKEN = "1"

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


answer = get_answer(question="who invented the reflecting telescope?", context="Title The History of the Telescope Text The telescope is one of the most important inventions in the history of science. It has allowed us to see objects in the universe that are too faint or too far away to be seen with the naked eye. The telescope has also helped us to learn more about the structure and evolution of the universe. The first telescopes were invented in the early 17th century. The earliest known working telescope was built by Hans Lippershey, a Dutch eyeglass maker, in 1608. Lippersheys telescope was a simple refracting telescope, which uses lenses to bend light and focus it into an image. Galileo Galilei, an Italian astronomer, was one of the first people to use a telescope to study the heavens. In 1609, Gato make a number of important discoveries, including the fact that the Moon is not perfectly smooth, that Jupiter has four moons, and that Venus goes through phases like the Moon. In the years that followed, many other astronomers made important discoveries using telescopes. Johannes Kepler, a German astronomer, used a telescope to study the planets and develop his laws of planetary motion. Isaac Newton, an English physicist, invented the reflecting telescope, which uses mirrors to focus light. Reflecting telescopes are now the most common type of telescope used by astronomers. Today, telescopes are used by astronomers all over the world to study the universe. Telescopes are now much more powerful than the early telescopes, and they can be used to see objects that are billions of lightyears away. Telescopes have helped us to learn a great deal about the universe, and they continue to be an essential tool for astronomers")

print(answer)