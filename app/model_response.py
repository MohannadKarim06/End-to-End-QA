import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

model_path = "model//roberta-finetuned-squad2"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForQuestionAnswering.from_pretrained(model_path)

def get_answer(question: str, context: str):
    
    inputs = tokenizer.encode_plus(
        question,
        context,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model(**inputs)

    start_scores = outputs.start_logits
    end_scores = outputs.end_logits

    # Get the most probable start and end of the answer
    start_idx = torch.argmax(start_scores)
    end_idx = torch.argmax(end_scores)

    # Confidence score = average of max logits
    confidence = (torch.max(start_scores) + torch.max(end_scores)) / 2
    confidence = torch.nn.functional.softmax(torch.tensor([confidence]), dim=0).item() * 100

    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start_idx:end_idx+1])
    )

    return {
        "answer": answer.strip(),
        "confidence": round(confidence, 2)
    }

question = "What is the policy?"
context = "The policy states that employees are eligible after 6 months."

answer, confidence = get_answer(question=question, context=context)

print(answer)