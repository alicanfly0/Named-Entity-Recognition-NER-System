from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
from .config import MODEL_SAVE_PATH, MODEL_NAME

def predict(text):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_SAVE_PATH)

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    return [(token, pred.item()) for token, pred in zip(tokens, predictions[0])]

if __name__ == "__main__":
    text = "Barack Obama was born in Hawaii."
    print(predict(text))
