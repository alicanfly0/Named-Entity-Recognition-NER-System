from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer
from .data_loader import get_datasets, tokenize_and_align_labels
from .config import MODEL_SAVE_PATH, MODEL_NAME
from datasets import load_metric

metric = load_metric("seqeval")

def main():
    raw_datasets = get_datasets()
    label_list = raw_datasets["train"].features["ner_tags"].feature.names
    tokenized_datasets = raw_datasets.map(tokenize_and_align_labels, batched=True)

    model = AutoModelForTokenClassification.from_pretrained(MODEL_SAVE_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    trainer = Trainer(model=model, tokenizer=tokenizer)

    predictions, labels, _ = trainer.predict(tokenized_datasets["test"])
    predictions = predictions.argmax(axis=-1)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    print(results)

if __name__ == "__main__":
    main()
