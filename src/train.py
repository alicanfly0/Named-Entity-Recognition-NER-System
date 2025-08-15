from transformers import TrainingArguments, Trainer, AutoTokenizer
from datasets import load_metric
from .data_loader import get_datasets, tokenize_and_align_labels
from .model import get_model
from .config import MODEL_SAVE_PATH, EPOCHS, BATCH_SIZE, LEARNING_RATE, MODEL_NAME

metric = load_metric("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = predictions.argmax(axis=-1)

    true_predictions = [
        [p for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [l for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

def main():
    print("📥 Loading dataset...")
    raw_datasets = get_datasets()
    label_list = raw_datasets["train"].features["ner_tags"].feature.names

    print("🔄 Tokenizing dataset...")
    tokenized_datasets = raw_datasets.map(tokenize_and_align_labels, batched=True)

    print("📦 Loading model...")
    model = get_model(len(label_list))

    training_args = TrainingArguments(
        output_dir=MODEL_SAVE_PATH,
        evaluation_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        save_strategy="epoch",
        logging_dir="./logs",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=AutoTokenizer.from_pretrained(MODEL_NAME),
        compute_metrics=compute_metrics
    )

    print("🚀 Starting training...")
    trainer.train()
    print("💾 Saving model...")
    model.save_pretrained(MODEL_SAVE_PATH)

if __name__ == "__main__":
    main()
