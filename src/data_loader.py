from datasets import load_dataset
from transformers import AutoTokenizer
from .config import MODEL_NAME, DATASET_NAME, MAX_LEN

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def get_datasets():
    """
    Automatically downloads and loads CoNLL-2003 dataset.
    """
    dataset = load_dataset(DATASET_NAME)
    return dataset

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        padding="max_length",
        max_length=MAX_LEN
    )

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx])
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs
