from transformers import AutoModelForTokenClassification
from .config import MODEL_NAME

def get_model(label_count):
    """
    Loads a pre-trained transformer and adapts it for token classification.
    """
    return AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=label_count
    )
