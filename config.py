import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_NAME = "bert-base-cased"
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models", "saved_model")
DATASET_NAME = "conll2003"

# Training Hyperparameters
EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 5e-5
MAX_LEN = 128
