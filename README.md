

# Named Entity Recognition (NER) System

A complete **Named Entity Recognition** (NER) pipeline built using **Hugging Face Transformers** with automatic download of the **CoNLL-2003** dataset. This project fine-tunes a BERT model for recognizing entities such as **persons, organizations, locations,** and **miscellaneous** entities from text.
**Author:** [@alicanfly0](https://github.com/alicanfly0)

## 📌 Features

* **Pretrained Transformer Model**: Fine-tunes `bert-base-cased` for token classification.
* **Automatic Dataset Handling**: Downloads and preprocesses CoNLL-2003 dataset on first run.
* **Evaluation Metrics**: Computes **Precision**, **Recall**, **F1-score**, and **Accuracy** using `seqeval`.
* **Easy Inference**: Predict entities from any custom input text.
* **Clean Modular Code**: Organized for scalability and maintainability.
* **Shell Script Support**: Train and evaluate in one command with `run.sh`.

## 📂 Project Structure

ner\_project/
│
├── src/
│   ├── config.py           # Configurations (paths, hyperparameters, etc.)
│   ├── data\_loader.py      # Dataset download & preprocessing
│   ├── model.py            # Model initialization
│   ├── train.py            # Training script
│   ├── evaluate.py         # Evaluation script
│   ├── inference.py        # Entity prediction on custom text
│
├── requirements.txt        # Python dependencies
├── run.sh                  # One-command training & evaluation
├── README.md               # Project documentation
└── models/                 # Trained model will be saved here

## 🚀 Installation

Clone the repository:
git clone [https://github.com/alicanfly0/ner\_project.git](https://github.com/alicanfly0/ner_project.git)
cd ner\_project

Install dependencies:
pip install -r requirements.txt

(Optional) Install spaCy for visualization:
pip install spacy

## 🏋️ Training the Model

Train and evaluate the model with a single command:
bash run.sh

This will:

1. Download the **CoNLL-2003** dataset automatically.
2. Tokenize and align labels with the BERT tokenizer.
3. Fine-tune `bert-base-cased` for **Named Entity Recognition**.
4. Save the trained model to `models/saved_model/`.

## 📊 Evaluation

The training pipeline will automatically run evaluation after each epoch. You can manually evaluate the trained model:
python -m src.evaluate

Example Output:
{'precision': 0.92, 'recall': 0.91, 'f1': 0.915, 'accuracy': 0.98}

## 🔍 Inference (Prediction on Custom Text)

Run predictions on any text:
python -m src.inference "Barack Obama was born in Hawaii."

Example Output:
\[('Barack', 'PERSON'), ('Obama', 'PERSON'), ('Hawaii', 'LOCATION')]

## 📈 Example Use Cases

* **News Article Analysis**: Extract key names, places, and organizations.
* **Business Intelligence**: Identify companies and stakeholders from reports.
* **Chatbot Enhancement**: Provide contextual awareness for conversations.
* **Information Retrieval**: Tag entities in large datasets for search indexing.

## 🤝 Contributing

1. Fork the repository
2. Create a new feature branch (git checkout -b feature-name)
3. Commit your changes (git commit -m "Add new feature")
4. Push to the branch (git push origin feature-name)
5. Open a Pull Request

## 📜 License

This project is licensed under the **MIT License** — see the LICENSE file for details.

## 💬 Contact

For questions or collaboration, reach out via GitHub: [@alicanfly0](https://github.com/alicanfly0)
