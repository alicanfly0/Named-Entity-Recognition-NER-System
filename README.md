# Named Entity Recognition (NER) System

A complete **Named Entity Recognition** (NER) pipeline built using **Hugging Face Transformers** with automatic download of the **CoNLL-2003** dataset.  
This project fine-tunes a BERT model for recognizing entities such as **persons, organizations, locations,** and **miscellaneous** entities from text.

**Author:** [@alicanfly0](https://github.com/alicanfly0)

---

## 📌 Features
- **Pretrained Transformer Model**: Fine-tunes `bert-base-cased` for token classification.
- **Automatic Dataset Handling**: Downloads and preprocesses CoNLL-2003 dataset on first run.
- **Evaluation Metrics**: Computes **Precision**, **Recall**, **F1-score**, and **Accuracy** using `seqeval`.
- **Easy Inference**: Predict entities from any custom input text.
- **Clean Modular Code**: Organized for scalability and maintainability.
- **Shell Script Support**: Train and evaluate in one command with `run.sh`.

---

## 📂 Project Structure
ner_project/
│
├── src/
│ ├── config.py # Configurations (paths, hyperparameters, etc.)
│ ├── data_loader.py # Dataset download & preprocessing
│ ├── model.py # Model initialization
│ ├── train.py # Training script
│ ├── evaluate.py # Evaluation script
│ ├── inference.py # Entity prediction on custom text
│
├── requirements.txt # Python dependencies
├── run.sh # One-command training & evaluation
├── README.md # Project documentation
└── models/ # Trained model will be saved here


---

## 🚀 Installation
1. Clone the repository:
```bash
git clone https://github.com/alicanfly0/ner_project.git
cd ner_project
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
(Optional) Install spaCy for visualization:

bash
Copy
Edit
pip install spacy
🏋️ Training the Model
Train and evaluate the model with a single command:

bash
Copy
Edit
bash run.sh
This will:

Download the CoNLL-2003 dataset automatically.

Tokenize and align labels with the BERT tokenizer.

Fine-tune bert-base-cased for Named Entity Recognition.

Save the trained model to models/saved_model/.

📊 Evaluation
The training pipeline will automatically run evaluation after each epoch.
You can manually evaluate the trained model:

bash
Copy
Edit
python -m src.evaluate
Example Output:

bash
Copy
Edit
{'precision': 0.92, 'recall': 0.91, 'f1': 0.915, 'accuracy': 0.98}
🔍 Inference (Prediction on Custom Text)
Run predictions on any text:

bash
Copy
Edit
python -m src.inference "Barack Obama was born in Hawaii."
Example Output:

css
Copy
Edit
[('Barack', 'PERSON'), ('Obama', 'PERSON'), ('Hawaii', 'LOCATION')]
📈 Example Use Cases
News Article Analysis: Extract key names, places, and organizations.

Business Intelligence: Identify companies and stakeholders from reports.

Chatbot Enhancement: Provide contextual awareness for conversations.

Information Retrieval: Tag entities in large datasets for search indexing.

🤝 Contributing
Fork the repository

Create a new feature branch (git checkout -b feature-name)

Commit your changes (git commit -m "Add new feature")

Push to the branch (git push origin feature-name)

Open a Pull Request

📜 License
This project is licensed under the MIT License — see the LICENSE file for details.

💬 Contact
For questions or collaboration, reach out via GitHub: @alicanfly0









Ask ChatGPT
