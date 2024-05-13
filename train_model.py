import pandas as pd
import torch
from transformers import BertModel, BertTokenizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# list of models

AI-Sweden-Models/roberta-large-1160k
KB/bert-base-swedish-cased
KBLab/megatron-bert-large-swedish-cased-165k	
AI-Sweden-Models/bert-large-nordic-pile-1M-steps
ltg/norbert3-large		

# Load the BERT model and tokenizer
model_name = 'KB/bert-base-swedish-cased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# Load the CSV file
data_path = 'patient_data.csv'
data = pd.read_csv(data_path)

# Extract texts and labels
texts = data['cookie_text'].tolist()
labels = data['ECAS_TOTAL'].tolist()

# Tokenization and embedding extraction
def embed_texts(texts):
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    # Use the embeddings of the [CLS] token
    cls_embeddings = outputs.last_hidden_state[:, 0, :].numpy()
    return cls_embeddings

embeddings = embed_texts(texts)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42)

# Train the SVM classifier
classifier = SVC(kernel='linear')
classifier.fit(X_train, y_train)

# Make predictions
y_pred = classifier.predict(X_test)

# Evaluate the classifier
print(classification_report(y_test, y_pred))