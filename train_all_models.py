import pandas as pd
import torch
from transformers import BertModel, BertTokenizer, AutoTokenizer, AutoModel
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# List of models to evaluate
models = [
    'AI-Sweden-Models/roberta-large-1160k',
    'KB/bert-base-swedish-cased',
    'KBLab/megatron-bert-large-swedish-cased-165k',
    'AI-Sweden-Models/bert-large-nordic-pile-1M-steps',
    'ltg/norbert3-large'
]

# Load the CSV file
data_path = 'patient_data.csv'
data = pd.read_csv(data_path)

# Group ECAS_TOTAL scores into categories
def categorize_ecas(score):
    if score < 46:
        return 'low'
    elif 46 <= score < 92:
        return 'medium'
    else:
        return 'high'

data['category'] = data['ECAS_TOTAL'].apply(categorize_ecas)

# Extract texts and labels
texts = data['cookie_text'].tolist()
labels = data['category'].tolist()

# Tokenization and embedding extraction
def embed_texts(model_name, texts):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    # Use the embeddings of the [CLS] token
    cls_embeddings = outputs.last_hidden_state[:, 0, :].numpy()
    return cls_embeddings

results = {}

# Evaluate each model
for model_name in models:
    print(f"Evaluating model: {model_name}")
    embeddings = embed_texts(model_name, texts)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42)
    
    # Train the SVM classifier
    classifier = SVC(kernel='linear')
    classifier.fit(X_train, y_train)
    
    # Make predictions
    y_pred = classifier.predict(X_test)
    
    # Evaluate the classifier
    report = classification_report(y_test, y_pred, output_dict=True)
    results[model_name] = report
    print(classification_report(y_test, y_pred))

# Display summary of results
for model_name, report in results.items():
    print(f"Model: {model_name}")
    print(f"Accuracy: {report['accuracy']}")
    print(f"Macro Avg F1-Score: {report['macro avg']['f1-score']}")
    print()
