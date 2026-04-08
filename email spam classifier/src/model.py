import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns
import os

# Download stopwords
nltk.download('stopwords', quiet=True)

# Load stopwords
stop_words = set(stopwords.words('english'))

# -------------------------------
# 1. TEXT PREPROCESSING
# -------------------------------
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  # remove special chars
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

# -------------------------------
# 2. LOAD DATA
# -------------------------------
def load_data(path):
    df = pd.read_csv(path, encoding='latin-1')[['v1', 'v2']]
    df.columns = ['label', 'message']
    df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
    df['clean'] = df['message'].apply(preprocess)
    return df

# -------------------------------
# 3. TRAIN MODEL
# -------------------------------
def train_model(df):
    vectorizer = TfidfVectorizer(max_features=3000)
    X = vectorizer.fit_transform(df['clean'])
    y = df['label_num']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearSVC()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"\nAccuracy: {acc*100:.2f}%")
    print("\nClassification Report:\n")
    print(classification_report(y_test, preds))

    # Save confusion matrix
    os.makedirs("outputs", exist_ok=True)
    cm = confusion_matrix(y_test, preds)

    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Ham', 'Spam'],
                yticklabels=['Ham', 'Spam'])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix.png")
    plt.close()

    return model, vectorizer

# -------------------------------
# 4. PREDICTION FUNCTION
# -------------------------------
def predict(text, model, vectorizer):
    clean_text = preprocess(text)
    vector = vectorizer.transform([clean_text])
    pred = model.predict(vector)[0]

    return "🚨 SPAM" if pred == 1 else "✅ HAM"