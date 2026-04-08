import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

# ─── 1. Load Dataset ────────────────────────────────────────────────────────

def load_data(filepath="data/spam.csv"):
    df = pd.read_csv(filepath, encoding='latin-1')[['v1', 'v2']]
    df.columns = ['label', 'message']
    df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
    print(f"Dataset loaded: {len(df)} emails | Spam: {df['label_num'].sum()} | Ham: {(df['label_num']==0).sum()}")
    return df

# ─── 2. Preprocess Text ─────────────────────────────────────────────────────

stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)          # remove non-letters
    words = text.split()
    words = [w for w in words if w not in stop_words]  # remove stopwords
    return ' '.join(words)

# ─── 3. Train & Evaluate Models ─────────────────────────────────────────────

def train_models(X_train, X_test, y_train, y_test):
    models = {
        "Naive Bayes":       MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM":               LinearSVC()
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        results[name] = {"model": model, "accuracy": acc, "preds": preds}
        print(f"\n{'─'*40}")
        print(f"Model: {name}  |  Accuracy: {acc*100:.2f}%")
        print(classification_report(y_test, preds, target_names=["Ham", "Spam"]))

    return results

# ─── 4. Visualizations ──────────────────────────────────────────────────────

def plot_results(df, results, vectorizer):
    os.makedirs("outputs", exist_ok=True)

    # 4a. Label Distribution
    plt.figure(figsize=(5, 4))
    df['label'].value_counts().plot(kind='bar', color=['steelblue', 'tomato'], edgecolor='black')
    plt.title("Email Label Distribution")
    plt.xlabel("Label"); plt.ylabel("Count")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("outputs/label_distribution.png")
    plt.close()
    print("Saved: outputs/label_distribution.png")

    # 4b. Model Accuracy Comparison
    names = list(results.keys())
    accs  = [results[m]['accuracy'] * 100 for m in names]
    plt.figure(figsize=(7, 4))
    bars = plt.barh(names, accs, color=['#4CAF50', '#2196F3', '#FF5722'])
    plt.xlabel("Accuracy (%)")
    plt.title("Model Accuracy Comparison")
    plt.xlim(90, 100)
    for bar, acc in zip(bars, accs):
        plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                 f"{acc:.2f}%", va='center')
    plt.tight_layout()
    plt.savefig("outputs/model_accuracy.png")
    plt.close()
    print("Saved: outputs/model_accuracy.png")

    # 4c. Confusion Matrix for best model
    best_name = max(results, key=lambda m: results[m]['accuracy'])
    best_preds = results[best_name]['preds']
    cm = confusion_matrix(y_test_global, best_preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    plt.title(f"Confusion Matrix — {best_name}")
    plt.ylabel("Actual"); plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix.png")
    plt.close()
    print("Saved: outputs/confusion_matrix.png")

    # 4d. Top Spam Keywords (Naive Bayes)
    nb_model = results["Naive Bayes"]["model"]
    feature_names = vectorizer.get_feature_names_out()
    spam_idx = 1  # class index for spam
    top_indices = np.argsort(nb_model.feature_log_prob_[spam_idx])[-15:]
    top_words = [feature_names[i] for i in top_indices]
    top_scores = nb_model.feature_log_prob_[spam_idx][top_indices]

    plt.figure(figsize=(7, 5))
    plt.barh(top_words, top_scores, color='tomato', edgecolor='black')
    plt.title("Top 15 Spam-Indicating Keywords (Naive Bayes)")
    plt.xlabel("Log Probability")
    plt.tight_layout()
    plt.savefig("outputs/spam_keywords.png")
    plt.close()
    print("Saved: outputs/spam_keywords.png")

# ─── 5. Real-Time Classifier ─────────────────────────────────────────────────

def classify_email(text, model, vectorizer):
    clean = preprocess(text)
    vec   = vectorizer.transform([clean])
    pred  = model.predict(vec)[0]
    return "🚨 SPAM" if pred == 1 else "✅ HAM (Not Spam)"

# ─── Main ────────────────────────────────────────────────────────────────────

y_test_global = None  # used in plot_results

if __name__ == "__main__":
    # 1. Load
    df = load_data("data/spam.csv")

    # 2. Preprocess
    df['clean_message'] = df['message'].apply(preprocess)

    # 3. Vectorize
    vectorizer = TfidfVectorizer(max_features=3000)
    X = vectorizer.fit_transform(df['clean_message'])
    y = df['label_num']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    y_test_global = y_test  # for plot_results

    # 4. Train
    results = train_models(X_train, X_test, y_train, y_test)

    # 5. Visualize
    plot_results(df, results, vectorizer)

    # 6. Real-time classification — user input loop
    best_model_name = max(results, key=lambda m: results[m]['accuracy'])
    model = results[best_model_name]['model']

    print("\n" + "═"*50)
    print(f"  REAL-TIME EMAIL SPAM CLASSIFIER")
    print(f"  Using: {best_model_name} (Best Model)")
    print("  Type 'quit' to exit")
    print("═"*50)

    while True:
        print()
        email = input("Enter email text: ").strip()
        if email.lower() == 'quit':
            print("Exiting classifier. Goodbye!")
            break
        if not email:
            print("Please enter some text.")
            continue
        result = classify_email(email, model, vectorizer)
        print(f"Result : {result}")
