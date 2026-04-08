# 📧 Email Spam Classifier

A simple ML-based spam classifier using NLP + Naive Bayes / Logistic Regression / SVM.

---

## 📁 File Structure

```
email_spam_classifier/
│
├── data/
│   └── spam.csv          ← Dataset goes here
│
├── outputs/              ← Auto-created; charts saved here
│
├── spam_classifier.py    ← Main script
├── requirements.txt      ← Dependencies
└── README.md
```

---

## 🚀 Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download the Dataset
- Go to: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
- Download `spam.csv` and place it inside the `data/` folder

### 3. Run
```bash
python spam_classifier.py
```

---

## 📊 What it does
1. Loads and cleans the email dataset
2. Converts text → TF-IDF features
3. Trains 3 models: Naive Bayes, Logistic Regression, SVM
4. Prints accuracy + classification report for each
5. Saves 4 charts to `outputs/`:
   - Label distribution
   - Model accuracy comparison
   - Confusion matrix (best model)
   - Top spam keywords
6. Classifies 4 demo emails in real-time

---

## ✅ Expected Accuracy
| Model                | Accuracy |
|---------------------|----------|
| Naive Bayes          | ~97%     |
| Logistic Regression  | ~98%     |
| SVM                  | ~98%     |
