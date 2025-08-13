import pandas as pd
import re
import nltk
import os
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib  # For saving and loading model and vectorizer

# Download stopwords if not already done
nltk.download('stopwords')

# Paths to saved files
MODEL_FILENAME = 'fake_news_model.joblib'
VECTORIZER_FILENAME = 'tfidf_vectorizer.joblib'
DATA_COMBINED_FILENAME = 'fake_news_combined.csv'

# -------------------------
# STEP 1: Load or create combined dataset
# -------------------------
if not os.path.exists(DATA_COMBINED_FILENAME):
    # Load individual files and combine
    fake = pd.read_csv('Fake.csv')
    real = pd.read_csv('True.csv')

    fake['label'] = 1
    real['label'] = 0

    data = pd.concat([fake, real], ignore_index=True)
    data = data[['text', 'label']]
    data.to_csv(DATA_COMBINED_FILENAME, index=False)
else:
    data = pd.read_csv(DATA_COMBINED_FILENAME)

print("Dataset preview:")
print(data.head())

# -------------------------
# STEP 2: Preprocess text
# -------------------------
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

data['cleaned_text'] = data['text'].apply(preprocess_text)

print("\nPreprocessed text preview:")
print(data['cleaned_text'].head())

# -------------------------
# STEP 3: Train-Test Split (Stratified)
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    data['cleaned_text'],
    data['label'],
    test_size=0.2,
    random_state=42,
    stratify=data['label']
)

# -------------------------
# STEP 4: Load or train TF-IDF vectorizer and model
# -------------------------
if os.path.exists(MODEL_FILENAME) and os.path.exists(VECTORIZER_FILENAME):
    print("\nLoading saved model and vectorizer...")
    model = joblib.load(MODEL_FILENAME)
    tfidf = joblib.load(VECTORIZER_FILENAME)
else:
    print("\nTraining new model and vectorizer...")
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    model = LogisticRegression(max_iter=200)
    model.fit(X_train_tfidf, y_train)

    # Save the trained model and vectorizer
    joblib.dump(model, MODEL_FILENAME)
    joblib.dump(tfidf, VECTORIZER_FILENAME)
    print("Model and vectorizer saved.")

# If vectorizer was loaded, transform test data now
if 'X_test_tfidf' not in locals():
    X_test_tfidf = tfidf.transform(X_test)
if 'X_train_tfidf' not in locals():
    X_train_tfidf = tfidf.transform(X_train)

# -------------------------
# STEP 5: Evaluate model
# -------------------------
y_pred = model.predict(X_test_tfidf)

print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# -------------------------
# STEP 6: Interactive Prediction with Confidence & Threshold
# -------------------------
while True:
    user_input = input("\nEnter a news article text (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        print("Exiting Fake News Detector. Goodbye!")
        break

    cleaned_input = preprocess_text(user_input)
    vectorized_input = tfidf.transform([cleaned_input])

    probs = model.predict_proba(vectorized_input)[0]
    print(f"Confidence - Real: {probs[0]:.3f}, Fake: {probs[1]:.3f}")

    if probs[1] > 0.6:
        print("🔴 Prediction: FAKE news")
    elif probs[0] > 0.6:
        print("🟢 Prediction: REAL news")
    else:
        print("⚠️ Prediction: UNCERTAIN — please provide more context or a longer article.")





