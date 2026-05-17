# My-Repository-
Implement Hyperparameter Tuning using GridSearchCV / RandomizedSearchCV
# baseline_model.py

import pandas as pd
import re
import nltk
import string

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Load dataset
df = pd.read_csv("dataset.csv")

# Initialize tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Text preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)

    words = text.split()

    words = [word for word in words if word not in stop_words]

    words = [lemmatizer.lemmatize(word) for word in words]

    return " ".join(words)

# Apply preprocessing
df['clean_text'] = df['text'].apply(preprocess_text)

# Features and labels
X = df['clean_text']
y = df['label']

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()

X_vectorized = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,

    test_size=0.2,
    random_state=42
)

# Baseline model
model = LogisticRegression()

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nBaseline Model Accuracy:", round(accuracy * 100, 2), "%")
# tuning.py

import pandas as pd
import re
import nltk
import string
import time
import joblib
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV
)

from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

# Download resources
nltk.download('stopwords')
nltk.download('wordnet')

# Load dataset
df = pd.read_csv("dataset.csv")

# Initialize preprocessing tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Preprocessing function
def preprocess_text(text):

    text = text.lower()

    text = re.sub(r'[^\w\s]', '', text)

    words = text.split()

    words = [word for word in words if word not in stop_words]

    words = [lemmatizer.lemmatize(word) for word in words]

    return " ".join(words)

# Apply preprocessing
df['clean_text'] = df['text'].apply(preprocess_text)

# Features and labels
X = df['clean_text']
y = df['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# BASELINE MODEL
# =========================

baseline_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('model', LogisticRegression())
])

baseline_pipeline.fit(X_train, y_train)

baseline_pred = baseline_pipeline.predict(X_test)

baseline_accuracy = accuracy_score(y_test, baseline_pred)

print("\nBefore Tuning Accuracy:",
      round(baseline_accuracy * 100, 2), "%")

# =========================
# GRID SEARCH CV
# =========================

grid_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('model', LogisticRegression())
])

param_grid = {
    'model__C': [0.01, 0.1, 1, 10],
    'model__solver': ['liblinear', 'lbfgs']
}

start_time = time.time()

grid_search = GridSearchCV(
    grid_pipeline,
    param_grid,
    cv=5,
    scoring='accuracy'
)

grid_search.fit(X_train, y_train)

grid_time = time.time() - start_time

best_grid_model = grid_search.best_estimator_

grid_pred = best_grid_model.predict(X_test)

grid_accuracy = accuracy_score(y_test, grid_pred)

print("\n===== GridSearchCV Results =====")

print("Best Parameters:", grid_search.best_params_)

print("Best Cross Validation Score:",
      round(grid_search.best_score_ * 100, 2), "%")

print("After Tuning Accuracy:",
      round(grid_accuracy * 100, 2), "%")

print("Training Time:", round(grid_time, 2), "seconds")

# =========================
# RANDOMIZED SEARCH CV
# =========================

random_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('model', LogisticRegression())
])

param_dist = {
    'model__C': [0.01, 0.1, 1, 10, 100],
    'model__penalty': ['l1', 'l2'],
    'model__solver': ['liblinear']
}

start_time = time.time()

random_search = RandomizedSearchCV(
    random_pipeline,
    param_distributions=param_dist,
    n_iter=5,
    cv=5,
    random_state=42,
    scoring='accuracy'
)

random_search.fit(X_train, y_train)

random_time = time.time() - start_time

best_random_model = random_search.best_estimator_

random_pred = best_random_model.predict(X_test)

random_accuracy = accuracy_score(y_test, random_pred)

print("\n===== RandomizedSearchCV Results =====")

print("Best Parameters:", random_search.best_params_)

print("Best Cross Validation Score:",
      round(random_search.best_score_ * 100, 2), "%")

print("After Tuning Accuracy:",
      round(random_accuracy * 100, 2), "%")

print("Training Time:", round(random_time, 2), "seconds")

# =========================
# SAVE BEST MODEL
# =========================

if grid_accuracy > random_accuracy:
    best_model = best_grid_model
else:
    best_model = best_random_model

joblib.dump(best_model, "best_model.pkl")

print("\nBest model saved as best_model.pkl")

# =========================
# ACCURACY COMPARISON GRAPH
# =========================

models = ['Baseline', 'GridSearchCV', 'RandomizedSearchCV']

accuracies = [
    baseline_accuracy * 100,
    grid_accuracy * 100,
    random_accuracy * 100
]

plt.figure(figsize=(8, 5))

plt.bar(models, accuracies)

plt.xlabel("Models")

plt.ylabel("Accuracy (%)")

plt.title("Model Accuracy Comparison")

plt.show()

# =========================
# PREDICTION SYSTEM
# =========================

while True:

    user_text = input("\nEnter text (or type quit): ")

    if user_text.lower() == 'quit':
        break

    cleaned_text = preprocess_text(user_text)

    prediction = best_model.predict([cleaned_text])

    print("Prediction (Tuned Model):", prediction[0])
# Hyperparameter Tuning using GridSearchCV & RandomizedSearchCV

## 📌 Objective
Improve NLP model performance using hyperparameter tuning techniques.

## 🛠 Technologies Used
- Python
- pandas
- scikit-learn
- nltk
- matplotlib
- joblib

## 📂 Features
- Text preprocessing
- TF-IDF vectorization
- Baseline Logistic Regression model
- GridSearchCV tuning
- RandomizedSearchCV tuning
- Accuracy comparison
- Model saving
- Prediction system

## ▶️ Run Project

### Install Libraries

pip install pandas scikit-learn nltk matplotlib joblib

### Run Baseline Model

python baseline_model.py

### Run Hyperparameter Tuning

python tuning.py

## 📊 Example Output

Before Tuning Accuracy: 82%

===== GridSearchCV Results =====

Best Parameters:
{'model__C': 10, 'model__solver': 'liblinear'}

After Tuning Accuracy: 89%

===== RandomizedSearchCV Results =====

Best Parameters:
{'model__solver': 'liblinear', 'model__penalty': 'l2', 'model__C': 10}

After Tuning Accuracy: 88%