import pandas as pd
import joblib
import numpy as np
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from utils.preprocess import clean_text

# ================= LOAD DATA =================
print("📥 Loading dataset...")

df = pd.read_json("data/arxiv-metadata-oai-snapshot.json", lines=True, nrows=50000)

df = df[['title', 'abstract', 'categories']].dropna()

# Combine text
df['text'] = (df['title'] + " " + df['abstract']).apply(clean_text)

# Extract category
df['category'] = df['categories'].apply(lambda x: x.split()[0])

# Keep only CS domains
df = df[df['category'].str.startswith('cs')]

print("✅ Data cleaned")

# ================= BALANCE DATA =================
print("⚖️ Balancing dataset...")

df = df.groupby('category').head(3000)

print("✅ Dataset balanced")

# ================= VECTORIZE =================
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')

X = vectorizer.fit_transform(df['text'])
y = df['category']

# ================= SPLIT =================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ================= MODEL =================
print("🤖 Training model...")

model = LogisticRegression(max_iter=2000, class_weight='balanced')

model.fit(X_train, y_train)

print("✅ Model trained")

# ================= SAVE =================
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("💾 Model saved")

# ================= METRICS =================
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

json.dump({"accuracy": float(acc)}, open("metrics.json", "w"))
np.save("confusion.npy", cm)

print(f"📊 Accuracy: {acc*100:.2f}%")
print("🎉 Training Complete!")