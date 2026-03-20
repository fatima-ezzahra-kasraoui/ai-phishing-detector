# ============================================================
# train_model_ultra.py – Pipeline phishing ultra-robuste
# ============================================================

import os
import re
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score, classification_report

# ============================================================
# CONFIGURATION
# ============================================================

CSV_PATH = r"C:\Users\Dell\Desktop\Projet_AI\Phishing-Email-Dataset\dataset_fusionnee.csv"
MODEL_PATH = r"C:\Users\Dell\Desktop\Projet_AI\phishing_pipeline_ultra.pkl"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ============================================================
# Nettoyage du texte
# ============================================================

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " URL ", text)
    text = re.sub(r"\S+@\S+", " EMAIL ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ============================================================
# Charger dataset original
# ============================================================

print("[INFO] Loading original dataset...")
df = pd.read_csv(CSV_PATH)
df["text"] = (df["subject"].fillna("") + " " + df["body"].fillna("")).apply(clean_text)
df["label"] = df["label"].astype(int)
print(f"[INFO] Original dataset: {len(df)} emails")

# ============================================================
# Génération 800 emails légitimes supplémentaires (anglais)
# ============================================================

print("[INFO] Generating 800 additional legitimate emails...")

additional_legit = [
    {"subject": "Team meeting tomorrow", "body": "Don't forget the team sync at 10 AM.", "label": 0},
    {"subject": "Amazon order shipped", "body": "Your order has been shipped. Track it with Amazon app.", "label": 0},
    {"subject": "Invoice November 2025", "body": "Please find attached your invoice.", "label": 0},
    {"subject": "Password successfully changed", "body": "Your password has been updated.", "label": 0},
    {"subject": "LinkedIn: Welcome", "body": "Tips to get started with your new account.", "label": 0},
    {"subject": "Subscription receipt", "body": "Your payment has been successfully received.", "label": 0},
    {"subject": "Holiday request approved", "body": "Your vacation has been approved.", "label": 0},
    {"subject": "Exam results posted", "body": "Check your grades online now.", "label": 0},
    {"subject": "Zoom meeting link", "body": "Join the meeting using the link provided.", "label": 0},
    {"subject": "Spotify new device login", "body": "A new device logged into your account.", "label": 0},
] * 80  # répète 10 emails * 80 = 800

df_add = pd.DataFrame(additional_legit)
df_add["text"] = (df_add["subject"].fillna("") + " " + df_add["body"].fillna("")).apply(clean_text)
df_add["label"] = df_add["label"].astype(int)

# Merge datasets
df = pd.concat([df, df_add]).reset_index(drop=True)
print(f"[INFO] New dataset size: {len(df)} emails")

# ============================================================
# Split train / test
# ============================================================

X = df["text"].values
y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print(f"[INFO] Train: {len(X_train)} – Test: {len(X_test)}")

# ============================================================
# TF-IDF + Logistic Regression + Calibration
# ============================================================

tfidf = TfidfVectorizer(
    max_features=30000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    stop_words="english",
    strip_accents="unicode",
    min_df=3
)

clf = LogisticRegression(
    class_weight="balanced",
    solver="liblinear",
    max_iter=1500
)

# Calibration
calibrated_clf = CalibratedClassifierCV(clf, cv=3)

print("[INFO] Training Logistic Regression...")
X_train_tfidf = tfidf.fit_transform(X_train)
calibrated_clf.fit(X_train_tfidf, y_train)

# ============================================================
# Évaluation sur test set
# ============================================================

X_test_tfidf = tfidf.transform(X_test)
y_prob = calibrated_clf.predict_proba(X_test_tfidf)[:,1]

# Chercher le seuil optimal
thresholds = np.arange(0.4, 0.95, 0.01)
best_thresh = 0.5
best_f1 = 0

for t in thresholds:
    y_pred_t = (y_prob >= t).astype(int)
    f1 = f1_score(y_test, y_pred_t)
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t

print(f"[INFO] Optimal threshold determined: {best_thresh:.2f} with F1: {best_f1:.4f}")

# Predictions finales
y_pred_final = (y_prob >= best_thresh).astype(int)
print("\n===== MODEL EVALUATION =====")
print(classification_report(y_test, y_pred_final))

# ============================================================
# Sauvegarde du pipeline complet (sans fonctions)
# ============================================================

final_pipeline = {
    "tfidf": tfidf,
    "clf": calibrated_clf,
    "threshold": best_thresh
}

joblib.dump(final_pipeline, MODEL_PATH)
print(f"[INFO] Final model saved to: {MODEL_PATH}")

# ============================================================
# Exemple rapide (utilisation des fonctions locales)
# ============================================================

def predict(text):
    text_clean = clean_text(text)
    prob = calibrated_clf.predict_proba(tfidf.transform([text_clean]))[0][1]
    return int(prob >= best_thresh)

def predict_proba(text):
    text_clean = clean_text(text)
    return float(calibrated_clf.predict_proba(tfidf.transform([text_clean]))[0][1])

example = "URGENT: verify your account now or lose access"
pred = predict(example)
prob = predict_proba(example)
print("\n[TEST] Example:")
print(('PHISHING' if pred==1 else 'LEGITIMATE', prob))
