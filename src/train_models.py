import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# 1. Load dataset
csv_path = (
    "data/PhiUSIIL_Phishing_URL_Dataset.csv"
    if os.path.exists("data/PhiUSIIL_Phishing_URL_Dataset.csv")
    else "PhiUSIIL_Phishing_URL_Dataset.csv"
)
df = pd.read_csv(csv_path)

# 2. Pure URL numeric features (excluding URLSimilarityIndex)
strict_url_features = [
    "URLLength",
    "DomainLength",
    "IsDomainIP",
    "CharContinuationRate",
    "TLDLegitimateProb",
    "URLCharProb",
    "TLDLength",
    "NoOfSubDomain",
    "HasObfuscation",
    "NoOfObfuscatedChar",
    "ObfuscationRatio",
    "NoOfLettersInURL",
    "LetterRatioInURL",
    "NoOfDegitsInURL",
    "DegitRatioInURL",
    "NoOfEqualsInURL",
    "NoOfQMarkInURL",
    "NoOfAmpersandInURL",
    "NoOfOtherSpecialCharsInURL",
    "SpacialCharRatioInURL",
    "IsHTTPS",
]

X = df[strict_url_features]
y = df["label"]

# 3. Same 70% Train / 15% Validation / 15% Test split (random_state=42)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

# 4. Define candidate models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss",
    ),
    "LightGBM": LGBMClassifier(
        n_estimators=200,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    ),
}

# 5. Train and compare on the Validation set
results = []
for name, clf in models.items():
    clf.fit(X_train, y_train)
    preds = clf.predict(X_val)
    results.append({
        "Model": name,
        "Accuracy": round(accuracy_score(y_val, preds), 4),
        "Precision": round(precision_score(y_val, preds), 4),
        "Recall": round(recall_score(y_val, preds), 4),
        "F1-Score": round(f1_score(y_val, preds), 4),
    })

comparison_df = pd.DataFrame(results)
print("\n--- Validation Comparison (URL-Only Features, No URLSimilarityIndex) ---")
print(comparison_df.to_string(index=False))