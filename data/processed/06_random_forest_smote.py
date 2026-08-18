"""
Phase ANALYZE - Etape 3 : Random Forest + SMOTE

Pourquoi SMOTE : le taux de fail est de 6.64% (fort desequilibre).
Sans traitement, un modele peut atteindre 93%+ d'accuracy en predisant
toujours "pass", sans aucune utilite reelle.

SMOTE genere des exemples synthetiques de la classe minoritaire (fail)
par interpolation entre plus proches voisins - applique UNIQUEMENT sur
le train set (jamais sur le test, pour eviter la fuite de donnees).

Metriques utilisees : Precision, Recall, F1, ROC-AUC (pas l'accuracy seule,
trompeuse en cas de desequilibre de classes).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, ConfusionMatrixDisplay
)
from imblearn.over_sampling import SMOTE
import joblib

RANDOM_STATE = 42

# ============================================================
# 1. CHARGEMENT
# ============================================================
data = pd.read_csv("data/processed/secom_clean.csv")
labels = pd.read_csv("data/processed/secom_labels_clean.csv")

with open("data/processed/capteurs_selectionnes_RF.txt") as f:
    capteurs = [l.strip() for l in f if l.strip()]

X = data[capteurs]
y = (labels["label"] == 1).astype(int)  # 1 = fail, 0 = pass

print(f"Features utilisees : {X.shape[1]} capteurs")
print(f"Distribution cible : {y.value_counts().to_dict()}")

# ============================================================
# 2. TRAIN/TEST SPLIT (stratifie pour garder le meme ratio fail/pass)
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
)
print(f"\nTrain : {X_train.shape[0]} lots ({y_train.sum()} fails)")
print(f"Test  : {X_test.shape[0]} lots ({y_test.sum()} fails)")

# ============================================================
# 3. SMOTE - UNIQUEMENT SUR LE TRAIN
# ============================================================
smote = SMOTE(random_state=RANDOM_STATE)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"\nApres SMOTE - Train : {X_train_smote.shape[0]} lots "
      f"({(y_train_smote==1).sum()} fails, {(y_train_smote==0).sum()} pass)")

# ============================================================
# 4. ENTRAINEMENT RANDOM FOREST
# ============================================================
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    class_weight=None,  # deja equilibre par SMOTE
    random_state=RANDOM_STATE,
    n_jobs=-1
)
rf.fit(X_train_smote, y_train_smote)

# ============================================================
# 5. EVALUATION SUR LE TEST SET (donnees reelles, jamais vues, sans SMOTE)
# ============================================================
y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:, 1]

print("\n" + "="*60)
print("RAPPORT DE CLASSIFICATION (test set)")
print("="*60)
print(classification_report(y_test, y_pred, target_names=["Pass", "Fail"]))

auc = roc_auc_score(y_test, y_proba)
print(f"ROC-AUC : {auc:.3f}")

# ============================================================
# 6. MATRICE DE CONFUSION
# ============================================================
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(cm, display_labels=["Pass", "Fail"]).plot(ax=ax, cmap="Blues")
plt.title("Matrice de confusion - Random Forest + SMOTE")
plt.tight_layout()
plt.savefig("reports/confusion_matrix.png", dpi=150)
print("\nMatrice de confusion sauvegardee : reports/confusion_matrix.png")

# ============================================================
# 7. COURBE ROC
# ============================================================
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"Random Forest (AUC={auc:.3f})", color="#4C72B0")
plt.plot([0, 1], [0, 1], "--", color="gray", label="Hasard (AUC=0.5)")
plt.xlabel("Taux de faux positifs")
plt.ylabel("Taux de vrais positifs")
plt.title("Courbe ROC - Prediction du fail")
plt.legend()
plt.tight_layout()
plt.savefig("reports/roc_curve.png", dpi=150)
print("Courbe ROC sauvegardee : reports/roc_curve.png")

# ============================================================
# 8. SAUVEGARDE DU MODELE
# ============================================================
joblib.dump(rf, "data/processed/random_forest_model.joblib")
X_test.to_csv("data/processed/X_test.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)
print("\nModele sauvegarde : data/processed/random_forest_model.joblib")