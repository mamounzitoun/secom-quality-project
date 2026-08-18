"""
Phase ANALYZE - Etape 3bis (revisee) : Scenarios de seuil de decision

CORRECTION METHODOLOGIQUE :
Il n'existe pas de seuil "optimal" universel - recall et precision sont
antagonistes par nature. Le choix du seuil est un ARBITRAGE BUSINESS
(capacite d'inspection de l'usine, tolerance aux fausses alertes), pas
une pure optimisation mathematique.

On presente donc plusieurs scenarios realistes avec leurs consequences
operationnelles concretes, plutot qu'un seuil unique impose.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import joblib

# ============================================================
# 1. CHARGEMENT
# ============================================================
rf = joblib.load("data/processed/random_forest_model.joblib")
X_test = pd.read_csv("data/processed/X_test.csv")
y_test = pd.read_csv("data/processed/y_test.csv").squeeze()
y_proba = rf.predict_proba(X_test)[:, 1]

n_total_test = len(y_test)

# ============================================================
# 2. SCENARIOS REALISTES (seuils choisis pour illustrer differents arbitrages)
# ============================================================
scenarios = {
    "Conservateur (defaut sklearn)": 0.50,
    "Equilibre": 0.30,
    "Sensible (priorise detection)": 0.20,
    "Tres sensible (risque de saturation)": 0.15,
}

print("="*95)
print("SCENARIOS DE SEUIL - IMPLICATIONS OPERATIONNELLES CONCRETES")
print("="*95)

resultats = []
for nom, seuil in scenarios.items():
    y_pred = (y_proba >= seuil).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    recall = tp / (tp + fn) if (tp+fn) > 0 else 0
    precision = tp / (tp + fp) if (tp+fp) > 0 else 0
    pct_lots_signales = (tp + fp) / n_total_test * 100

    resultats.append({
        "scenario": nom, "seuil": seuil,
        "fails_detectes": f"{tp}/{tp+fn}", "recall_pct": round(recall*100,1),
        "fausses_alertes": fp, "precision_pct": round(precision*100,1),
        "pct_lots_total_signales": round(pct_lots_signales, 1)
    })

df = pd.DataFrame(resultats)
print(df.to_string(index=False))

print("\n" + "="*95)
print("LECTURE OPERATIONNELLE (pas juste mathematique)")
print("="*95)
for r in resultats:
    print(f"\n[{r['scenario']}] seuil={r['seuil']}")
    print(f"  -> Detecte {r['fails_detectes']} fails reels ({r['recall_pct']}%)")
    print(f"  -> {r['pct_lots_total_signales']}% de TOUS les lots production seraient signales/inspectes")

# ============================================================
# 3. ANALYSE DE SENSIBILITE : et si le cout d'une fausse alerte
#    etait sous-estime ? (200 EUR etait une hypothese fragile)
# ============================================================
print("\n" + "="*95)
print("SENSIBILITE : seuil optimal selon differentes hypotheses de cout fausse alerte")
print("="*95)

COUT_FN = 5000
hypotheses_cout_fp = [200, 500, 1000, 2000]

for cout_fp in hypotheses_cout_fp:
    couts_par_seuil = []
    for seuil in np.arange(0.05, 0.95, 0.05):
        y_pred = (y_proba >= seuil).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        cout = fn * COUT_FN + fp * cout_fp
        couts_par_seuil.append((round(seuil,2), cout))
    meilleur = min(couts_par_seuil, key=lambda x: x[1])
    print(f"Si cout fausse alerte = {cout_fp:>5} EUR -> seuil optimal = {meilleur[0]} (cout = {meilleur[1]:,} EUR)")

print("\n=> Le seuil 'optimal' change fortement selon l'hypothese de cout.")
print("=> Conclusion : ce n'est PAS une verite mathematique fixe, c'est une")
print("   decision business a valider avec les responsables qualite/production,")
print("   en fonction de la vraie capacite d'inspection de l'usine.")

df.to_csv("data/processed/threshold_scenarios.csv", index=False)