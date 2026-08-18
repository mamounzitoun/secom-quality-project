"""
Phase ANALYZE - Etape 4 : SHAP (explicabilite du modele)

TreeSHAP calcule les valeurs de Shapley exactement pour les modeles a
base d'arbres (Random Forest). Pour chaque capteur, la valeur SHAP
represente sa contribution reelle a la prediction, pour chaque lot.

Deux niveaux :
1. Global : moyenne des |SHAP| par capteur -> Pareto #2 (impact reel
   sur le fail, a comparer avec le Pareto #1 base sur la variabilite SPC)
2. Local : explication d'un lot fail precis (waterfall plot)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib

# ============================================================
# 1. CHARGEMENT DU MODELE ET DES DONNEES DE TEST
# ============================================================
rf = joblib.load("data/processed/random_forest_model.joblib")
X_test = pd.read_csv("data/processed/X_test.csv")
y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

# ============================================================
# 2. CALCUL DES VALEURS SHAP (TreeExplainer, exact pour Random Forest)
# ============================================================
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test)

# Pour classification binaire, shap_values peut etre une liste [classe0, classe1]
# ou un array 3D selon la version - on isole la classe "Fail" (1)
if isinstance(shap_values, list):
    shap_values_fail = shap_values[1]
elif shap_values.ndim == 3:
    shap_values_fail = shap_values[:, :, 1]
else:
    shap_values_fail = shap_values

print(f"Valeurs SHAP calculees : {shap_values_fail.shape}")

# ============================================================
# 3. IMPORTANCE GLOBALE - PARETO #2
# ============================================================
importance_moyenne = np.abs(shap_values_fail).mean(axis=0)
pareto2_df = pd.DataFrame({
    "capteur": X_test.columns,
    "importance_shap": importance_moyenne
}).sort_values("importance_shap", ascending=False).reset_index(drop=True)

pareto2_df["cumsum"] = pareto2_df["importance_shap"].cumsum()
pareto2_df["cumsum_pct"] = pareto2_df["cumsum"] / pareto2_df["importance_shap"].sum() * 100

print("\n" + "="*60)
print("TOP 15 CAPTEURS - PARETO #2 (importance SHAP)")
print("="*60)
print(pareto2_df[["capteur", "importance_shap", "cumsum_pct"]].head(15).to_string(index=False))

n_capteurs_80 = (pareto2_df["cumsum_pct"] <= 80).sum() + 1
print(f"\n{n_capteurs_80} capteurs sur {len(pareto2_df)} expliquent 80% de l'importance SHAP totale")

# ============================================================
# 4. COMPARAISON PARETO #1 (SPC) vs PARETO #2 (SHAP)
# ============================================================
pareto1_df = pd.read_csv("data/processed/pareto_1_spc.csv")
top10_pareto1 = set(pareto1_df["capteur"].head(10))
top10_pareto2 = set(pareto2_df["capteur"].head(10))
communs = top10_pareto1 & top10_pareto2

print(f"\nCapteurs communs entre Top 10 Pareto#1 (SPC) et Top 10 Pareto#2 (SHAP) : {len(communs)}")
if communs:
    print(f"  -> {sorted(communs)}")
print("(Un faible recoupement confirme que variabilite statistique et impact")
print(" reel sur le fail sont deux informations differentes et complementaires)")

# ============================================================
# 5. GRAPHIQUE SUMMARY PLOT (vue d'ensemble)
# ============================================================
plt.figure()
shap.summary_plot(shap_values_fail, X_test, show=False, max_display=15)
plt.tight_layout()
plt.savefig("reports/shap_summary_plot.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSummary plot sauvegarde : reports/shap_summary_plot.png")

# ============================================================
# 6. GRAPHIQUE PARETO #2 (bar + cumul, comme Pareto #1)
# ============================================================
top20 = pareto2_df.head(20)
fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.bar(range(len(top20)), top20["importance_shap"], color="#55A868")
ax1.set_xlabel("Capteurs (classes par importance SHAP)")
ax1.set_ylabel("Importance SHAP moyenne (|valeur|)", color="#55A868")
ax1.set_xticks(range(len(top20)))
ax1.set_xticklabels(top20["capteur"], rotation=90, fontsize=7)

ax2 = ax1.twinx()
ax2.plot(range(len(top20)), top20["cumsum_pct"], color="#C44E52", marker="o")
ax2.set_ylabel("Pourcentage cumule (%)", color="#C44E52")
ax2.axhline(80, color="gray", linestyle="--", linewidth=1)

plt.title("Diagramme de Pareto #2 - Importance SHAP (Top 20)")
plt.tight_layout()
plt.savefig("reports/pareto_2_shap.png", dpi=150)
plt.close()
print("Pareto #2 sauvegarde : reports/pareto_2_shap.png")

# ============================================================
# 7. EXPLICATION LOCALE - un cas de fail correctement detecte
# ============================================================
y_proba = rf.predict_proba(X_test)[:, 1]
idx_candidats = np.where((y_test.values == 1) & (y_proba >= 0.5))[0]

if len(idx_candidats) > 0:
    idx = idx_candidats[0]
    plt.figure()
    exp = shap.Explanation(
        values=shap_values_fail[idx],
        base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value,
        data=X_test.iloc[idx],
        feature_names=X_test.columns.tolist()
    )
    shap.plots.waterfall(exp, show=False, max_display=12)
    plt.tight_layout()
    plt.savefig("reports/shap_waterfall_exemple.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Waterfall plot (lot test #{idx}, fail correctement detecte) sauvegarde : reports/shap_waterfall_exemple.png")

# ============================================================
# 8. SAUVEGARDE
# ============================================================
pareto2_df.to_csv("data/processed/pareto_2_shap.csv", index=False)
print("\nResultats sauvegardes : data/processed/pareto_2_shap.csv")