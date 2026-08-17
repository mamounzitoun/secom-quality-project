"""
Phase ANALYZE - Etape 2 : Analyse de correlation

1. Correlation capteur <-> fail (point-bicseriale = Pearson sur variable binaire)
   Calculee sur les 442 capteurs pour identifier lesquels sont statistiquement
   lies au defaut (independamment de leur instabilite SPC vue en Measure).

2. Correlation capteur <-> capteur (parmi les capteurs critiques) pour
   detecter la multicolinearite avant le Random Forest (Etape 3).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# 1. CHARGEMENT
# ============================================================
data = pd.read_csv("data/processed/secom_clean.csv")
labels = pd.read_csv("data/processed/secom_labels_clean.csv")
pareto1 = pd.read_csv("data/processed/pareto_1_spc.csv")

# fail binaire : 1 = fail, 0 = pass (label original: 1=fail, -1=pass)
fail_binaire = (labels["label"] == 1).astype(int)

# ============================================================
# 2. CORRELATION CAPTEUR <-> FAIL (sur les 442 capteurs)
# ============================================================
correlations_fail = data.corrwith(fail_binaire)
corr_fail_df = pd.DataFrame({
    "capteur": correlations_fail.index,
    "correlation_avec_fail": correlations_fail.values,
    "correlation_abs": correlations_fail.abs().values
}).sort_values("correlation_abs", ascending=False).reset_index(drop=True)

print("="*70)
print("TOP 15 CAPTEURS - CORRELATION AVEC LE FAIL (sur 442 capteurs)")
print("="*70)
print(corr_fail_df[["capteur", "correlation_avec_fail"]].head(15).to_string(index=False))

# ============================================================
# 3. COMPARAISON AVEC LE PARETO #1 (SPC) - meme capteurs ou pas ?
# ============================================================
top20_fail_corr = set(corr_fail_df["capteur"].head(20))
top20_pareto_spc = set(pareto1["capteur"].head(20))
intersection = top20_fail_corr & top20_pareto_spc

print("\n" + "="*70)
print("COMPARAISON : Top 20 correlation-fail VS Top 20 Pareto SPC (Measure)")
print("="*70)
print(f"Capteurs communs aux deux classements : {len(intersection)} / 20")
if intersection:
    print(f"Capteurs : {sorted(intersection)}")

# ============================================================
# 4. MATRICE DE CORRELATION ENTRE CAPTEURS CRITIQUES (multicolinearite)
#    On prend l'union des deux Top 20 (SPC + correlation-fail) pour la suite
# ============================================================
capteurs_analyse = sorted(top20_fail_corr | top20_pareto_spc)
print(f"\nCapteurs retenus pour le Random Forest (union des 2 criteres) : {len(capteurs_analyse)}")

corr_matrix = data[capteurs_analyse].corr()

# Detection des paires fortement correlees (multicolinearite, |r| > 0.85)
paires_fortes = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        r = corr_matrix.iloc[i, j]
        if abs(r) > 0.85:
            paires_fortes.append((corr_matrix.columns[i], corr_matrix.columns[j], round(r, 3)))

print(f"\nPaires de capteurs fortement correlees entre elles (|r|>0.85) : {len(paires_fortes)}")
for p in paires_fortes[:10]:
    print(f"  {p[0]} <-> {p[1]} : r = {p[2]}")

# ============================================================
# 5. HEATMAP
# ============================================================
plt.figure(figsize=(14, 12))
sns.heatmap(corr_matrix, cmap="coolwarm", center=0, square=True,
            xticklabels=True, yticklabels=True, cbar_kws={"label": "Coefficient r"})
plt.title("Matrice de correlation - Capteurs critiques (union SPC + correlation-fail)")
plt.tight_layout()
plt.savefig("reports/correlation_heatmap.png", dpi=150)
print("\nHeatmap sauvegardee : reports/correlation_heatmap.png")

# ============================================================
# 6. SAUVEGARDE
# ============================================================
corr_fail_df.to_csv("data/processed/correlation_fail.csv", index=False)
corr_matrix.to_csv("data/processed/correlation_matrix_critiques.csv")
with open("data/processed/capteurs_selectionnes_RF.txt", "w") as f:
    f.write("\n".join(capteurs_analyse))
print("Resultats sauvegardes dans data/processed/")