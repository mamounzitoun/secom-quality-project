"""
Phase MEASURE - Etape 2 : Cartes de controle SPC (I-MR) + Diagramme de Pareto #1

Methode I-MR (Individuals - Moving Range) :
  - Adaptee car une seule mesure par lot (pas de sous-groupes)
  - MR = |x_i - x_(i-1)|
  - Sigma estime = MR_barre / d2   (d2 = 1.128 pour n=2)
  - UCL = x_barre + 3*sigma ; LCL = x_barre - 3*sigma

Pareto #1 : classement des capteurs par frequence de depassement des
limites de controle (nombre de points hors UCL/LCL).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

D2_CONSTANT = 1.128  # constante statistique standard pour n=2 (tables SPC)

# ============================================================
# 1. CHARGEMENT DES DONNEES NETTOYEES
# ============================================================
data = pd.read_csv("data/processed/secom_clean.csv")
print(f"Donnees chargees : {data.shape[0]} lots x {data.shape[1]} capteurs")

# ============================================================
# 2. CALCUL SPC (I-MR) POUR CHAQUE CAPTEUR
# ============================================================
resultats = []

for col in data.columns:
    x = data[col].values
    x_barre = x.mean()

    mr = np.abs(np.diff(x))
    mr_barre = mr.mean()
    sigma_est = mr_barre / D2_CONSTANT

    ucl = x_barre + 3 * sigma_est
    lcl = x_barre - 3 * sigma_est

    hors_controle = ((x > ucl) | (x < lcl)).sum()

    resultats.append({
        "capteur": col,
        "moyenne": x_barre,
        "sigma_estime": sigma_est,
        "UCL": ucl,
        "LCL": lcl,
        "nb_points_hors_controle": hors_controle,
        "pct_hors_controle": hors_controle / len(x) * 100
    })

spc_df = pd.DataFrame(resultats)

# ============================================================
# 3. CONSTRUCTION DU DIAGRAMME DE PARETO #1
# ============================================================
pareto_df = spc_df[spc_df["nb_points_hors_controle"] > 0].copy()
pareto_df = pareto_df.sort_values("nb_points_hors_controle", ascending=False).reset_index(drop=True)
pareto_df["cumsum"] = pareto_df["nb_points_hors_controle"].cumsum()
pareto_df["cumsum_pct"] = pareto_df["cumsum"] / pareto_df["nb_points_hors_controle"].sum() * 100

print("\n" + "="*55)
print("TOP 15 CAPTEURS - PARETO #1 (SPC)")
print("="*55)
print(pareto_df[["capteur", "nb_points_hors_controle", "pct_hors_controle", "cumsum_pct"]].head(15).to_string(index=False))

n_capteurs_80pct = (pareto_df["cumsum_pct"] <= 80).sum() + 1
pct_capteurs_80 = n_capteurs_80pct / len(pareto_df) * 100

print(f"\n{n_capteurs_80pct} capteurs sur {len(pareto_df)} ({pct_capteurs_80:.1f}%) "
      f"expliquent 80% des depassements de limites de controle.")

# ============================================================
# 4. GRAPHIQUE PARETO (top 20)
# ============================================================
top20 = pareto_df.head(20)

fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.bar(range(len(top20)), top20["nb_points_hors_controle"], color="#4C72B0")
ax1.set_xlabel("Capteurs (classes par frequence de depassement)")
ax1.set_ylabel("Nombre de points hors controle", color="#4C72B0")
ax1.set_xticks(range(len(top20)))
ax1.set_xticklabels(top20["capteur"], rotation=90, fontsize=7)

ax2 = ax1.twinx()
ax2.plot(range(len(top20)), top20["cumsum_pct"], color="#C44E52", marker="o")
ax2.set_ylabel("Pourcentage cumule (%)", color="#C44E52")
ax2.axhline(80, color="gray", linestyle="--", linewidth=1)

plt.title("Diagramme de Pareto #1 - Capteurs hors controle SPC (Top 20)")
plt.tight_layout()
plt.savefig("reports/pareto_1_spc.png", dpi=150)
print("\nGraphique sauvegarde : reports/pareto_1_spc.png")

# ============================================================
# 5. SAUVEGARDE DES RESULTATS
# ============================================================
spc_df.to_csv("data/processed/spc_results.csv", index=False)
pareto_df.to_csv("data/processed/pareto_1_spc.csv", index=False)
print("Resultats sauvegardes dans data/processed/")