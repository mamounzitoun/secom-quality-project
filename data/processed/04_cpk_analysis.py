"""
Phase ANALYZE - Etape 1 : Capabilite process Cp/Cpk vs Pp/Ppk

DISTINCTION METHODOLOGIQUE (standard en ingenierie qualite) :
  - sigma court terme (sigma_within) : variation inherente/bruit point-a-point,
    estimee via I-MR (Moving Range / d2) - deja calculee en phase Measure
  - sigma long terme (sigma_overall) : ecart-type classique sur toute la serie,
    inclut les derives et changements du process dans le temps

  Cp/Cpk  -> utilise sigma court terme -> "potentiel" du process
  Pp/Ppk  -> utilise sigma long terme  -> "performance reelle" observee

HYPOTHESE (le dataset ne fournit aucune specification produit reelle) :
  USL assumee = moyenne + 4 x sigma_long_terme
  LSL assumee = moyenne - 4 x sigma_long_terme
  (definit une tolerance a partir de l'etendue historique reelle du process)

Avec cette hypothese, Ppk sera par construction ~1.333 (sert de reference),
et Cpk variera reellement selon le ratio sigma_court/sigma_long de chaque
capteur -> c'est cet ecart Cpk vs Ppk qui est informatif : plus l'ecart est
grand, plus le capteur souffre de causes speciales de variation (derives).

Formules :
  Cp/Ppk = (USL-LSL) / (6 x sigma)
  Cpk/Ppk_k = min( (USL-moyenne)/(3 x sigma) ; (moyenne-LSL)/(3 x sigma) )
"""

import pandas as pd
import numpy as np

K_TOLERANCE = 4

# ============================================================
# 1. CHARGEMENT (donnees + resultats SPC deja calcules en Measure)
# ============================================================
data = pd.read_csv("data/processed/secom_clean.csv")
pareto1 = pd.read_csv("data/processed/pareto_1_spc.csv")
spc_results = pd.read_csv("data/processed/spc_results.csv")  # contient sigma_estime (court terme)

capteurs_critiques = pareto1["capteur"].head(20).tolist()
print(f"Analyse Cp/Cpk vs Pp/Ppk sur les {len(capteurs_critiques)} capteurs critiques (Pareto #1)")

# ============================================================
# 2. CALCUL
# ============================================================
resultats = []

for col in capteurs_critiques:
    x = data[col].values
    moyenne = x.mean()

    sigma_long = x.std(ddof=1)  # long terme (Pp/Ppk)
    sigma_court = spc_results.loc[spc_results["capteur"] == col, "sigma_estime"].values[0]  # court terme (Cp/Cpk)

    usl = moyenne + K_TOLERANCE * sigma_long
    lsl = moyenne - K_TOLERANCE * sigma_long

    # Pp/Ppk (performance reelle, long terme)
    ppk = min((usl - moyenne) / (3 * sigma_long), (moyenne - lsl) / (3 * sigma_long))

    # Cp/Cpk (potentiel, court terme)
    cpk = min((usl - moyenne) / (3 * sigma_court), (moyenne - lsl) / (3 * sigma_court))

    ecart_potentiel = cpk - ppk  # si positif et grand -> beaucoup de causes speciales/derive

    resultats.append({
        "capteur": col,
        "sigma_court_terme": round(sigma_court, 4),
        "sigma_long_terme": round(sigma_long, 4),
        "ratio_long_sur_court": round(sigma_long / sigma_court, 2),
        "Ppk_performance_reelle": round(ppk, 3),
        "Cpk_potentiel": round(cpk, 3),
        "ecart_potentiel_cache": round(ecart_potentiel, 3)
    })

cpk_df = pd.DataFrame(resultats).sort_values("ecart_potentiel_cache", ascending=False)

print("\n" + "="*90)
print("Cp/Cpk (potentiel) vs Pp/Ppk (performance reelle) - tries par ecart decroissant")
print("="*90)
print(cpk_df.to_string(index=False))

print("\nInterpretation : un grand 'ecart_potentiel_cache' signale un capteur ou la")
print("variation long terme (derives) domine largement le bruit court terme ->")
print("fort potentiel d'amelioration en stabilisant le process (causes speciales).")

# ============================================================
# 3. SAUVEGARDE
# ============================================================
cpk_df.to_csv("data/processed/cpk_results.csv", index=False)
print("\nResultats sauvegardes : data/processed/cpk_results.csv")