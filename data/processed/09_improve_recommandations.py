"""
Phase IMPROVE - Recommandations qualitatives basees sur donnees

Pour chaque capteur prioritaire (Top 10 SHAP), on croise :
- Le ratio sigma_long/sigma_court (deja calculable via SPC, Measure)
  -> ratio eleve = derive dans le temps = maintenance/recalibration
  -> ratio proche de 1 = bruit stable = resserrement du controle
- Le sens de la correlation avec le fail (deja calcule, Analyze Etape 2)
  -> positif = valeur haute du capteur associee au fail
  -> negatif = valeur basse du capteur associee au fail

Ceci transforme le Pareto SHAP en plan d'action concret, pas une liste
de noms de capteurs sans contexte.
"""

import pandas as pd
import numpy as np

N_CAPTEURS = 10

# ============================================================
# 1. CHARGEMENT DE TOUT CE QUI A DEJA ETE CALCULE
# ============================================================
pareto2 = pd.read_csv("data/processed/pareto_2_shap.csv")
spc_results = pd.read_csv("data/processed/spc_results.csv")
data_full = pd.read_csv("data/processed/secom_clean.csv")
corr_fail = pd.read_csv("data/processed/correlation_fail.csv")

capteurs_prioritaires = pareto2["capteur"].head(N_CAPTEURS).tolist()

# ============================================================
# 2. CALCUL DU RATIO DERIVE POUR CES CAPTEURS PRECIS
# ============================================================
lignes = []
for capteur in capteurs_prioritaires:
    sigma_court = spc_results.loc[spc_results["capteur"] == capteur, "sigma_estime"].values[0]
    sigma_long = data_full[capteur].std(ddof=1)
    ratio = sigma_long / sigma_court

    sens_corr = corr_fail.loc[corr_fail["capteur"] == capteur, "correlation_avec_fail"].values[0]
    direction = "Valeur ELEVEE -> risque fail" if sens_corr > 0 else "Valeur BASSE -> risque fail"

    importance = pareto2.loc[pareto2["capteur"] == capteur, "importance_shap"].values[0]

    if ratio > 2.5:
        action = "Maintenance preventive / recalibration (derive dans le temps dominante)"
    elif ratio > 1.8:
        action = "Surveillance renforcee + recalibration periodique"
    else:
        action = "Resserrement des limites de controle (bruit stable, pas de derive)"

    lignes.append({
        "capteur": capteur,
        "importance_shap": round(importance, 4),
        "ratio_derive_long_court": round(ratio, 2),
        "direction_risque": direction,
        "action_recommandee": action
    })

reco_df = pd.DataFrame(lignes).sort_values("importance_shap", ascending=False)

print("="*100)
print("PLAN D'ACTION - CAPTEURS PRIORITAIRES (Top 10 SHAP)")
print("="*100)
for _, r in reco_df.iterrows():
    print(f"\n{r['capteur']}  (importance SHAP: {r['importance_shap']})")
    print(f"  Ratio derive (long/court terme) : {r['ratio_derive_long_court']}")
    print(f"  {r['direction_risque']}")
    print(f"  -> Action recommandee : {r['action_recommandee']}")

print("\n" + "="*100)
print("SYNTHESE - REPARTITION DES ACTIONS")
print("="*100)
print(reco_df["action_recommandee"].value_counts().to_string())

reco_df.to_csv("data/processed/improve_recommandations.csv", index=False)
print("\nRecommandations sauvegardees : data/processed/improve_recommandations.csv")
