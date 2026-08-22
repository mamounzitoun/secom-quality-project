"""
Phase CONTROL - Export des KPIs de synthese pour Power BI

Regroupe les indicateurs cles de tout le projet DMAIC (Define -> Control)
dans un seul fichier compact, pour alimenter des cartes KPI dans le dashboard.
"""

import pandas as pd

pareto2 = pd.read_csv("data/processed/pareto_2_shap.csv")
improve_sim = pd.read_csv("data/processed/improve_simulation_resultats.csv")
alertes = pd.read_csv("data/processed/alertes_control.csv")

capteur_top = pareto2.iloc[0]["capteur"]

kpis = pd.DataFrame([
    {"indicateur": "Taux de fail baseline (Define)", "valeur": "6.64%"},
    {"indicateur": "Objectif SMART (Define)", "valeur": "5.31%"},
    {"indicateur": "Capteurs analyses (initial)", "valeur": "590"},
    {"indicateur": "Capteurs retenus apres nettoyage (Measure)", "valeur": "442"},
    {"indicateur": "Capteur le plus critique (Analyze - SHAP)", "valeur": capteur_top},
    {"indicateur": "ROC-AUC du modele (Analyze)", "valeur": "0.785"},
    {"indicateur": "Nouveau taux de fail estime (Improve)", "valeur": "5.27%"},
    {"indicateur": "Gain financier annuel estime (Improve)", "valeur": "136 779 EUR"},
    {"indicateur": "Total alertes SPC actives (Control)", "valeur": str(len(alertes))},
    {"indicateur": "Capteurs sous surveillance perenne (Control)", "valeur": "10"},
])

kpis.to_csv("dashboard/kpis_synthese_dmaic.csv", index=False, sep=";", decimal=",")
print("KPIs sauvegardes : dashboard/kpis_synthese_dmaic.csv")
print(kpis.to_string(index=False))