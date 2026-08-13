"""
Phase DEFINE - Script complet
1. Calcul du taux de rejet baseline
2. Analyse financiere (COPQ) - hypotheses assumees et labellisees
3. Objectif SMART chiffre

Ce script regroupe toute la Phase Define en un seul endroit,
pour eviter la dispersion du code entre plusieurs fichiers.
"""

import pandas as pd

# ============================================================
# 1. CHARGEMENT ET TAUX DE REJET BASELINE
# ============================================================
labels = pd.read_csv(
    "data/raw/secom_labels.data",
    sep=" ",
    header=None,
    names=["label", "timestamp"]
)
labels["timestamp"] = pd.to_datetime(
    labels["timestamp"].str.strip('"'), format="%d/%m/%Y %H:%M:%S"
)

n_total = len(labels)
n_fail = (labels["label"] == 1).sum()
n_pass = (labels["label"] == -1).sum()
fail_rate = (n_fail / n_total) * 100

print("="*55)
print("1. BASELINE QUALITE")
print("="*55)
print(f"Nombre total de lots      : {n_total}")
print(f"Nombre de lots Fail       : {n_fail}")
print(f"Nombre de lots Pass       : {n_pass}")
print(f"Taux de fail (baseline)   : {fail_rate:.2f}%")

# ============================================================
# 2. ANALYSE FINANCIERE (COPQ) - HYPOTHESES ASSUMEES
#    Le dataset SECOM ne fournit aucune donnee financiere reelle.
#    Ces valeurs sont des hypotheses sectorielles, clairement
#    labellisees comme telles (pas des donnees d'entreprise reelles).
# ============================================================
COUT_PAR_DEFAUT_EUROS = 5000   # Hypothese: matiere premiere + arret machine + reusinage
LOTS_PAR_AN = 2000             # Hypothese: volume annuel de production

cout_total_defauts = n_fail * COUT_PAR_DEFAUT_EUROS
cout_moyen_par_lot = cout_total_defauts / n_total
pertes_annuelles_estimees = cout_moyen_par_lot * LOTS_PAR_AN

print("\n" + "="*55)
print("2. ANALYSE FINANCIERE (COPQ) - hypotheses assumees")
print("="*55)
print(f"Cout par defaut (hypothese)              : {COUT_PAR_DEFAUT_EUROS:,} EUR")
print(f"Cout total des defauts (sur ce dataset)  : {cout_total_defauts:,.0f} EUR")
print(f"Perte annuelle estimee ({LOTS_PAR_AN} lots/an)    : {pertes_annuelles_estimees:,.0f} EUR")

# ============================================================
# 3. OBJECTIF SMART - reduction relative de 20% (methode Pareto)
# ============================================================
REDUCTION_VISEE = 0.20
TAUX_CIBLE = round(fail_rate * (1 - REDUCTION_VISEE), 2)
reduction_points = fail_rate - TAUX_CIBLE
nb_defauts_a_eliminer = int(n_total * (reduction_points / 100))
gain_financier = nb_defauts_a_eliminer * COUT_PAR_DEFAUT_EUROS

print("\n" + "="*55)
print("3. OBJECTIF SMART - PHASE DEFINE")
print("="*55)
print(f"Taux de fail actuel       : {fail_rate:.2f}%")
print(f"Taux de fail cible        : {TAUX_CIBLE}%  (reduction relative de {int(REDUCTION_VISEE*100)}%)")
print(f"Reduction necessaire      : {reduction_points:.2f} points de %")
print(f"Lots defectueux a eliminer: {nb_defauts_a_eliminer}")
print(f"Gain financier potentiel  : {gain_financier:,.0f} EUR")