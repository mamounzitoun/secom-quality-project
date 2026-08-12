"""
Etape 1 - Phase DEFINE
Objectif : quantifier le taux de rejet (fail rate) actuel du processus.

Formule utilisee (indicateur qualite standard) :
    Taux de fail (%) = (Nb lots Fail / Nb total lots) x 100
"""

from mimetypes import init

import pandas as pd

labels = pd.read_csv(
    "data/raw/secom_labels.data",
    sep=" ",
    header=None,
    names=["label", "timestamp"]
)

n_total = len(labels)
n_fail = (labels["label"] == 1).sum()
n_pass = (labels["label"] == -1).sum()

fail_rate = (n_fail / n_total) * 100
pass_rate = (n_pass / n_total) * 100

print(f"Nombre total de lots      : {n_total}")
print(f"Nombre de lots Fail       : {n_fail}")
print(f"Nombre de lots Pass       : {n_pass}")
print(f"Taux de fail (baseline)   : {fail_rate:.2f}%")
print(f"Taux de pass              : {pass_rate:.2f}%")
git init
git add .
git commit -m "Define - Etape 1: calcul du taux de rejet baseline (6.64%)"

git remote add origin https://github.com/mamounzitoun/secom-quality-project.git
git branch -M main
git push -u origin main