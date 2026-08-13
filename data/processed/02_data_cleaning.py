"""
Phase MEASURE - Etape 1 : Nettoyage des donnees

Criteres d'exclusion d'un capteur (objectifs, pas arbitraires) :
- Plus de 40% de valeurs manquantes (NaN) -> imputation non fiable
- Variance = 0 -> capteur constant, aucune information exploitable
"""

import pandas as pd

# ============================================================
# 1. CHARGEMENT
# ============================================================
data = pd.read_csv("data/raw/secom.data", sep=" ", header=None)
data.columns = [f"Sensor_{i+1:03d}" for i in range(data.shape[1])]

labels = pd.read_csv(
    "data/raw/secom_labels.data", sep=" ", header=None, names=["label", "timestamp"]
)

n_obs, n_features_initial = data.shape
print("="*55)
print("ETAT INITIAL")
print("="*55)
print(f"Observations : {n_obs}")
print(f"Capteurs (features) : {n_features_initial}")

# ============================================================
# 2. CRITERE 1 : suppression des capteurs >40% NaN
# ============================================================
SEUIL_NAN = 0.40
nan_ratio = data.isna().mean()
capteurs_trop_nan = nan_ratio[nan_ratio > SEUIL_NAN].index.tolist()

data_step1 = data.drop(columns=capteurs_trop_nan)

print("\n" + "="*55)
print(f"CRITERE 1 : suppression capteurs >{int(SEUIL_NAN*100)}% NaN")
print("="*55)
print(f"Capteurs supprimes : {len(capteurs_trop_nan)}")
print(f"Capteurs restants  : {data_step1.shape[1]}")

# ============================================================
# 3. CRITERE 2 : suppression des capteurs a variance nulle
# ============================================================
variance = data_step1.var()
capteurs_variance_nulle = variance[variance == 0].index.tolist()

data_step2 = data_step1.drop(columns=capteurs_variance_nulle)

print("\n" + "="*55)
print("CRITERE 2 : suppression capteurs a variance nulle")
print("="*55)
print(f"Capteurs supprimes : {len(capteurs_variance_nulle)}")
print(f"Capteurs restants  : {data_step2.shape[1]}")

# ============================================================
# 4. IMPUTATION DES NaN RESTANTS (mediane, robuste aux outliers)
# ============================================================
nan_avant_imputation = data_step2.isna().sum().sum()
data_clean = data_step2.fillna(data_step2.median())
nan_apres_imputation = data_clean.isna().sum().sum()

print("\n" + "="*55)
print("IMPUTATION DES VALEURS MANQUANTES RESTANTES (mediane)")
print("="*55)
print(f"NaN avant imputation : {nan_avant_imputation}")
print(f"NaN apres imputation : {nan_apres_imputation}")

# ============================================================
# 5. RESUME FINAL
# ============================================================
print("\n" + "="*55)
print("RESUME DU NETTOYAGE")
print("="*55)
print(f"Capteurs initiaux         : {n_features_initial}")
print(f"Capteurs supprimes (NaN)  : {len(capteurs_trop_nan)}")
print(f"Capteurs supprimes (var=0): {len(capteurs_variance_nulle)}")
print(f"Capteurs finaux           : {data_clean.shape[1]}")
print(f"Reduction dimensionnelle  : {(1 - data_clean.shape[1]/n_features_initial)*100:.1f}%")

# ============================================================
# 6. SAUVEGARDE
# ============================================================
data_clean.to_csv("data/processed/secom_clean.csv", index=False)
labels.to_csv("data/processed/secom_labels_clean.csv", index=False)
print("\nFichiers sauvegardes dans data/processed/")