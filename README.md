# 🏭 Système de Détection Prédictive des Défauts Qualité
### SECOM Semiconductor Dataset — Pipeline DMAIC Complet IE + Data Science + IA Explicable

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Random%20Forest-orange)
![SHAP](https://img.shields.io/badge/SHAP-Explainable%20AI-purple)
![Power BI](https://img.shields.io/badge/PowerBI-Dashboard-yellow)
![Status](https://img.shields.io/badge/Status-Completed-green)

---

## 📋 Table des matières
- [Contexte et objectif](#-contexte-et-objectif)
- [Dataset](#-dataset)
- [Architecture du projet](#️-architecture-du-projet)
- [Étape 1 — Define](#-étape-1--define)
- [Étape 2 — Measure](#-étape-2--measure)
- [Étape 3 — Analyze](#-étape-3--analyze)
- [Étape 4 — Improve](#-étape-4--improve)
- [Étape 5 — Control](#-étape-5--control)
- [Résultats et conclusions](#-résultats-et-conclusions)
- [Découvertes importantes](#-découvertes-importantes)
- [Choix techniques et justifications](#-choix-techniques-et-justifications)
- [Défis rencontrés](#️-défis-rencontrés)
- [Améliorations possibles](#-améliorations-possibles)
- [Structure finale du projet](#-structure-finale-du-projet)
- [Installation](#️-installation)

---

## 🏭 Contexte et objectif

Ce projet applique une démarche **DMAIC** (Lean Six Sigma) complète, augmentée de **Machine Learning explicable (SHAP)**, sur un cas réel de fabrication de semi-conducteurs.

### Problème industriel adressé

```
1. Taux de rejet mesure de 6.64% sur la ligne de production
2. 590 capteurs surveilles, mais aucune hierarchisation de leur impact reel
3. Aucune specification produit fournie (confidentialite industrielle)
4. Aucune donnee financiere fournie pour justifier un investissement qualite
```

### Objectifs du projet

```
✅ Quantifier precisement le taux de rejet et son cout (COPQ)
✅ Identifier statistiquement les capteurs instables (SPC + Pareto)
✅ Identifier les capteurs causalement lies au defaut (ML + SHAP + Pareto)
✅ Simuler l'impact chiffre d'une action corrective
✅ Mettre en place un systeme de surveillance perenne (Control + Power BI)
```

### Ce qui différencie ce projet des notebooks Kaggle classiques

| Notebooks Kaggle typiques | Ce projet |
|---|---|
| Un seul modèle ML, accuracy affichée telle quelle | Métriques adaptées au déséquilibre (ROC-AUC, recall) + seuil discuté |
| Feature importance brute | SHAP + comparaison croisée avec un Pareto SPC indépendant |
| Pas de contexte financier | Analyse COPQ avec hypothèses explicitement labellisées |
| Résultats jamais remis en question | 3 tentatives de simulation documentées, 2 rejetées et expliquées |
| Notebook isolé | Pipeline DMAIC complet + Dashboard Power BI interactif |
| Specs supposées disponibles | Absence de specs traitée comme contrainte réelle (Cp/Cpk vs Pp/Ppk) |

---

## 📊 Dataset

**Source :** [SECOM Dataset — UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/179/secom)

| Caractéristique | Valeur |
|---|---|
| Observations (lots de production) | 1567 |
| Capteurs (features) | 590, anonymisés (`Sensor_001` à `Sensor_590`) |
| Cible | Pass (-1) / Fail (1) |
| Taux de fail réel | 6.64% (104 fails) |
| Cellules manquantes | 4.5% du dataset |
| Durée de collecte couverte | ~89 jours |

### Contrainte majeure du dataset

```
Aucun nom physique de capteur (confidentialite industrielle)
Aucune limite de specification produit (USL/LSL) fournie
Aucune donnee de cout reelle fournie

=> Traite tout au long du projet via des hypotheses explicitement
   labellisees, jamais presentees comme des faits etablis.
```

---

## 🗂️ Architecture du projet

```
projet_lean_qualite/
│
├── 📄 Data
│   ├── data/raw/                         ← secom.data, secom_labels.data
│   └── data/processed/                   ← Tous les resultats intermediaires (.csv)
│
├── 🐍 Scripts Python (executes dans l'ordre)
│   ├── 01_define_baseline_and_copq.py    ← Baseline + COPQ + objectif SMART
│   ├── 02_data_cleaning.py               ← Nettoyage (590 → 442 capteurs)
│   ├── 03_spc_pareto.py                  ← Cartes SPC (I-MR) + Pareto #1
│   ├── 04_cpk_analysis.py                ← Cp/Cpk vs Pp/Ppk
│   ├── 05_correlation_analysis.py        ← Correlation capteur-fail + multicolinearite
│   ├── 06_random_forest_smote.py         ← Random Forest + SMOTE
│   ├── 06b_threshold_scenarios.py        ← Scenarios de seuil + sensibilite
│   ├── 07_shap_analysis.py               ← SHAP + Pareto #2
│   ├── 08_improve_simulation.py          ← Simulation d'impact (methode V3)
│   ├── 09_improve_recommandations.py     ← Plan d'action differencie par capteur
│   ├── 10_control_charts.py              ← Cartes de controle perennes + regles Nelson
│   └── 11_export_kpis_powerbi.py         ← Export KPIs de synthese
│
├── 📊 reports/
│   ├── pareto_1_spc.png
│   ├── correlation_heatmap.png
│   ├── confusion_matrix.png / roc_curve.png
│   ├── threshold_cost_optimization.png
│   ├── shap_summary_plot.png / pareto_2_shap.png / shap_waterfall_exemple.png
│   ├── control_chart_exemple.png
│   └── screenshots/                      ← Captures du dashboard Power BI
│
├── 📈 dashboard/
│   ├── donnees_controle_powerbi.csv
│   └── kpis_synthese_dmaic.csv
│
└── 📝 README.md
```

---

## 🎯 Étape 1 — Define

### Méthodologie

```
Chargement des labels (1567 lots)
        ↓
Calcul du taux de fail = (Nb Fail / Nb Total) x 100
        ↓
Analyse financiere COPQ (hypotheses assumees, labellisees)
        ↓
Objectif SMART base sur le principe de Pareto (regle 80/20)
```

### Résultats obtenus

| Indicateur | Valeur |
|---|---|
| Lots Fail / Pass | 104 / 1463 |
| **Taux de fail baseline** | **6.64%** |
| Coût par défaut (hypothèse sectorielle) | 5 000 € |
| Coût total des défauts (sur le dataset) | 520 000 € |
| Perte annuelle estimée (2000 lots/an) | 663 689 € |
| **Objectif SMART (cible)** | **5.31%** (réduction relative de 20%) |
| Gain financier visé | 100 000 € |

### Formule de l'objectif SMART

```
Objectif (%) = Taux de fail baseline x (1 - Reduction visee)
Objectif = 6.64 x (1 - 0.20) = 5.31%
```

---

## 📏 Étape 2 — Measure

### Nettoyage des données

```
590 capteurs initiaux
        ↓
Suppression capteurs >40% NaN        → 32 capteurs supprimes
        ↓
Suppression capteurs variance nulle  → 116 capteurs supprimes
        ↓
Imputation des NaN restants (mediane)
        ↓
442 capteurs retenus (reduction de 25.1%)
```

### Cartes de contrôle SPC — méthode I-MR

```
Individuals - Moving Range (adaptee : une seule mesure par lot)

MR = |x_i - x_(i-1)|
Sigma estime = MR_barre / d2         (d2 = 1.128 pour n=2)

UCL = x_barre + 3 x sigma
LCL = x_barre - 3 x sigma
```

### Pareto #1 — Capteurs les plus souvent hors contrôle

| Rang | Capteur | Points hors contrôle | % |
|---|---|---|---|
| 1 | Sensor_587 | 356 | 22.7% |
| 2 | Sensor_406 | 336 | 21.4% |
| 3 | Sensor_407 | 329 | 21.0% |
| 4 | Sensor_130 | 329 | 21.0% |
| 5 | Sensor_268 | 320 | 20.4% |

**Observation clé :** contrairement à un Pareto 80/20 classique, il faut **180 capteurs sur 428** pour expliquer 80% des dépassements — l'instabilité SPC est **diffuse**, pas concentrée. Ce constat justifie à lui seul la nécessité d'une analyse complémentaire (Analyze).

---

## 🔬 Étape 3 — Analyze

### 3.1 — Capabilité process : Cp/Cpk vs Pp/Ppk

```
sigma court terme (I-MR)  → "potentiel" du process  → Cp / Cpk
sigma long terme (std)    → "performance reelle"     → Pp / Ppk

USL/LSL assumees = moyenne ± 4 x sigma_long_terme

Ecart Cpk - Ppk eleve = le capteur DERIVE dans le temps
Ecart faible           = bruit stable, pas de derive
```

| Capteur | Ratio long/court terme | Diagnostic |
|---|---|---|
| Sensor_590 | 3.44 | Forte dérive — priorité recalibration |
| Sensor_041 | 1.49 | Relativement stable |

**Tous les 20 capteurs SPC critiques ont un ratio > 1.4** → dérive réelle, pas juste du bruit aléatoire.

### 3.2 — Corrélation et multicolinéarité

| Analyse | Résultat |
|---|---|
| Corrélation max capteur↔fail (442 capteurs) | ~0.156 (faible) |
| Recoupement Top 20 SPC vs Top 20 corrélation-fail | 2 / 20 seulement |
| Paires de capteurs fortement corrélées entre elles (\|r\|>0.85) | 32 paires |
| Capteurs retenus pour le ML (union des 2 critères) | 38 |

**Conclusion :** corrélations linéaires trop faibles pour expliquer le fail seules → nécessité d'un modèle capable de capter des interactions non-linéaires.

### 3.3 — Random Forest + SMOTE

```
Train : 1175 lots (78 fails)  |  Test : 392 lots (26 fails)
SMOTE applique UNIQUEMENT sur le train (evite la fuite de donnees)
Apres SMOTE : 2194 lots (1097 fails / 1097 pass)
```

| Métrique (seuil 0.5) | Valeur |
|---|---|
| ROC-AUC | 0.785 |
| Recall (Fail) | 27% |
| Precision (Fail) | 28% |
| Accuracy (trompeuse) | 91% |

### 3.4 — Scénarios de seuil de décision

| Scénario | Seuil | Recall | % lots signalés |
|---|---|---|---|
| Conservateur (défaut) | 0.50 | 27% | faible |
| **Équilibré (retenu)** | **0.30** | **65%** | **26%** |
| Sensible | 0.20 | élevé | modéré |
| Très sensible | 0.15 | 96% | 58% (irréaliste) |

**Analyse de sensibilité (seuil optimal selon coût fausse alerte assumé) :**

| Coût fausse alerte | Seuil "optimal" |
|---|---|
| 200 € | 0.15 |
| 500 € | 0.30 |
| 1000 € | 0.45 |
| 2000 € | 0.60 |

→ Le seuil n'est **pas une vérité mathématique fixe**, c'est un arbitrage business.

### 3.5 — SHAP (TreeExplainer) et Pareto #2

```
prediction(x) = valeur_de_base + somme(valeurs SHAP de tous les capteurs)
```

| Rang | Capteur | Importance SHAP |
|---|---|---|
| 1 | Sensor_060 | 0.0587 |
| 2 | Sensor_104 | 0.0313 |
| 3 | Sensor_041 | 0.0313 |
| 4 | Sensor_131 | 0.0267 |
| 5 | Sensor_122 | 0.0253 |

**20 capteurs sur 38 expliquent 80%** de l'importance SHAP totale — Pareto net, contrairement au Pareto #1 (SPC).

**Résultat le plus important du projet : 0 capteur en commun entre le Top 10 Pareto #1 (SPC) et le Top 10 Pareto #2 (SHAP).** L'instabilité statistique et l'impact réel sur le fail sont deux informations différentes et complémentaires.

---

## 🔧 Étape 4 — Improve

### Simulation d'impact — 3 tentatives, 1 retenue

```
V1 - Correction simultanee de 10 capteurs (valeurs mediane)
     → taux de fail predit EN HAUSSE (26.02% → 32.14%) — REJETEE
       Cause : combinaisons hors distribution d'entrainement

V2 - Neutralisation des contributions SHAP (mise a zero)
     → resultat encore pire (jusqu'a 83%) — REJETEE
       Cause : erreur conceptuelle (effet protecteur supprime a tort)

V3 - Correction INDIVIDUELLE, un capteur a la fois — RETENUE
     → Sensor_060 seul : 26.02% → 20.66% (test set)
     → 25 lots sur 392 basculent de Fail predit a Pass predit
```

### Projection financière (méthode prudente)

| Indicateur | Valeur |
|---|---|
| Taux de fail baseline | 6.64% |
| Nouveau taux de fail estimé | **5.27%** (cohérent avec l'objectif SMART 5.31%) |
| Lots défectueux évités / an | ~27 |
| **Gain financier annuel estimé** | **136 779 €** |

### Plan d'action différencié (Top 10 SHAP)

| Type d'action | Nombre de capteurs | Exemple |
|---|---|---|
| Resserrement des limites de contrôle | 6 / 10 | Sensor_060 (ratio dérive 1.77) |
| Recalibration périodique | 4 / 10 | Sensor_131, 122, 117, 125 |

---

## 📡 Étape 5 — Control

### Règles d'alerte (Nelson, 2 premières règles)

```
Regle 1 : 1 point au-dela de 3 sigma (deja utilisee en SPC)
Regle 2 : 9 points consecutifs du meme cote de la moyenne (derive precoce)
```

| Indicateur | Valeur |
|---|---|
| Capteurs sous surveillance pérenne | 10 (Top SHAP) |
| Total alertes détectées (Règle 1 + 2) | 4 951 |
| Capteur le plus alerté | Sensor_117 (913 alertes) |
| Lignes exportées pour Power BI | 15 670 |

**Observation :** Sensor_117 est le plus alerté en fréquence, mais **pas** le plus important en SHAP (Sensor_060) — confirme une dernière fois la distinction fréquence d'alerte ≠ impact réel.

### Dashboard Power BI

Cartes KPI (baseline, objectif, AUC, gain financier) + carte de contrôle interactive filtrable par capteur + Pareto des alertes + table de suivi des lots à risque.

![Vue d'ensemble du dashboard](reports/screenshots/dashboard_overview.png)
![Carte de contrôle interactive](reports/screenshots/dashboard_control_chart.png)
![Pareto des alertes](reports/screenshots/dashboard_pareto_alertes.png)

---

## 📈 Résultats et conclusions

| Indicateur | Avant projet | Après projet |
|---|---|---|
| Taux de fail | 6.64% (non expliqué) | 5.27% estimé (capteur prioritaire identifié) |
| Capteurs surveillés | 590 (aucune priorisation) | 10 capteurs prioritaires, plan d'action différencié |
| Détection du risque | Aucune | Modèle ROC-AUC=0.785, seuil ajustable selon business |
| Coût qualité | Inconnu | ~663 689 €/an estimé, 136 779 €/an de gain visé |

---

## 💡 Découvertes importantes

### Découverte 1 — Instabilité statistique ≠ impact réel sur le défaut

```
0 capteur commun entre Top 10 Pareto SPC (Measure) et Top 10 SHAP (Analyze)

Lecon : un capteur peut beaucoup varier sans jamais causer de fail,
et inversement, un capteur stable peut etre determinant.
```

### Découverte 2 — Tous les capteurs critiques dérivent dans le temps

```
Les 20 capteurs SPC critiques ont TOUS un ratio sigma_long/sigma_court > 1.4
→ Le probleme n'est pas que du bruit aleatoire, ce sont de vraies derives
  (recalibration/maintenance preventive justifiee, pas juste surveillance)
```

### Découverte 3 — Corrélations linéaires trop faibles, interactions nécessaires

```
Correlation max capteur-fail : ~0.156 seulement
→ Justifie le recours au Random Forest + SHAP plutot qu'une
  analyse de correlation seule
```

### Découverte 4 — Anomalie de comportement sur Sensor_060 (capteur #1 SHAP)

```
Carte de controle montre une zone tres instable entre les lots 0-300,
puis une stabilisation nette apres le lot ~300
→ Suggere un evenement reel (recalibration, changement de composant)
  survenu en cours de collecte des donnees
```

---

## 🔧 Choix techniques et justifications

### Pourquoi I-MR plutôt que X-bar/R pour le SPC ?

```
X-bar/R  → necessite des sous-groupes de mesures (plusieurs pieces/heure)
I-MR     → adapte aux mesures individuelles (1 valeur par lot)

Notre dataset : 1 mesure par capteur par lot → I-MR est la methode correcte
```

### Pourquoi la distinction Cp/Cpk vs Pp/Ppk ?

```
Sans specifications produit reelles, un Cp/Cpk classique base sur les
memes donnees que ses propres limites devient soit biaise (percentiles),
soit tautologique (meme sigma des deux cotes de la formule).

La distinction court terme (I-MR) / long terme (ecart-type classique)
resout ce probleme ET revele une information utile : la derive.
```

### Pourquoi SMOTE uniquement sur le train set ?

```
Appliquer SMOTE avant le split train/test creerait des versions
synthetiques d'une meme observation dans le train ET le test
→ fuite de donnees → resultats faussement excellents et non fiables
```

### Pourquoi pas un seuil de décision unique "optimal" ?

```
Recall et precision sont mathematiquement antagonistes.
Un seuil "optimal" par minimisation de cout depend entierement
d'hypotheses de cout non verifiees (ex: cout d'une fausse alerte).

Presenter plusieurs scenarios + une analyse de sensibilite est plus
honnete et plus utile a un vrai decideur qualite.
```

### Pourquoi corriger les capteurs UN A LA FOIS dans la simulation Improve ?

```
Corriger plusieurs capteurs correles simultanement cree des combinaisons
de valeurs jamais vues par le modele a l'entrainement ("hors distribution"),
qui produisent des predictions non fiables (verifie empiriquement : le
taux de fail predit augmentait au lieu de baisser).

Corriger un seul capteur a la fois reste proche des donnees reelles
connues du modele → estimation plus prudente mais plus credible.
```

---

## ⚠️ Défis rencontrés

### Défi 1 — Absence de spécifications produit (Cp/Cpk)

**Problème :** le Cp/Cpk nécessite des limites de spécification (USL/LSL), absentes du dataset SECOM (anonymisé).

**Tentatives rejetées :**
```
V1 : USL/LSL = percentiles 1-99%
     → biais systematique, tous les capteurs "non capables"
V2 : USL/LSL = moyenne ± 4 sigma (meme sigma que le calcul)
     → tautologie mathematique, Cpk=1.333 partout
```

**Solution retenue :** distinction Cp/Cpk (σ court terme) vs Pp/Ppk (σ long terme) — méthode reconnue en ingénierie qualité, révèle les dérives réelles au lieu d'un artefact de calcul.

---

### Défi 2 — Déséquilibre extrême des classes (6.64% de fail)

**Problème :** un modèle naïf atteint 91-93% d'accuracy en prédisant toujours "Pass", sans détecter un seul fail.

**Solution :** SMOTE sur le train uniquement + métriques adaptées (ROC-AUC, recall, precision) au lieu de l'accuracy seule.

---

### Défi 3 — Seuil de décision arbitraire (0.5) et son coût opérationnel caché

**Problème :** au seuil par défaut, seulement 27% des fails sont détectés — insuffisant pour une simulation d'Improve fiable. Une première correction (seuil unique optimisé par coût) donnait un seuil de 0.15, signalant **58% de tous les lots** comme suspects — opérationnellement irréaliste.

**Analyse :** l'hypothèse de coût de fausse alerte (200€) était trop simpliste, ignorant la saturation de la ligne de contrôle qualité.

**Solution :** présentation de plusieurs scénarios + analyse de sensibilité, laissant le choix final à un vrai arbitrage business.

---

### Défi 4 — Simulation d'impact aberrante (Phase Improve)

**Problème :** corriger 10 capteurs critiques simultanément **augmentait** le taux de fail prédit (26% → 32%) au lieu de le réduire.

**Causes identifiées :**
```
1. Combinaisons de valeurs jamais vues a l'entrainement (hors distribution)
2. (Tentative 2) Neutraliser une contribution SHAP positive ET negative
   supprime aussi les effets protecteurs de certains capteurs pour
   certains lots
```

**Solution :** correction individuelle capteur par capteur, résultat cohérent et défendable (6.64% → 5.27% estimé).

---

### Défi 5 — Format décimal Power BI

**Problème :** Power BI en français utilise la virgule comme séparateur décimal, incompatible avec les CSV pandas standard (point décimal) → colonnes importées comme texte, agrégations ("Moyenne") indisponibles.

**Solution :** exporter tous les CSV avec `sep=";"` et `decimal=","` dans pandas.

---

## 🚀 Améliorations possibles

### Court terme

```
1. Tester d'autres modeles (XGBoost, LightGBM)
   → Comparer AUC et recall face au Random Forest actuel

2. Validation croisee stratifiee (k-fold)
   → Plus robuste qu'un split unique train/test (392 lots test seulement)

3. Affiner l'hypothese de cout de fausse alerte
   → Enqueter un vrai cout aupres d'un responsable qualite terrain
```

### Moyen terme

```
4. SHAP interaction values
   → Etudier les interactions entre paires de capteurs, pas seulement
     l'importance individuelle

5. Recalibration progressive multi-capteurs
   → Tester la correction de 2-3 capteurs a la fois (au lieu de 1 ou 10),
     en verifiant a chaque etape que le modele reste dans sa distribution
     d'entrainement (ex: score de distance de Mahalanobis)

6. Vraie analyse de rentabilite COPQ
   → Remplacer les hypotheses assumees par des couts reels d'usine
     si le projet etait deploye en conditions industrielles
```

### Long terme

```
7. Deploiement temps reel
   → API (Flask/FastAPI) servant les predictions du modele en continu
   → Integration avec le systeme MES/SCADA de la ligne de production

8. Boucle de reentrainement automatique
   → Reentrainer periodiquement le modele avec les nouveaux lots reels
   → Suivre la derive du modele lui-meme (model drift), pas seulement
     celle des capteurs

9. Dashboard Power BI connecte en direct
   → Remplacer les exports CSV statiques par une connexion base de
     donnees en temps reel
```

---

## 📁 Structure finale du projet

```
projet_lean_qualite/
├── data/
│   ├── raw/
│   │   ├── secom.data
│   │   ├── secom_labels.data
│   │   └── secom.names
│   └── processed/
│       ├── secom_clean.csv
│       ├── secom_labels_clean.csv
│       ├── spc_results.csv
│       ├── pareto_1_spc.csv
│       ├── cpk_results.csv
│       ├── correlation_fail.csv
│       ├── correlation_matrix_critiques.csv
│       ├── capteurs_selectionnes_RF.txt
│       ├── random_forest_model.joblib
│       ├── X_test.csv / y_test.csv
│       ├── threshold_scenarios.csv
│       ├── pareto_2_shap.csv
│       ├── improve_simulation_resultats.csv
│       ├── improve_recommandations.csv
│       └── alertes_control.csv
├── reports/
│   ├── pareto_1_spc.png
│   ├── correlation_heatmap.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── threshold_cost_optimization.png
│   ├── shap_summary_plot.png
│   ├── pareto_2_shap.png
│   ├── shap_waterfall_exemple.png
│   ├── control_chart_exemple.png
│   └── screenshots/
│       ├── dashboard_overview.png
│       ├── dashboard_control_chart.png
│       └── dashboard_pareto_alertes.png
├── dashboard/
│   ├── donnees_controle_powerbi.csv
│   └── kpis_synthese_dmaic.csv
├── 01_define_baseline_and_copq.py
├── 02_data_cleaning.py
├── 03_spc_pareto.py
├── 04_cpk_analysis.py
├── 05_correlation_analysis.py
├── 06_random_forest_smote.py
├── 06b_threshold_scenarios.py
├── 07_shap_analysis.py
├── 08_improve_simulation.py
├── 09_improve_recommandations.py
├── 10_control_charts.py
├── 11_export_kpis_powerbi.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### Prérequis

```bash
pip install pandas numpy matplotlib seaborn
pip install scikit-learn imbalanced-learn shap joblib
```

### Exécution dans l'ordre

```bash
python 01_define_baseline_and_copq.py    # Baseline + COPQ + SMART
python 02_data_cleaning.py               # Nettoyage → secom_clean.csv
python 03_spc_pareto.py                  # SPC + Pareto #1
python 04_cpk_analysis.py                # Cp/Cpk vs Pp/Ppk
python 05_correlation_analysis.py        # Correlations + multicolinearite
python 06_random_forest_smote.py         # Random Forest + SMOTE
python 06b_threshold_scenarios.py        # Scenarios de seuil
python 07_shap_analysis.py               # SHAP + Pareto #2
python 08_improve_simulation.py          # Simulation d'impact
python 09_improve_recommandations.py     # Plan d'action
python 10_control_charts.py              # Cartes de controle + Nelson
python 11_export_kpis_powerbi.py         # Export KPIs Power BI
```

### Test rapide

```python
import pandas as pd

# Verifier le Top 5 SHAP (capteurs les plus critiques)
pareto2 = pd.read_csv("data/processed/pareto_2_shap.csv")
print(pareto2.head(5))

# Verifier la simulation d'impact retenue
sim = pd.read_csv("data/processed/improve_simulation_resultats.csv")
print(sim.sort_values("reduction_points", ascending=False).head(3))
```

---

## 👤 Auteur

**Mamoun** — Étudiant Ingénieur Génie Industriel, ENIB Bizerte
Spécialisation : Qualité Industrielle, Data Science appliquée à l'IE, IA Explicable

---

## 📄 Licence

Ce projet est open source — libre d'utilisation à des fins éducatives et professionnelles.

---

*Projet réalisé dans le cadre du développement d'un portfolio en génie industriel — combinant une démarche DMAIC complète (Define, Measure, Analyze, Improve, Control), du Machine Learning appliqué à la détection de défauts (Random Forest, SMOTE, SHAP) et un dashboard de monitoring qualité (Power BI). Chaque choix méthodologique a été documenté, y compris les tentatives rejetées et les raisons de leur échec.*
