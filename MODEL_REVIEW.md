# Audit du modèle de prévision d'affluence (`ml_service.py`)

Auteur : revue senior — modèles probabilistes / workplace analytics (FR)
Date : 2026-06-16
Périmètre : `backend/app/services/ml_service.py`, `backend/app/data_generator.py`, `backend/app/models.py`
Statut : **CRITIQUE / AUDIT UNIQUEMENT — aucun code modifié.**

---

## 1) Résumé exécutif

Le module est propre, bien documenté et **fonctionnellement opérationnel** : il agrège correctement les événements RFID en affluence journalière (visiteurs distincts), matérialise les jours à zéro (point fort souvent oublié), et expose une API de prédiction lisible avec un intervalle et un niveau probabiliste. Le calcul de la moyenne mobile est correctement **décalé d'un jour** (`shift(1)`, `build_dataset` l.211-213), ce qui évite la fuite la plus évidente.

**Mais sur le plan méthodologique, le modèle n'est pas valide en l'état pour une série temporelle**, et la qualité annoncée (R²≈0.77, MAE≈1.2) est **optimiste / trompeuse** pour trois raisons structurelles :

1. **Validation par `train_test_split` aléatoire** sur des données temporelles (l.266-268) → fuite temporelle : des jours du futur servent à prédire le passé. Le R² réel en production sera plus bas.
2. **`room_id` injecté comme entier dans un KNN à distance euclidienne** (l.55-66, l.263) → la salle 2 est « à distance 1 » de la salle 3 et « à distance 10 » de la salle 12, ce qui n'a aucun sens métier. Idem pour `mois`, `jour_semaine`, `semaine_annee` traités comme variables linéaires alors qu'elles sont **cycliques** (décembre est voisin de janvier).
3. **`predict()` du pipeline et l'intervalle ne sont pas cohérents** : `predicted` vient de `pipeline.predict` (KNN pondéré distance, l.462) alors que `lower/upper/confidence` viennent d'une **moyenne non pondérée** des mêmes voisins (l.471-481). Deux estimateurs différents pour le même point → l'intervalle ne contient pas toujours la prédiction centrale.

Aucune de ces corrections ne nécessite d'enrichir le dataset : **elles sont applicables immédiatement** (encodage one-hot/cyclique, `TimeSeriesSplit`, alignement prédiction/intervalle). Les gains les plus forts en justesse métier (météo, télétravail, réunions planifiées, ponts) nécessitent en revanche d'**enrichir les données**.

Verdict : **viable comme démonstrateur, non déployable tel quel comme outil de décision.** Priorité absolue : corriger la validation temporelle et l'encodage de `room_id`.

---

## 2) Correctness technique — bugs et risques méthodologiques par sévérité

### 🔴 BLOQUANT

**B1. Validation temporelle absente — `train_test_split` aléatoire sur une série temporelle**
`train()`, l.266-268 et l.305-307.
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
```
Le split mélange aléatoirement les jours. Un point de test du 2025-03-10 peut avoir comme voisin d'entraînement le 2025-03-11. Avec `weights="distance"` et la feature `affluence_moy_mobile` (très autocorrélée), le modèle **mémorise le voisinage temporel** → métriques gonflées. La sélection de `best_k` sur ce test (l.281) est donc elle aussi biaisée.
**Correction :** utiliser `sklearn.model_selection.TimeSeriesSplit` (ou un hold-out chronologique : 80 % premiers jours en train, 20 % derniers en test, par salle). Sélectionner `k` par CV temporelle. Reporter la MAE/RMSE/R² **sur le hold-out chronologique** comme métrique officielle.

**B2. `room_id` (et variables cycliques) en entrée brute d'un KNN euclidien**
`FEATURE_COLUMNS` l.55-66 ; `X = df[FEATURE_COLUMNS].to_numpy(dtype=float)` l.263.
- `room_id` est un **identifiant catégoriel**, pas une grandeur ordonnée. Après `StandardScaler`, la distance entre salles devient arbitraire et le KNN mélange des salles de capacités/usages radicalement différents (Amphi 200 places vs salle réunion 6 places).
- `mois`, `jour_semaine`, `jour_du_mois`, `semaine_annee` sont **cycliques** : `mois=12` et `mois=1` sont voisins dans la réalité mais à distance maximale pour le scaler.
**Correction :**
  - `room_id` → **one-hot** (12 salles, parfaitement gérable) ou, mieux, remplacer par des **attributs de salle** (`capacity`, `kind`, `batiment`, `etage`) qui généralisent à de nouvelles salles. Aujourd'hui `capacity` est la seule présente (l.64) ; `kind`/`batiment` manquent alors qu'ils sont déterminants.
  - variables temporelles → **encodage cyclique sin/cos** :
    `jour_sin = sin(2π·jour_semaine/7)`, `jour_cos = cos(...)` ; idem mois/7→12, semaine→52.
  - Construire un `ColumnTransformer` (OneHot pour catégoriel, sin/cos pour cyclique, StandardScaler pour numérique) plutôt qu'un `StandardScaler` global.

**B3. Prédiction centrale et intervalle de confiance incohérents**
`predict_room_day`, l.462 vs l.471-481.
- `central = pipeline.predict(x)` → moyenne **pondérée par distance** des voisins.
- `moyenne = np.mean(voisins)` → moyenne **non pondérée** des mêmes voisins, qui sert à `lower`, `upper`, `confidence` ET à `predicted_mean` (l.490).
- `predicted` (l.483) est arrondi depuis `central`, mais l'intervalle est centré sur `moyenne`. Résultat : `predicted` peut tomber **hors** de `[lower, upper]`. L'utilisateur voit une prédiction et un intervalle qui se contredisent → perte de crédibilité.
**Correction :** choisir UN estimateur central. Recommandé : centrer l'intervalle sur la même quantité que `predicted`. Pour un intervalle honnête, préférer des **quantiles empiriques des voisins** (`np.percentile(voisins, [10, 90])`) plutôt que `±1.96σ` qui suppose une normalité fausse sur des comptages bornés/asymétriques.

### 🟠 MAJEUR

**M1. Intervalle `±1.96σ` non valide sur des comptages**
l.474-475. L'affluence est un **comptage positif, borné, souvent asymétrique** (beaucoup de jours pleins, queue à gauche). Le `±1.96σ` gaussien produit des bornes irréalistes (et `lower` est clampé à 0, ce qui déforme la couverture). De plus `std = np.std(voisins)` est l'écart-type de la **population des voisins**, pas l'incertitude de la prédiction.
**Correction :** intervalle par **quantiles empiriques des k voisins** (p10/p90 ou p5/p95), naturellement borné et asymétrique. Documenter le niveau de couverture visé et **le vérifier** (taux de couverture observé sur hold-out, voir M4).

**M2. `_build_features_for` rappelle `build_dataset(db)` à CHAQUE prédiction**
l.390. `build_dataset` recharge **tous** les événements et reconstruit toute la grille salle×jour à chaque appel unitaire — y compris depuis `predict_affluence_level` (l.523) qui le rappelle une 2e fois dans le même `predict_room_day` (l.500). Coût O(tout l'historique) par prédiction. Non tenable pour un dashboard ou des prédictions en batch.
**Correction :** calculer/mettre en cache `affluence_moy_mobile` par salle (table de features ou cache mémoire), ou passer le dataset déjà construit. Mutualiser l'appel entre régression et classification.

**M3. Pas de calibration des probabilités du classifieur**
`predict_affluence_level`, l.530 utilise `predict_proba` brut d'un `KNeighborsClassifier`. Avec `weights="distance"` et petit k, les probas KNN sont **mal calibrées** (souvent 0/1 tranchés). Les afficher comme « probabilité d'affluence forte » est trompeur.
**Correction :** `CalibratedClassifierCV` (méthode `isotonic` ou `sigmoid`) entraîné sur un pli temporel ; vérifier la calibration (courbe de fiabilité / Brier score).

**M4. Aucune métrique d'incertitude/couverture évaluée**
Le module produit un intervalle mais ne **mesure jamais** s'il est crédible. Un intervalle « 95 % » qui ne couvre la vérité que 60 % du temps est pire qu'inutile.
**Correction :** ajouter, sur le hold-out temporel, le **taux de couverture empirique** de `[lower, upper]` et la **largeur moyenne** ; viser couverture ≈ niveau nominal. Ajouter aussi le **pinball loss** si passage en quantiles.

**M5. Seuils de niveaux calculés sur tout le dataset → fuite dans le classifieur**
`_niveaux_from_target` l.352-370 calcule les terciles sur **l'ensemble** des jours (train+test confondus), puis l.305-307 split aléatoire. Les bornes de classes encodent donc de l'information du test.
**Correction :** calculer les terciles **sur le train uniquement** (après split temporel), puis appliquer au test.

### 🟡 MINEUR / ROBUSTESSE

- **m1. `best_k` peut différer entre régression et classification mais le clf réutilise `best['k']`** (l.311) sans propre sélection — acceptable mais à documenter.
- **m2. `np.std` sans `ddof=1`** (l.472) : biais sur petit k ; mineur.
- **m3. `confidence = 1/(1+std/moyenne)`** (l.480-481) est une heuristique non bornée métier (sensible quand `moyenne→0`, jours vides). Préférer une confiance dérivée de la **largeur d'intervalle relative à la capacité**.
- **m4. `est_vacances` / jours fériés codés en dur** (l.109-124) : approximation nationale, ignore les **zones académiques A/B/C** (essentiel en FR multi-sites Paris/Lyon/Marseille — voir §3).
- **m5. `MLData.feature_3` = durée de présence simulée aléatoire** (`data_generator` l.394) : bruit pur, ne pas utiliser comme feature.
- **m6. `trained_at` via `datetime.utcnow()`** (l.328, déprécié en Python récent) ; cosmétique.
- **m7. Le bundle stocke tout `X_train`/`y_train`** (l.324-325) : OK pour 12 salles × ~180 jours, mais ne scalera pas ; prévoir un référentiel séparé.

---

## 3) Paramètres additionnels recommandés (réalisme métier FR)

Inspiré des pratiques des acteurs FR d'affluence / workplace analytics (ex. occupation de bâtiments, prévision de fréquentation, flex office). Hiérarchisé par rapport coût/bénéfice. **Légende coût** : ⚙ faible (données déjà là), ⚙⚙ moyen (source externe simple), ⚙⚙⚙ élevé (intégration tierce).

| Paramètre | Justification métier FR | Coût implémentation | Gain attendu |
|---|---|---|---|
| **Attributs de salle** (`kind`, `batiment`, `etage`) | Un amphi et une salle de réunion n'ont pas la même dynamique ; permet de généraliser à une salle neuve sans historique | ⚙ (déjà en base, `ROOMS_SPEC`) | **Élevé** — corrige B2, meilleure généralisation |
| **Encodage cyclique temporel** (sin/cos) | Décembre↔janvier, dimanche↔lundi voisins ; respecte la réalité des cycles | ⚙ (calcul pur) | **Élevé** — corrige B2 |
| **Affluence N-1 (même jour semaine dernière)** | Forte autocorrélation hebdomadaire des bureaux ; lundi ressemble au lundi précédent | ⚙ (dérivable de l'historique) | **Élevé** |
| **Veille/lendemain de férié + détection de pont** | Le « pont » (jeudi férié → vendredi désert) est un effet FR massif non capté par `est_ferie` seul | ⚙ (calendrier) | **Élevé** |
| **Tendance / effet rentrée** (jours depuis rentrée sept., index de tendance) | Reprise de septembre, montée/descente de charge saisonnière | ⚙ | Moyen-élevé |
| **Zone de vacances scolaires A/B/C** | Paris (zone C), Lyon (A), Marseille (B) n'ont pas les mêmes dates → `est_vacances` national est faux par site | ⚙⚙ (table de dates par zone) | **Élevé** (multi-sites) |
| **Taux de télétravail / jour TT** (lundi & vendredi télétravaillés) | Pratique dominante post-2020 en FR ; explique le creux structurel lun/ven | ⚙⚙ (politique RH ou inféré) | **Élevé** |
| **Réunions / événements planifiés** (agenda, résa de salle) | Une résa de 50 pers. dans l'amphi est *connue à l'avance* → la meilleure feature prédictive possible | ⚙⚙⚙ (connecteur calendrier/résa) | **Très élevé** |
| **Météo prévue** (pluie, canicule, neige, température) | Pluie ↑ présence cafétéria/intérieur ; canicule/grève ↓ présence | ⚙⚙ (API météo) | Moyen |
| **Jour de paie / fin de mois** | Léger effet sur présence et fréquentation cafétéria | ⚙ | Faible-moyen |
| **Grèves / perturbations transport** (RATP/SNCF) | Impact fort et brutal sur la présence en IDF | ⚙⚙⚙ (flux externe) | Moyen (événementiel) |
| **Capacité parking / affluence transports** | Contrainte d'accès physique au bâtiment | ⚙⚙⚙ | Faible-moyen |

**Recommandation de séquencement** : les 5 premières lignes (⚙) sont **gratuites en données** et corrigent en partie la méthodo → à faire en premier. Réunions planifiées et zones de vacances offrent le meilleur gain/effort parmi les enrichissements externes.

---

## 4) Nouvelles cibles / sorties probabilistes utiles

Au-delà du « nombre de personnes / salle / jour », ce qui rend un outil d'affluence réellement exploitable :

1. **Prévision horaire (par créneau)** — pas seulement journalière. Les données existent : `Event.timestamp` porte l'heure (`data_generator` génère des pics 8-10h / 12-14h / 17-19h). Agréger par (salle, jour, **tranche horaire**) débloque la sortie ci-dessous. *Faisable avec données actuelles.*
2. **Pic d'occupation simultanée** — différent du cumul journalier de visiteurs distincts. Nécessite de reconstruire la **présence instantanée** (entrées − sorties au fil de la journée). Métrique reine pour la sécurité incendie / jauge / flex office. *Faisable avec données actuelles* (entrées + sorties cohérentes sont générées).
3. **Taux de remplissage prédit = affluence / capacity** — sortie directement actionnable (vert/orange/rouge), plus parlante qu'un nombre brut. `capacity` est déjà disponible. *Faisable immédiatement.*
4. **Heure de pointe prévue** — quand le pic survient (utile planning ménage, clim, staff cafétéria). *Faisable* après agrégation horaire (point 1).
5. **Détection d'anomalies** — écart fort prédiction vs réel (panne lecteur, événement non planifié, jour atypique). Le schéma `Prediction.predicted_anomaly` existe déjà (`models.py` l.140) mais n'est **pas alimenté**. *Faisable.*
6. **No-show rate** (réunions/résas) — taux de salles réservées mais non occupées. **Nécessite la donnée de réservation** (absente aujourd'hui). *Nécessite enrichissement.*
7. **Sortie probabiliste honnête** — remplacer `±1.96σ` par des **quantiles** (p10/p50/p90) cohérents avec la prédiction centrale, avec couverture vérifiée (§M1/M4). *Faisable immédiatement.*

---

## 5) Roadmap d'amélioration priorisée

### Phase 0 — Correctifs méthodo (données actuelles, aucune nouvelle source)
1. **Validation temporelle** : `TimeSeriesSplit` / hold-out chronologique par salle ; sélection de `k` et métriques officielles dessus (corrige **B1, M5**).
2. **Encodage** : `ColumnTransformer` = one-hot `room_id` (+ `kind`/`batiment`) + sin/cos temporel + scaler numérique (corrige **B2**).
3. **Cohérence prédiction/intervalle** : un seul estimateur central ; intervalle par **quantiles des voisins** ; aligner `predicted` ∈ `[lower, upper]` (corrige **B3, M1**).
4. **Évaluation de l'incertitude** : taux de couverture + largeur moyenne sur hold-out (corrige **M4**).
5. **Performance** : mutualiser/cacher `build_dataset` au lieu de le rappeler à chaque prédiction (corrige **M2**).

### Phase 1 — Features gratuites + cibles à fort impact (données actuelles)
6. Features ⚙ : **affluence N-1 hebdo**, **détection de pont**, **veille/lendemain de férié**, **tendance/rentrée**.
7. **Taux de remplissage** + **prévision horaire** + **pic d'occupation simultanée** + **heure de pointe** (§4 points 1-4) — reconstruits depuis `timestamp` entrées/sorties.
8. **Calibration** des probas du classifieur (`CalibratedClassifierCV`) (corrige **M3**).
9. Alimenter `Prediction.predicted_anomaly` (détection d'écart prédiction/réel).

### Phase 2 — Enrichissement du dataset (sources externes)
10. **Zones de vacances A/B/C** par bâtiment ; **politique télétravail** (jours TT).
11. **Réunions / réservations de salles** (la feature la plus prédictive) + **no-show rate**.
12. **Météo** prévue ; puis, si ROI prouvé, **transports/grèves** et **parking**.

### Phase 3 — Industrialisation / modèle
13. Si le volume croît, **comparer le KNN à un gradient boosting quantile** (LightGBM `objective=quantile`) qui gère nativement catégoriel + intervalles + non-linéarités et scalera mieux que le stockage de `X_train` complet (§m7).
14. Suivi de **drift** et ré-entraînement périodique ; versionner les métriques de couverture.

---

### Annexe — Faisable maintenant vs nécessite d'enrichir les données

| Faisable avec données actuelles | Nécessite d'enrichir le dataset |
|---|---|
| Validation temporelle (B1) | Météo prévue |
| One-hot `room_id` + `kind`/`batiment` (B2) | Réunions/réservations planifiées + no-show |
| Encodage cyclique sin/cos (B2) | Zones vacances A/B/C |
| Cohérence prédiction/intervalle + quantiles (B3, M1) | Taux de télétravail (politique RH) |
| Couverture/largeur d'intervalle (M4) | Grèves / transports / parking |
| Features N-1, pont, tendance, rentrée | Jour de paie (calendrier RH) |
| Prévision horaire, pic simultané, remplissage, heure de pointe | |
| Calibration du classifieur (M3) | |
| Détection d'anomalies (`predicted_anomaly`) | |
