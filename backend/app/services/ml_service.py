"""Service IA de prévision d'affluence par salle et par jour (KNN).

Ce module fournit un pipeline complet de Machine Learning, le plus probabiliste
possible, pour PRÉVOIR LE NOMBRE DE PERSONNES PRÉSENTES dans chaque salle pour
une journée donnée, à partir de l'historique des événements RFID.

Refonte méthodologique (audit MODEL_REVIEW.md, phase 0 + phase 1)
----------------------------------------------------------------
Cette version corrige les défauts bloquants de la version initiale :

  * **Validation TEMPORELLE** (B1, M5) : plus de `train_test_split` aléatoire.
    On effectue un hold-out chronologique (les derniers `TEST_FRACTION` jours
    servent de test) ; `k` est sélectionné sur ce hold-out et les métriques
    officielles (MAE/RMSE/R2) sont mesurées dessus — honnêtes, sans fuite.
  * **Encodage correct** (B2) via un `ColumnTransformer`/`Pipeline` :
    - `room_id`, `kind`, `batiment`, `etage` → one-hot ;
    - `jour_semaine`, `mois`, `jour_du_mois`, `semaine_annee` → encodage
      cyclique sin/cos (décembre est voisin de janvier) ;
    - features numériques → `StandardScaler`.
  * **Cible propre** (point 3) : nombre de badges DISTINCTS par (salle, jour)
    sur les entrées autorisées ; les passages 'cafeteria' sont dédoublonnés
    (un passant = 1, pas N passages) ; les jours à 0 sont matérialisés ; les
    événements d'anomalie (`uid_scanne` préfixé 'ANOM:') sont exclus.
  * **Intervalle probabiliste COHÉRENT** (B3, M1, M4) : prédiction centrale =
    MÉDIANE des k voisins ; intervalle = QUANTILES EMPIRIQUES des voisins
    (p10/p90 par défaut), borné à [0, capacity]. La prédiction centrale est
    toujours dans l'intervalle. Le taux de COUVERTURE de l'intervalle est
    mesuré sur le hold-out temporel.
  * **Classifieur de niveau** (M3, M5) : terciles calculés sur le TRAIN
    uniquement (corrige la fuite) puis appliqués au test ; `predict_proba`
    CALIBRÉ via `CalibratedClassifierCV`.
  * **Features métier gratuites** (phase 1) : affluence N-1 même jour de
    semaine, veille/lendemain de férié, détection de pont, index de tendance.
  * **Performance** (M2) : le dataset construit est mis en cache mémoire (clé =
    nombre d'événements) ; il n'est plus reconstruit à chaque prédiction.

Utilisation :
    from app.services.ml_service import train, predict_room_day
    metrics = train(db)
    pred = predict_room_day(db, room_id=1, date=datetime.date(2026, 6, 17))
"""
from __future__ import annotations

import datetime as _dt
import os
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from app.models import Event, Reader, Room

# --------------------------------------------------------------------------- #
# Paramètres & chemins
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42                       # graine fixe pour reproductibilité
TEST_FRACTION = 0.20                    # 20 % des jours LES PLUS RÉCENTS en test
# k minimum élevé : indispensable pour des QUANTILES empiriques d'intervalle
# stables (p10/p90 sur 3 voisins n'a aucun sens). Compromis biais/couverture.
K_CANDIDATES = [9, 11, 15, 21, 31]      # valeurs de k testées (régression)
MOVING_AVG_WINDOW = 7                   # fenêtre (jours) de la moyenne mobile
LOWER_Q = 10                            # quantile bas de l'intervalle (%)
UPPER_Q = 90                            # quantile haut de l'intervalle (%)

_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ml_models")
MODEL_PATH = os.path.join(_MODELS_DIR, "affluence_knn.joblib")
CLF_MODEL_PATH = os.path.join(_MODELS_DIR, "affluence_knn_clf.joblib")

# Colonnes de features par nature (consommées par le ColumnTransformer).
CATEGORICAL_FEATURES: List[str] = ["room_id", "kind", "batiment", "etage"]
CYCLIC_FEATURES: Dict[str, int] = {
    "jour_semaine": 7,
    "mois": 12,
    "jour_du_mois": 31,
    "semaine_annee": 53,
}
NUMERIC_FEATURES: List[str] = [
    "est_weekend",
    "est_ferie",
    "est_vacances",
    "est_pont",
    "veille_ferie",
    "lendemain_ferie",
    "capacity",
    "affluence_moy_mobile",
    "affluence_n1_hebdo",
    "tendance",
]
# Ordre canonique des colonnes du DataFrame de features (X).
FEATURE_COLUMNS: List[str] = (
    CATEGORICAL_FEATURES + list(CYCLIC_FEATURES.keys()) + NUMERIC_FEATURES
)
TARGET_COLUMN = "nb_personnes"

# Cache mémoire du dataset construit (corrige M2 : pas de reconstruction par
# prédiction). Clé = nombre d'événements en base (invalide le cache si la base
# change). Valeur = DataFrame complet retourné par build_dataset.
_DATASET_CACHE: Dict[int, pd.DataFrame] = {}


# --------------------------------------------------------------------------- #
# Calendrier français (réutilise la même logique que le générateur)
# --------------------------------------------------------------------------- #
def _paques(an: int) -> _dt.date:
    """Calcule le dimanche de Pâques (algorithme de Butcher)."""
    a = an % 19
    b, c = divmod(an, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    m = (32 + 2 * e + 2 * i - h - k) % 7
    n = (a + 11 * h + 22 * m) // 451
    mois, jour = divmod(h + m - 7 * n + 114, 31)
    return _dt.date(an, mois, jour + 1)


def _jours_feries(an: int) -> set:
    """Retourne l'ensemble des jours fériés français pour une année."""
    paques = _paques(an)
    fixes = [
        _dt.date(an, 1, 1), _dt.date(an, 5, 1), _dt.date(an, 5, 8),
        _dt.date(an, 7, 14), _dt.date(an, 8, 15), _dt.date(an, 11, 1),
        _dt.date(an, 11, 11), _dt.date(an, 12, 25),
    ]
    mobiles = [
        paques + _dt.timedelta(days=1),    # lundi de Pâques
        paques + _dt.timedelta(days=39),   # Ascension
        paques + _dt.timedelta(days=50),   # lundi de Pentecôte
    ]
    return set(fixes + mobiles)


def _est_ferie(jour: _dt.date) -> bool:
    """Indique si la date est un jour férié français."""
    return jour in _jours_feries(jour.year)


def _est_vacances(jour: _dt.date) -> bool:
    """Indique si la date tombe dans une période de vacances scolaires."""
    m, d = jour.month, jour.day
    if m == 12 and d >= 20:
        return True
    if m == 1 and d <= 3:
        return True
    if m == 2 and 8 <= d <= 24:
        return True
    if m == 4 and 8 <= d <= 24:
        return True
    if m in (7, 8):
        return True
    if m == 10 and 19 <= d <= 31:
        return True
    return False


def _est_pont(jour: _dt.date) -> bool:
    """Détecte un pont FR : vendredi après jeudi férié, ou lundi avant mardi férié."""
    wd = jour.weekday()
    feries = _jours_feries(jour.year)
    if wd == 4 and (jour - _dt.timedelta(days=1)) in feries:
        return True
    if wd == 0 and (jour + _dt.timedelta(days=1)) in feries:
        return True
    return False


def _veille_ferie(jour: _dt.date) -> bool:
    """Indique si le LENDEMAIN est férié (veille de férié, présence souvent réduite)."""
    lendemain = jour + _dt.timedelta(days=1)
    return lendemain in _jours_feries(lendemain.year)


def _lendemain_ferie(jour: _dt.date) -> bool:
    """Indique si la VEILLE est fériée (lendemain de férié)."""
    veille = jour - _dt.timedelta(days=1)
    return veille in _jours_feries(veille.year)


# --------------------------------------------------------------------------- #
# 1) Construction du dataset
# --------------------------------------------------------------------------- #
def _room_attributs(db: Session) -> Dict[int, Dict]:
    """Retourne {room_id: {capacity, kind, batiment, etage}} pour toutes les salles."""
    return {
        r.id: {
            "capacity": int(r.capacity or 0),
            "kind": r.kind or "inconnu",
            "batiment": r.batiment or "inconnu",
            "etage": int(r.etage) if r.etage is not None else -1,
        }
        for r in db.query(Room).all()
    }


def _ajouter_features_calendaires(df: pd.DataFrame, attributs: Dict[int, Dict]) -> pd.DataFrame:
    """Ajoute toutes les features (calendaires, salle, dérivées) à la grille salle×jour.

    `df` doit contenir au minimum les colonnes ['room_id', 'date', TARGET_COLUMN]
    triées par (room_id, date). Modifie et retourne `df`.
    """
    dts = pd.to_datetime(df["date"])
    df["jour_semaine"] = dts.dt.weekday
    df["mois"] = dts.dt.month
    df["jour_du_mois"] = dts.dt.day
    df["est_weekend"] = (df["jour_semaine"] >= 5).astype(int)
    df["semaine_annee"] = dts.dt.isocalendar().week.astype(int)
    df["est_ferie"] = df["date"].map(lambda d: int(_est_ferie(d)))
    df["est_vacances"] = df["date"].map(lambda d: int(_est_vacances(d)))
    df["est_pont"] = df["date"].map(lambda d: int(_est_pont(d)))
    df["veille_ferie"] = df["date"].map(lambda d: int(_veille_ferie(d)))
    df["lendemain_ferie"] = df["date"].map(lambda d: int(_lendemain_ferie(d)))

    # Attributs de salle (one-hot en aval) — généralisent à de nouvelles salles.
    df["capacity"] = df["room_id"].map(lambda rid: attributs.get(rid, {}).get("capacity", 0))
    df["kind"] = df["room_id"].map(lambda rid: attributs.get(rid, {}).get("kind", "inconnu"))
    df["batiment"] = df["room_id"].map(lambda rid: attributs.get(rid, {}).get("batiment", "inconnu"))
    df["etage"] = df["room_id"].map(lambda rid: attributs.get(rid, {}).get("etage", -1))

    # Moyenne mobile récente d'affluence par salle (tendance), décalée d'un jour
    # pour ne PAS inclure la cible du jour courant (pas de fuite de données).
    df["affluence_moy_mobile"] = (
        df.groupby("room_id")[TARGET_COLUMN]
        .transform(lambda s: s.shift(1).rolling(MOVING_AVG_WINDOW, min_periods=1).mean())
    )
    moy_salle = df.groupby("room_id")[TARGET_COLUMN].transform("mean")
    df["affluence_moy_mobile"] = df["affluence_moy_mobile"].fillna(moy_salle)

    # Affluence N-1 hebdomadaire : même jour de la semaine précédente (lag 7),
    # décalée pour ne lire que le passé. Forte autocorrélation des bureaux.
    df["affluence_n1_hebdo"] = (
        df.groupby("room_id")[TARGET_COLUMN].transform(lambda s: s.shift(7))
    )
    df["affluence_n1_hebdo"] = df["affluence_n1_hebdo"].fillna(df["affluence_moy_mobile"])

    # Index de tendance : position relative du jour dans l'historique de la salle
    # (0 au plus ancien, 1 au plus récent) → capte une montée/descente de charge.
    df["tendance"] = (
        df.groupby("room_id")["date"].transform(lambda s: s.rank(method="dense"))
    )
    rang_max = df.groupby("room_id")["tendance"].transform("max").clip(lower=1)
    df["tendance"] = (df["tendance"] - 1) / rang_max.where(rang_max > 0, 1)

    return df


def build_dataset(db: Session) -> pd.DataFrame:
    """Agrège les événements en affluence journalière par salle (cible propre).

    L'affluence d'un jour pour une salle = nombre de BADGES DISTINCTS (visiteurs
    uniques) ayant réalisé une entrée AUTORISÉE ce jour-là. Les passages
    cafétéria ('passage') sont comptés mais DÉDOUBLONNÉS (un passant = 1, pas N
    passages). Les événements d'anomalie (`uid_scanne` préfixé 'ANOM:') sont
    exclus (vérité terrain de détection, pas d'affluence normale). Les jours
    sans visite sont matérialisés à 0.

    Args:
        db: session SQLAlchemy ouverte.

    Returns:
        DataFrame avec une ligne par (room_id, date) et les colonnes :
        FEATURE_COLUMNS + [TARGET_COLUMN, 'date'].
        DataFrame vide si aucun événement exploitable.
    """
    # Entrées + passages cafétéria autorisés, rattachés à une salle via le lecteur.
    rows = (
        db.query(
            Event.timestamp,
            Event.badge_id,
            Event.uid_scanne,
            Reader.room_id,
        )
        .join(Reader, Event.reader_mac == Reader.mac_address)
        .filter(
            Event.type_evenement.in_(["entree", "passage"]),
            Event.resultat.is_(True),
        )
        .all()
    )
    if not rows:
        return pd.DataFrame(columns=FEATURE_COLUMNS + [TARGET_COLUMN, "date"])

    raw = pd.DataFrame(rows, columns=["timestamp", "badge_id", "uid_scanne", "room_id"])
    raw = raw.dropna(subset=["room_id"])

    # Exclut les anomalies plantées (préfixe 'ANOM:'), qui ne sont pas de
    # l'affluence « normale » mais de la vérité terrain pour la détection.
    uid_str = raw["uid_scanne"].astype("string").fillna("")
    raw = raw[~uid_str.str.startswith("ANOM:")]
    if raw.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS + [TARGET_COLUMN, "date"])

    raw["date"] = pd.to_datetime(raw["timestamp"]).dt.date
    # Identité du visiteur : badge_id si présent (badge distinct), sinon UID brut.
    raw["visiteur"] = raw["badge_id"].astype("object").where(
        raw["badge_id"].notna(), raw["uid_scanne"]
    )

    # Affluence = nb de visiteurs DISTINCTS par (salle, jour). Le nunique
    # dédoublonne nativement les multiples passages cafétéria d'un même badge.
    agg = (
        raw.groupby(["room_id", "date"])["visiteur"]
        .nunique()
        .reset_index(name=TARGET_COLUMN)
    )
    agg["room_id"] = agg["room_id"].astype(int)

    attributs = _room_attributs(db)

    # Grille complète (salle x jours) pour matérialiser les jours à 0 visite
    # (week-ends, fériés) : essentiel pour que le modèle apprenne l'absence.
    frames = []
    for room_id, grp in agg.groupby("room_id"):
        d_min, d_max = grp["date"].min(), grp["date"].max()
        full = pd.DataFrame({
            "date": pd.date_range(d_min, d_max, freq="D").date,
            "room_id": room_id,
        })
        full = full.merge(grp[["date", TARGET_COLUMN]], on="date", how="left")
        full[TARGET_COLUMN] = full[TARGET_COLUMN].fillna(0).astype(int)
        frames.append(full)
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["room_id", "date"]).reset_index(drop=True)

    df = _ajouter_features_calendaires(df, attributs)

    return df[FEATURE_COLUMNS + [TARGET_COLUMN, "date"]].reset_index(drop=True)


def _get_dataset_cached(db: Session) -> pd.DataFrame:
    """Renvoie le dataset construit, mis en cache mémoire (corrige M2).

    La clé de cache est le nombre d'événements en base : si la base change
    (regénération, nouveaux events), le cache est automatiquement invalidé.
    """
    from sqlalchemy import func

    n_events = db.query(func.count(Event.id)).scalar() or 0
    cached = _DATASET_CACHE.get(n_events)
    if cached is None:
        cached = build_dataset(db)
        _DATASET_CACHE.clear()           # ne garde qu'une version en mémoire
        _DATASET_CACHE[n_events] = cached
    return cached


# --------------------------------------------------------------------------- #
# 2) Encodage : ColumnTransformer (one-hot + cyclique + scaler)
# --------------------------------------------------------------------------- #
def _encode_cyclique(arr: np.ndarray, periods: np.ndarray) -> np.ndarray:
    """Encode chaque colonne périodique en (sin, cos) selon sa période.

    Fonction de MODULE (et non closure) pour rester picklable par joblib.
    `periods` est l'ordre des colonnes (cf. CYCLIC_FEATURES).
    """
    arr = np.asarray(arr, dtype=float)
    ang = 2.0 * np.pi * arr / np.asarray(periods, dtype=float)
    return np.hstack([np.sin(ang), np.cos(ang)])


def _make_cyclic_transformer():
    """Construit le transformer d'encodage cyclique sin/cos des variables périodiques."""
    from sklearn.preprocessing import FunctionTransformer

    periods = np.array([CYCLIC_FEATURES[c] for c in CYCLIC_FEATURES])
    return FunctionTransformer(
        _encode_cyclique, validate=False, kw_args={"periods": periods}
    )


def _make_preprocessor():
    """Construit le ColumnTransformer : one-hot catégoriel + cyclique + scaler numérique."""
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    cyclic_pipe = Pipeline([
        ("cyclic", _make_cyclic_transformer()),
        ("scale", StandardScaler()),
    ])

    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("cyc", cyclic_pipe, list(CYCLIC_FEATURES.keys())),
            ("num", StandardScaler(), NUMERIC_FEATURES),
        ],
        remainder="drop",
    )


def _make_regressor(k: int):
    """Construit le pipeline complet de régression (préproc + KNN pondéré distance)."""
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.pipeline import Pipeline

    return Pipeline([
        ("prep", _make_preprocessor()),
        ("knn", KNeighborsRegressor(n_neighbors=k, weights="distance")),
    ])


# --------------------------------------------------------------------------- #
# Intervalle probabiliste par quantiles empiriques des voisins
# --------------------------------------------------------------------------- #
def _voisins_dun_point(pipeline, X_ref: pd.DataFrame, y_ref: np.ndarray,
                       x_row: pd.DataFrame) -> np.ndarray:
    """Renvoie les affluences (cibles) des k plus proches voisins d'un point.

    Les voisins sont cherchés dans l'espace TRANSFORMÉ par le préprocesseur du
    pipeline, garantissant la cohérence avec l'entraînement.
    """
    prep = pipeline.named_steps["prep"]
    knn = pipeline.named_steps["knn"]
    x_t = prep.transform(x_row)
    _, indices = knn.kneighbors(x_t)
    return np.asarray(y_ref)[indices[0]]


def _interval_from_neighbors(voisins: np.ndarray, capacity: int) -> Tuple[float, float, float]:
    """Calcule (central, lower, upper) à partir des voisins par quantiles empiriques.

    - central = MÉDIANE des voisins (estimateur central robuste) ;
    - lower/upper = quantiles empiriques (LOWER_Q / UPPER_Q) ;
    - le tout borné à [0, capacity] ; on garantit lower <= central <= upper.
    """
    central = float(np.median(voisins))
    lower = float(np.percentile(voisins, LOWER_Q))
    upper = float(np.percentile(voisins, UPPER_Q))

    cap = float(capacity) if capacity and capacity > 0 else None
    central = max(0.0, central)
    lower = max(0.0, lower)
    upper = max(0.0, upper)
    if cap is not None:
        central = min(central, cap)
        lower = min(lower, cap)
        upper = min(upper, cap)
    # Cohérence : la prédiction centrale est toujours dans l'intervalle.
    lower = min(lower, central)
    upper = max(upper, central)
    return central, lower, upper


# --------------------------------------------------------------------------- #
# 3) Entraînement
# --------------------------------------------------------------------------- #
def _split_temporel(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Hold-out CHRONOLOGIQUE : derniers TEST_FRACTION jours en test (anti-fuite).

    Le split porte sur les DATES (toutes salles confondues) : les jours
    strictement postérieurs au seuil constituent le test. Aucune information du
    futur ne se trouve dans le train.
    """
    dates_uniques = np.sort(df["date"].unique())
    n_test = max(1, int(round(len(dates_uniques) * TEST_FRACTION)))
    seuil = dates_uniques[-n_test]            # première date du bloc de test
    train = df[df["date"] < seuil].reset_index(drop=True)
    test = df[df["date"] >= seuil].reset_index(drop=True)
    return train, test


def _terciles_par_salle(df_train: pd.DataFrame) -> Dict[int, Tuple[float, float]]:
    """Calcule les seuils (terciles) de niveau PAR SALLE sur le TRAIN uniquement.

    Renvoie {room_id: (q33, q66)} sur les jours non nuls. Corrige la fuite M5 :
    aucune information du test n'entre dans les bornes de classes.
    """
    seuils: Dict[int, Tuple[float, float]] = {}
    for room_id, grp in df_train.groupby("room_id"):
        non_nuls = grp[grp[TARGET_COLUMN] > 0][TARGET_COLUMN]
        if len(non_nuls) < 3:
            seuils[room_id] = (0.0, 0.0)
        else:
            q1, q2 = non_nuls.quantile([0.33, 0.66])
            seuils[room_id] = (float(q1), float(q2))
    return seuils


def _label_niveau(valeur: float, seuils: Tuple[float, float]) -> str:
    """Affecte un niveau faible/moyen/fort à partir des seuils (terciles) d'une salle."""
    q1, q2 = seuils
    if valeur <= q1:
        return "faible"
    if valeur <= q2:
        return "moyen"
    return "fort"


def _niveaux_serie(df: pd.DataFrame, seuils: Dict[int, Tuple[float, float]]) -> pd.Series:
    """Applique les seuils (calculés sur le train) pour étiqueter chaque ligne."""
    return df.apply(
        lambda r: _label_niveau(r[TARGET_COLUMN], seuils.get(r["room_id"], (0.0, 0.0))),
        axis=1,
    )


def train(db: Session) -> Dict:
    """Entraîne les modèles de prévision d'affluence et les sauvegarde.

    Méthodologie (sans fuite temporelle) :
      1. Construit le dataset (cible = badges distincts/salle/jour, jours à 0).
      2. Hold-out CHRONOLOGIQUE : derniers 20 % des jours en test.
      3. Sélectionne `k` (régression) sur la MAE du hold-out temporel.
      4. Mesure MAE/RMSE/R2 + TAUX DE COUVERTURE de l'intervalle [p10, p90] sur
         le hold-out (métriques honnêtes).
      5. Ré-entraîne le pipeline gagnant sur tout l'historique pour la production.
      6. Classifieur de niveau : terciles sur le TRAIN seulement, probas
         calibrées (`CalibratedClassifierCV`).

    Args:
        db: session SQLAlchemy ouverte.

    Returns:
        Dictionnaire de métriques (hold-out temporel) : best_k, mae, rmse, r2,
        couverture & largeur d'intervalle, tailles, features, classification.

    Raises:
        ValueError: si le dataset est vide ou trop petit pour entraîner.
    """
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import (
        accuracy_score,
        mean_absolute_error,
        mean_squared_error,
        r2_score,
    )
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import Pipeline

    df = build_dataset(db)
    _DATASET_CACHE.clear()  # le dataset vient de changer : invalide le cache
    if df.empty or len(df) < 20:
        raise ValueError(
            "Dataset insuffisant pour l'entraînement : générez d'abord des "
            "événements (data_generator.generate)."
        )

    # --- Split TEMPOREL (corrige B1) --------------------------------------- #
    df_train, df_test = _split_temporel(df)
    if df_train.empty or df_test.empty:
        raise ValueError("Historique trop court pour un hold-out chronologique.")

    X_train = df_train[FEATURE_COLUMNS]
    y_train = df_train[TARGET_COLUMN].to_numpy(dtype=float)
    X_test = df_test[FEATURE_COLUMNS]
    y_test = df_test[TARGET_COLUMN].to_numpy(dtype=float)
    caps_test = df_test["capacity"].to_numpy(dtype=float)

    # --- Sélection de k sur le hold-out temporel --------------------------- #
    best = None
    for k in K_CANDIDATES:
        k_eff = min(k, len(X_train))
        pipe = _make_regressor(k_eff)
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        mae = mean_absolute_error(y_test, pred)
        if best is None or mae < best["mae"]:
            best = {"k": k_eff, "mae": float(mae), "pipeline": pipe, "pred": pred}

    pipe = best["pipeline"]
    pred = best["pred"]
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    r2 = float(r2_score(y_test, pred))

    # --- Couverture de l'intervalle [p10, p90] sur le hold-out (corrige M4) - #
    lowers, uppers = [], []
    for i in range(len(X_test)):
        voisins = _voisins_dun_point(
            pipe, X_train, y_train, X_test.iloc[[i]]
        )
        _, lo, up = _interval_from_neighbors(voisins, int(caps_test[i]))
        lowers.append(lo)
        uppers.append(up)
    lowers = np.array(lowers)
    uppers = np.array(uppers)
    couverture = float(np.mean((y_test >= lowers) & (y_test <= uppers)))
    largeur_moy = float(np.mean(uppers - lowers))

    # --- Ré-entraînement sur tout l'historique pour la production ---------- #
    X_all = df[FEATURE_COLUMNS]
    y_all = df[TARGET_COLUMN].to_numpy(dtype=float)
    final_pipe = _make_regressor(best["k"])
    final_pipe.fit(X_all, y_all)

    # --- Classifieur de niveau calibré (corrige M3, M5) -------------------- #
    seuils = _terciles_par_salle(df_train)         # TRAIN uniquement (anti-fuite)
    yc_train = _niveaux_serie(df_train, seuils).to_numpy()
    yc_test = _niveaux_serie(df_test, seuils).to_numpy()

    clf_metrics: Dict = {}
    clf_bundle = None
    if len(np.unique(yc_train)) >= 2:
        base_clf = Pipeline([
            ("prep", _make_preprocessor()),
            ("knn", KNeighborsClassifier(
                n_neighbors=min(best["k"], len(X_train)), weights="distance")),
        ])
        # Calibration des probas (sigmoïde, robuste sur petits effectifs).
        n_classes_min = min(np.bincount(
            pd.factorize(yc_train)[0]).min(), len(X_train))
        cv = max(2, min(3, int(n_classes_min)))
        clf = CalibratedClassifierCV(base_clf, method="sigmoid", cv=cv)
        clf.fit(X_train, yc_train)
        clf_metrics = {
            "accuracy": float(accuracy_score(yc_test, clf.predict(X_test))),
            "classes": [str(c) for c in clf.classes_],
            "calibration": "sigmoid",
        }
        # Ré-entraînement complet sur seuils recalculés sur tout l'historique.
        seuils_full = _terciles_par_salle(df)
        yc_all = _niveaux_serie(df, seuils_full).to_numpy()
        if len(np.unique(yc_all)) >= 2:
            clf_full = CalibratedClassifierCV(
                Pipeline([
                    ("prep", _make_preprocessor()),
                    ("knn", KNeighborsClassifier(
                        n_neighbors=min(best["k"], len(X_all)), weights="distance")),
                ]),
                method="sigmoid", cv=cv,
            )
            clf_full.fit(X_all, yc_all)
            clf_bundle = {"clf": clf_full, "seuils": seuils_full, "k": best["k"]}

    # --- Sauvegarde -------------------------------------------------------- #
    os.makedirs(_MODELS_DIR, exist_ok=True)
    joblib.dump(
        {
            "pipeline": final_pipe,
            "X_train": X_all,            # référentiel des voisins (probabiliste)
            "y_train": y_all,
            "features": FEATURE_COLUMNS,
            "k": best["k"],
            "lower_q": LOWER_Q,
            "upper_q": UPPER_Q,
            "trained_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        },
        MODEL_PATH,
    )
    if clf_bundle is not None:
        joblib.dump(clf_bundle, CLF_MODEL_PATH)
    elif os.path.exists(CLF_MODEL_PATH):
        os.remove(CLF_MODEL_PATH)

    return {
        "validation": "hold-out temporel (chronologique)",
        "best_k": best["k"],
        "mae": round(best["mae"], 3),
        "rmse": round(rmse, 3),
        "r2": round(r2, 3),
        "interval_coverage": round(couverture, 3),
        "interval_nominal": (UPPER_Q - LOWER_Q) / 100.0,
        "interval_width_mean": round(largeur_moy, 2),
        "n_samples": int(len(df)),
        "n_train": int(len(df_train)),
        "n_test": int(len(df_test)),
        "features": FEATURE_COLUMNS,
        "classification": clf_metrics,
        "model_path": MODEL_PATH,
    }


# --------------------------------------------------------------------------- #
# 4) Construction des features pour une prédiction unitaire
# --------------------------------------------------------------------------- #
def _build_features_for(db: Session, room_id: int, jour: _dt.date) -> Optional[pd.DataFrame]:
    """Construit la ligne de features (DataFrame 1×n) pour une salle et une date.

    Les features dérivées (moyenne mobile, N-1 hebdo, tendance) sont calculées à
    partir de l'historique réel de la salle, en n'utilisant QUE le passé. Le
    dataset est lu depuis le cache (corrige M2).

    Returns:
        DataFrame d'une ligne avec FEATURE_COLUMNS, ou None si la salle est inconnue.
    """
    attributs = _room_attributs(db)
    if room_id not in attributs:
        return None
    attr = attributs[room_id]

    df = _get_dataset_cached(db)
    hist = df[df["room_id"] == room_id]
    passe = hist[pd.to_datetime(hist["date"]) < pd.Timestamp(jour)].sort_values("date")

    if not passe.empty:
        recents = passe.tail(MOVING_AVG_WINDOW)
        moy_mobile = float(recents[TARGET_COLUMN].mean())
        # Affluence N-1 hebdo : même jour de semaine la semaine précédente.
        cible_date = jour - _dt.timedelta(days=7)
        ligne_n1 = passe[passe["date"] == cible_date]
        n1 = float(ligne_n1[TARGET_COLUMN].iloc[0]) if not ligne_n1.empty else moy_mobile
        tendance = 1.0  # prédiction = futur immédiat → extrémité récente
    elif not hist.empty:
        moy_mobile = float(hist[TARGET_COLUMN].mean())
        n1 = moy_mobile
        tendance = 1.0
    else:
        moy_mobile = n1 = 0.0
        tendance = 1.0

    feats = {
        "room_id": room_id,
        "kind": attr["kind"],
        "batiment": attr["batiment"],
        "etage": attr["etage"],
        "jour_semaine": jour.weekday(),
        "mois": jour.month,
        "jour_du_mois": jour.day,
        "semaine_annee": int(jour.isocalendar()[1]),
        "est_weekend": int(jour.weekday() >= 5),
        "est_ferie": int(_est_ferie(jour)),
        "est_vacances": int(_est_vacances(jour)),
        "est_pont": int(_est_pont(jour)),
        "veille_ferie": int(_veille_ferie(jour)),
        "lendemain_ferie": int(_lendemain_ferie(jour)),
        "capacity": attr["capacity"],
        "affluence_moy_mobile": moy_mobile,
        "affluence_n1_hebdo": n1,
        "tendance": tendance,
    }
    return pd.DataFrame([{c: feats[c] for c in FEATURE_COLUMNS}])


def predict_room_day(db: Session, room_id: int, date: _dt.date) -> Dict:
    """Prédit le nombre de personnes présentes dans une salle un jour donné.

    L'aspect probabiliste s'appuie sur la distribution des k plus proches
    voisins (dans l'espace transformé) : la prédiction centrale est leur
    MÉDIANE et l'intervalle leurs QUANTILES EMPIRIQUES (p10/p90), borné à
    [0, capacity]. La prédiction centrale est toujours dans l'intervalle.

    Args:
        db: session SQLAlchemy ouverte.
        room_id: identifiant de la salle.
        date: jour à prévoir (datetime.date).

    Returns:
        Dictionnaire contenant au moins :
          - predicted (int), lower (int), upper (int) : prévision + intervalle
          - confidence (float) : confiance dans [0, 1] (resserrement de l'intervalle)
          - level (str), level_proba (dict) : niveau probabiliste calibré
          - taux_remplissage (float) : predicted / capacity
          + room_id, room_nom, date, capacity, neighbors, lower_q, upper_q.
        En cas d'erreur : {"error": "..."}.
    """
    if not os.path.exists(MODEL_PATH):
        return {
            "error": "Modèle introuvable. Lancez d'abord ml_service.train(db) "
                     "pour entraîner et sauvegarder le modèle."
        }

    room = db.query(Room).filter(Room.id == room_id).first()
    if room is None:
        return {"error": f"Salle {room_id} inconnue."}

    bundle = joblib.load(MODEL_PATH)
    pipeline = bundle["pipeline"]
    X_ref = bundle["X_train"]
    y_ref = bundle["y_train"]
    capacity = int(room.capacity or 0)

    x_row = _build_features_for(db, room_id, date)
    if x_row is None:
        return {"error": f"Impossible de construire les features pour la salle {room_id}."}

    # Distribution des voisins → prédiction centrale (médiane) + intervalle (quantiles).
    voisins = _voisins_dun_point(pipeline, X_ref, y_ref, x_row)
    central, lower, upper = _interval_from_neighbors(voisins, capacity)
    predicted = int(round(central))

    # Confiance : intervalle d'autant plus resserré (relativement à la capacity)
    # que les voisins sont concordants. Bornée à [0, 1].
    echelle = float(capacity) if capacity > 0 else max(1.0, float(np.max(voisins) + 1))
    largeur_rel = (upper - lower) / echelle
    confidence = float(np.clip(1.0 - largeur_rel, 0.0, 1.0))

    taux = round(predicted / capacity, 3) if capacity > 0 else None

    result: Dict = {
        "room_id": room_id,
        "room_nom": room.nom,
        "date": date.isoformat(),
        "predicted": predicted,
        "lower": int(round(lower)),
        "upper": int(round(upper)),
        "confidence": round(confidence, 3),
        "capacity": capacity,
        "taux_remplissage": taux,
        "neighbors": [int(round(v)) for v in voisins],
        "lower_q": LOWER_Q,
        "upper_q": UPPER_Q,
    }

    # Niveau d'affluence probabiliste calibré (si classifieur entraîné).
    niveau = predict_affluence_level(db, room_id, date)
    if "error" not in niveau:
        result["level"] = niveau["level"]
        result["level_proba"] = niveau["level_proba"]

    return result


# --------------------------------------------------------------------------- #
# 5) Classification probabiliste calibrée du niveau d'affluence
# --------------------------------------------------------------------------- #
def predict_affluence_level(db: Session, room_id: int, date: _dt.date) -> Dict:
    """Prédit le niveau d'affluence (faible/moyen/fort) avec probabilités CALIBRÉES.

    S'appuie sur le `CalibratedClassifierCV` sauvegardé lors de l'entraînement
    (probas fiables, contrairement au predict_proba KNN brut).

    Returns:
        {"level": str, "level_proba": {classe: proba}} ou {"error": "..."}.
    """
    if not os.path.exists(CLF_MODEL_PATH):
        return {"error": "Classifieur de niveau indisponible (non entraîné)."}

    x_row = _build_features_for(db, room_id, date)
    if x_row is None:
        return {"error": f"Salle {room_id} inconnue."}

    bundle = joblib.load(CLF_MODEL_PATH)
    clf = bundle["clf"]
    classes = clf.classes_
    probas = clf.predict_proba(x_row)[0]
    level = str(classes[int(np.argmax(probas))])
    return {
        "room_id": room_id,
        "date": date.isoformat(),
        "level": level,
        "level_proba": {str(c): round(float(p), 3) for c, p in zip(classes, probas)},
    }
