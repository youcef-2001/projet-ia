"""Générateur de dataset réaliste pour le projet IoT RFID (Smart Campus).

Peuple la base avec un historique volumineux (~6 mois) d'accès RFID simulant
un environnement de travail professionnel français, exploitable pour entraîner
un modèle ML de prévision d'affluence PAR SALLE et de détection d'anomalies.

Principes de modélisation (refonte « par salle / par individu ») :
  - Génération centrée SALLE/CRÉNEAU : pour chaque salle et chaque heure ouvrée,
    les arrivées suivent une loi de Poisson d'intensité dépendant du type de
    salle (`kind`), de l'heure et du jour. L'occupation instantanée est suivie
    et plafonnée à `capacity` (les arrivées excédentaires deviennent no-show).
  - Profils horaires PAR TYPE de salle (`ROOM_PROFILES`) : cloche pour les
    open spaces, créneaux discrets pour meeting/amphi, double pic pour cafétéria.
  - Habitudes INDIVIDUELLES stables : chaque user a une salle de rattachement,
    des heures personnelles récurrentes, des jours sur site vs télétravail, et
    des blocs de congés/RTT multi-jours. Plus de présence i.i.d.
  - Saisonnalité française fine : ponts, creux estival progressif, rentrée.
  - Anomalies étiquetées plantées (badge cloné, accès nocturne, rafale de
    refus, anti-passback, volume aberrant) → vérité terrain pour la détection.

Le module est idempotent (reset=True) et reproductible (graine fixe `SEED`).

Lancement :
    docker compose exec backend python -m app.data_generator
    # ou en code :
    from app.data_generator import generate
    generate(reset=True, months=6)
"""
from __future__ import annotations

import math
import random
from datetime import date, datetime, time, timedelta
from typing import Dict, List, Optional, Tuple

from sqlalchemy import func

from .database import Base, SessionLocal, engine
from .models import Event, MLData, RFIDBadge, Reader, Room, User

# --------------------------------------------------------------------------- #
# Paramètres exposés (modifiables sans toucher au reste du code)
# --------------------------------------------------------------------------- #
SEED = 42                     # graine random pour reproductibilité
NB_USERS = 60                 # nombre d'utilisateurs générés
NB_MONTHS = 6                 # profondeur d'historique par défaut (mois)

# Répartition réaliste des types d'utilisateurs (poids relatifs)
TYPE_WEIGHTS: Dict[str, float] = {
    "employe": 0.50,
    "etudiant": 0.35,
    "admin": 0.05,
    "visiteur": 0.10,
}

PROBA_SECOND_BADGE = 0.20     # proba qu'un user ait un 2e badge
PROBA_BADGE_INACTIF = 0.08    # proba qu'un badge soit inactif
GEN_ML_DATA = True            # génère aussi des lignes MLData dérivées
BULK_CHUNK = 5000             # taille des lots d'insertion bulk

# Taux de présence sur les jours « éligibles » (sur site, hors congé), par type.
# Remplace l'ancienne Bernoulli i.i.d. PROBA_PRESENCE_JOUR.
PRESENCE_PAR_TYPE: Dict[str, float] = {
    "employe": 0.92,
    "etudiant": 0.78,
    "admin": 0.95,
    "visiteur": 0.0,   # géré séparément (venues ponctuelles)
}

# Modèle télétravail français : jours typiquement sur site selon le type.
# (0=lundi .. 4=vendredi). Chaque employé tire ensuite SON propre sous-ensemble.
JOURS_SITE_BASE: Dict[str, List[int]] = {
    "employe": [1, 3],          # socle mardi + jeudi, +jours aléatoires
    "etudiant": [0, 1, 2, 3, 4],
    "admin": [0, 1, 2, 3, 4],
}

# Profils horaires par TYPE de salle (résout C1/C2).
#   slots     : créneaux discrets (heure, minute) pour meeting/amphi/cafeteria
#   arr_peak  : (mu, sigma) heure d'arrivée pour openspace
#   dep_peak  : (mu, sigma) heure de départ pour openspace
#   fill      : (min, max) taux de remplissage de la capacity
#   dwell_min : (min, max) durée de présence en minutes
#   no_show   : proba qu'une arrivée prévue ne se produise pas
#   open      : (h_ouverture, h_fermeture)
ROOM_PROFILES: Dict[str, Dict] = {
    "meeting": {
        "slots": [(9, 0), (10, 30), (14, 0), (16, 0)],
        "fill": (0.40, 0.80), "dwell_min": (30, 90),
        "no_show": 0.15, "open": (8, 19),
    },
    "openspace": {
        "arr_peak": (8.5, 0.8), "dep_peak": (17.8, 0.9),
        "fill": (0.45, 0.75), "dwell_min": (360, 540),
        "no_show": 0.05, "open": (7, 20),
    },
    "cafeteria": {
        "slots": [(12, 0), (12, 30), (13, 0), (13, 30)],
        "fill": (0.25, 0.60), "dwell_min": (20, 40),
        "no_show": 0.05, "open": (11, 15),
    },
    "amphi": {
        "slots": [(8, 0), (10, 15), (13, 30), (15, 45)],
        "fill": (0.10, 0.90), "dwell_min": (90, 105),
        "no_show": 0.20, "open": (8, 18),
    },
}

# Décalage des services de déjeuner par site (minutes) — résout M4.
LUNCH_OFFSET_PAR_SITE: Dict[str, int] = {
    "Paris-Nord": 0,
    "Lyon-Tech": 10,
    "Marseille-Sud": 20,
}

TAUX_ANOMALIES = 0.003        # densité d'anomalies plantées (rapport au volume)

# Prénoms / noms français pour des identités plausibles (doublons dédupliqués)
_PRENOMS = [
    "Alice", "Bob", "Chloe", "David", "Emma", "Lucas", "Lea", "Hugo", "Manon",
    "Nathan", "Camille", "Louis", "Sarah", "Jules", "Ines", "Gabriel", "Jade",
    "Raphael", "Louise", "Arthur", "Paul", "Anna", "Tom", "Eva",
    "Theo", "Lina", "Adam", "Rose", "Maxime", "Julie", "Antoine", "Clara",
    "Romain", "Marie", "Pierre", "Sophie", "Nicolas", "Laura", "Mehdi",
]
_NOMS = [
    "Durand", "Martin", "Petit", "Roux", "Bernard", "Dubois", "Thomas",
    "Robert", "Richard", "Moreau", "Laurent", "Simon", "Michel", "Lefebvre",
    "Leroy", "Garcia", "David", "Bertrand", "Morel", "Fournier", "Girard",
    "Bonnet", "Dupont", "Lambert", "Fontaine", "Rousseau", "Vincent", "Muller",
    "Faure", "Andre", "Mercier", "Blanc", "Guerin", "Boyer", "Garnier",
]

# Salles réparties sur 3 sites/bâtiments, capacités et usages variés
ROOMS_SPEC: List[Dict] = [
    # batiment, nom, etage, capacity, kind, nb_readers
    {"nom": "Salle réunion Aristote", "batiment": "Paris-Nord", "etage": 1, "capacity": 8, "kind": "meeting", "readers": 1},
    {"nom": "Salle réunion Newton", "batiment": "Paris-Nord", "etage": 1, "capacity": 12, "kind": "meeting", "readers": 1},
    {"nom": "Open Space Atlas", "batiment": "Paris-Nord", "etage": 2, "capacity": 60, "kind": "openspace", "readers": 2},
    {"nom": "Open Space Mercure", "batiment": "Paris-Nord", "etage": 3, "capacity": 45, "kind": "openspace", "readers": 2},
    {"nom": "Cafétéria Le Forum", "batiment": "Paris-Nord", "etage": 0, "capacity": 120, "kind": "cafeteria", "readers": 3},
    {"nom": "Amphi Curie", "batiment": "Lyon-Tech", "etage": 0, "capacity": 200, "kind": "amphi", "readers": 2},
    {"nom": "Amphi Pasteur", "batiment": "Lyon-Tech", "etage": 0, "capacity": 150, "kind": "amphi", "readers": 2},
    {"nom": "Labo Informatique", "batiment": "Lyon-Tech", "etage": 1, "capacity": 30, "kind": "openspace", "readers": 1},
    {"nom": "Salle réunion Lovelace", "batiment": "Lyon-Tech", "etage": 2, "capacity": 10, "kind": "meeting", "readers": 1},
    {"nom": "Open Space Phoenix", "batiment": "Marseille-Sud", "etage": 1, "capacity": 50, "kind": "openspace", "readers": 2},
    {"nom": "Cafétéria La Calanque", "batiment": "Marseille-Sud", "etage": 0, "capacity": 80, "kind": "cafeteria", "readers": 2},
    {"nom": "Salle réunion Fermat", "batiment": "Marseille-Sud", "etage": 2, "capacity": 6, "kind": "meeting", "readers": 1},
]


# --------------------------------------------------------------------------- #
# Calendrier français : jours fériés + vacances scolaires (simplifiés)
# --------------------------------------------------------------------------- #
def _paques(an: int) -> date:
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
    return date(an, mois, jour + 1)


def _jours_feries(an: int) -> set:
    """Retourne l'ensemble des jours fériés français pour une année."""
    paques = _paques(an)
    fixes = [
        date(an, 1, 1), date(an, 5, 1), date(an, 5, 8), date(an, 7, 14),
        date(an, 8, 15), date(an, 11, 1), date(an, 11, 11), date(an, 12, 25),
    ]
    mobiles = [
        paques + timedelta(days=1),    # lundi de Pâques
        paques + timedelta(days=39),   # Ascension
        paques + timedelta(days=50),   # lundi de Pentecôte
    ]
    return set(fixes + mobiles)


def _est_vacances(jour: date) -> bool:
    """Indique si la date tombe dans une période de vacances scolaires."""
    m, d = jour.month, jour.day
    if m == 12 and d >= 20:      # vacances de Noël
        return True
    if m == 1 and d <= 3:
        return True
    if m == 2 and 8 <= d <= 24:  # vacances d'hiver
        return True
    if m == 4 and 8 <= d <= 24:  # vacances de printemps
        return True
    if m == 7 or m == 8:         # vacances d'été
        return True
    if m == 10 and 19 <= d <= 31:  # vacances de la Toussaint
        return True
    return False


def _est_pont(jour: date, feries: set) -> bool:
    """Détecte un pont : vendredi après un jeudi férié, ou lundi avant mardi férié."""
    wd = jour.weekday()
    if wd == 4 and (jour - timedelta(days=1)) in feries:   # vendredi après jeudi férié
        return True
    if wd == 0 and (jour + timedelta(days=1)) in feries:   # lundi avant mardi férié
        return True
    return False


def _facteur_saison(jour: date, feries: set) -> float:
    """Facteur d'affluence DÉTERMINISTE lié au calendrier (résout M5).

    Cumule : ponts, creux estival progressif, rentrée chargée, molle de fin
    d'année, vacances scolaires générales. Séparé de l'aléa événementiel pour
    que le ML puisse capter la tendance.
    """
    f = 1.0
    m, d = jour.month, jour.day

    if _est_pont(jour, feries):
        f *= 0.40
    # Été progressif
    if m == 7:
        f *= 0.70
    elif m == 8:
        if d <= 14:
            f *= 0.50
        elif d <= 20:           # semaine du 15 août : creux maximal
            f *= 0.35
        else:                   # dernière semaine d'août : remontée
            f *= 0.70
    elif _est_vacances(jour):   # autres vacances scolaires
        f *= 0.50
    # Rentrée de septembre chargée
    if m == 9 and d <= 15:
        f *= 1.10
    # Molle de fin d'année (23 déc -> 2 jan)
    if (m == 12 and d >= 23) or (m == 1 and d <= 2):
        f *= 0.30
    # Vendredi structurellement plus calme
    if jour.weekday() == 4:
        f *= 0.85
    return f


# --------------------------------------------------------------------------- #
# Génération des entités statiques (salles, readers, users, badges, profils)
# --------------------------------------------------------------------------- #
def _gen_uid(rng: random.Random) -> str:
    """Génère un UID RFID hexadécimal sur 8 caractères (style ESP32)."""
    return "".join(rng.choice("0123456789ABCDEF") for _ in range(8))


def _choisir_type(rng: random.Random) -> str:
    """Tire un type d'utilisateur selon la répartition pondérée."""
    types = list(TYPE_WEIGHTS.keys())
    poids = list(TYPE_WEIGHTS.values())
    return rng.choices(types, weights=poids, k=1)[0]


def _creer_salles_et_readers(
    db, rng: random.Random
) -> List[Tuple[Room, List[Reader], Dict]]:
    """Crée les salles et leurs lecteurs ESP32.

    Retourne la liste (salle, readers, spec) où `spec` est le dictionnaire
    `ROOMS_SPEC` correspondant (contient `kind`).
    """
    resultat: List[Tuple[Room, List[Reader], Dict]] = []
    mac_counter = 0
    for spec in ROOMS_SPEC:
        room = Room(nom=spec["nom"], batiment=spec["batiment"],
                    etage=spec["etage"], capacity=spec["capacity"],
                    kind=spec["kind"])
        db.add(room)
        db.flush()  # pour obtenir room.id
        readers: List[Reader] = []
        for n in range(spec["readers"]):
            mac_counter += 1
            mac = f"AA:BB:CC:{mac_counter:02X}:{room.id:02X}:{n:02X}"
            reader = Reader(
                mac_address=mac,
                nom=f"ESP-{spec['nom'][:12]}-{n + 1}",
                ip_address=f"10.0.{room.id}.{n + 10}",
                statut="online",
                room_id=room.id,
            )
            db.add(reader)
            readers.append(reader)
        resultat.append((room, readers, spec))
    db.flush()
    return resultat


def _tirer_jours_site(type_u: str, rng: random.Random) -> List[int]:
    """Tire l'emploi du temps hebdomadaire stable d'un user (jours sur site).

    Modélise le télétravail français : un employé est par défaut sur site
    mardi + jeudi, plus un jour aléatoire ; étudiants/admins quasi tous les
    jours ouvrés.
    """
    base = list(JOURS_SITE_BASE.get(type_u, [0, 1, 2, 3, 4]))
    if type_u == "employe":
        candidats = [j for j in range(5) if j not in base]
        rng.shuffle(candidats)
        # 1 à 2 jours supplémentaires sur site (présence 3-4 j/semaine)
        base += candidats[: rng.randint(1, 2)]
    return sorted(set(base))


def _tirer_conges(
    rng: random.Random, debut: date, fin: date
) -> set:
    """Génère 1 à 3 blocs d'absence (congés/RTT) et retourne l'ensemble des jours.

    Mélange semaines complètes (été, Noël) et RTT isolés.
    """
    jours_off: set = set()
    nb_blocs = rng.randint(1, 3)
    total_days = (fin - debut).days
    if total_days <= 0:
        return jours_off
    for _ in range(nb_blocs):
        longueur = rng.choice([1, 1, 2, 5, 5, 10])  # RTT isolés ou semaines
        start = debut + timedelta(days=rng.randint(0, max(0, total_days - 1)))
        for k in range(longueur):
            jours_off.add(start + timedelta(days=k))
    return jours_off


def _creer_users_et_badges(
    db,
    rng: random.Random,
    debut: date,
    fin: date,
    salles: List[Tuple[Room, List[Reader], Dict]],
) -> List[Dict]:
    """Crée les utilisateurs, leurs badges et leur PROFIL COMPORTEMENTAL stable.

    Chaque profil contient des attributs persistants tirés une seule fois :
    salle de rattachement, heures personnelles, jours sur site, congés. Ce sont
    eux qui produisent le signal hebdomadaire exploitable par le ML.
    """
    # Salles de rattachement candidates par type d'utilisateur
    openspaces = [(r, spec) for (r, _, spec) in salles if spec["kind"] == "openspace"]
    amphis = [(r, spec) for (r, _, spec) in salles if spec["kind"] == "amphi"]
    meetings = [(r, spec) for (r, _, spec) in salles if spec["kind"] == "meeting"]

    profils: List[Dict] = []
    emails_vus = set()
    for i in range(NB_USERS):
        prenom = rng.choice(_PRENOMS)
        nom = rng.choice(_NOMS)
        base_email = f"{prenom.lower()}.{nom.lower()}"
        email = f"{base_email}@campus.fr"
        suffixe = 1
        while email in emails_vus:
            suffixe += 1
            email = f"{base_email}{suffixe}@campus.fr"
        emails_vus.add(email)

        type_u = _choisir_type(rng)
        user = User(nom=nom, prenom=prenom, email=email, type_utilisateur=type_u)
        db.add(user)
        db.flush()

        # Badges : 1 ou 2 par utilisateur
        nb_badges = 2 if rng.random() < PROBA_SECOND_BADGE else 1
        actifs: List[Tuple[int, str]] = []
        inactifs: List[Tuple[int, str]] = []
        for _ in range(nb_badges):
            uid = _gen_uid(rng)
            statut = "inactif" if rng.random() < PROBA_BADGE_INACTIF else "actif"
            attribution = datetime.combine(
                debut - timedelta(days=rng.randint(0, 90)), time(9, 0)
            )
            badge = RFIDBadge(uid=uid, statut=statut, user_id=user.id,
                              date_attribution=attribution)
            db.add(badge)
            db.flush()
            if statut == "actif":
                actifs.append((badge.id, uid))
            else:
                inactifs.append((badge.id, uid))

        # Salle de rattachement : openspace pour employés/admins, amphi/openspace
        # pour étudiants, meeting pour visiteurs.
        if type_u == "etudiant" and amphis:
            home_room, home_spec = rng.choice(amphis + openspaces)
        elif type_u == "visiteur" and meetings:
            home_room, home_spec = rng.choice(meetings)
        elif openspaces:
            home_room, home_spec = rng.choice(openspaces)
        else:
            home_room, home_spec = rng.choice(amphis + meetings)

        # Heures personnelles récurrentes (moyenne propre à l'individu)
        h_in_mu = rng.gauss(8 * 60 + 45, 35) if type_u != "etudiant" else rng.gauss(9 * 60, 40)
        h_out_mu = rng.gauss(18 * 60, 45) if type_u != "etudiant" else rng.gauss(17 * 60, 60)

        profils.append({
            "user": user,
            "type": type_u,
            "actifs": actifs,
            "inactifs": inactifs,
            "home_room_id": home_room.id,
            "home_batiment": home_spec["batiment"],
            "h_in_mu": h_in_mu,
            "h_out_mu": h_out_mu,
            "jours_site": _tirer_jours_site(type_u, rng),
            "conges": _tirer_conges(rng, debut, fin),
            "cafet_proba": rng.uniform(0.55, 0.85),
        })
    db.flush()
    return profils


# --------------------------------------------------------------------------- #
# Profils horaires par type de salle + occupation plafonnée
# --------------------------------------------------------------------------- #
def _poids_horaire(kind: str, heure: float) -> float:
    """Poids relatif d'arrivées à l'heure `heure` (float) pour un type de salle.

    - openspace : cloche d'arrivée le matin (vidage géré par les départs) ;
    - cafeteria : double/multi-pic concentré sur le midi ;
    - meeting/amphi : pics autour des créneaux discrets du profil.
    """
    profil = ROOM_PROFILES[kind]
    h_open, h_close = profil["open"]
    if heure < h_open or heure >= h_close:
        return 0.0

    if kind == "openspace":
        mu, sigma = profil["arr_peak"]
        return math.exp(-0.5 * ((heure - mu) / sigma) ** 2)

    # meeting / cafeteria / amphi : somme de cloches étroites sur les créneaux
    poids = 0.0
    for (sh, sm) in profil["slots"]:
        centre = sh + sm / 60.0
        sigma = 0.35
        poids += math.exp(-0.5 * ((heure - centre) / sigma) ** 2)
    return poids


def _h_to_time(h_float: float) -> time:
    """Convertit une heure flottante (ex. 8.75) en objet time borné [0, 23:59]."""
    h_float = max(0.0, min(23 + 59 / 60.0, h_float))
    minutes = int(round(h_float * 60))
    return time(minutes // 60, minutes % 60)


def _facteur_alea_jour(rng: random.Random) -> float:
    """Aléa événementiel multiplicatif du jour (séparé du déterministe)."""
    if rng.random() < 0.04:
        return rng.uniform(1.3, 1.8)     # réunion générale, conférence
    if rng.random() < 0.04:
        return rng.uniform(0.5, 0.7)     # creux (météo, événement externe)
    return 1.0


def _generer_salle_jour(
    room: Room,
    spec: Dict,
    readers: List[Reader],
    jour: date,
    facteur: float,
    rng: random.Random,
    events: List[Event],
    home_users: List[Dict],
) -> None:
    """Génère les flux d'UNE salle sur UN jour, avec occupation plafonnée.

    Pour chaque heure ouverte : nombre d'arrivées ~ Poisson(λ), λ dépendant de
    la capacity, du type de salle, de l'heure et du jour. L'occupation
    instantanée est suivie (départs planifiés) et bornée à `capacity` : toute
    arrivée au-delà devient un no-show (non émis). Entrée et sortie portent le
    MÊME lecteur (cohérence dwell time par salle).
    """
    kind = spec["kind"]
    profil = ROOM_PROFILES[kind]
    capacity = max(1, room.capacity)
    h_open, h_close = profil["open"]
    fill = rng.uniform(*profil["fill"])
    no_show = profil.get("no_show", 0.05)
    offset_min = LUNCH_OFFSET_PAR_SITE.get(spec["batiment"], 0) if kind == "cafeteria" else 0

    # Normalise les poids horaires pour répartir la « charge » cible sur la journée.
    heures = [h for h in range(h_open, h_close)]
    poids = [_poids_horaire(kind, h + 0.5) for h in heures]
    total_poids = sum(poids) or 1.0
    # Charge cible journalière (nombre d'arrivées attendu) ≈ capacity * fill *
    # rotation, modulée par le facteur jour.
    rotation = 3.0 if kind == "cafeteria" else 1.2
    charge_cible = capacity * fill * rotation * facteur

    occupation = 0                      # occupation instantanée
    departs: Dict[int, int] = {}        # heure_depart -> nb partants

    for h, w in zip(heures, poids):
        # Libère les places des départs prévus à cette heure
        occupation = max(0, occupation - departs.pop(h, 0))

        lam = charge_cible * (w / total_poids)
        if lam <= 0:
            continue
        n_arrivees = _poisson(rng, lam)
        for _ in range(n_arrivees):
            if rng.random() < no_show:
                continue                # no-show : arrivée prévue annulée
            if occupation >= capacity:
                continue                # refoulement / no-show (plafond capacity)

            profil_u = rng.choice(home_users) if home_users else None
            if profil_u is None:
                continue
            badge_id, uid = rng.choice(profil_u["actifs"])
            reader = rng.choice(readers)

            minute = rng.randint(0, 59)
            ts_in = datetime.combine(jour, _h_to_time(h + (minute + offset_min) / 60.0))
            events.append(Event(
                timestamp=ts_in,
                type_evenement="passage" if kind == "cafeteria" else "entree",
                resultat=True, uid_scanne=uid, badge_id=badge_id,
                reader_mac=reader.mac_address,
            ))
            occupation += 1

            # Planifie la sortie (même lecteur) selon le dwell time du type
            dwell = rng.randint(*profil["dwell_min"])
            ts_out = ts_in + timedelta(minutes=dwell)
            if ts_out.date() != jour or ts_out.hour >= 23:
                ts_out = datetime.combine(jour, time(22, 30))
            events.append(Event(
                timestamp=ts_out,
                type_evenement="sortie", resultat=True, uid_scanne=uid,
                badge_id=badge_id, reader_mac=reader.mac_address,
            ))
            h_dep = min(h_close, ts_out.hour)
            departs[h_dep] = departs.get(h_dep, 0) + 1


def _poisson(rng: random.Random, lam: float) -> int:
    """Tire un entier selon une loi de Poisson(λ) (algorithme de Knuth)."""
    if lam <= 0:
        return 0
    if lam > 30:  # approximation gaussienne au-delà (Knuth devient lent)
        val = int(round(rng.gauss(lam, math.sqrt(lam))))
        return max(0, val)
    seuil = math.exp(-lam)
    k = 0
    p = 1.0
    while True:
        k += 1
        p *= rng.random()
        if p <= seuil:
            return k - 1


# --------------------------------------------------------------------------- #
# Génération des événements (cœur du dataset) — centrée salle + individu
# --------------------------------------------------------------------------- #
def _est_present(profil: Dict, jour: date, facteur: float, rng: random.Random) -> bool:
    """Indique si un user (selon son profil stable) est présent ce jour."""
    if jour in profil["conges"]:
        return False
    if jour.weekday() not in profil["jours_site"]:
        return False
    proba = PRESENCE_PAR_TYPE.get(profil["type"], 0.0) * min(1.2, max(0.2, facteur))
    return rng.random() < min(proba, 0.98)


def _generer_evenements(
    rng: random.Random,
    profils: List[Dict],
    salles: List[Tuple[Room, List[Reader], Dict]],
    debut: date,
    fin: date,
) -> List[Event]:
    """Construit la liste complète des événements sur la période [debut, fin].

    Pipeline par jour ouvré :
      1. Calcule le facteur d'affluence (déterministe saisonnier × aléa).
      2. Détermine les users présents (habitudes individuelles stables).
      3. Pour chaque SALLE non-cafétéria, génère les flux par créneau (Poisson,
         occupation plafonnée), en piochant parmi les users rattachés présents.
      4. Génère les passages cafétéria des présents (déjeuner échelonné par site).
      5. Injecte des refus crédibles (badges inactifs + UID inconnus), aux
         heures de pointe.
    """
    annees = {debut.year, fin.year}
    feries = set()
    for an in annees:
        feries |= _jours_feries(an)

    # Indexation
    salles_par_room_id = {r.id: (r, rd, sp) for (r, rd, sp) in salles}
    cafeterias = [(r, rd, sp) for (r, rd, sp) in salles if sp["kind"] == "cafeteria"]
    cafet_par_site: Dict[str, List[Tuple]] = {}
    for c in cafeterias:
        cafet_par_site.setdefault(c[2]["batiment"], []).append(c)
    all_readers = [rd for (_, rds, _) in salles for rd in rds]

    profils_actifs = [p for p in profils if p["actifs"]]
    profils_inactifs = [p for p in profils if p["inactifs"]]
    events: List[Event] = []

    jour = debut
    while jour <= fin:
        ouvre = jour.weekday() < 5 and jour not in feries
        if not ouvre:
            jour += timedelta(days=1)
            continue

        facteur = _facteur_saison(jour, feries) * _facteur_alea_jour(rng)

        # 2) Présence individuelle stable
        presents = [p for p in profils_actifs if _est_present(p, jour, facteur, rng)]
        presents_par_room: Dict[int, List[Dict]] = {}
        for p in presents:
            presents_par_room.setdefault(p["home_room_id"], []).append(p)

        # 3) Flux par salle (hors cafétéria), centrés sur les users rattachés
        for room_id, (room, readers, spec) in salles_par_room_id.items():
            if spec["kind"] == "cafeteria":
                continue
            home_users = presents_par_room.get(room_id, [])
            if not home_users:
                # Salle peu fréquentée ce jour : flux résiduel via tous présents
                if not presents or rng.random() < 0.5:
                    continue
                home_users = presents
            _generer_salle_jour(room, spec, readers, jour, facteur, rng,
                                 events, home_users)

        # 4) Cafétéria : passages déjeuner des présents, sur le site de l'user
        for p in presents:
            if rng.random() >= p["cafet_proba"]:
                continue
            cafs = cafet_par_site.get(p["home_batiment"]) or cafeterias
            if not cafs:
                continue
            room, readers, spec = rng.choice(cafs)
            badge_id, uid = rng.choice(p["actifs"])
            reader = rng.choice(readers)
            offset = LUNCH_OFFSET_PAR_SITE.get(spec["batiment"], 0)
            sh, sm = rng.choice(ROOM_PROFILES["cafeteria"]["slots"])
            base = sh * 60 + sm + offset + int(rng.gauss(0, 8))
            base = max(11 * 60, min(14 * 60 + 30, base))
            ts = datetime.combine(jour, time(base // 60, base % 60))
            events.append(Event(
                timestamp=ts, type_evenement="passage", resultat=True,
                uid_scanne=uid, badge_id=badge_id, reader_mac=reader.mac_address,
            ))

        # 5) Refus : badges inactifs + UID inconnus, concentrés aux heures de pointe
        #    (~2-3 % du volume légitime, en moyenne).
        nb_jour = max(2, int(len(presents) * 0.22) + 1)
        for _ in range(nb_jour):
            if rng.random() < 0.6 and profils_inactifs:
                prof = rng.choice(profils_inactifs)
                badge_id, uid = rng.choice(prof["inactifs"])
            else:
                badge_id, uid = None, _gen_uid(rng)
            # Heure suivant le flux légitime (pics 8-10h / 17-19h)
            h = rng.choice([8, 8, 9, 9, 10, 12, 14, 17, 18, 18, 19])
            ts = datetime.combine(jour, time(h, rng.randint(0, 59)))
            reader = rng.choice(all_readers)
            events.append(Event(
                timestamp=ts, type_evenement="refus", resultat=False,
                uid_scanne=uid, badge_id=badge_id, reader_mac=reader.mac_address,
            ))

        jour += timedelta(days=1)

    return events


# --------------------------------------------------------------------------- #
# Anomalies plantées étiquetées (vérité terrain pour la détection) — résout M3
# --------------------------------------------------------------------------- #
def _injecter_anomalies(
    rng: random.Random,
    profils: List[Dict],
    salles: List[Tuple[Room, List[Reader], Dict]],
    debut: date,
    fin: date,
    nb_events_legit: int,
) -> Tuple[List[Event], Dict[str, int]]:
    """Plante des anomalies étiquetées et retourne (events, compteur par type).

    Les événements anormaux sont marqués par un préfixe `ANOM:` dans
    `uid_scanne` pour fournir une vérité terrain traçable sans modifier le
    schéma. Types injectés :
      1. badge cloné multi-sites en < 5 min (impossible physiquement) ;
      2. accès nocturne (2h-4h) un jour ouvré ;
      3. rafale de refus (brute force) du même UID inconnu ;
      4. anti-passback : deux entrées consécutives sans sortie (talonnage) ;
      5. volume aberrant : dépassement ponctuel de capacity (capteur défaillant).
    """
    readers_par_site: Dict[str, List[Reader]] = {}
    for (_, rds, sp) in salles:
        readers_par_site.setdefault(sp["batiment"], []).extend(rds)
    sites = list(readers_par_site.keys())
    all_readers = [rd for rds in readers_par_site.values() for rd in rds]
    profils_actifs = [p for p in profils if p["actifs"]]

    anomalies: List[Event] = []
    compteur = {"clone": 0, "nocturne": 0, "rafale_refus": 0,
                "anti_passback": 0, "volume_aberrant": 0}

    def _jour_ouvre(rng_: random.Random) -> date:
        for _ in range(50):
            j = debut + timedelta(days=rng_.randint(0, max(1, (fin - debut).days)))
            if j.weekday() < 5:
                return j
        return debut

    nb_total = max(8, int(nb_events_legit * TAUX_ANOMALIES))
    for _ in range(nb_total):
        choix = rng.choice(["clone", "nocturne", "rafale_refus",
                            "anti_passback", "volume_aberrant"])
        jour = _jour_ouvre(rng)

        if choix == "clone" and len(sites) >= 2:
            prof = rng.choice(profils_actifs)
            badge_id, uid = rng.choice(prof["actifs"])
            site_a, site_b = rng.sample(sites, 2)
            ra = rng.choice(readers_par_site[site_a])
            rb = rng.choice(readers_par_site[site_b])
            t0 = datetime.combine(jour, time(rng.randint(9, 16), rng.randint(0, 59)))
            anomalies.append(Event(timestamp=t0, type_evenement="entree",
                                   resultat=True, uid_scanne=f"ANOM:clone:{uid}",
                                   badge_id=badge_id, reader_mac=ra.mac_address))
            anomalies.append(Event(timestamp=t0 + timedelta(minutes=rng.randint(1, 4)),
                                   type_evenement="entree", resultat=True,
                                   uid_scanne=f"ANOM:clone:{uid}", badge_id=badge_id,
                                   reader_mac=rb.mac_address))
            compteur["clone"] += 1

        elif choix == "nocturne":
            prof = rng.choice(profils_actifs)
            badge_id, uid = rng.choice(prof["actifs"])
            reader = rng.choice(all_readers)
            ts = datetime.combine(jour, time(rng.randint(2, 4), rng.randint(0, 59)))
            anomalies.append(Event(timestamp=ts, type_evenement="entree",
                                   resultat=True, uid_scanne=f"ANOM:nocturne:{uid}",
                                   badge_id=badge_id, reader_mac=reader.mac_address))
            compteur["nocturne"] += 1

        elif choix == "rafale_refus":
            uid = _gen_uid(rng)
            reader = rng.choice(all_readers)
            t0 = datetime.combine(jour, time(rng.randint(8, 20), rng.randint(0, 59)))
            for k in range(rng.randint(5, 15)):
                anomalies.append(Event(
                    timestamp=t0 + timedelta(seconds=20 * k),
                    type_evenement="refus", resultat=False,
                    uid_scanne=f"ANOM:rafale_refus:{uid}", badge_id=None,
                    reader_mac=reader.mac_address))
            compteur["rafale_refus"] += 1

        elif choix == "anti_passback":
            prof = rng.choice(profils_actifs)
            badge_id, uid = rng.choice(prof["actifs"])
            reader = rng.choice(all_readers)
            t0 = datetime.combine(jour, time(rng.randint(9, 17), rng.randint(0, 59)))
            anomalies.append(Event(timestamp=t0, type_evenement="entree",
                                   resultat=True,
                                   uid_scanne=f"ANOM:anti_passback:{uid}",
                                   badge_id=badge_id, reader_mac=reader.mac_address))
            anomalies.append(Event(timestamp=t0 + timedelta(minutes=rng.randint(2, 8)),
                                   type_evenement="entree", resultat=True,
                                   uid_scanne=f"ANOM:anti_passback:{uid}",
                                   badge_id=badge_id, reader_mac=reader.mac_address))
            compteur["anti_passback"] += 1

        else:  # volume_aberrant : pic d'entrées dépassant capacity sur une salle
            room, readers, spec = rng.choice(
                [s for s in salles if s[2]["kind"] != "cafeteria"])
            reader = rng.choice(readers)
            t0 = datetime.combine(jour, time(rng.randint(10, 15), rng.randint(0, 59)))
            surcharge = int(room.capacity * rng.uniform(1.1, 1.4))
            for k in range(surcharge):
                prof = rng.choice(profils_actifs)
                badge_id, uid = rng.choice(prof["actifs"])
                anomalies.append(Event(
                    timestamp=t0 + timedelta(seconds=15 * k),
                    type_evenement="entree", resultat=True,
                    uid_scanne=f"ANOM:volume_aberrant:{uid}", badge_id=badge_id,
                    reader_mac=reader.mac_address))
            compteur["volume_aberrant"] += 1

    return anomalies, compteur


# --------------------------------------------------------------------------- #
# Génération MLData (features exploitables, résout m1/m2)
# --------------------------------------------------------------------------- #
def _generer_ml_data(
    events: List[Event],
    salles: List[Tuple[Room, List[Reader], Dict]],
) -> List[MLData]:
    """Dérive des lignes MLData à partir des événements autorisés.

    feature_1 = heure de la journée ; feature_2 = jour de semaine ;
    feature_3 = durée de présence RÉELLE par salle (dwell, en heures) calculée
    en appariant entrée→sortie du même UID sur le même lecteur. Remplace le
    bruit `rng.uniform`. `label` = type d'événement.
    """
    # Index reader_mac -> kind (non utilisé en feature ici mais conservé pour clarté)
    rows: List[MLData] = []

    # Apparie entrée/sortie pour estimer la durée réelle par (uid, reader, jour)
    in_times: Dict[Tuple, datetime] = {}
    for ev in sorted(events, key=lambda e: e.timestamp):
        if ev.type_evenement in ("entree", "passage") and ev.resultat:
            key = (ev.uid_scanne, ev.reader_mac, ev.timestamp.date())
            in_times.setdefault(key, ev.timestamp)

    for ev in events:
        if ev.id is None:
            continue
        heure = ev.timestamp.hour + ev.timestamp.minute / 60.0
        duree = 0.0
        if ev.type_evenement == "sortie" and ev.resultat:
            key = (ev.uid_scanne, ev.reader_mac, ev.timestamp.date())
            t_in = in_times.get(key)
            if t_in is not None:
                duree = round((ev.timestamp - t_in).total_seconds() / 3600.0, 2)
        rows.append(MLData(
            event_id=ev.id,
            feature_1=round(heure, 2),                  # heure de la journée
            feature_2=float(ev.timestamp.weekday()),    # jour de semaine
            feature_3=duree,                            # durée présence réelle (h)
            label=ev.type_evenement,
            prediction=None,
        ))
    return rows


# --------------------------------------------------------------------------- #
# Insertion bulk + orchestration
# --------------------------------------------------------------------------- #
def _bulk_add(db, objets: List, chunk: int = BULK_CHUNK) -> None:
    """Insère les objets par lots pour limiter la mémoire et accélérer."""
    for i in range(0, len(objets), chunk):
        db.add_all(objets[i:i + chunk])
        db.flush()


def _reset(db) -> None:
    """Purge toutes les tables peuplées par le générateur (ordre FK respecté)."""
    db.query(MLData).delete()
    db.query(Event).delete()
    db.query(RFIDBadge).delete()
    db.query(Reader).delete()
    db.query(User).delete()
    db.query(Room).delete()
    db.commit()


def _afficher_stats(db, anomalies: Optional[Dict[str, int]] = None) -> Dict:
    """Calcule et affiche les statistiques du dataset généré."""
    nb_users = db.query(func.count(User.id)).scalar()
    nb_badges = db.query(func.count(RFIDBadge.id)).scalar()
    nb_badges_inactifs = db.query(func.count(RFIDBadge.id)).filter(
        RFIDBadge.statut == "inactif").scalar()
    nb_rooms = db.query(func.count(Room.id)).scalar()
    nb_readers = db.query(func.count(Reader.mac_address)).scalar()
    nb_events = db.query(func.count(Event.id)).scalar()
    nb_refus = db.query(func.count(Event.id)).filter(
        Event.type_evenement == "refus").scalar()
    pct_refus = round(100.0 * nb_refus / nb_events, 2) if nb_events else 0.0

    par_jour = [0] * 7
    par_heure = [0] * 24
    for (ts,) in db.query(Event.timestamp).all():
        par_jour[ts.weekday()] += 1
        par_heure[ts.hour] += 1

    jours_fr = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]

    # Diversité d'affluence par TYPE de salle (entrées + passages autorisés)
    par_kind = (
        db.query(Room.kind, func.count(Event.id))
        .join(Reader, Reader.room_id == Room.id)
        .join(Event, Event.reader_mac == Reader.mac_address)
        .filter(Event.type_evenement.in_(["entree", "passage"]))
        .group_by(Room.kind)
        .all()
    )
    affluence_par_kind = {k: c for k, c in par_kind}

    print("=" * 60)
    print("STATS DATASET GÉNÉRÉ")
    print("=" * 60)
    print(f"Users           : {nb_users}")
    print(f"Badges          : {nb_badges} (dont {nb_badges_inactifs} inactifs)")
    print(f"Salles          : {nb_rooms}")
    print(f"Lecteurs ESP32  : {nb_readers}")
    print(f"Events          : {nb_events}")
    print(f"Refus           : {nb_refus} ({pct_refus} %)")
    print("-" * 60)
    print("Affluence (entrées/passages) par TYPE de salle :")
    for k, c in sorted(affluence_par_kind.items(), key=lambda x: -x[1]):
        print(f"  {k:<10} : {c:>7}")
    print("-" * 60)
    print("Distribution par jour de semaine (signal télétravail) :")
    for i, j in enumerate(jours_fr):
        bar = "#" * (par_jour[i] // max(1, max(par_jour) // 40))
        print(f"  {j} : {par_jour[i]:>7}  {bar}")
    print("-" * 60)
    print("Distribution par heure :")
    for h in range(7, 21):
        bar = "#" * (par_heure[h] // max(1, max(par_heure) // 40))
        print(f"  {h:02d}h : {par_heure[h]:>6}  {bar}")
    if anomalies:
        print("-" * 60)
        print("Anomalies plantées (vérité terrain) :")
        for k, c in anomalies.items():
            print(f"  {k:<16} : {c}")
        print(f"  TOTAL épisodes   : {sum(anomalies.values())}")
    print("=" * 60)

    return {
        "users": nb_users, "badges": nb_badges, "rooms": nb_rooms,
        "readers": nb_readers, "events": nb_events, "refus": nb_refus,
        "pct_refus": pct_refus, "par_jour": dict(zip(jours_fr, par_jour)),
        "par_heure": par_heure, "affluence_par_kind": affluence_par_kind,
        "anomalies": anomalies or {},
    }


def generate(reset: bool = False, months: int = NB_MONTHS) -> Dict:
    """Peuple la base avec un dataset réaliste d'accès RFID.

    Args:
        reset: si True, purge les tables avant de regénérer (idempotent).
        months: profondeur d'historique en mois (par défaut NB_MONTHS).

    Returns:
        Dictionnaire de statistiques du dataset généré.
    """
    Base.metadata.create_all(bind=engine)
    rng = random.Random(SEED)
    db = SessionLocal()
    try:
        if reset:
            _reset(db)
        elif db.query(func.count(Event.id)).scalar():
            print("Base déjà peuplée — utilisez generate(reset=True) pour regénérer.")
            return _afficher_stats(db)

        fin = date.today()
        debut = fin - timedelta(days=int(months * 30.4))

        print(f"Génération sur {months} mois : {debut} -> {fin}")
        salles = _creer_salles_et_readers(db, rng)
        profils = _creer_users_et_badges(db, rng, debut, fin, salles)
        db.commit()

        print("Génération des événements (centrée salle + individu)...")
        events = _generer_evenements(rng, profils, salles, debut, fin)

        print("Injection des anomalies étiquetées...")
        anomalies, compteur = _injecter_anomalies(
            rng, profils, salles, debut, fin, len(events))
        events.extend(anomalies)

        _bulk_add(db, events)
        db.commit()

        if GEN_ML_DATA:
            print("Génération des données ML...")
            ml_rows = _generer_ml_data(events, salles)
            _bulk_add(db, ml_rows)
            db.commit()

        return _afficher_stats(db, compteur)
    finally:
        db.close()


if __name__ == "__main__":
    generate(reset=True, months=NB_MONTHS)
