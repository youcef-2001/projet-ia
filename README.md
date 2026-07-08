# Smart Campus — Système IoT RFID / ESP32 / Machine Learning

> Projet M2 IoT (sujet 3). Système complet de contrôle d'accès : des lecteurs
> **ESP32 + RFID** identifient les utilisateurs par badge, transmettent les scans
> à un **backend FastAPI** via WebSocket/HTTP (JSON), qui décide de l'autorisation,
> **enregistre chaque passage** en base **PostgreSQL**, expose des **métriques
> Prometheus** et un **dashboard Grafana**, et alimente un module **Machine
> Learning** (détection d'anomalies, prédiction de fréquentation).

## Présentation (en bref)

- **Identification RFID** : chaque utilisateur possède un ou plusieurs badges.
- **Multi-ESP / multi-salle** : un ESP32 est identifié par sa **MAC address** ;
  une salle regroupe plusieurs ESP. Le lecteur RFID partage l'id de l'ESP.
- **Flux temps réel** : l'ESP envoie `{mac_address, rfid_uid}`, le serveur répond
  `autorisé / refusé` et journalise l'événement (`entrée / sortie / refus`).
- **Observabilité** : logs structurés, métriques Prometheus, dashboards Grafana.
- **ML** : dataset construit depuis les événements pour anomalies & prévisions.
- **Stack** : FastAPI · SQLAlchemy · PostgreSQL · Prometheus · Grafana · Vue ·
  Docker Compose.

---

## 1. Lancement pas à pas

```bash
# 1. Cloner puis se placer à la racine du projet
cd projet-ia

# 2. Construire et démarrer toute la stack
sudo docker compose up --build

# 3. (auto) Les tables sont créées et la base est peuplée au démarrage
#    (utilisateurs Alice, Bob… + badges + salles + lecteurs).
#    Pour re-seed manuellement :
sudo docker compose exec backend python -m app.seed

# 4. Accès aux services
#    Backend  API + docs ... http://localhost:8000/docs
#    Frontend (Monitoring Live) ... http://localhost:8080
#    Prometheus ............. http://localhost:9090
#    Grafana ................ http://localhost:3000  (anonyme = Admin)
```

> Postgres n'expose **aucun port hôte** (accès interne `db:5432` uniquement) afin
> d'éviter les conflits de port. Le backend attend que la base soit *healthy*
> (`pg_isready`) avant de démarrer.

### Tester un scan sans ESP

```bash
curl -X POST http://localhost:8000/scan \
  -H "Content-Type: application/json" \
  -d '{"mac_address":"14:08:08:A4:C9:28","rfid_uid":"47C12E06"}'
# -> {"status":"success","authorized":true,"event":"entree","user":"Alice Durand", ...}
```

---

## 2. Architecture globale

```
┌──────────────┐   WiFi / WebSocket (JSON)   ┌─────────────────────────────┐
│  ESP32 + RFID│ ──────────────────────────► │        Backend FastAPI       │
│ (lecteur)    │   {mac_address, rfid_uid}   │  controllers → services → DB │
│  MAC = id    │ ◄────────────────────────── │  /ws/esp  /scan  /predict    │
└──────────────┘   {authorized, event, ...}  └──────────┬──────────────────┘
                                                         │ SQLAlchemy
                                              ┌──────────▼──────────┐
                                              │   PostgreSQL (db)   │
                                              └──────────┬──────────┘
   ┌─────────────┐   scrape /metrics                     │
   │ Prometheus  │ ◄─────────────────────────────────────┤
   └──────┬──────┘                                        │
          │ datasource                          datasource (SQL)
   ┌──────▼──────┐                                        │
   │   Grafana   │ ◄──────────────────────────────────────┘
   └─────────────┘            ┌──────────────┐
                              │ Frontend Vue │  fetch http://localhost:8000
                              └──────────────┘
```

**Couches backend** (`backend/app/`)
- `controllers/` — points d'entrée HTTP/WebSocket (validation, I/O).
- `services/` — logique métier (autorisation, gestion lecteurs, ML).
- `models.py` — schéma SQLAlchemy (MCD). `schemas.py` — DTO Pydantic.
- `monitoring.py` — logger + métriques Prometheus. `seed.py` — données de démo.

---

## 3. Base de données (MCD)

```
UTILISATEUR (1,N) ── possède    ── (1,1) BADGE_RFID
BADGE_RFID  (1,N) ── génère     ── (1,1) ÉVÉNEMENT
LECTEUR     (1,N) ── enregistre ── (1,1) ÉVÉNEMENT
ÉVÉNEMENT   (1,1) ── alimente   ── (0,1) DONNEE_ML
SALLE       (1,N) ── héberge    ── (1,1) LECTEUR
```

| Table | Clé | Champs clés | Rôle |
|-------|-----|-------------|------|
| `users` | `id` | nom, prenom, email (unique), type_utilisateur | Personnes |
| `rfid_badges` | `id` | uid (unique), statut, date_attribution, `user_id` | Badges (1..N / user) |
| `rooms` | `id` | nom, batiment, etage, capacity | Salles / localisation |
| `readers` | **`mac_address`** | nom, ip_address, statut, last_seen, `room_id` | ESP32 (= lecteur RFID) |
| `events` | `id` | timestamp, type_evenement, resultat, uid_scanne, `badge_id`, `reader_mac` | Passages |
| `ml_data` | `id` | feature_1/2/3, label, prediction, `event_id` | Dataset ML |
| `predictions` | `id` | predicted_occupancy, predicted_anomaly, confidence, `room_id` | Prévisions |

**Choix d'efficacité** : clé primaire naturelle MAC pour les lecteurs ;
index sur `uid`, `email`, `timestamp`, et toutes les FK (`badge_id`, `reader_mac`,
`room_id`, `event_id`) pour des recherches/jointures rapides ;
`cascade delete` user → badges ; relation `event ↔ ml_data` en 1,1 via clé unique.

---

## 4. API

| Méthode | Route | Entrée | Sortie |
|---------|-------|--------|--------|
| WS | `/ws/esp` | `{mac_address, rfid_uid}` | `{status, authorized, event, user, message}` |
| POST | `/scan` | `{mac_address, rfid_uid}` | idem (alternative HTTP) |
| POST | `/predict/{room_id}` | — | prévision de fréquentation |
| GET | `/api/stats` | — | compteurs (succès / refus / ESP en ligne) |
| GET | `/api/readers` | — | ESP32 connus (salle, MAC, IP, statut) |
| GET | `/api/events?limit=N` | — | historique des scans (live monitoring) |
| GET | `/api/ml/status` | — | modèle IA entraîné ? |
| POST | `/api/ml/train` | — | (ré)entraîne le modèle KNN |
| GET | `/api/ml/predict/{room_id}?date=YYYY-MM-DD` | — | prévision d'affluence d'une salle |
| GET | `/api/ml/predict?date=YYYY-MM-DD` | — | prévision de toutes les salles |
| GET | `/health` | — | `{status: ok}` |
| GET | `/metrics` | — | métriques Prometheus |
| GET | `/docs` | — | Swagger UI |

---

## 5. Workflow détaillé d'un scan

1. L'ESP32 lit l'UID RFID et envoie `{"mac_address","rfid_uid"}` sur `/ws/esp`.
2. `iot_service.register_or_update_reader` crée/actualise le lecteur (par MAC,
   statut `online`, `last_seen`, IP).
3. `rfid_service.process_scan` cherche le badge par `uid` :
   - badge **actif** → accès autorisé, type `entrée`/`sortie` (alterné selon le
     dernier passage), résolution de l'utilisateur.
   - badge **inactif/inconnu** → `refus`.
4. Un `Event` est inséré (résultat, type, badge, lecteur, UID brut).
5. Métriques mises à jour (`badge_scans_total`, `device_status`…).
6. Le serveur renvoie le verdict à l'ESP (`authorized: true/false`).
7. Prometheus scrape `/metrics`, Grafana visualise ; les `events` alimentent
   `ml_data` pour l'analyse (anomalies, clustering, prédiction).

---

## 6. Machine Learning — prévision d'affluence (KNN)

Le service IA ([ml_service.py](backend/app/services/ml_service.py)) prévoit le
**nombre de personnes par salle et par jour**.

- **Dataset** : généré par [data_generator.py](backend/app/data_generator.py)
  (~6 mois, ~70k événements réalistes : horaires de bureau, télétravail
  mardi/jeudi, week-ends/fériés vides, saisonnalité française, anomalies plantées,
  profils par type de salle). Régénérable : `docker compose exec backend python -m app.data_generator`.
- **Modèle** : `KNeighborsRegressor` dans un `Pipeline` (one-hot salle/type,
  encodage cyclique des variables temporelles, `StandardScaler`).
  Validation **temporelle** (hold-out chronologique, sans fuite).
- **Features (18)** : salle, type, bâtiment, étage, jour/mois/semaine (cycliques),
  week-end, férié, vacances, pont, veille/lendemain de férié, capacité,
  moyenne mobile, affluence N-1 hebdo, tendance.
- **Sortie probabiliste** : valeur prévue, **intervalle p10–p90** (quantiles des
  voisins), score de confiance, **niveau** faible/moyen/fort calibré
  (`predict_proba`), taux de remplissage.
- **Performance (hold-out temporel)** : MAE ≈ 1.7 · R² ≈ 0.73 · couverture
  d'intervalle ≈ 0.88.

> **Premier démarrage** : à la première exécution, le backend génère le dataset
> et entraîne le modèle **en tâche de fond** (~1 min). L'API reste disponible ;
> `GET /api/ml/status` indique quand le modèle est prêt. Les redémarrages suivants
> sont instantanés (dataset + modèle persistés).

La page **Prévisions IA** du frontend affiche ces prédictions par salle (sélecteur
de date, jauge de remplissage, niveau + probabilités, intervalle de confiance).

Les tables `events` / `ml_data` permettent aussi la détection d'anomalies
(épisodes étiquetés : badge cloné, accès nocturne, anti-passback…).

---

## 7. Structure du projet

```
projet-ia/
├── docker-compose.yml          # orchestration (db, backend, frontend, prometheus, grafana)
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app/
│       ├── main.py             # bootstrap FastAPI + seed
│       ├── database.py         # engine / session SQLAlchemy
│       ├── models.py           # MCD (SQLAlchemy)
│       ├── schemas.py          # DTO Pydantic
│       ├── monitoring.py       # logs + métriques Prometheus
│       ├── seed.py             # données de démo (Alice, Bob…)
│       ├── controllers/        # rfid_controller, iot_controller
│       └── services/           # rfid_service, iot_service, prediction_service
├── frontend/                   # dashboard Vue
├── prometheus/prometheus.yml
└── grafana/provisioning/       # datasources + dashboards
```

## 8. Téléverser le firmware ESP32 (Arduino IDE)

Le firmware du lecteur se trouve dans [get_and_send_uuid/get_and_send_uuid.ino](get_and_send_uuid/get_and_send_uuid.ino). Il tourne sur un **ESP32 (NodeMCU-32S)** relié à un **RFID RC522** en SPI.

### 8.1 Câblage RC522 → ESP32 (SPI)

| RC522 | ESP32 (GPIO) |
|-------|--------------|
| SDA / SS | 5 |
| SCK      | 18 |
| MOSI     | 23 |
| MISO     | 19 |
| RST      | (non utilisé par le driver, laisser libre ou 22) |
| 3.3V     | 3V3 (**jamais 5V**) |
| GND      | GND |

Le pin SS est fixé à `5` dans le code (`MFRC522DriverPinSimple ss_pin(5)`). Les broches SCK/MOSI/MISO sont celles du bus VSPI par défaut de l'ESP32.

### 8.2 Prérequis Arduino IDE

1. **Board manager** : ajouter l'URL ESP32 dans *Fichier → Préférences → URLs de gestionnaire de cartes* :
   `https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json`
   puis installer **esp32 by Espressif Systems** via *Outils → Type de carte → Gestionnaire de cartes*.
2. **Bibliothèques** (*Croquis → Inclure une bibliothèque → Gérer les bibliothèques*) :
   - `MFRC522v2` (Rfid RC522, API v2)
   - `WiFiManager` (tzapu)
   - `WebSockets` (Markus Sattler — fournit `WebSocketsClient`)
   - `WiFi` et `Preferences` sont inclus dans le core ESP32.

### 8.3 Téléversement

1. Ouvrir `get_and_send_uuid/get_and_send_uuid.ino`.
2. *Outils → Type de carte* → **NodeMCU-32S** (ou « ESP32 Dev Module »).
3. *Outils → Port* → sélectionner le port série de l'ESP32 (ex. `/dev/ttyUSB0`, `COMx`). Vitesse moniteur : **115200**.
4. Cliquer sur **Téléverser** (→). Maintenir le bouton **BOOT** si le flash ne démarre pas.
5. Ouvrir le **Moniteur série** (115200 baud) pour suivre les logs.

### 8.4 Configuration réseau au premier démarrage

Aucune modification du code n'est nécessaire : la config Wi-Fi et l'IP du serveur passent par le **portail captif** (WiFiManager).

1. Au premier boot, l'ESP32 crée un point d'accès **`ESP32_RFID_AP`**.
2. S'y connecter depuis un téléphone/PC, le portail s'ouvre automatiquement.
3. Choisir le réseau Wi-Fi, saisir le mot de passe, et renseigner le champ **« IP du Serveur Master »** avec l'IP de la machine qui héberge le backend FastAPI (port `8000`, chemin WebSocket `/ws/esp`).
4. Valider : l'IP est sauvegardée en mémoire flash et réutilisée aux redémarrages suivants.

Une fois connecté, l'ESP32 s'enregistre auprès du backend via son adresse MAC, puis envoie chaque UID de badge scanné au format JSON sur le WebSocket.
