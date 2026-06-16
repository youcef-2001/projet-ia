# Audit d'architecture — Smart Campus (IoT RFID)

> Revue d'architecture de niveau entreprise. Périmètre : backend FastAPI + SQLAlchemy + PostgreSQL, frontend Vue, Prometheus, Grafana, docker-compose.
> Date : 2026-06-16. Aucun code n'a été modifié.

---

## 1. Résumé exécutif

Le projet est **fonctionnellement conforme au cahier des charges (sujet 3)** : MCD complet (`UTILISATEUR / BADGE_RFID / LECTEUR / EVENEMENT / DONNEE_ML`), ESP identifié par MAC, salle 1..N ESP, badge 1..N par utilisateur, flux WebSocket de scan avec autorisation + journalisation, et un module ML. L'organisation en couches (`controllers / services / models / schemas`) est saine et lisible. C'est une **excellente base pédagogique**, mais **pas un système prêt pour la production**.

### Forces majeures
- Séparation des responsabilités claire et idiomatique (controllers fins → services métier → ORM).
- MCD bien modélisé : clé naturelle MAC pour les lecteurs, index sur FK/colonnes de recherche, `cascade delete` user→badges, relation 1,1 `event↔ml_data` via clé unique (`models.py:121`).
- Observabilité présente dès le départ : métriques custom Prometheus + instrumentation FastAPI + Grafana provisionné.
- Gestion d'erreurs avec `rollback` + compteur `SYSTEM_ERRORS` dans les services (`rfid_service.py:72`, `iot_service.py:32`).
- DTO Pydantic distincts du modèle ORM ; healthcheck DB et `depends_on: service_healthy`.

### Faiblesses majeures
- **Sécurité quasi absente** : aucune authentification/autorisation sur les endpoints, CORS `*` avec `allow_credentials=True`, secrets en clair (DB, Grafana), Grafana anonyme en rôle Admin, WebSocket non authentifié — n'importe quel ESP usurpé peut écrire en base.
- **Gestion du cycle de vie applicatif fragile** : `create_all()` + `seed()` exécutés à l'import du module (`main.py:9-10`), pas de migrations (Alembic), pas de `lifespan`.
- **Le « ML » est un placeholder** (`random.randint`) : la table `DONNEE_ML` n'est jamais alimentée, aucune feature réelle n'est calculée.
- **Testabilité faible** : un seul test (`/health`), aucun test de la logique d'autorisation, du WebSocket ou des services.
- **Concurrence WebSocket** : une `Session` SQLAlchemy unique partagée pour toute la durée de la connexion ESP, jamais raffraîchie en cas d'erreur transitoire.

**Verdict** : maquette solide niveau M2, à durcir significativement (sécurité, migrations, ML réel, tests) avant tout usage réel.

---

## 2. Analyse par couche

### 2.1 Controllers (`controllers/`)
- **Bon** : controllers minces, validation déléguée à Pydantic, injection `Depends(get_db)` pour le REST.
- `rfid_controller.py:18` — `request.client.host` peut être `None` derrière un proxy (pas de prise en compte de `X-Forwarded-For`).
- `iot_controller.py` — le WebSocket **ne passe pas par `get_db`** mais ouvre `SessionLocal()` à la main (`:44`) et la garde ouverte pour toute la session. En cas d'exception non-`WebSocketDisconnect`, la boucle peut laisser la session dans un état incohérent ; `process_scan` fait un `rollback` interne mais relance (`raise`), ce qui tue la connexion silencieusement (pas de `send_json` d'erreur au client).
- Pas de gestion du heartbeat/timeout : un ESP « fantôme » reste `online` indéfiniment si la déconnexion TCP n'est pas propre. Aucune tâche de fond ne passe les lecteurs muets en `offline`.

### 2.2 Services (`services/`)
- **Bon** : logique métier isolée et testable en théorie. `_next_event_type` (`rfid_service.py:11`) encapsule bien la règle entrée/sortie.
- **Couplage** : `rfid_service` importe et appelle `register_or_update_reader` de `iot_service` (`:32`), qui **commit** la session. Du coup `process_scan` produit **deux commits** (lecteur, puis événement) — pas atomique : un crash entre les deux laisse un lecteur enregistré sans événement.
- **Cohérence de labels Prometheus** : `BADGE_SCANS` est labellisé `esp_ip=mac_address` (`rfid_service.py:61`) — on met une MAC dans un label nommé `esp_ip`. Incohérent et trompeur pour les dashboards.
- **Cardinalité métriques** : labels `room_id`, `ip_address`, `mac_address` sur des Counters/Gauges → risque d'explosion de cardinalité en production (chaque IP/MAC crée une série temporelle). Anti-pattern Prometheus connu.
- `prediction_service.py` — **fausse IA** : `random.randint/choice/uniform`. Aucune lecture de `events`/`ml_data`. Le cahier des charges (features : heure, fréquence, durée) n'est pas réellement implémenté. Écrit une ligne `Prediction` en base à chaque appel GET/POST (croissance non bornée, pas de purge).
- `mark_reader_offline` n'est appelé qu'à la déconnexion WS ; un ESP qui n'utilise que `/scan` HTTP reste `online` à vie.

### 2.3 Modèles (`models.py`)
- **Très bon** dans l'ensemble. Index pertinents, contraintes d'unicité, relations cohérentes.
- `datetime.utcnow` est **déprécié** en Python 3.12+ (préférer `datetime.now(timezone.utc)`) et stocke des datetimes **naïfs** (`timestamp`, `last_seen`) — risque de confusion fuseau horaire.
- `Event.reader_mac` et `badge_id` sont `nullable=True` sans `ondelete` explicite : supprimer un lecteur/badge échouera ou laissera des FK orphelines (cascade non défini côté `events`).
- `type_utilisateur`, `statut`, `type_evenement`, `resultat` sont des `String`/`Boolean` libres : pas d'`Enum` ni de contrainte `CHECK`. Risque d'incohérence (`"actif"` vs `"ACTIF"`).
- Pas de champ `created_at/updated_at` systématique ni de soft-delete.

### 2.4 Persistance (`database.py`)
- **Bon** : pattern `SessionLocal` + `get_db` générateur standard.
- **Pas de pool tuning** (`pool_size`, `max_overflow`, `pool_pre_ping`) — `pool_pre_ping=True` est quasi obligatoire avec Postgres pour éviter les `connection closed` après inactivité.
- **Aucune migration** : `Base.metadata.create_all()` ne gère pas les évolutions de schéma. Alembic est indispensable pour l'entreprise.
- L'URL DB par défaut contient les credentials en dur (`database.py:5`).

### 2.5 Config / bootstrap (`main.py`, docker-compose)
- `create_all()` + `seed()` au **niveau module** (`main.py:9-10`) : exécutés à l'import, donc aussi lors des tests (`test_main.py` importe `app.main`), ce qui couple les tests à une vraie DB. À déplacer dans un `lifespan` FastAPI.
- `requirements.txt` **sans versions épinglées** : builds non reproductibles, risque de régression silencieuse.
- `--reload` activé dans `docker-compose.yml:29` (mode dev en « prod »).
- Secrets en clair partout : `POSTGRES_PASSWORD: password`, `GF_SECURITY_ADMIN_PASSWORD=admin`, password Postgres dans `datasource.yml:17`. Aucun usage de `.env`/secrets Docker.
- `pydantic-settings` est installé mais **jamais utilisé** : aucune classe `Settings` centralisant la config (tout passe par `os.getenv` épars).

### 2.6 Observabilité (`monitoring.py`, prometheus, grafana)
- **Bon point de départ** : logger structuré-ish, 4 métriques custom, scrape configuré, Grafana provisionné avec datasources Prometheus + Postgres.
- Logs en **format texte** (pas JSON) → difficile à parser/ingérer (Loki/ELK). Pas de `correlation_id`/`trace_id`.
- Pas de **tracing distribué** (OpenTelemetry) alors que le flux ESP→backend→DB s'y prête.
- Pas d'alerting (Alertmanager) ni de SLO définis.
- Datasource Postgres dans Grafana avec credentials en clair et accès Admin anonyme → exfiltration de données triviale.

### 2.7 Frontend (`App.vue`)
- **Largement maquetté** : « État Actuel » est **codé en dur** (`Occupation: 24/50`, `10:45 AM` — `App.vue:11-12`). Seul `fetchPrediction` appelle réellement l'API.
- URL backend **en dur** `http://localhost:8000` (`App.vue:38`) → ne marche qu'en local, pas de variable d'env Vite.
- `axios` est dans les dépendances mais le code utilise `fetch`.
- Aucune connexion WebSocket temps réel (le cœur du sujet est « temps réel ») ; pas de gestion d'état (Pinia), pas de routing, pas de composants découpés, pas de tests.
- Monté en volume + `--reload` : config de dev, pas de build de prod (`npm run build` + nginx).

### 2.8 Déploiement (`docker-compose.yml`, Dockerfiles)
- **Bon** : healthcheck DB, `depends_on: condition`, Postgres non exposé sur l'hôte, volume nommé `pgdata`.
- Backend Dockerfile : tourne en **root**, pas de `USER` non-privilégié, pas de multi-stage, dépendances non figées.
- Pas de `restart: unless-stopped`, pas de limites de ressources (`mem_limit`, `cpus`), pas de réseaux séparés.
- Un seul `docker-compose.yml` pour dev et prod (pas d'override). `--reload` en commande par défaut.
- Pas de CI/CD ni de scan de vulnérabilités d'image.

---

## 3. Points critiques classés par sévérité

### CRITIQUE

| # | Constat | Fichier | Recommandation |
|---|---------|---------|----------------|
| C1 | **Aucune authentification/autorisation** sur `/scan`, `/ws/esp`, `/predict`. N'importe qui peut injecter des scans et fausser les accès/événements. | `rfid_controller.py`, `iot_controller.py` | Authentifier les ESP (token/clé partagée ou mTLS), protéger les routes d'admin par JWT/OAuth2, valider que la MAC est un lecteur connu avant d'accepter un scan. |
| C2 | **CORS `allow_origins=["*"]` avec `allow_credentials=True`** — combinaison interdite par la spec et dangereuse. | `main.py:14-20` | Lister explicitement les origines autorisées ; ne pas combiner `*` et credentials. |
| C3 | **Secrets en clair** (DB, Grafana admin, datasource Postgres) commités dans le repo. | `docker-compose.yml:7-8,53`, `datasource.yml:17`, `database.py:5` | Externaliser via `.env`/Docker secrets/vault ; ne jamais committer de mots de passe. |
| C4 | **Grafana anonyme avec rôle Admin** + datasource Postgres en clair → lecture/écriture totale des données sans login. | `docker-compose.yml:54-55` | Désactiver l'accès anonyme ou le limiter à `Viewer` ; restreindre la datasource Postgres (utilisateur read-only). |
| C5 | **« ML » factice** (`random`) — l'exigence ML n'est pas réellement satisfaite ; `DONNEE_ML` n'est jamais peuplée. | `prediction_service.py` | Calculer de vraies features depuis `events`, persister `MLData`, et implémenter au minimum un modèle simple (détection d'anomalie par seuil, prévision par moyenne mobile). |

### MAJEUR

| # | Constat | Fichier | Recommandation |
|---|---------|---------|----------------|
| M1 | **Pas de migrations** ; `create_all()`+`seed()` au niveau module, exécutés même pendant les tests. | `main.py:9-10`, `database.py` | Introduire Alembic ; déplacer init/seed dans un `lifespan` ; seed conditionné par variable d'env. |
| M2 | **Opération de scan non atomique** (commit lecteur puis commit événement). | `rfid_service.py:32-54`, `iot_service.py:28` | Une seule transaction par scan ; les services ne committent pas, c'est l'appelant (controller) qui commit. |
| M3 | **Couverture de tests quasi nulle** (seul `/health`). | `tests/test_main.py` | Tests unitaires des services (autorisation, alternance entrée/sortie, badge inactif), tests d'intégration WS, DB de test isolée (SQLite/pg testcontainers). |
| M4 | **Cardinalité Prometheus** : labels IP/MAC sur Counters/Gauges + label `esp_ip` contenant une MAC. | `monitoring.py:14-34`, `rfid_service.py:61` | Réduire la cardinalité (pas d'IP en label), corriger le nom `esp_ip`→`mac`, déplacer l'identité device hors des labels. |
| M5 | **Requirements non épinglés** + `--reload` + root container. | `requirements.txt`, `Dockerfile`, `docker-compose.yml:29` | Épingler les versions, retirer `--reload` en prod, ajouter `USER appuser`, multi-stage build. |
| M6 | **Session WS unique longue durée** sans `pool_pre_ping`, pas de gestion de panne DB transitoire. | `iot_controller.py:44`, `database.py:7` | `pool_pre_ping=True`, session par message ou re-création sur erreur, renvoyer une erreur structurée au client au lieu de couper. |
| M7 | **Frontend en dur** (données mockées, URL backend en dur, pas de WS temps réel). | `App.vue:11-12,38` | Variabiliser l'URL (`import.meta.env`), brancher les vraies données, ouvrir une connexion WS pour le temps réel exigé. |

### MINEUR

| # | Constat | Fichier | Recommandation |
|---|---------|---------|----------------|
| m1 | `datetime.utcnow` déprécié, datetimes naïfs. | `models.py`, services | `datetime.now(timezone.utc)`, colonnes `DateTime(timezone=True)`. |
| m2 | Champs statut/type en `String` libre, pas d'`Enum`. | `models.py` | `Enum` SQLAlchemy + Pydantic + contraintes `CHECK`. |
| m3 | `pydantic-settings` installé mais inutilisé. | `requirements.txt` | Créer une classe `Settings` centralisant toute la config. |
| m4 | Logs texte non structurés, pas de `correlation_id`. | `monitoring.py` | Logging JSON (structlog) + corrélation de requête. |
| m5 | `Event` FK sans `ondelete` ; suppression lecteur/badge problématique. | `models.py:107-108` | Définir la stratégie (`SET NULL`/`RESTRICT`) explicitement. |
| m6 | `axios` en dépendance non utilisé ; pas de build de prod frontend. | `package.json`, `frontend/Dockerfile` | Nettoyer la dépendance, ajouter étape `build`+nginx pour la prod. |
| m7 | Pas de `restart`, ni limites ressources, ni réseaux séparés. | `docker-compose.yml` | Ajouter `restart: unless-stopped`, limites CPU/mémoire, réseau backend/monitoring séparé. |
| m8 | `Prediction`/`MLData` croissent sans purge ni rétention. | `prediction_service.py` | Politique de rétention / agrégation. |

---

## 4. Roadmap priorisée

### Phase 0 — Quick wins (jours, fort impact / faible effort)
1. **Sécuriser la config** : déplacer tous les secrets dans un `.env` non commité ; restreindre CORS (C2) ; passer Grafana anonyme en `Viewer` ou désactiver (C4). 
2. **Épingler `requirements.txt`** et retirer `--reload` du compose de prod (M5).
3. **Corriger le label `esp_ip`→`mac`** et retirer les IP des labels Prometheus (M4).
4. **Variabiliser l'URL backend** du frontend via `import.meta.env` (M7 partiel).
5. **`pool_pre_ping=True`** sur l'engine (M6 partiel).

### Phase 1 — Fiabilité & qualité (1-2 semaines)
6. **Introduire Alembic** et déplacer `create_all`/`seed` dans un `lifespan` ; seed gardé par env (M1).
7. **Rendre le scan atomique** : services sans commit, transaction unique gérée par le controller (M2).
8. **Suite de tests** : services (autorisation, entrée/sortie, badge inactif), intégration WS, DB de test isolée ; viser >70% sur les services (M3).
9. **Enums + contraintes** sur statut/type (m2), datetimes tz-aware (m1).
10. **Settings centralisés** via `pydantic-settings` (m3).

### Phase 2 — Sécurité & production (2-4 semaines)
11. **AuthN/AuthZ** : token/clé ou mTLS pour les ESP, validation MAC connue avant scan ; JWT/OAuth2 pour l'admin (C1).
12. **Hardening Docker** : utilisateur non-root, multi-stage, limites ressources, `restart`, réseaux séparés, build prod frontend + nginx (M5, m6, m7).
13. **Heartbeat / mise offline automatique** des lecteurs muets (tâche de fond / TTL).
14. **Logging JSON + corrélation**, et idéalement OpenTelemetry tracing + Alertmanager (m4).

### Phase 3 — Chantiers de fond (1 mois+)
15. **ML réel** : pipeline features depuis `events`, peuplement de `MLData`, modèle d'anomalie + prévision, séparation entraînement/inférence, versioning de modèle (C5).
16. **Frontend complet** : WebSocket temps réel, état (Pinia), composants, données réelles, tests (M7).
17. **CI/CD** : lint, tests, scan d'images, déploiement ; observabilité SLO/alerting.

---

## 5. Bonnes pratiques déjà respectées

- **Architecture en couches** controllers → services → ORM, controllers minces, logique métier isolée.
- **DTO Pydantic distincts** des modèles ORM (`schemas.py`) avec `from_attributes`.
- **MCD soigné** : clé naturelle MAC, index sur FK et colonnes de recherche, unicité (`uid`, `email`), `cascade delete` user→badges, relation 1,1 `event↔ml_data`.
- **Gestion d'erreurs** des services avec `rollback` + métrique `SYSTEM_ERRORS` + log.
- **Idempotence du seed** (`seed.py` vérifie l'existence avant insertion).
- **Healthcheck DB** + `depends_on: condition: service_healthy` ; Postgres non exposé sur l'hôte ; volume nommé persistant.
- **Observabilité dès le départ** : métriques custom + instrumentation FastAPI + Grafana/Prometheus provisionnés as-code.
- **Documentation** : README riche et clair (architecture, MCD, workflow, API).
- **WebSocket** : gestion `WebSocketDisconnect` + nettoyage du `ConnectionManager` + mise offline du lecteur.
