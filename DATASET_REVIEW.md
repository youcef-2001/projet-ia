# Audit du dataset RFID généré — `backend/app/data_generator.py`

Auditeur : Data Scientist senior (simulation de données)
Date : 2026-06-16
Périmètre : critique uniquement (aucune modification de code). Objectif : rendre le dataset plus réaliste d'un environnement de travail français et mieux mocké **par salle** et **par type d'utilisateur/de salle**, avec du signal exploitable pour le ML (prévision d'affluence + détection d'anomalies).

---

## 1) Résumé

Le générateur est propre, bien paramétré et déjà crédible au niveau **macro** (calendrier français, fériés mobiles via Butcher `_paques` l.90, vacances `_est_vacances` l.121, pics 8-10h/12-14h/17-19h, week-ends/fériés vides, refus ~3.5%). C'est une bonne base.

Mais au niveau **micro** — c'est-à-dire exactement ce que le ML doit apprendre — il y a un défaut structurel majeur : **il n'existe aucun signal différenciant par salle ni par individu.** Concrètement :

- Les événements ne sont **pas affectés à une salle précise de façon cohérente** : `_generer_evenements` (l.283-380) pioche un lecteur au hasard via `rng.choice(entree_readers)` (l.334, 355) parmi TOUS les lecteurs de type openspace+amphi+meeting confondus. Une salle de réunion de 6 places et un amphi de 200 reçoivent donc le **même profil de flux**. Le type de salle (`kind`) sert seulement à router vers cafétéria vs reste, puis est jeté.
- Il n'y a **aucune habitude individuelle stable** : chaque jour présent, chaque user retire de nouvelles heures depuis les mêmes gaussiennes globales (`_heure_entree` l.238, `_heure_sortie` l.245). Le « Bob » du lundi n'a aucun lien comportemental avec le « Bob » du mardi. Pas de télétravail, pas d'emploi du temps étudiant, pas de congés multi-jours.
- La **capacity n'est jamais respectée ni même lue** au moment de générer les flux (elle est seulement stockée l.159-160).
- **Aucune anomalie n'est plantée volontairement** (badge volé, accès nocturne, anti-passback), alors que le projet vise explicitement la détection d'anomalies (`Prediction.predicted_anomaly`, models.py l.140).

Conséquence ML : un modèle de prévision d'affluence **par salle** ne pourra rien apprendre de discriminant entre salles (le bruit domine le signal), et un détecteur d'anomalies n'a aucun positif à apprendre/évaluer.

---

## 2) Faiblesses par sévérité

### CRITIQUE (bloquant pour l'objectif « par salle » / ML exploitable)

**C1 — Pas d'affectation cohérente événement → salle.**
`_generer_evenements` l.334/355 : `reader_in = rng.choice(entree_readers)` et `reader_out = rng.choice(entree_readers)` où `entree_readers = openspace + amphi + meeting` (l.297-298). Un même user entre dans une salle aléatoire et sort d'une autre. Il n'y a ni notion de salle de rattachement, ni de réunion, ni de cours. Toutes les salles non-cafétéria sont statistiquement **interchangeables** → zéro signal inter-salles.

**C2 — Aucun profil horaire/occupation par TYPE de salle.**
La forme du flux est unique (`_heure_entree`/`_heure_sortie`/`_heure_cafeteria`). Or :
- meeting : créneaux discrets (9h, 10h, 14h, 16h), pics courts, occupation 40-80% de capacity pendant 30-90 min, beaucoup de no-shows ;
- openspace : remplissage progressif 8-10h, plateau journée, vidage 17-19h, occupation moyenne 50-70% ;
- cafeteria : double pic court 12h-13h30, rotation très rapide (dwell 20-40 min), débit >> occupation instantanée ;
- amphi : créneaux de cours en blocs (8h/10h15/13h30/15h45), quasi-vide hors créneaux, taux de remplissage très variable selon le cours.
Rien de tout cela n'existe : `_facteur_jour` (l.259) module seulement un scalaire global.

**C3 — Aucune habitude individuelle ni télétravail/absences.**
`PROBA_PRESENCE_JOUR` (l.49) est une simple Bernoulli i.i.d. par jour (l.321-326). Pas de profil persistant par user (horaire récurrent, salle d'affectation, jours sur site). Pas de télétravail (ex. employé sur site mar/jeu), pas de congés/RTT en blocs, pas d'emploi du temps étudiant. Le ML ne peut pas exploiter la régularité hebdomadaire individuelle, qui est pourtant le signal #1 d'un vrai bâtiment.

**C4 — `kind` non persisté en base.**
`Room` (models.py l.65-76) n'a pas de colonne `kind`. Le type de salle, central pour le réalisme, est perdu : impossible pour le ML/dashboard de stratifier par type de salle sans heuristique sur le nom. (NB : modification de schéma → à arbitrer, mais à signaler.)

### MAJEUR

**M1 — Capacity jamais respectée.**
Aucune borne « occupation instantanée ≤ capacity » (cf. l.159-160 stockée, jamais relue). Avec une affectation salle correcte, il faudra plafonner et générer des no-shows/refoulements crédibles.

**M2 — Pas de loi d'arrivée par salle (Poisson).**
Le nombre de présents par jour découle d'une somme de Bernoulli au niveau user, jamais d'un processus d'arrivée par salle/créneau. Une intensité de Poisson non-homogène λ(salle, heure, jour) donnerait une variance et des pics réalistes et directement labellisables.

**M3 — Aucune anomalie plantée.**
Pas de badge volé (même UID sur 2 sites en 5 min), pas d'accès hors horaire (3h du matin), pas de rafale de refus (tentative d'intrusion), pas d'anti-passback violé (2 entrées sans sortie). `Prediction.predicted_anomaly` n'a donc aucune vérité terrain.

**M4 — Pas de pauses déjeuner échelonnées par site.**
`_heure_cafeteria` (l.252) est une gaussienne unique (µ=12h45, σ=30). En vrai, le déjeuner est échelonné (services), varie par site, et Marseille déjeune un peu plus tard. Pas de fuseau/décalage par bâtiment.

**M5 — Saisonnalité grossière.**
`_est_vacances` applique un -55% binaire (`f *= 0.45`, l.263). Manquent : ponts français (vendredi après un jeudi férié, lundi avant mardi férié), creux estival progressif (juillet < août < creux mi-août), rentrée de septembre chargée, RTT individuels, « molle » de fin décembre.

### MINEUR

**m1 — Cafétéria comptée comme « entree » pas comme passage.** l.344 : `type_evenement="entree"`. Sémantiquement un passage cafétéria n'est pas une entrée bâtiment ; brouille les comptages d'occupation par salle.

**m2 — `MLData.feature_3` (durée) est un pur bruit `rng.uniform(0.5, 9.0)` (l.394)**, décorrélé de l'entrée/sortie réelle. Inutile, voire trompeur pour le ML.

**m3 — Sorties sans lien physique avec l'entrée** (salle de sortie ≠ salle d'entrée), incohérent et empêche le calcul de dwell time par salle.

**m4 — Doublons de prénoms** dans `_PRENOMS` (« Alice » l.55 et l.57) — cosmétique.

**m5 — Refus à heure uniforme** `time(randint(7,19), …)` (l.370) : devrait suivre le flux légitime (plus de refus aux heures de pointe).

---

## 3) Améliorations concrètes (pistes d'implémentation)

### A. Persister et exploiter le type de salle (résout C1/C4)
- **Schéma** : ajouter `kind = Column(String, index=True)` à `Room` (models.py). Renseigner depuis `ROOMS_SPEC` dans `_creer_salles_et_readers` (l.159-160).
- **Constante** `ROOM_PROFILES: Dict[str, Dict]` indexée par `kind`, ex. :
  ```
  ROOM_PROFILES = {
    "meeting":   {"slots": [(9,0),(10,30),(14,0),(16,0)], "fill": (0.4,0.8),
                  "dwell_min": (30,90), "no_show": 0.15, "open": (8,19)},
    "openspace": {"arr_peak": (8.0,10.0), "dep_peak": (17.0,19.0),
                  "fill": (0.45,0.75), "dwell_min": (360,540), "open": (7,20)},
    "cafeteria": {"slots": [(12,0),(12,45),(13,15)], "fill": (0.2,0.6),
                  "dwell_min": (20,40), "rotation": True, "open": (11,15)},
    "amphi":     {"slots": [(8,0),(10,15),(13,30),(15,45)], "fill": (0.1,0.9),
                  "dwell_min": (90,105), "no_show": 0.2, "open": (8,18)},
  }
  ```
- **Routage** : remplacer `rng.choice(entree_readers)` (l.334/355) par un choix de salle pondéré, puis un lecteur **de cette salle**. Conserver la salle choisie pour générer l'entrée ET la sortie cohérentes (résout m3).

### B. Génération centrée salle/créneau avec Poisson (résout M2/M1)
- Nouvelle fonction `_generer_salle_jour(room, kind, profil_jour, rng)` : pour chaque créneau/heure, tirer un nombre d'arrivées `n ~ Poisson(λ)` avec `λ = capacity * fill * facteur_jour * profil_horaire(h)`.
- **Plafond capacity** : suivre l'occupation instantanée (file d'arrivées/départs) ; si `occ ≥ capacity`, convertir l'arrivée excédentaire en **no-show** ou refoulement (résout M1). Fonction `_occupation_courante(...)`.
- Profils horaires par kind via une fonction `_poids_horaire(kind, heure) -> float` (cloche pour openspace, créneaux discrets pour meeting/amphi, double pic pour cafeteria).

### C. Habitudes individuelles + télétravail + congés (résout C3)
- Enrichir le profil créé dans `_creer_users_et_badges` (l.225-230) avec des attributs **persistants par user** tirés une seule fois :
  - `home_room_id` (salle de rattachement : openspace pour employés, amphi/labo pour étudiants) ;
  - `h_in_mu`, `h_out_mu` (heures personnelles, ex. `rng.gauss(8h45, 25)`) → puis bruit journalier faible σ≈10 min autour de SA moyenne (au lieu de la gaussienne globale l.240/247) ;
  - `jours_site` (set de weekdays) : pour `employe`, modèle télétravail français typique « présent mar+jeu, +1 jour aléatoire » ; `_est_sur_site(profil, jour)`.
  - `etudiant` : `emploi_du_temps` = mapping {weekday → liste de créneaux amphi/labo}.
  - `visiteur` : 1-3 venues ponctuelles sur toute la période, plage horaire diurne, souvent salle meeting.
  - `admin` : présence élevée et stable, horaires larges, accès multi-bâtiments.
- **Congés/RTT** : générer par user 1-3 blocs d'absence (`_tirer_conges(rng, debut, fin)`) de 1-15 jours ouvrés (semaine d'été, semaine de Noël, RTT isolés). Pendant un bloc → user absent.
- Constante `PRESENCE_PAR_TYPE: Dict[str, float]` (taux de présence sur jours « éligibles ») remplaçant les multiplicateurs en dur l.322-325.

### D. Saisonnalité fine française (résout M5)
- `_facteur_saison(jour) -> float` séparé de l'aléa, additionnant :
  - **ponts** : si `jour` est un vendredi suivant un jeudi férié ou un lundi précédant un mardi férié → `*0.4` (`_est_pont`).
  - **été progressif** : juillet `*0.7`, première quinzaine d'août `*0.5`, semaine du 15 août `*0.35`, dernière semaine d'août remontée `*0.7`.
  - **rentrée** : 1ʳᵉ quinzaine de septembre `*1.1`.
  - **fin d'année** : 23 déc → 2 jan `*0.3`.
- Garder l'aléa événementiel (`_facteur_jour` l.267-271) mais le séparer du déterministe pour que le ML capte la tendance.

### E. Pauses déjeuner échelonnées / fuseau site (résout M4)
- `LUNCH_OFFSET_PAR_SITE = {"Paris-Nord": 0, "Lyon-Tech": +10, "Marseille-Sud": +20}` (minutes) appliqué dans `_heure_cafeteria`.
- Échelonner par « service » : tirer le créneau dans `cafeteria.slots` plutôt qu'une seule gaussienne, pour étaler 12h-14h.

### F. Anomalies plantées (résout M3) — vérité terrain pour la détection
Fonction `_injecter_anomalies(events, profils, salles, rng, taux≈0.003)` ajoutant, avec un **flag traçable** (ex. via un UID/marqueur ou une table/colonne dédiée si autorisé) :
1. **Badge volé / clonage** : même UID scanné sur 2 bâtiments distants à <5 min d'intervalle (impossible physiquement).
2. **Accès hors horaire** : entrée à 2h-4h du matin un jour ouvré ou un dimanche.
3. **Rafale de refus** : 5-15 refus du même UID inconnu sur un même lecteur en quelques minutes (brute force).
4. **Anti-passback violé** : 2 « entree » consécutives sans « sortie » (talonnage).
5. **Volume aberrant** : une salle qui dépasse ponctuellement sa capacity (capteur défaillant).
Documenter le compte d'anomalies dans `_afficher_stats` (l.422) pour pouvoir évaluer precision/recall.

### G. Cohérence MLData (résout m1/m2)
- Cafétéria : `type_evenement="passage"` (ou `"cafeteria"`) au lieu de `"entree"` (l.344).
- `feature_3` : calculer la **vraie durée** = `ts_out - ts_in` par user/jour (l.394), au lieu de `rng.uniform`. Ajouter `feature_4 = room.kind` (encodé) et `feature_5 = occupation_salle_au_moment` pour donder du signal exploitable.

---

## 4) Priorisation

| Prio | Item | Effort | Impact ML | Dépend de |
|------|------|--------|-----------|-----------|
| P0 | **A** Persister `kind` + routage événement→salle cohérent | M | Très élevé (débloque tout le « par salle ») | — |
| P0 | **B** Génération par salle/créneau + Poisson + plafond capacity | L | Très élevé | A |
| P0 | **C** Habitudes individuelles + télétravail + congés | L | Très élevé (signal hebdo) | — |
| P1 | **F** Anomalies plantées étiquetées | M | Élevé (détection d'anomalies) | A |
| P1 | **D** Saisonnalité fine (ponts/été/rentrée) | S | Élevé | — |
| P2 | **E** Déjeuners échelonnés par site | S | Moyen | A |
| P2 | **G** Cohérence MLData (durée réelle, type cafétéria) | S | Moyen | A,B |
| P3 | **m4** dédup prénoms, **m5** refus aux heures de pointe | XS | Faible | — |

Ordre d'exécution recommandé : **A → B → C** (le trio qui crée le signal), puis **F → D** (richesse + anomalies), puis **E → G**, finitions P3.

---

### Annexe — pointeurs code clés
- Routage aléatoire à corriger : `data_generator.py` l.296-299, l.334, l.343, l.355.
- Heures globales à individualiser : `_heure_entree` l.238, `_heure_sortie` l.245, `_heure_cafeteria` l.252.
- Présence i.i.d. à remplacer : l.319-326.
- Facteur jour à scinder (déterministe vs aléa) : `_facteur_jour` l.259-272.
- Profils users à enrichir : `_creer_users_et_badges` l.225-230.
- Capacity stockée mais inutilisée : l.159-160.
- `kind` jeté : `_readers_par_kind` l.275-280.
- MLData bruité : `_generer_ml_data` l.383-398.
- Schéma `Room` sans `kind` : `models.py` l.65-76.
