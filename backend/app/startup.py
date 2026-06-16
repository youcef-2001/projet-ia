"""Initialisation applicative : schéma, données, entraînement du modèle IA.

Exécutée en tâche de fond au démarrage pour ne pas bloquer l'API :
  1. Création des tables.
  2. Seed des badges/ESP de démonstration (UIDs réels pour la démo live).
  3. Génération d'un historique réaliste si la base est vide (pour le ML).
  4. Entraînement du modèle d'affluence si aucun modèle n'est encore sauvegardé.

Tout est idempotent : au redémarrage, si les données/le modèle existent déjà,
les étapes coûteuses sont sautées.
"""
import os
import threading

from app.database import Base, SessionLocal, engine
from app.models import Event
from app.monitoring import logger

# En dessous de ce volume d'événements, on (re)génère l'historique de démo.
MIN_EVENTS_FOR_ML = 500


def _initialize() -> None:
    Base.metadata.create_all(bind=engine)

    # 2) Seed démo (Alice, Bob… + ESP réel) — léger et idempotent.
    try:
        from app.seed import seed

        seed()
    except Exception as e:  # noqa: BLE001
        logger.error(f"Seed de démonstration échoué: {e}")

    # 3) Historique réaliste pour le ML (uniquement si la base est quasi vide).
    db = SessionLocal()
    try:
        nb_events = db.query(Event).count()
    finally:
        db.close()

    if nb_events < MIN_EVENTS_FOR_ML:
        try:
            from app.data_generator import generate

            logger.info("Génération de l'historique d'affluence (peut prendre ~1 min)…")
            generate(reset=False, months=6)
        except Exception as e:  # noqa: BLE001
            logger.error(f"Génération du dataset échouée: {e}")

    # 4) Entraînement du modèle d'affluence si absent.
    try:
        from app.services import ml_service

        if not os.path.exists(ml_service.MODEL_PATH):
            db = SessionLocal()
            try:
                logger.info("Entraînement du modèle d'affluence…")
                metrics = ml_service.train(db)
                logger.info(f"Modèle entraîné: {metrics}")
            finally:
                db.close()
        else:
            logger.info("Modèle d'affluence déjà présent, entraînement sauté.")
    except Exception as e:  # noqa: BLE001
        logger.error(f"Entraînement du modèle échoué: {e}")


def run_in_background() -> None:
    """Lance l'initialisation dans un thread démon (API immédiatement servie)."""
    threading.Thread(target=_initialize, name="startup-init", daemon=True).start()
