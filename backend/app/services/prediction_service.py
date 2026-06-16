"""Module ML simulé : prédiction de fréquentation / anomalie par salle."""
import random
from typing import Optional

from sqlalchemy.orm import Session

from app.models import Prediction, Room
from app.monitoring import SYSTEM_ERRORS, logger


def generate_predictions(db: Session, room_id: int) -> Optional[Prediction]:
    """
    Génère une prédiction factice (occupation + anomalie) pour une salle.

    Retourne None si la salle n'existe pas. Le modèle est un simple tirage
    aléatoire borné par la capacité de la salle (placeholder du futur module ML).
    """
    room = db.query(Room).filter(Room.id == room_id).first()
    if not room:
        return None

    # capacity a un défaut de 0 mais peut être NULL sur d'anciennes lignes.
    capacity = room.capacity or 0
    pred_occupancy = random.randint(0, capacity)
    anomaly = random.choice([True, False, False, False])  # ~25% de chance
    confidence = round(random.uniform(0.7, 0.99), 2)

    prediction = Prediction(
        room_id=room_id,
        predicted_occupancy=pred_occupancy,
        predicted_anomaly=anomaly,
        confidence=confidence,
    )
    try:
        db.add(prediction)
        db.commit()
        db.refresh(prediction)
        return prediction
    except Exception as e:
        db.rollback()
        logger.error(f"Erreur génération prédiction salle {room_id}: {e}")
        SYSTEM_ERRORS.labels(service="prediction_service", error_type=type(e).__name__).inc()
        raise
