"""API du service IA : prévision d'affluence par salle/jour (KNN probabiliste)."""
import datetime as _dt
import os

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Room
from app.monitoring import logger
from app.services import ml_service

router = APIRouter(prefix="/api/ml")


def _parse_date(date: str | None) -> _dt.date:
    if not date:
        return _dt.date.today()
    try:
        return _dt.date.fromisoformat(date)
    except ValueError:
        raise HTTPException(status_code=400, detail="Format de date attendu: YYYY-MM-DD")


@router.get("/status")
def ml_status():
    """Indique si le modèle est entraîné et où il est stocké."""
    return {
        "model_trained": os.path.exists(ml_service.MODEL_PATH),
        "classifier_trained": os.path.exists(ml_service.CLF_MODEL_PATH),
        "model_path": ml_service.MODEL_PATH,
    }


@router.post("/train")
def ml_train(db: Session = Depends(get_db)):
    """(Ré)entraîne le modèle KNN sur l'historique des événements."""
    logger.info("Entraînement du modèle d'affluence demandé via API")
    return ml_service.train(db)


@router.get("/predict/{room_id}")
def ml_predict(room_id: int, date: str | None = Query(default=None), db: Session = Depends(get_db)):
    """Prévision du nombre de personnes pour une salle à une date donnée."""
    result = ml_service.predict_room_day(db, room_id, _parse_date(date))
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@router.get("/predict")
def ml_predict_all(
    date: str | None = Query(default=None),
    db: Session = Depends(get_db),
):
    """Prévision pour TOUTES les salles à une date (vue dashboard)."""
    jour = _parse_date(date)
    rooms = db.query(Room).order_by(Room.id).all()
    out = []
    for room in rooms:
        res = ml_service.predict_room_day(db, room.id, jour)
        if "error" not in res:
            res["kind"] = room.kind
            res["batiment"] = room.batiment
            out.append(res)
    return {"date": jour.isoformat(), "predictions": out}
