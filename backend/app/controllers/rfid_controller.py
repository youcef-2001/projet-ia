"""Routes REST : scan RFID (alternative HTTP au WebSocket) + prédictions ML."""
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from app.database import get_db
from app.monitoring import logger
from app.schemas import Prediction, RFIDScan, ScanResult
from app.services.prediction_service import generate_predictions
from app.services.rfid_service import process_scan

router = APIRouter()


@router.post("/scan", response_model=ScanResult)
def scan_rfid(scan: RFIDScan, request: Request, db: Session = Depends(get_db)):
    """Reçoit un scan {mac_address, rfid_uid} et renvoie le verdict d'accès."""
    logger.info(f"Scan HTTP {scan.rfid_uid} via {scan.mac_address}")
    return process_scan(db, mac_address=scan.mac_address, rfid_uid=scan.rfid_uid, ip=request.client.host)


@router.post("/predict/{room_id}", response_model=Prediction)
def predict_room(room_id: int, db: Session = Depends(get_db)):
    """Prédiction de fréquentation pour une salle."""
    pred = generate_predictions(db, room_id)
    if not pred:
        raise HTTPException(status_code=404, detail="Salle introuvable")
    return pred
