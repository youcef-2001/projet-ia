"""Gestion du cycle de vie des lecteurs ESP32 (identifiés par MAC)."""
from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session

from app.models import Reader
from app.monitoring import DEVICE_STATUS, SYSTEM_ERRORS, logger


def register_or_update_reader(
    db: Session, mac_address: str, ip_address: Optional[str] = None
) -> Reader:
    """Crée le lecteur s'il est inconnu, sinon met à jour ip / last_seen / statut."""
    try:
        reader = db.query(Reader).filter(Reader.mac_address == mac_address).first()
        if not reader:
            reader = Reader(
                mac_address=mac_address,
                ip_address=ip_address,
                statut="online",
            )
            db.add(reader)
            logger.info(f"Nouveau lecteur enregistré: {mac_address}")
        else:
            reader.last_seen = datetime.utcnow()
            reader.statut = "online"
            if ip_address:
                reader.ip_address = ip_address

        db.commit()
        db.refresh(reader)
        DEVICE_STATUS.labels(ip_address=reader.ip_address or "?", mac_address=mac_address).set(1)
        return reader
    except Exception as e:
        db.rollback()
        logger.error(f"Erreur enregistrement lecteur {mac_address}: {e}")
        SYSTEM_ERRORS.labels(service="iot_service", error_type=type(e).__name__).inc()
        raise


def mark_reader_offline(db: Session, mac_address: str) -> None:
    """Passe un lecteur à 'offline' (no-op si la MAC est inconnue)."""
    try:
        reader = db.query(Reader).filter(Reader.mac_address == mac_address).first()
        if reader:
            reader.statut = "offline"
            reader.last_seen = datetime.utcnow()
            db.commit()
            DEVICE_STATUS.labels(ip_address=reader.ip_address or "?", mac_address=mac_address).set(0)
            logger.info(f"Lecteur {mac_address} hors ligne")
    except Exception as e:
        db.rollback()
        logger.error(f"Erreur mise hors ligne {mac_address}: {e}")
        SYSTEM_ERRORS.labels(service="iot_service", error_type=type(e).__name__).inc()
        raise
