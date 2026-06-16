"""Logique métier RFID : autorisation d'accès + enregistrement des événements."""
from datetime import datetime

from sqlalchemy.orm import Session

from app.models import Event, RFIDBadge
from app.monitoring import BADGE_SCANS, SYSTEM_ERRORS, logger
from app.services.iot_service import register_or_update_reader


def _next_event_type(db: Session, badge_id: int) -> str:
    """Alterne entree/sortie selon le dernier passage autorisé du badge."""
    last = (
        db.query(Event)
        .filter(Event.badge_id == badge_id, Event.resultat.is_(True))
        .order_by(Event.timestamp.desc())
        .first()
    )
    return "sortie" if last and last.type_evenement == "entree" else "entree"


def process_scan(db: Session, mac_address: str, rfid_uid: str, ip: str = "unknown") -> dict:
    """
    Traite un scan RFID transmis par un ESP32.

    1. Enregistre / met à jour le lecteur (par MAC).
    2. Cherche le badge par UID et décide de l'autorisation.
    3. Crée l'événement (entree/sortie/refus) en base.
    4. Retourne le verdict à renvoyer à l'ESP.
    """
    try:
        reader = register_or_update_reader(db, mac_address=mac_address, ip_address=ip)

        badge = db.query(RFIDBadge).filter(RFIDBadge.uid == rfid_uid).first()
        granted = bool(badge and badge.is_active)

        if granted:
            event_type = _next_event_type(db, badge.id)
            # badge.user_id est NOT NULL, mais on reste défensif si la relation
            # n'est pas chargeable (donnée incohérente) pour ne pas planter le scan.
            user = badge.user
            user_label = f"{user.prenom} {user.nom}" if user else f"badge {badge.uid}"
            message = f"Accès {event_type} autorisé pour {user_label}"
        else:
            event_type = "refus"
            user_label = None
            message = "Badge inconnu ou inactif" if not badge else "Badge inactif"

        event = Event(
            type_evenement=event_type,
            resultat=granted,
            uid_scanne=rfid_uid,
            badge_id=badge.id if badge else None,
            reader_mac=reader.mac_address,
        )
        db.add(event)
        db.commit()
        db.refresh(event)

        room_id = str(reader.room_id) if reader.room_id else "unknown"
        BADGE_SCANS.labels(
            status="success" if granted else "denied",
            room_id=room_id,
            mac=mac_address,
        ).inc()
        logger.info(f"Scan UID={rfid_uid} via {mac_address} -> {event_type} ({granted})")

        return {
            "status": "success" if granted else "denied",
            "authorized": granted,
            "event": event_type,
            "user": user_label,
            "message": message,
        }
    except Exception as e:
        db.rollback()
        logger.error(f"Erreur traitement scan {rfid_uid}@{mac_address}: {e}")
        SYSTEM_ERRORS.labels(service="rfid_service", error_type=type(e).__name__).inc()
        raise
