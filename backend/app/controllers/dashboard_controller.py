"""API de monitoring temps réel : lecteurs ESP32, historique des scans, stats."""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Event, RFIDBadge, Reader, Room, User

router = APIRouter(prefix="/api")


@router.get("/readers")
def list_readers(db: Session = Depends(get_db)):
    """Liste des ESP32 connus avec leur salle et leur état."""
    readers = db.query(Reader).all()
    out = []
    for r in readers:
        out.append({
            "mac_address": r.mac_address,
            "nom": r.nom,
            "ip_address": r.ip_address,
            "statut": r.statut,
            "last_seen": r.last_seen.isoformat() if r.last_seen else None,
            "salle": r.room.nom if r.room else None,
            "batiment": r.room.batiment if r.room else None,
            "etage": r.room.etage if r.room else None,
        })
    return out


@router.get("/events")
def list_events(limit: int = 50, db: Session = Depends(get_db)):
    """Historique des derniers passages (badge, utilisateur, salle, résultat)."""
    rows = (
        db.query(Event, RFIDBadge, User, Reader, Room)
        .outerjoin(RFIDBadge, Event.badge_id == RFIDBadge.id)
        .outerjoin(User, RFIDBadge.user_id == User.id)
        .outerjoin(Reader, Event.reader_mac == Reader.mac_address)
        .outerjoin(Room, Reader.room_id == Room.id)
        .order_by(Event.timestamp.desc())
        .limit(min(limit, 200))
        .all()
    )
    out = []
    for event, badge, user, reader, room in rows:
        out.append({
            "id": event.id,
            "timestamp": event.timestamp.isoformat() if event.timestamp else None,
            "type_evenement": event.type_evenement,
            "resultat": event.resultat,
            "rfid_uid": event.uid_scanne,
            "user": f"{user.prenom} {user.nom}" if user else None,
            "salle": room.nom if room else None,
            "mac_address": reader.mac_address if reader else event.reader_mac,
            "reader_nom": reader.nom if reader else None,
        })
    return out


@router.get("/stats")
def stats(db: Session = Depends(get_db)):
    """Compteurs globaux pour le dashboard."""
    total = db.query(Event).count()
    success = db.query(Event).filter(Event.resultat.is_(True)).count()
    refus = total - success
    readers_total = db.query(Reader).count()
    readers_online = db.query(Reader).filter(Reader.statut == "online").count()
    return {
        "total_scans": total,
        "total_success": success,
        "total_refus": refus,
        "readers_total": readers_total,
        "readers_online": readers_online,
        "users": db.query(User).count(),
        "badges": db.query(RFIDBadge).count(),
    }
