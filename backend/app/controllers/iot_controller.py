"""WebSocket /ws/esp — réception des scans RFID transmis par les ESP32.

Protocole (JSON, format recommandé par le sujet) :
  ESP  -> serveur : {"mac_address": "14:08:08:A4:C9:28", "rfid_uid": "47C12E06"}
  serveur -> ESP  : {"status": "...", "authorized": bool, "event": "...",
                     "user": "...", "message": "..."}
"""
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.database import SessionLocal
from app.monitoring import IOT_MESSAGES, logger
from app.services.iot_service import mark_reader_offline, register_or_update_reader
from app.services.rfid_service import process_scan

router = APIRouter()


class ConnectionManager:
    """Garde une trace des lecteurs connectés (clé = MAC)."""

    def __init__(self):
        self.active: dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()

    def bind(self, mac: str, websocket: WebSocket):
        self.active[mac] = websocket

    def disconnect(self, mac: str):
        self.active.pop(mac, None)


manager = ConnectionManager()


@router.websocket("/ws/esp")
async def websocket_esp(websocket: WebSocket):
    await manager.connect(websocket)
    client_ip = websocket.client.host if websocket.client else "unknown"
    mac: str | None = None
    db = SessionLocal()
    logger.info(f"ESP connecté depuis {client_ip}")

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"status": "error", "message": "JSON invalide"})
                continue
            if not isinstance(payload, dict):
                await websocket.send_json({"status": "error", "message": "Payload JSON invalide"})
                continue

            mac = payload.get("mac_address", mac)
            rfid_uid = payload.get("rfid_uid")

            if not mac:
                await websocket.send_json({"status": "error", "message": "mac_address requis"})
                continue

            # Toute erreur DB est isolée au message courant : on log, on notifie
            # l'ESP, mais on garde la connexion WebSocket ouverte.
            try:
                if rfid_uid:
                    # process_scan enregistre/maj le lecteur lui-même : pas de
                    # double appel à register_or_update_reader.
                    result = process_scan(db, mac_address=mac, rfid_uid=rfid_uid, ip=client_ip)
                else:
                    # Heartbeat / handshake sans scan.
                    register_or_update_reader(db, mac_address=mac, ip_address=client_ip)
                    result = {"status": "success", "message": "Lecteur enregistré"}
            except Exception as e:  # noqa: BLE001 - on protège la boucle WS
                db.rollback()
                logger.error(f"Erreur traitement message ESP {mac}: {e}")
                IOT_MESSAGES.labels(ip_address=client_ip, status="error").inc()
                await websocket.send_json(
                    {"status": "error", "message": "Erreur interne serveur"}
                )
                continue

            manager.bind(mac, websocket)
            IOT_MESSAGES.labels(ip_address=client_ip, status="success").inc()
            await websocket.send_json(result)

    except WebSocketDisconnect:
        logger.info(f"ESP déconnecté: {mac or client_ip}")
    finally:
        if mac:
            manager.disconnect(mac)
            try:
                # La session a pu rester dans un état incohérent ; on repart propre.
                db.rollback()
                mark_reader_offline(db, mac)
            except Exception as e:  # noqa: BLE001
                logger.error(f"Erreur passage offline {mac}: {e}")
        db.close()
