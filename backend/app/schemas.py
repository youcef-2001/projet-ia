from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel


# ---------- Utilisateurs ----------
class UserBase(BaseModel):
    nom: str
    prenom: str
    email: str
    type_utilisateur: str = "visiteur"


class UserCreate(UserBase):
    pass


class User(UserBase):
    id: int

    class Config:
        from_attributes = True


# ---------- Badges ----------
class BadgeBase(BaseModel):
    uid: str
    statut: str = "actif"


class BadgeCreate(BadgeBase):
    user_id: int


class Badge(BadgeBase):
    id: int
    user_id: int
    date_attribution: datetime

    class Config:
        from_attributes = True


# ---------- Salles ----------
class RoomBase(BaseModel):
    nom: str
    batiment: Optional[str] = None
    etage: Optional[int] = None
    capacity: int = 0
    kind: Optional[str] = None  # meeting, openspace, cafeteria, amphi


class RoomCreate(RoomBase):
    pass


class Room(RoomBase):
    id: int

    class Config:
        from_attributes = True


# ---------- Lecteurs (ESP32) ----------
class ReaderBase(BaseModel):
    mac_address: str
    nom: Optional[str] = None
    room_id: Optional[int] = None


class Reader(ReaderBase):
    ip_address: Optional[str] = None
    statut: str
    last_seen: datetime

    class Config:
        from_attributes = True


# ---------- Scan RFID (payload envoyé par l'ESP32) ----------
class RFIDScan(BaseModel):
    mac_address: str
    rfid_uid: str


class ScanResult(BaseModel):
    status: str           # success | denied
    authorized: bool
    event: str            # entree | sortie | refus
    user: Optional[str] = None
    message: str


# ---------- Événements ----------
class Event(BaseModel):
    id: int
    timestamp: datetime
    type_evenement: str
    resultat: bool
    uid_scanne: Optional[str]
    badge_id: Optional[int]
    reader_mac: Optional[str]

    class Config:
        from_attributes = True


# ---------- Prédictions ----------
class Prediction(BaseModel):
    id: int
    room_id: int
    timestamp: datetime
    predicted_occupancy: int
    predicted_anomaly: bool
    confidence: float

    class Config:
        from_attributes = True
