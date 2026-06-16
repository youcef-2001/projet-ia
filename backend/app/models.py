"""
Modèle de données Smart Campus (MCD sujet 3).

Entités principales :
  UTILISATEUR (1,N) ── possède   ── (1,1) BADGE_RFID
  BADGE_RFID  (1,N) ── génère    ── (1,1) EVENEMENT
  LECTEUR     (1,N) ── enregistre── (1,1) EVENEMENT
  EVENEMENT   (1,1) ── alimente  ── (0,1) DONNEE_ML
  SALLE       (1,N) ── héberge   ── (1,1) LECTEUR

Choix techniques :
  - Le LECTEUR (ESP32) est identifié par sa MAC address (clé primaire naturelle).
    Le lecteur RFID étant physiquement unique à l'ESP, il partage le même identifiant.
  - Une SALLE regroupe un nom + plusieurs ESP (relation 1,N vers LECTEUR).
"""
from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
)
from sqlalchemy.orm import relationship

from .database import Base


class User(Base):
    """UTILISATEUR — une personne pouvant détenir plusieurs badges."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    nom = Column(String, nullable=False, index=True)
    prenom = Column(String, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    type_utilisateur = Column(String, default="visiteur")  # admin, employe, visiteur, etudiant

    badges = relationship("RFIDBadge", back_populates="user", cascade="all, delete-orphan")


class RFIDBadge(Base):
    """BADGE_RFID — carte physique. Un utilisateur peut en posséder plusieurs."""

    __tablename__ = "rfid_badges"

    id = Column(Integer, primary_key=True, index=True)
    uid = Column(String, unique=True, index=True, nullable=False)  # UID lu par l'ESP
    date_attribution = Column(DateTime, default=datetime.utcnow)
    statut = Column(String, default="actif", index=True)  # actif, inactif
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)

    user = relationship("User", back_populates="badges")
    events = relationship("Event", back_populates="badge")

    @property
    def is_active(self) -> bool:
        return self.statut == "actif"


class Room(Base):
    """SALLE / LOCALISATION — un nom + un ensemble d'ESP (lecteurs)."""

    __tablename__ = "rooms"

    id = Column(Integer, primary_key=True, index=True)
    nom = Column(String, nullable=False, index=True)
    batiment = Column(String, nullable=True)
    etage = Column(Integer, nullable=True)
    capacity = Column(Integer, default=0)
    # Type de salle : 'meeting', 'openspace', 'cafeteria', 'amphi'.
    # Central pour stratifier l'affluence par type de salle (ML / dashboard).
    kind = Column(String, nullable=True, index=True)

    readers = relationship("Reader", back_populates="room")


class Reader(Base):
    """LECTEUR (ESP32) — identifié par sa MAC. Lecteur RFID intégré = même id."""

    __tablename__ = "readers"

    mac_address = Column(String, primary_key=True, index=True)  # identifiant naturel
    nom = Column(String, nullable=True)
    ip_address = Column(String, index=True, nullable=True)
    statut = Column(String, default="online")  # online, offline
    registered_at = Column(DateTime, default=datetime.utcnow)
    last_seen = Column(DateTime, default=datetime.utcnow)
    room_id = Column(Integer, ForeignKey("rooms.id"), nullable=True, index=True)

    room = relationship("Room", back_populates="readers")
    events = relationship("Event", back_populates="reader")


class Event(Base):
    """EVENEMENT — un passage de badge enregistré par un lecteur."""

    __tablename__ = "events"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    type_evenement = Column(String, nullable=False)  # entree, sortie, refus
    resultat = Column(Boolean, nullable=False)  # True=autorisé, False=refusé
    uid_scanne = Column(String, index=True)  # UID brut (utile si badge inconnu)

    badge_id = Column(Integer, ForeignKey("rfid_badges.id"), nullable=True, index=True)
    reader_mac = Column(String, ForeignKey("readers.mac_address"), nullable=True, index=True)

    badge = relationship("RFIDBadge", back_populates="events")
    reader = relationship("Reader", back_populates="events")
    ml_data = relationship("MLData", back_populates="event", uselist=False)


class MLData(Base):
    """DONNEE_ML — features dérivées d'un événement pour le module ML."""

    __tablename__ = "ml_data"

    id = Column(Integer, primary_key=True, index=True)
    event_id = Column(Integer, ForeignKey("events.id"), unique=True, nullable=False, index=True)
    feature_1 = Column(Float, nullable=True)  # ex: heure de la journée
    feature_2 = Column(Float, nullable=True)  # ex: fréquence d'accès
    feature_3 = Column(Float, nullable=True)  # ex: durée de présence simulée
    label = Column(String, nullable=True)     # optionnel (supervised)
    prediction = Column(String, nullable=True)  # optionnel (sortie modèle)

    event = relationship("Event", back_populates="ml_data")


class Prediction(Base):
    """Prédictions de fréquentation par salle (module ML / dashboard)."""

    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    room_id = Column(Integer, ForeignKey("rooms.id"), index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    predicted_occupancy = Column(Integer)
    predicted_anomaly = Column(Boolean, default=False)
    confidence = Column(Float)

    room = relationship("Room")
