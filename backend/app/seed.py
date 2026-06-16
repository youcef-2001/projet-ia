"""Peuplement initial de la base (idempotent).

Lancement :
    docker compose exec backend python -m app.seed
"""
from app.database import Base, SessionLocal, engine
from app.models import RFIDBadge, Reader, Room, User

# Données de démonstration : utilisateurs + badges (1..N badges par user)
USERS = [
    {
        "nom": "Durand", "prenom": "Alice", "email": "alice@campus.fr",
        "type_utilisateur": "etudiant",
        "badges": [("47C12E06", "actif"), ("FA1C0DB1", "actif")],
    },
    {
        "nom": "Martin", "prenom": "Bob", "email": "bob@campus.fr",
        "type_utilisateur": "employe",
        "badges": [("EE32B126", "actif")],
    },
    {
        "nom": "Petit", "prenom": "Chloe", "email": "chloe@campus.fr",
        "type_utilisateur": "admin",
        "badges": [("92D584EA", "actif")],
    },
    {
        "nom": "Roux", "prenom": "David", "email": "david@campus.fr",
        "type_utilisateur": "visiteur",
        "badges": [("088016A0", "inactif")],  # badge désactivé -> accès refusé
    },
]

# Salles avec plusieurs ESP possibles (relation 1,N)
ROOMS = [
    {"nom": "Salle 101", "batiment": "A", "etage": 1, "capacity": 50,
     "kind": "openspace",
     "readers": [("14:08:08:A4:C9:28", "ESP-Entrée-101")]},
    {"nom": "Amphi B", "batiment": "B", "etage": 0, "capacity": 200,
     "kind": "amphi",
     "readers": [("AA:BB:CC:DD:EE:01", "ESP-Porte-Nord"),
                 ("AA:BB:CC:DD:EE:02", "ESP-Porte-Sud")]},
]


def seed():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        # Salles + lecteurs
        for r in ROOMS:
            room = db.query(Room).filter(Room.nom == r["nom"]).first()
            if not room:
                room = Room(nom=r["nom"], batiment=r["batiment"],
                            etage=r["etage"], capacity=r["capacity"],
                            kind=r.get("kind"))
                db.add(room)
                db.flush()
            for mac, nom in r["readers"]:
                if not db.query(Reader).filter(Reader.mac_address == mac).first():
                    db.add(Reader(mac_address=mac, nom=nom, room_id=room.id, statut="offline"))

        # Utilisateurs + badges
        for u in USERS:
            user = db.query(User).filter(User.email == u["email"]).first()
            if not user:
                user = User(nom=u["nom"], prenom=u["prenom"],
                            email=u["email"], type_utilisateur=u["type_utilisateur"])
                db.add(user)
                db.flush()
            for uid, statut in u["badges"]:
                if not db.query(RFIDBadge).filter(RFIDBadge.uid == uid).first():
                    db.add(RFIDBadge(uid=uid, statut=statut, user_id=user.id))

        db.commit()
        print("✅ Base peuplée : "
              f"{db.query(User).count()} users, "
              f"{db.query(RFIDBadge).count()} badges, "
              f"{db.query(Room).count()} salles, "
              f"{db.query(Reader).count()} lecteurs.")
    finally:
        db.close()


if __name__ == "__main__":
    seed()
