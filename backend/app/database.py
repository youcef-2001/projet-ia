import os

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@db:5432/presence_db")

# SQLite (utilisé pour les tests / smoke tests) interdit par défaut le partage
# d'une connexion entre threads : FastAPI sert pourtant les requêtes sur des
# threads différents, d'où check_same_thread=False. Inoffensif pour Postgres.
_connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(
    DATABASE_URL,
    connect_args=_connect_args,
    pool_pre_ping=True,  # détecte les connexions mortes (Postgres) avant usage
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    """Dépendance FastAPI : fournit une session et la ferme systématiquement."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()