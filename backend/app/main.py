from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.monitoring import HTTP_REQUESTS

from app.controllers import (
    dashboard_controller,
    iot_controller,
    ml_controller,
    rfid_controller,
)
from app.startup import run_in_background

app = FastAPI(title="Smart Campus API", version="1.0.0")


@app.on_event("startup")
def _on_startup():
    # Schéma + seed + génération du dataset + entraînement du modèle IA,
    # en tâche de fond pour que l'API soit servie immédiatement.
    run_in_background()

# CORS : l'application n'utilise pas de cookies/credentials côté navigateur,
# donc on garde `allow_origins=["*"]` (valide) AVEC `allow_credentials=False`.
# La combinaison `allow_origins=["*"]` + `allow_credentials=True` est interdite
# par la spec CORS (un navigateur rejette `Access-Control-Allow-Origin: *`
# dès que les credentials sont demandés). En la corrigeant, le backend renvoie
# un `Access-Control-Allow-Origin: *` propre, exploitable par n'importe quel front.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def compter_requetes(request: Request, call_next):
    """Compte chaque requête HTTP pour Prometheus (méthode, route, statut)."""
    response = await call_next(request)
    route = request.scope.get("route")
    endpoint = getattr(route, "path", request.url.path)
    HTTP_REQUESTS.labels(
        method=request.method,
        endpoint=endpoint,
        status=response.status_code,
    ).inc()
    return response


app.include_router(rfid_controller.router)
app.include_router(iot_controller.router)
app.include_router(dashboard_controller.router)
app.include_router(ml_controller.router)

@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/metrics", include_in_schema=False)
def metrics():
    """Expose les métriques Prometheus (scans, refus, état des lecteurs…)."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

if __name__ == "__main__":
    import uvicorn
    # En écoute sur 0.0.0.0 pour permettre les connexions réseau (ex: ESP32)
    uvicorn.run(app, host="0.0.0.0", port=8000)
