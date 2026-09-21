from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(
    title="IBVAP API",
    description="Intelligent Border Video Analytics Platform API",
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "name": "IBVAP",
        "status": "online",
        "message": "Intelligent Border Video Analytics Platform API",
    }


@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "service": "ibvap-backend",
    }