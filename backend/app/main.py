"""
FastAPI Application Entry Point
================================
Sets up CORS, registers all routers, and initializes the database.
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import CORS_ORIGINS
from app.database.db import init_db
from app.api import auth, dataset, training, prediction, dashboard


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database tables on startup."""
    print("Initializing database...")
    init_db()
    print("Database initialized.")
    yield
    print("Application shutting down.")


app = FastAPI(
    title="GNSS Spoofing Detection API",
    description=(
        "Deep Learning-Based Detection of GNSS Spoofing Attacks "
        "Using a Bidirectional Gated Recurrent Unit (Bi-GRU) Network. "
        "MCA Final Year Project."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS – allow React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(auth.router)
app.include_router(dataset.router)
app.include_router(training.router)
app.include_router(prediction.router)
app.include_router(dashboard.router)


@app.get("/")
def root():
    return {
        "message": "GNSS Spoofing Detection API is running.",
        "docs": "/docs",
        "version": "1.0.0",
    }


@app.get("/health")
def health():
    return {"status": "healthy"}
