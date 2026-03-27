"""
FastAPI Application Entry Point - Passive Crowd Intelligence System Backend
"""
# Load .env file FIRST so all env vars are available before module imports
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.video_routes import router as video_router
from api.cellular_routes import router as cellular_router

app = FastAPI(
    title="Passive Crowd Intelligence System API",
    description="Video analytics (YOLO26) + Cellular network density analytics",
    version="1.0.0",
)

# Allow Next.js frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(video_router)
app.include_router(cellular_router)


@app.get("/")
async def root():
    return {
        "name": "Passive Crowd Intelligence System",
        "status": "running",
        "docs": "/docs",
    }
