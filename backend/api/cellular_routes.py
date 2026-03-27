"""
Cellular Network Density API Routes.
Provides REST and WebSocket endpoints for population analytics and anomaly detection.
"""
import asyncio
import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from core.baseline_model import analyze_zones, ZONE_METADATA

router = APIRouter()

# Store for active WebSocket connections
_active_connections: list[WebSocket] = []

# Track any manually injected surge zone
_surge_zone: str | None = None


@router.get("/api/zones")
async def get_zones():
    """Return static zone metadata (name, location, etc.)."""
    return {"zones": ZONE_METADATA}


@router.get("/api/cellular/snapshot")
async def cellular_snapshot(surge_zone: str | None = None):
    """Return a one-shot snapshot of cellular analytics for all zones."""
    results = analyze_zones(inject_surge_zone=surge_zone)
    return {"snapshot": results}


@router.get("/api/test-telegram")
async def test_telegram():
    """Send a test Telegram message to verify the bot credentials are correct."""
    from core.baseline_model import send_telegram_alert, ALERT_PHONE
    import os
    token   = os.environ.get("TELEGRAM_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        return {
            "status": "error",
            "message": "TELEGRAM_TOKEN or TELEGRAM_CHAT_ID not set in .env file",
        }
    send_telegram_alert(
        zone_name="TEST ZONE",
        lat=28.6139,
        lon=77.2090,
        rt_pop=9999,
        p_base=500.0,
        ratio=19.9,
    )
    return {"status": "sent", "chat_id": chat_id, "phone": ALERT_PHONE}


@router.post("/api/simulate-surge")
async def simulate_surge(payload: dict):
    """
    Trigger a surge simulation for a given zone.
    Body: { "zone_id": "zone_A" }
    The surge will persist for one broadcast cycle.
    """
    global _surge_zone
    _surge_zone = payload.get("zone_id")
    return {"status": "surge_triggered", "zone": _surge_zone}


@router.websocket("/ws/cellular")
async def cellular_websocket(websocket: WebSocket):
    """
    WebSocket that broadcasts cellular zone analytics every 2 seconds.
    Sends anomaly alerts including GPS coordinates for affected zones.
    """
    global _surge_zone
    await websocket.accept()
    _active_connections.append(websocket)

    try:
        while True:
            results = analyze_zones(inject_surge_zone=_surge_zone)
            _surge_zone = None  # Reset surge after one use

            # Build alert notifications for anomaly zones
            alerts = [
                {
                    "type": "police_alert",
                    "zone_id": r["zone_id"],
                    "zone_name": r["zone_name"],
                    "lat": r["lat"],
                    "lon": r["lon"],
                    "real_time_pop": r["real_time_pop"],
                    "p_base": r["p_base"],
                    "ratio": r["anomaly_ratio"],
                    "message": (
                        f"CRITICAL: Zone '{r['zone_name']}' has {r['real_time_pop']} people "
                        f"({r['anomaly_ratio']}x baseline). Dispatch to GPS ({r['lat']}, {r['lon']})."
                    ),
                }
                for r in results
                if r["is_anomaly"]
            ]

            payload = {
                "zones": results,
                "alerts": alerts,
            }
            await websocket.send_text(json.dumps(payload))
            await asyncio.sleep(2.0)

    except WebSocketDisconnect:
        _active_connections.remove(websocket)
    except Exception:
        if websocket in _active_connections:
            _active_connections.remove(websocket)
