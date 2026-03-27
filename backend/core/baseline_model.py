"""
Network-Based Population Analytics.
Fixed baseline (P_base) per zone - never updated from live observations.
Anomalies trigger a Telegram alert to the configured phone number.
"""
from __future__ import annotations
import math
import random
import os
from datetime import datetime
from typing import Optional


# ------- Mocked Cellular Data Generator -------

def generate_mock_cellular_data(zones: list[str], base_populations: dict[str, int]) -> dict:
    """
    Generate synthetic cellular signaling data for each zone.
    Returns real-time population estimate per zone, mimicking natural variation.
    """
    data = {}
    hour = datetime.now().hour
    # Simulate daily patterns: low at night, high during day
    diurnal_factor = 0.4 + 0.6 * math.sin(math.pi * (hour - 6) / 14) if 6 <= hour <= 20 else 0.2

    for zone in zones:
        base = base_populations.get(zone, 500)
        noise = random.gauss(0, base * 0.05)
        population = max(0, int(base * diurnal_factor + noise))
        data[zone] = population
    return data


def inject_crowd_event(data: dict, zone: str, multiplier: float = 12.0) -> dict:
    """Simulate a surge event by multiplying zone population."""
    if zone in data:
        data[zone] = int(data[zone] * multiplier)
    return data


# ------- Exponential Smoothing Baseline -------

class ZoneBaseline:
    """
    Fixed baseline per zone. P_base is set once at initialization
    from the zone's known historical average and NEVER updated from
    live observations - it is a stable reference point.
    """

    def __init__(self, zone_id: str, initial_estimate: int = 500):
        self.zone_id = zone_id
        self.baseline = float(initial_estimate)   # Fixed forever

    def get_baseline(self) -> float:
        """Return the fixed baseline. Always the same value."""
        return self.baseline

    def get_dynamic(self, real_time_pop: int) -> float:
        """P_dynamic = P_real_time - P_base"""
        return real_time_pop - self.baseline


# ------- Anomaly Detection & Alerting -------

ANOMALY_MULTIPLIER = 10.0  # Alert when P_real_time > 10 * P_base

# ------- Telegram Notification -------
# To activate:
# 1. Message @BotFather on Telegram -> /newbot -> copy the token
# 2. Start a chat with your bot, then visit:
#    https://api.telegram.org/bot<TOKEN>/getUpdates
#    to get your chat_id
# 3. Set env vars: TELEGRAM_TOKEN and TELEGRAM_CHAT_ID
ALERT_PHONE      = "9342844932"   # For display in alert messages

# Debounce: track last alert time per zone to avoid spam
_last_alert_times: dict[str, float] = {}
ALERT_COOLDOWN_SECONDS = 60  # Only re-alert at most once per minute per zone


def send_telegram_alert(zone_name: str, lat: float, lon: float,
                        rt_pop: int, p_base: float, ratio: float) -> None:
    """
    Send a Telegram message when a zone goes critical.
    Reads TELEGRAM_TOKEN and TELEGRAM_CHAT_ID at call time (lazy)
    so .env values loaded by dotenv are always available.
    """
    token   = os.environ.get("TELEGRAM_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()

    if not token or not chat_id:
        print("[Alert] Telegram skipped - TELEGRAM_TOKEN or TELEGRAM_CHAT_ID not set in .env")
        return
    try:
        import urllib.request
        import urllib.parse
        msg = (
            f"\U0001f6a8 CROWD ALERT - {zone_name}\n"
            f"Population : {rt_pop} ({ratio:.1f}x baseline of {int(p_base)})\n"
            f"GPS        : {lat:.4f}, {lon:.4f}\n"
            f"Notify     : +91 {ALERT_PHONE}\n"
            f"Action     : Dispatch police immediately!"
        )
        params = urllib.parse.urlencode({
            "chat_id": chat_id,
            "text": msg,
        }).encode()
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        req = urllib.request.Request(url, data=params)
        urllib.request.urlopen(req, timeout=5)
        print(f"[Alert] Telegram message sent for {zone_name}")
    except Exception as e:
        print(f"[Alert] Telegram FAILED: {e}")


ZONE_METADATA = {
    "zone_A": {"name": "City Center", "lat": 28.6139, "lon": 77.2090, "base_pop": 500},
    "zone_B": {"name": "Railway Station", "lat": 28.6420, "lon": 77.2172, "base_pop": 800},
    "zone_C": {"name": "Market Area", "lat": 28.6280, "lon": 77.2190, "base_pop": 600},
    "zone_D": {"name": "Stadium", "lat": 28.6363, "lon": 77.2162, "base_pop": 300},
}

# Global state for baseline models
_zone_baselines: dict[str, ZoneBaseline] = {}


def get_or_create_baseline(zone_id: str) -> ZoneBaseline:
    if zone_id not in _zone_baselines:
        meta = ZONE_METADATA.get(zone_id, {"base_pop": 500})
        _zone_baselines[zone_id] = ZoneBaseline(zone_id, initial_estimate=meta["base_pop"])
    return _zone_baselines[zone_id]


def analyze_zones(inject_surge_zone: Optional[str] = None) -> list:
    """
    Run the cellular analytics pipeline for all zones.
    P_base is FIXED (never changes). P_dynamic = P_real_time - P_base.
    Fires Telegram alert (with cooldown) when a zone is anomalous.
    """
    zones = list(ZONE_METADATA.keys())
    base_pops = {z: ZONE_METADATA[z]["base_pop"] for z in zones}

    real_time = generate_mock_cellular_data(zones, base_pops)
    if inject_surge_zone and inject_surge_zone in real_time:
        real_time = inject_crowd_event(real_time, inject_surge_zone, multiplier=12.0)

    results = []
    import time
    now = time.monotonic()

    for zone_id in zones:
        meta = ZONE_METADATA[zone_id]
        baseline_model = get_or_create_baseline(zone_id)
        rt_pop = real_time[zone_id]

        # Use fixed baseline - no update call
        p_base = baseline_model.get_baseline()
        p_dynamic = baseline_model.get_dynamic(rt_pop)

        is_anomaly = rt_pop > ANOMALY_MULTIPLIER * p_base
        alert_level = "critical" if is_anomaly else ("warning" if p_dynamic > 0.5 * p_base else "normal")
        ratio = round(rt_pop / p_base, 2) if p_base > 0 else 0

        # Send Telegram alert (with cooldown)
        if is_anomaly:
            last = _last_alert_times.get(zone_id, 0)
            if now - last > ALERT_COOLDOWN_SECONDS:
                _last_alert_times[zone_id] = now
                send_telegram_alert(
                    zone_name=meta["name"],
                    lat=meta["lat"],
                    lon=meta["lon"],
                    rt_pop=rt_pop,
                    p_base=p_base,
                    ratio=ratio,
                )

        results.append({
            "zone_id": zone_id,
            "zone_name": meta["name"],
            "lat": meta["lat"],
            "lon": meta["lon"],
            "real_time_pop": rt_pop,
            "p_base": round(p_base, 1),
            "p_dynamic": round(p_dynamic, 1),
            "alert_level": alert_level,
            "is_anomaly": is_anomaly,
            "anomaly_ratio": ratio,
            "timestamp": datetime.now().isoformat(),
        })

    return results
