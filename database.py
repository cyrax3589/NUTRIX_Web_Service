import os
import uuid
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.server_api import ServerApi

# Load environment variables from .env file if present
load_dotenv()

MONGO_URI = os.getenv("MONGODB_URI")


# ============================================================================
# In-Memory Collection Mock for Local Testing (Zero DB Setup Required)
# ============================================================================

class InMemoryCollection:
    """Lightweight in-memory MongoDB-compatible collection for local testing."""
    def __init__(self, name: str):
        self.name = name
        self._docs: List[Dict[str, Any]] = []

    def insert_one(self, document: Dict[str, Any]):
        doc = dict(document)
        if "_id" not in doc:
            doc["_id"] = str(uuid.uuid4())
        self._docs.append(doc)
        class InsertResult:
            inserted_id = doc["_id"]
        return InsertResult()

    def find_one(self, query: Optional[Dict[str, Any]] = None, sort=None) -> Optional[Dict[str, Any]]:
        if not self._docs:
            return None
        filtered = self._docs
        if query:
            filtered = [
                d for d in self._docs
                if all(str(d.get(k)) == str(v) for k, v in query.items() if k != "_id")
            ]
        if not filtered:
            return None
        return dict(filtered[-1])

    def find(self, query: Optional[Dict[str, Any]] = None, sort=None):
        filtered = list(self._docs)
        if query:
            filtered = [
                d for d in filtered
                if all(str(d.get(k)) == str(v) for k, v in query.items() if k != "_id")
            ]
        filtered.reverse()

        class MockCursor(list):
            def limit(self, n: int):
                return [dict(x) for x in self[:n]]

        return MockCursor(filtered)


# ============================================================================
# Database Client Initialization
# ============================================================================

IS_LOCAL_MOCK_MODE = False

if MONGO_URI:
    try:
        client = MongoClient(
            MONGO_URI,
            serverSelectionTimeoutMS=5000,
            server_api=ServerApi("1"),
            tz_aware=True,
        )
        db = client["NUTRIX"]
        readings_collection = db["readings"]
        devices_collection = db["devices"]
        recommendations_collection = db["recommendations"]
        IS_LOCAL_MOCK_MODE = False
    except Exception as e:
        print(f"WARNING: Could not connect to MongoDB URI ({e}). Falling back to In-Memory Mock Mode.")
        client = None
        db = None
        readings_collection = InMemoryCollection("readings")
        devices_collection = InMemoryCollection("devices")
        recommendations_collection = InMemoryCollection("recommendations")
        IS_LOCAL_MOCK_MODE = True
else:
    client = None
    db = None
    readings_collection = InMemoryCollection("readings")
    devices_collection = InMemoryCollection("devices")
    recommendations_collection = InMemoryCollection("recommendations")
    IS_LOCAL_MOCK_MODE = True

    # Pre-populate sample telemetry matching the Android app screenshot
    sample_reading = {
        "_id": "mock_telemetry_001",
        "device_id": "NUTRIX_001",
        "sensor": "nutrix_multiparameter",
        "raw": 400,
        "unit": "adc",
        "moisture_percent": 8.0,
        "ph_value": 2.8,
        "temperature_c": 29.6,
        "pressure_hpa": 922.0,
        "nitrogen_mg_kg": 0,
        "phosphorus_mg_kg": 0,
        "potassium_mg_kg": 0,
        "timestamp": datetime.now(timezone.utc),
    }
    readings_collection.insert_one(sample_reading)


def check_database():
    """Verify that MongoDB Atlas is reachable or report local mock mode."""
    if IS_LOCAL_MOCK_MODE or client is None:
        return "local_in_memory_mode"
    client.admin.command("ping")
    return True


def save_recommendation(doc: Dict[str, Any]) -> str:
    """Save an AI recommendation document to MongoDB or In-Memory store."""
    result = recommendations_collection.insert_one(doc)
    return str(result.inserted_id)


def get_latest_recommendation(device_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Retrieve the latest AI recommendation."""
    query = {}
    if device_id:
        query["device_id"] = device_id
    return recommendations_collection.find_one(
        query,
        sort=[("timestamp", -1), ("_id", -1)]
    )


def get_recommendations_history(device_id: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
    """Retrieve recent AI recommendations history."""
    query = {}
    if device_id:
        query["device_id"] = device_id
    cursor = recommendations_collection.find(
        query,
        sort=[("timestamp", -1), ("_id", -1)]
    ).limit(limit)
    return list(cursor)
