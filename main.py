from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

from bson import ObjectId
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from database import (
    readings_collection,
    recommendations_collection,
    check_database,
    save_recommendation,
    get_latest_recommendation,
    get_recommendations_history,
)
from ai_service import (
    generate_recommendations_with_gemini,
    calculate_heuristic_recommendation,
    format_recommendation_response,
    RecommendationResponse,
    AIRecommendationData,
    SoilHealthScore,
    FertilizerGuidance,
    CropSuitability,
    SoilContext,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Verify MongoDB when the API starts
    try:
        check_database()
        print("INFO: Connected to MongoDB Atlas successfully.")
    except Exception as e:
        print(f"WARNING: MongoDB Atlas connection check deferred: {e}")
    yield


app = FastAPI(
    title="NUTRIX Smart Soil Care API",
    description="IoT Soil Telemetry & Gemini AI Precision Agronomy Backend",
    version="1.2.0",
    lifespan=lifespan,
)

# Enable CORS for mobile apps and web dashboards
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Request & Response Models
# ============================================================================

class SoilReading(BaseModel):
    device_id: str = "NUTRIX_001"
    sensor: str = "nutrix_multiparameter"
    raw: int = Field(ge=0, le=4095)
    unit: str = "adc"
    moisture_percent: Optional[float] = None

    # pH field
    ph_value: Optional[float] = None

    # BMP280 fields (optional: the ESP32 omits them if the sensor fails).
    temperature_c: Optional[float] = Field(default=None, ge=-40, le=85)
    pressure_hpa: Optional[float] = Field(default=None, ge=300, le=1100)

    # NPK fields (optional: omitted if Modbus reading fails).
    nitrogen_mg_kg: Optional[int] = None
    phosphorus_mg_kg: Optional[int] = None
    potassium_mg_kg: Optional[int] = None

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class GenerateRecommendationRequest(BaseModel):
    """Input payload to trigger a fresh Gemini AI analysis."""
    device_id: str = "NUTRIX_001"
    selected_crop: str = "Rice (Paddy)"
    soil_type: Optional[str] = "Loamy Clay"
    reading_id: Optional[str] = None

    # Optional overrides for testing or direct app submissions
    moisture_percent: Optional[float] = None
    ph_value: Optional[float] = None
    temperature_c: Optional[float] = None
    pressure_hpa: Optional[float] = None
    nitrogen_mg_kg: Optional[int] = None
    phosphorus_mg_kg: Optional[int] = None
    potassium_mg_kg: Optional[int] = None


# ============================================================================
# Health Check Endpoint
# ============================================================================

@app.get("/health")
def health():
    try:
        check_database()
        return {
            "api": "ok",
            "database": "connected",
            "ai_engine": "ready",
        }
    except Exception as e:
        return {
            "api": "ok",
            "database": f"offline: {str(e)}",
            "ai_engine": "ready (heuristic mode active)",
        }


# ============================================================================
# Telemetry Readings Endpoints
# ============================================================================

@app.post("/api/v1/readings", status_code=201)
def create_reading(
    reading: SoilReading,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    auto_ai: bool = False,
):
    """
    Ingest real-time soil telemetry from ESP32 / sensors.
    Optionally triggers AI recommendations in the background.
    """
    document = reading.model_dump()

    # Calculate moisture_percent if only raw ADC was supplied
    if document.get("moisture_percent") is None and document.get("raw") is not None:
        raw_val = document["raw"]
        document["moisture_percent"] = max(0.0, min(100.0, round((4095 - raw_val) / 40.95, 1)))

    if readings_collection is not None:
        result = readings_collection.insert_one(document)
        doc_id = str(result.inserted_id)
    else:
        doc_id = "local_mock_id"

    # Optional background AI recommendation generation
    if auto_ai:
        background_tasks.add_task(
            _generate_and_persist_recommendation,
            reading_dict=document,
            device_id=reading.device_id,
            reading_id=doc_id,
            selected_crop="Rice (Paddy)",
        )

    return {
        "message": "Reading saved",
        "id": doc_id,
        "device_id": reading.device_id,
        "soil_moisture_raw": reading.raw,
        "moisture_percent": document.get("moisture_percent"),
        "ph_value": reading.ph_value,
        "nitrogen_mg_kg": reading.nitrogen_mg_kg,
        "phosphorus_mg_kg": reading.phosphorus_mg_kg,
        "potassium_mg_kg": reading.potassium_mg_kg,
        "temperature_c": reading.temperature_c,
        "pressure_hpa": reading.pressure_hpa,
        "timestamp": reading.timestamp.isoformat(),
    }


@app.get("/api/v1/readings/latest")
def get_latest_reading(include_soil_health: bool = True):
    """
    Fetch the latest soil telemetry reading.
    Also attaches soil health score & status for the Android Home Screen card.
    """
    document = None
    if readings_collection is not None:
        document = readings_collection.find_one(
            sort=[("timestamp", -1), ("_id", -1)]
        )

    if document is None:
        # Provide sample baseline reading matching the app UI if DB is empty
        now = datetime.now(timezone.utc)
        document = {
            "_id": "mock_latest_001",
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
            "timestamp": now,
        }

    document["_id"] = str(document["_id"])
    if isinstance(document.get("timestamp"), datetime):
        document["timestamp"] = document["timestamp"].isoformat()

    # Attach soil health score and status for Android Home Screen
    if include_soil_health:
        latest_rec = None
        if recommendations_collection is not None:
            latest_rec = get_latest_recommendation(document.get("device_id"))

        if latest_rec and "soil_health" in latest_rec:
            document["soil_health"] = latest_rec["soil_health"]
            document["soil_health_score"] = latest_rec.get("soil_health_score", 5.8)
            document["soil_status"] = latest_rec.get("soil_status", "Needs Attention")
        else:
            # Generate quick heuristic soil health score for the current reading
            ai_data = calculate_heuristic_recommendation(
                reading=document,
                selected_crop="Rice (Paddy)",
            )
            document["soil_health"] = ai_data.soil_health.model_dump()
            document["soil_health_score"] = ai_data.soil_health.score
            document["soil_status"] = ai_data.soil_health.status

    return document


# ============================================================================
# Gemini AI Recommendations Endpoints (Soil Health, Fertilizer, Crop Tips)
# ============================================================================

def _generate_and_persist_recommendation(
    reading_dict: Dict[str, Any],
    device_id: str = "NUTRIX_001",
    reading_id: Optional[str] = None,
    selected_crop: str = "Rice (Paddy)",
    soil_type: Optional[str] = "Loamy Clay",
) -> Dict[str, Any]:
    """Helper to call Gemini, format response, and persist to MongoDB."""
    ai_data = generate_recommendations_with_gemini(
        reading=reading_dict,
        selected_crop=selected_crop,
        soil_type=soil_type,
    )

    now = datetime.now(timezone.utc)
    resp = format_recommendation_response(
        ai_data=ai_data,
        reading=reading_dict,
        device_id=device_id,
        selected_crop=selected_crop,
        reading_id=reading_id,
        timestamp=now,
    )

    doc_to_save = resp.model_dump()
    if recommendations_collection is not None:
        try:
            inserted_id = save_recommendation(doc_to_save)
            doc_to_save["id"] = inserted_id
        except Exception as e:
            print(f"Error persisting recommendation: {e}")

    return doc_to_save


@app.get("/api/v1/recommendations/latest", response_model=RecommendationResponse)
def get_recommendation_latest(
    device_id: str = Query("NUTRIX_001", description="Device identifier"),
    selected_crop: str = Query("Rice (Paddy)", description="Target crop for the advisory"),
    soil_type: str = Query("Loamy Clay", description="Soil texture type"),
    force_refresh: bool = Query(False, description="Force a new Gemini AI generation"),
):
    """
    Get the latest AI recommendations for the Android app.
    Returns:
    1. Soil health score out of 10 & status badge
    2. Fertilizer guidance bullet points & advisory note
    3. Ranked crop suitability tips with regional vernacular names
    """
    # 1. Check if an existing recommendation is available and force_refresh is False
    if not force_refresh and recommendations_collection is not None:
        cached = get_latest_recommendation(device_id)
        if cached:
            cached["id"] = str(cached.get("_id", cached.get("id", "")))
            if "_id" in cached:
                del cached["_id"]
            return cached

    # 2. Retrieve latest reading from DB to pass to Gemini
    latest_reading = None
    reading_id = None
    if readings_collection is not None:
        latest_reading = readings_collection.find_one(
            {"device_id": device_id} if device_id else {},
            sort=[("timestamp", -1), ("_id", -1)]
        )
        if latest_reading:
            reading_id = str(latest_reading["_id"])

    # If no reading in DB, construct realistic default reading
    if latest_reading is None:
        latest_reading = {
            "device_id": device_id,
            "moisture_percent": 8.0,
            "ph_value": 2.8,
            "temperature_c": 29.6,
            "pressure_hpa": 922.0,
            "nitrogen_mg_kg": 0,
            "phosphorus_mg_kg": 0,
            "potassium_mg_kg": 0,
        }

    # 3. Generate with Gemini AI
    recommendation = _generate_and_persist_recommendation(
        reading_dict=latest_reading,
        device_id=device_id,
        reading_id=reading_id,
        selected_crop=selected_crop,
        soil_type=soil_type,
    )

    return recommendation


@app.post("/api/v1/recommendations/generate", response_model=RecommendationResponse)
def generate_recommendation_on_demand(payload: GenerateRecommendationRequest):
    """
    Trigger immediate on-demand Gemini AI analysis.
    Accepts custom crop selection or sensor overrides.
    """
    reading_dict: Dict[str, Any] = {
        "device_id": payload.device_id,
        "moisture_percent": payload.moisture_percent,
        "ph_value": payload.ph_value,
        "temperature_c": payload.temperature_c,
        "pressure_hpa": payload.pressure_hpa,
        "nitrogen_mg_kg": payload.nitrogen_mg_kg,
        "phosphorus_mg_kg": payload.phosphorus_mg_kg,
        "potassium_mg_kg": payload.potassium_mg_kg,
    }

    reading_id = payload.reading_id

    # If any telemetry field is omitted, fill from latest database reading
    if any(v is None for v in [payload.ph_value, payload.moisture_percent, payload.nitrogen_mg_kg]):
        if readings_collection is not None:
            latest = readings_collection.find_one(
                {"device_id": payload.device_id},
                sort=[("timestamp", -1), ("_id", -1)]
            )
            if latest:
                reading_id = str(latest["_id"])
                for k in ["moisture_percent", "ph_value", "temperature_c", "pressure_hpa", "nitrogen_mg_kg", "phosphorus_mg_kg", "potassium_mg_kg"]:
                    if reading_dict.get(k) is None:
                        reading_dict[k] = latest.get(k)

    # Fallback defaults if still None
    if reading_dict.get("ph_value") is None:
        reading_dict["ph_value"] = 2.8
    if reading_dict.get("moisture_percent") is None:
        reading_dict["moisture_percent"] = 8.0

    return _generate_and_persist_recommendation(
        reading_dict=reading_dict,
        device_id=payload.device_id,
        reading_id=reading_id,
        selected_crop=payload.selected_crop,
        soil_type=payload.soil_type,
    )


@app.get("/api/v1/recommendations/history")
def get_recommendation_history(
    device_id: Optional[str] = Query(None, description="Filter by device ID"),
    limit: int = Query(10, ge=1, le=50, description="Number of past records to return"),
):
    """Retrieve history of AI recommendations for the app's History tab."""
    if recommendations_collection is None:
        return {"history": []}

    records = get_recommendations_history(device_id=device_id, limit=limit)
    formatted = []
    for r in records:
        r["id"] = str(r["_id"])
        del r["_id"]
        formatted.append(r)

    return {
        "count": len(formatted),
        "history": formatted,
    }


@app.get("/api/v1/recommendations/{rec_id}")
def get_recommendation_by_id(rec_id: str):
    """Retrieve a specific recommendation by its ID."""
    if recommendations_collection is None:
        raise HTTPException(status_code=503, detail="Database unavailable")

    doc = None
    try:
        if ObjectId.is_valid(rec_id):
            doc = recommendations_collection.find_one({"_id": ObjectId(rec_id)})
    except Exception:
        pass

    if doc is None:
        doc = recommendations_collection.find_one({"_id": rec_id})
    if doc is None:
        doc = recommendations_collection.find_one({"id": rec_id})

    if not doc:
        raise HTTPException(status_code=404, detail="Recommendation not found")

    doc["id"] = str(doc.get("_id", doc.get("id", "")))
    if "_id" in doc:
        del doc["_id"]
    return doc
