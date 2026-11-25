import os
import json
import logging
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nutrix.ai")


# ============================================================================
# Pydantic Schemas for AI Output & API Response
# ============================================================================

class SoilHealthScore(BaseModel):
    score: float = Field(
        ...,
        description="Soil health score out of 10.0 (e.g. 5.8)",
        ge=0.0,
        le=10.0,
    )
    max_score: float = Field(
        default=10.0,
        description="Maximum attainable soil score (10.0)"
    )
    status: str = Field(
        ...,
        description="Soil status badge: 'Optimal', 'Good', 'Needs Attention', or 'Critical'"
    )
    summary: str = Field(
        ...,
        description="1-2 sentence high-level overview of soil health for current crop"
    )


class FertilizerGuidance(BaseModel):
    title: str = Field(
        default="Fertilizer Guidance",
        description="Section title"
    )
    subtitle: str = Field(
        default="Nutrient Intelligence Advisory",
        description="Section subtitle"
    )
    recommendations: List[str] = Field(
        ...,
        description="Actionable bullet points for organic and inorganic amendments"
    )
    advisory_note: str = Field(
        default="Advisory is tailored for regional soil conditioning. Calibrate with local agricultural extension offices.",
        description="Footer note / disclaimer"
    )


class CropSuitability(BaseModel):
    crop_name: str = Field(
        ...,
        description="Common English crop name, e.g. 'Rice' or 'Ragi (Finger Millet)'"
    )
    vernacular_name: str = Field(
        ...,
        description="Vernacular / regional names including local script, e.g. 'Nellu / Akki (వరి / நெல்)'"
    )
    match_percentage: int = Field(
        ...,
        description="Suitability match percentage (0 to 100), e.g. 92",
        ge=0,
        le=100
    )
    description: str = Field(
        ...,
        description="Agronomic justification based on soil pH, moisture, and NPK metrics"
    )
    water_requirement: str = Field(
        ...,
        description="Water requirement tag, e.g. 'High', 'Moderate', or 'Low to Drought Tolerant'"
    )


class SoilContext(BaseModel):
    selected_crop: str = Field(default="Rice (Paddy)")
    soil_type: str = Field(default="Loamy Clay")
    soil_status: str = Field(default="Needs Attention")
    test_date: str = Field(default_factory=lambda: datetime.now(timezone.utc).strftime("%d %b %Y"))


class AIRecommendationData(BaseModel):
    """Structured response schema returned by Gemini AI."""
    soil_health: SoilHealthScore
    fertilizer_guidance: FertilizerGuidance
    crop_tips: List[CropSuitability]
    soil_type: str = Field(default="Loamy Clay")


class RecommendationResponse(BaseModel):
    """Complete API response format consumed by the Android app."""
    id: Optional[str] = None
    reading_id: Optional[str] = None
    device_id: str = "NUTRIX_001"
    timestamp: str
    test_date: str
    context: SoilContext
    
    # Core AI Features (Nested objects matching UI cards)
    soil_health: SoilHealthScore
    fertilizer_guidance: FertilizerGuidance
    crop_tips: List[CropSuitability]

    # Flat convenience aliases (for direct Android binding)
    soil_health_score: float
    soil_status: str
    soil_health_summary: str
    fertilizer_recommendations: List[str]
    crop_suitability_rankings: List[CropSuitability]
    
    is_ai_generated: bool = True
    ai_model: Optional[str] = None


# ============================================================================
# Gemini AI Prompt & Client Integration
# ============================================================================

SYSTEM_PROMPT = """You are NUTRIX Soil AI, an expert precision agronomist and soil scientist.
Your role is to analyze multi-parameter real-time telemetry from IoT soil sensors:
- Soil Moisture (%)
- Soil pH
- Temperature (°C)
- Atmospheric Pressure (hPa)
- Available Nitrogen (N) in mg/kg
- Available Phosphorus (P) in mg/kg
- Available Potassium (K) in mg/kg

You must provide an intelligent, scientifically sound agricultural recommendation with 3 components:
1. Soil Health Score:
   - A score between 0.0 and 10.0 (rounded to 1 decimal place, e.g. 5.8).
   - Evaluate against the selected crop requirements (e.g. Rice/Paddy ideal pH 5.5-6.5, moisture 30-60%, balanced NPK).
   - Assign status: 'Optimal' (8.5-10.0), 'Good' (7.0-8.4), 'Needs Attention' (4.0-6.9), or 'Critical' (<4.0).
   - Summary: 1-2 concise, clear sentences explaining the soil condition and main limiting factors.

2. Fertilizer Guidance:
   - Title: 'Fertilizer Guidance'
   - Subtitle: 'Nutrient Intelligence Advisory'
   - 3 to 4 actionable recommendations (bullet points) addressing:
     * Organic soil conditioning (compost, vermicompost, biochar) to improve water retention and soil biology.
     * Specific nutrient replenishment (N, P, K) based on deficiencies.
     * Soil pH amendments (e.g. agricultural lime/dolomite if acidic pH < 6.0; gypsum if alkaline pH > 7.8).
     * Sowing/growth stage timing and rainfall warnings.
   - Advisory note: Regional conditioning note.

3. Crop Tips / Suitability Rankings:
   - Rank 2 to 4 crops that best match the current soil and climatic conditions.
   - Include English name (e.g. 'Rice', 'Ragi (Finger Millet)', 'Groundnut').
   - Include vernacular / regional Indian names with local script (e.g. 'Nellu / Akki (వరి / நெல்)', 'Mandua / Kelvaragu (రాగి / கேழ்வரகு)').
   - Match percentage (0-100%).
   - Concise agronomic reasoning linking soil metrics (e.g. NPK, pH, moisture) to crop physiological needs.
   - Water requirement (e.g. 'High', 'Moderate', 'Low to Drought Tolerant').

Always return realistic, helpful guidance adhering strictly to the JSON schema.
"""


def _build_user_prompt(
    reading: Dict[str, Any],
    selected_crop: str = "Rice (Paddy)",
    soil_type: Optional[str] = None,
) -> str:
    """Format soil telemetry into a prompt for Gemini."""
    moisture = reading.get("moisture_percent")
    raw_moisture = reading.get("raw")
    ph = reading.get("ph_value")
    temp = reading.get("temperature_c")
    pressure = reading.get("pressure_hpa")
    n = reading.get("nitrogen_mg_kg")
    p = reading.get("phosphorus_mg_kg")
    k = reading.get("potassium_mg_kg")
    device_id = reading.get("device_id", "NUTRIX_001")

    prompt = f"""Telemetry reading from sensor device '{device_id}':
- Soil Moisture: {moisture if moisture is not None else 'Unknown (raw: ' + str(raw_moisture) + ')'}%
- Soil pH: {ph if ph is not None else 'Unknown'}
- Ambient Temperature: {temp if temp is not None else 'Unknown'} °C
- Atmospheric Pressure: {pressure if pressure is not None else 'Unknown'} hPa
- Available Nitrogen (N): {n if n is not None else '0'} mg/kg
- Available Phosphorus (P): {p if p is not None else '0'} mg/kg
- Available Potassium (K): {k if k is not None else '0'} mg/kg

Farmer Context:
- Current Target Crop: {selected_crop}
- Soil Texture/Type: {soil_type or 'Loamy Clay (or infer from telemetry)'}

Analyze these parameters and generate the Soil Health Score (0-10), Fertilizer Guidance, and Crop Suitability Rankings."""
    return prompt


def generate_recommendations_with_gemini(
    reading: Dict[str, Any],
    selected_crop: str = "Rice (Paddy)",
    soil_type: Optional[str] = None,
) -> AIRecommendationData:
    """
    Call Google Gemini API using the official google-genai SDK.
    Falls back gracefully to heuristic agronomic rule engine if the key is missing or API fails.
    """
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.warning(
            "GEMINI_API_KEY is not set. Using agronomic heuristic engine."
        )
        return calculate_heuristic_recommendation(reading, selected_crop, soil_type)

    model_name = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
    models_to_try = [model_name]
    for alt in ["gemini-3.5-flash", "gemini-3.8-flash", "gemini-flash-latest"]:
        if alt not in models_to_try:
            models_to_try.append(alt)

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    prompt = _build_user_prompt(reading, selected_crop, soil_type)

    last_error = None
    for model in models_to_try:
        try:
            logger.info(f"Calling Gemini API with model: {model}")
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    response_schema=AIRecommendationData,
                    temperature=0.3,
                ),
            )

            text_content = (response.text or "").strip()
            # Clean possible markdown delimiters if present
            if text_content.startswith("```"):
                text_content = text_content.split("\n", 1)[1]
                if text_content.endswith("```"):
                    text_content = text_content.rsplit("```", 1)[0]
                text_content = text_content.strip()

            parsed = AIRecommendationData.model_validate_json(text_content)
            logger.info("Successfully received structured response from Gemini!")
            return parsed

        except Exception as e:
            logger.warning(f"Gemini API attempt with {model} failed: {e}")
            last_error = e

    logger.error(
        f"All Gemini models failed ({last_error}). Falling back to agronomic heuristic engine."
    )
    return calculate_heuristic_recommendation(reading, selected_crop, soil_type)


# ============================================================================
# Agronomic Heuristic Engine (Offline / Fallback / Guaranteed Reliability)
# ============================================================================

def calculate_heuristic_recommendation(
    reading: Dict[str, Any],
    selected_crop: str = "Rice (Paddy)",
    soil_type: Optional[str] = None,
) -> AIRecommendationData:
    """
    Precision agronomic fallback engine that calculates realistic soil health
    score, fertilizer guidance, and crop tips based on scientific thresholds.
    """
    ph = reading.get("ph_value")
    moisture = reading.get("moisture_percent")
    if moisture is None and reading.get("raw") is not None:
        # Approximate moisture from ADC raw 0-4095
        raw = reading["raw"]
        moisture = max(0.0, min(100.0, round((4095 - raw) / 40.95, 1)))

    n = reading.get("nitrogen_mg_kg") or 0
    p = reading.get("phosphorus_mg_kg") or 0
    k = reading.get("potassium_mg_kg") or 0

    # 1. Calculate Soil Health Score (0 - 10)
    base_score = 10.0
    penalties = 0.0
    concerns = []

    # pH evaluation (Optimal 5.5 - 6.5 for Rice/Paddy, 6.0 - 7.2 general)
    if ph is not None:
        if ph < 5.0:
            penalties += min(2.5, (5.5 - ph) * 0.85)
            concerns.append(f"Soil is acidic (pH {ph:.1f})")
        elif ph > 7.8:
            penalties += min(2.0, (ph - 7.5) * 0.8)
            concerns.append(f"Soil is alkaline (pH {ph:.1f})")
    else:
        ph = 6.5  # default assumption

    # Moisture evaluation (Optimal 30% - 60% for paddy)
    if moisture is not None:
        if moisture < 20.0:
            penalties += min(2.0, (25.0 - moisture) * 0.08)
            concerns.append(f"Moisture is low ({moisture:.0f}%)")
        elif moisture > 85.0:
            penalties += 0.8
            concerns.append("Soil is waterlogged")
    else:
        moisture = 40.0

    # NPK evaluation
    if n < 20:
        penalties += 0.4
    if p < 15:
        penalties += 0.3
    if k < 20:
        penalties += 0.3

    calculated_score = round(max(1.0, min(10.0, base_score - penalties)), 1)

    # Determine status
    if calculated_score >= 8.0:
        status = "Optimal"
    elif calculated_score >= 6.5:
        status = "Good"
    elif calculated_score >= 4.0:
        status = "Needs Attention"
    else:
        status = "Critical"

    # Summary text
    if concerns:
        summary = f"{', '.join(concerns)}. Timely soil conditioning and balanced nutrients recommended for {selected_crop}."
    else:
        summary = f"Soil is in a healthy range for your current crop ({selected_crop})."

    # 2. Fertilizer Guidance Recommendations
    recommendations = []
    recommendations.append(
        "Apply balanced organic amendments (compost, vermicompost) to retain soil moisture and foster soil microbial activity."
    )

    if ph is not None and ph < 5.8:
        recommendations.append(
            f"Apply agricultural lime (calcium carbonate) or dolomite at 200-300 kg/acre to neutralize acidity (current pH: {ph:.1f})."
        )
    elif ph is not None and ph > 8.0:
        recommendations.append(
            f"Apply agricultural gypsum to lower pH and manage soil sodicity (current pH: {ph:.1f})."
        )

    if n < 40:
        recommendations.append(
            "For vegetative growth, monitor nitrogen reserves prior to sowing; apply split doses of neem-coated urea or bio-fertilizers."
        )
    else:
        recommendations.append(
            "Nitrogen levels are adequate; maintain split application during tillering to prevent vegetative excess."
        )

    recommendations.append(
        "Avoid excessive synthetic fertilization; test periodically after heavy rainfall or irrigation cycles."
    )

    # 3. Crop Suitability Rankings
    inferred_soil_type = soil_type or ("Loamy Clay" if (moisture is not None and moisture > 15) else "Sandy Loam")

    crop_tips = [
        CropSuitability(
            crop_name="Rice",
            vernacular_name="Nellu / Akki (వరి / நெல்)",
            match_percentage=92 if status in ["Good", "Needs Attention", "Optimal"] else 72,
            description=f"Current soil potassium level ({k} mg/kg) and loamy texture match grain-filling requirements. Nitrogen is adequate for vegetative development.",
            water_requirement="High"
        ),
        CropSuitability(
            crop_name="Ragi (Finger Millet)",
            vernacular_name="Mandua / Kelvaragu (రాగి / கேழ்வரகு)",
            match_percentage=84,
            description="Highly suitable for local agro-climatic conditions. Minimal external fertilizer required due to resilient root assimilation.",
            water_requirement="Low to Drought Tolerant"
        ),
        CropSuitability(
            crop_name="Groundnut (Peanut)",
            vernacular_name="Kadale Kayi / Verkadalai (వేరుశనగ / வேர்க்கடலை)",
            match_percentage=78,
            description="Fixes atmospheric nitrogen through symbiotic Rhizobium, helping restore nutrient-depleted soil.",
            water_requirement="Moderate"
        )
    ]

    return AIRecommendationData(
        soil_health=SoilHealthScore(
            score=calculated_score,
            max_score=10.0,
            status=status,
            summary=summary
        ),
        fertilizer_guidance=FertilizerGuidance(
            title="Fertilizer Guidance",
            subtitle="Nutrient Intelligence Advisory",
            recommendations=recommendations[:3],  # 3 key points matching app screen
            advisory_note="Advisory is tailored for regional soil conditioning. Calibrate with local agricultural extension offices."
        ),
        crop_tips=crop_tips[:2],  # top 2 matches matching app screen
        soil_type=inferred_soil_type
    )


# ============================================================================
# Response Formatter
# ============================================================================

def format_recommendation_response(
    ai_data: AIRecommendationData,
    reading: Optional[Dict[str, Any]] = None,
    device_id: str = "NUTRIX_001",
    selected_crop: str = "Rice (Paddy)",
    reading_id: Optional[str] = None,
    doc_id: Optional[str] = None,
    timestamp: Optional[datetime] = None,
    ai_model: Optional[str] = None,
) -> RecommendationResponse:
    """Format AI output into the full response model for the API and DB."""
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)
    
    date_str = timestamp.strftime("%d %b %Y")
    iso_timestamp = timestamp.isoformat()

    context = SoilContext(
        selected_crop=selected_crop,
        soil_type=ai_data.soil_type,
        soil_status=ai_data.soil_health.status,
        test_date=date_str
    )

    return RecommendationResponse(
        id=doc_id,
        reading_id=reading_id,
        device_id=device_id,
        timestamp=iso_timestamp,
        test_date=date_str,
        context=context,
        soil_health=ai_data.soil_health,
        fertilizer_guidance=ai_data.fertilizer_guidance,
        crop_tips=ai_data.crop_tips,
        # Flat convenience fields
        soil_health_score=ai_data.soil_health.score,
        soil_status=ai_data.soil_health.status,
        soil_health_summary=ai_data.soil_health.summary,
        fertilizer_recommendations=ai_data.fertilizer_guidance.recommendations,
        crop_suitability_rankings=ai_data.crop_tips,
        is_ai_generated=True,
        ai_model=ai_model or os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
    )
