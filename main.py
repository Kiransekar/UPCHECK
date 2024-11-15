import os
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Any
import logging
import time
from pathlib import Path
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator, Field
from enum import Enum
import yaml
import structlog
from prometheus_client import Counter, Histogram, start_http_server
from cachetools import TTLCache
import requests
from pymongo import MongoClient, IndexModel
from pymongo.errors import PyMongoError
from contextlib import contextmanager
from dataclasses import dataclass
import json
from typing_extensions import Protocol
from datetime import datetime

# Initialize structured logging
logger = structlog.get_logger()

# Load configuration
def load_config() -> Dict:
    config_path = Path("config.yaml")
    if not config_path.exists():
        raise FileNotFoundError("Configuration file not found")
    
    with open(config_path) as f:
        return yaml.safe_load(f)

CONFIG = load_config()

# Domain Models
class WaterColor(str, Enum):
    CLEAR = "Clear"
    GREEN = "Green"
    BROWN = "Brown"
    OTHER = "Other"

class ShrimpBehavior(str, Enum):
    ACTIVE = "Active"
    LETHARGIC = "Lethargic"
    SURFACE = "Coming to Surface"

@dataclass
class WeatherData:
    temperature: float
    condition: str
    wind_speed: float
    precipitation: float
    timestamp: datetime

class PondParameters(BaseModel):
    area: float = Field(..., description="Pond area in square meters")
    depth: float = Field(..., description="Pond depth in meters")
    stocking_density: int = Field(..., description="Number of shrimp per square meter")
    culture_start_date: datetime = Field(..., description="Date when culture started")
    water_color: WaterColor = Field(..., description="Color of pond water")
    shrimp_behavior: ShrimpBehavior = Field(..., description="Observed shrimp behavior")
    secchi_disk: float = Field(..., description="Secchi disk reading in centimeters")
    ph: float = Field(..., description="pH level of pond water")
    location: str = Field(..., description="Geographic location of pond")

    @validator('depth')
    def validate_depth(cls, v):
        ranges = CONFIG['measurement_ranges']['depth']
        if not ranges['min'] <= v <= ranges['max']:
            raise ValueError(f"Depth must be between {ranges['min']} and {ranges['max']} meters")
        return v

    @validator('ph')
    def validate_ph(cls, v):
        ranges = CONFIG['measurement_ranges']['ph']
        if not ranges['min'] <= v <= ranges['max']:
            raise ValueError(f"pH must be between {ranges['min']} and {ranges['max']}")
        return v

    @validator('culture_start_date')
    def validate_start_date(cls, v):
        if v > datetime.now():
            raise ValueError("Culture start date cannot be in the future")
        return v

# Weather Service
class WeatherService(Protocol):
    def get_weather_data(self, location: str) -> Optional[WeatherData]:
        pass

class WeatherAPIAdapter:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.cache = TTLCache(
            maxsize=CONFIG['api']['weather']['batch_size'],
            ttl=CONFIG['api']['weather']['cache_ttl']
        )
        self.request_counter = Counter('weather_api_requests_total', 'Total weather API requests')
        self.error_counter = Counter('weather_api_errors_total', 'Total weather API errors')

    def get_weather_data(self, location: str) -> Optional[WeatherData]:
        if location in self.cache:
            return self.cache[location]

        self.request_counter.inc()
        retries = CONFIG['api']['weather']['retries']
        
        for attempt in range(retries):
            try:
                url = f"http://api.weatherapi.com/v1/current.json"
                response = requests.get(
                    url,
                    params={'key': self.api_key, 'q': location},
                    timeout=CONFIG['api']['weather']['timeout']
                )
                
                if response.status_code == 200:
                    data = response.json()
                    weather_data = WeatherData(
                        temperature=data['current']['temp_c'],
                        condition=data['current']['condition']['text'],
                        wind_speed=data['current']['wind_kph'],
                        precipitation=data['current']['precip_mm'],
                        timestamp=datetime.now()
                    )
                    self.cache[location] = weather_data
                    return weather_data
                
                if response.status_code == 429:  # Rate limit
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                    
            except Exception as e:
                logger.error("weather_api_error", error=str(e), attempt=attempt)
                self.error_counter.inc()
                if attempt == retries - 1:
                    return None
                time.sleep(2 ** attempt)
        
        return None

# Growth Calculator
class GrowthCalculator:
    def __init__(self):
        self.calc_duration = Histogram(
            'growth_calculation_duration_seconds',
            'Time spent calculating growth'
        )

    def calculate_growth(self, params: PondParameters, weather: Optional[WeatherData]) -> Dict:
        with self.calc_duration.time():
            factors = CONFIG['growth_factors']
            base_growth = factors['base_growth']
            
            # Density factor
            if params.stocking_density < factors['density']['low_threshold']:
                density_factor = factors['density']['factor_low']
            elif params.stocking_density > factors['density']['high_threshold']:
                density_factor = factors['density']['factor_high']
            else:
                density_factor = 1.0
            
            # Depth factor
            if params.depth < factors['depth']['shallow_threshold']:
                depth_factor = factors['depth']['factor_shallow']
            elif params.depth > factors['depth']['deep_threshold']:
                depth_factor = factors['depth']['factor_deep']
            else:
                depth_factor = 1.0
            
            # Temperature factor
            if weather:
                if weather.temperature < factors['temperature']['optimal_min']:
                    temp_factor = factors['temperature']['factor_low']
                elif weather.temperature > factors['temperature']['optimal_max']:
                    temp_factor = factors['temperature']['factor_high']
                else:
                    temp_factor = 1.0
            else:
                temp_factor = 1.0
            
            adjusted_growth = base_growth * density_factor * depth_factor * temp_factor
            days_of_culture = (datetime.now() - params.culture_start_date).days
            
            return {
                'daily_growth': round(adjusted_growth, 4),
                'weekly_growth': round(adjusted_growth * 7, 4),
                'estimated_size': round(0.002 + (adjusted_growth * days_of_culture), 4)
            }

# Database Management
class DatabaseManager:
    def __init__(self, mongo_uri: str):
        self.client = MongoClient(mongo_uri)
        self.db = self.client.aquaculture_db
        self._setup_indexes()

    def _setup_indexes(self):
        indexes = [
            IndexModel([("timestamp", -1)]),
            IndexModel([("pond_params.location", 1)]),
            IndexModel([("pond_params.culture_start_date", 1)])
        ]
        self.db.pond_analyses.create_indexes(indexes)

    async def save_analysis(self, analysis: Dict) -> str:
        try:
            result = self.db.pond_analyses.insert_one({
                **analysis,
                "timestamp": datetime.now()
            })
            return str(result.inserted_id)
        except PyMongoError as e:
            logger.error("database_error", error=str(e))
            raise HTTPException(status_code=500, detail="Database error")

    async def get_analysis_history(self, location: str, days: int = 30) -> List[Dict]:
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            cursor = self.db.pond_analyses.find({
                "pond_params.location": location,
                "timestamp": {"$gte": cutoff_date}
            }).sort("timestamp", -1)
            return list(cursor)
        except PyMongoError as e:
            logger.error("database_error", error=str(e))
            raise HTTPException(status_code=500, detail="Database error")

# Main Analysis Service
class AquacultureAnalyzer:
    def __init__(
        self,
        weather_service: WeatherService,
        growth_calculator: GrowthCalculator,
        db_manager: DatabaseManager
    ):
        self.weather_service = weather_service
        self.growth_calculator = growth_calculator
        self.db_manager = db_manager

    def _calculate_confidence(self, params: PondParameters, weather: Optional[WeatherData]) -> int:
        confidence = 100
        
        if not weather:
            confidence -= 20
        
        ranges = CONFIG['measurement_ranges']
        if not ranges['ph']['min'] + 0.5 <= params.ph <= ranges['ph']['max'] - 0.5:
            confidence -= 10
        
        if not ranges['secchi_disk']['min'] + 5 <= params.secchi_disk <= ranges['secchi_disk']['max'] - 5:
            confidence -= 10
            
        if params.stocking_density > ranges['stocking_density']['max'] - 30:
            confidence -= 10
            
        culture_age = (datetime.now() - params.culture_start_date).days
        if culture_age > 100:
            confidence -= 5
            
        return max(confidence, 0)

    def _assess_water_quality(self, params: PondParameters) -> Dict:
        quality_score = 100
        issues = []
        
        if params.ph < 7.0 or params.ph > 8.5:
            quality_score -= 20
            issues.append("pH outside optimal range")
            
        if params.secchi_disk < 20:
            quality_score -= 15
            issues.append("Low water transparency")
            
        if params.water_color == WaterColor.BROWN:
            quality_score -= 10
            issues.append("Suboptimal water color")
            
        if params.shrimp_behavior == ShrimpBehavior.SURFACE:
            quality_score -= 25
            issues.append("Abnormal shrimp behavior")
        
        return {
            "score": quality_score,
            "issues": issues,
            "status": "Good" if quality_score >= 80 else "Fair" if quality_score >= 60 else "Poor"
        }

    def _calculate_feeding(self, biomass: float) -> Dict:
        base_rate = 0.03  # 3% of biomass
        feeding_frequency = 4  # times per day
        
        daily_feed = biomass * base_rate
        per_feeding = daily_feed / feeding_frequency
        
        return {
            "daily_feed_kg": round(daily_feed, 2),
            "feeding_frequency": feeding_frequency,
            "feed_per_time_kg": round(per_feeding, 2)
        }

    async def analyze_pond(self, params: PondParameters) -> Dict:
        # Get weather data
        weather = self.weather_service.get_weather_data(params.location)
        
        # Calculate growth
        growth = self.growth_calculator.calculate_growth(params, weather)
        
        # Calculate biomass
        estimated_survival = 0.8  # 80% survival rate
        current_size = growth['estimated_size']
        total_shrimp = params.area * params.stocking_density * estimated_survival
        biomass = total_shrimp * current_size
        
        # Compile analysis
        analysis = {
            "pond_params": params.dict(),
            "weather_data": weather.__dict__ if weather else None,
            "growth_prediction": growth,
            "biomass_estimation": {
                "estimated_biomass": round(biomass, 2),
                "estimated_survival": estimated_survival,
                "total_shrimp": round(total_shrimp, 0)
            },
            "water_quality": self._assess_water_quality(params),
            "feeding_recommendation": self._calculate_feeding(biomass),
            "confidence_score": self._calculate_confidence(params, weather)
        }
        
        # Save to database
        analysis_id = await self.db_manager.save_analysis(analysis)
        analysis['id'] = analysis_id
        
        return analysis

# FastAPI Application
app = FastAPI(title="Aquaculture Management System")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

def get_analyzer():
    weather_service = WeatherAPIAdapter(os.getenv("WEATHER_API_KEY"))
    growth_calculator = GrowthCalculator()
    db_manager = DatabaseManager(os.getenv("MONGO_URI"))
    return AquacultureAnalyzer(weather_service, growth_calculator, db_manager)

@app.post("/analyze_pond")
async def analyze_pond(
    params: PondParameters,
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    try:
        analyzer = get_analyzer()
        result = await analyzer.analyze_pond(params)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("analysis_error")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/analysis_history/{location}")
async def get_history(
    location: str,
    days: int = 30,
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    try:
        db_manager = DatabaseManager(os.getenv("MONGO_URI"))
        history = await db_manager.get_analysis_history(location, days)
        return {"history": history}
    except Exception as e:
        logger.exception("history_error")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health")
async def health_check():
    try:
        # Check MongoDB connection
        db_manager = DatabaseManager(os.getenv("MONGO_URI"))
        db_manager.db.command("ping")
        
        # Check Weather API
        weather_service = WeatherAPIAdapter(os.getenv("WEATHER_API_KEY"))
        weather = weather_service.get_weather_data("London")  # Test location
        
        return {
            "status": "healthy",
            "database": "connected",
            "weather_api": "operational" if weather else "degraded"
        }
    except Exception as e:
        logger.exception("health_check_error")
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.on_event("startup")
async def startup_event():
    # Start Prometheus metrics server
    start_http_server(8000)
    
    # Initialize database
    db_manager = DatabaseManager(os.getenv("MONGO_URI"))
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Application started")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Application shutting down")

if __name__ == "__main__":
    import uvicorn
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Validate required environment variables
    required_env_vars = ["MONGO_URI", "WEATHER_API_KEY"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    # Run application
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 5000)),
        reload=True if os.getenv("ENVIRONMENT") == "development" else False,
        workers=int(os.getenv("WORKERS", 1))
    )
