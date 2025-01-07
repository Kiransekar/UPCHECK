from dataclasses import dataclass
from typing import Dict, List, Tuple
from datetime import datetime
import math

@dataclass
class DetailedFeedingRecommendation:
    pond_id: str
    date: datetime
    daily_feed_kg: float
    shrimp_size_g: float
    current_fcr: float
    feeding_schedule: List[Dict]
    environmental_status: Dict
    alerts: List[str]

class AquaFeedSystem:
    def _init_(self):
        self.base_feeding_rates = {
            # Size in grams: feeding rate as % of biomass
            1: 0.08,  # 8% for 1g shrimp
            5: 0.06,  # 6% for 5g shrimp
            10: 0.04, # 4% for 10g shrimp
            15: 0.03, # 3% for 15g shrimp
            20: 0.025 # 2.5% for 20g+ shrimp
        }
        
        self.optimal_params = {
            'do': (4.0, 6.0),
            'ph': (7.2, 8.3),
            'temp': (28, 32),
            'salinity': (15, 25)
        }

    def calculate_feeding_details(
        self,
        pond_id: str,
        biomass_kg: float,
        avg_size_g: float,
        days_of_culture: int,
        water_params: Dict,
        lunar_phase: str
    ) -> DetailedFeedingRecommendation:
        # Calculate base feeding rate
        base_rate = self._get_feeding_rate(avg_size_g)
        daily_feed = biomass_kg * base_rate
        
        # Apply environmental adjustments
        env_factor, env_status = self._assess_environment(water_params)
        adjusted_feed = daily_feed * env_factor
        
        # Calculate FCR (assuming historical data available)
        current_fcr = self._calculate_fcr(pond_id, days_of_culture)
        
        # Generate feeding schedule
        schedule = self._create_feeding_schedule(adjusted_feed, avg_size_g, lunar_phase)
        
        # Generate alerts if needed
        alerts = self._generate_alerts(env_status, current_fcr)
        
        return DetailedFeedingRecommendation(
            pond_id=pond_id,
            date=datetime.now(),
            daily_feed_kg=round(adjusted_feed, 2),
            shrimp_size_g=avg_size_g,
            current_fcr=round(current_fcr, 2),
            feeding_schedule=schedule,
            environmental_status=env_status,
            alerts=alerts
        )

    def _get_feeding_rate(self, size_g: float) -> float:
        """Get appropriate feeding rate based on shrimp size."""
        sizes = sorted(self.base_feeding_rates.keys())
        for ref_size in sizes:
            if size_g <= ref_size:
                return self.base_feeding_rates[ref_size]
        return self.base_feeding_rates[sizes[-1]]

    def _create_feeding_schedule(
        self, 
        total_feed: float,
        size_g: float,
        lunar_phase: str
    ) -> List[Dict]:
        """Create detailed feeding schedule with amounts per feeding."""
        # Base feeding times (adjust based on size)
        if size_g < 5:
            times = ['06:00', '10:00', '14:00', '18:00', '22:00', '02:00']
        else:
            times = ['06:00', '11:00', '16:00', '21:00']
            
        # Adjust for lunar phase
        if lunar_phase == "Full Moon":
            times = times[::2]  # Feed less frequently during full moon
            
        feed_per_time = total_feed / len(times)
        
        return [
            {
                'time': time,
                'amount_kg': round(feed_per_time, 3),
                'check_points': self._get_checkpoints(time)
            }
            for time in times
        ]

    def _assess_environment(self, params: Dict) -> Tuple[float, Dict]:
        """Assess environmental parameters and return adjustment factor."""
        status = {}
        total_factor = 1.0
        
        for param, value in params.items():
            if param in self.optimal_params:
                min_val, max_val = self.optimal_params[param]
                if min_val <= value <= max_val:
                    status[param] = {'value': value, 'status': 'optimal'}
                else:
                    factor = max(0.5, 1 - abs(value - (min_val + max_val)/2)/(max_val - min_val))
                    total_factor *= factor
                    status[param] = {
                        'value': value,
                        'status': 'suboptimal',
                        'adjustment': factor
                    }
        
        return total_factor, status

    def _calculate_fcr(self, pond_id: str, doc: int) -> float:
        """Calculate FCR based on historical data."""
        # This would normally fetch from database
        # For demonstration, using simple calculation
        base_fcr = 1.5
        if doc < 30:
            return base_fcr * 0.9
        elif doc < 60:
            return base_fcr
        else:
            return base_fcr * 1.1

    def _get_checkpoints(self, time: str) -> List[str]:
        """Get specific checkpoints for each feeding time."""
        checks = ["Check feeding tray after 2 hours"]
        hour = int(time.split(':')[0])
        
        if 5 <= hour <= 7:
            checks.append("Measure morning DO")
        elif 13 <= hour <= 15:
            checks.append("Check afternoon temperature")
        
        return checks

    def _generate_alerts(self, env_status: Dict, fcr: float) -> List[str]:
        """Generate alerts based on conditions."""
        alerts = []
        
        for param, details in env_status.items():
            if details.get('status') == 'suboptimal':
                alerts.append(f"Suboptimal {param}: {details['value']}")
                
        if fcr > 1.8:
            alerts.append("High FCR detected - check feeding efficiency")
            
        return alerts
from fastapi import FastAPI, HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime, timedelta
import logging

class AquaDatabase:
    def _init_(self, mongo_uri: str):
        self.client = AsyncIOMotorClient(mongo_uri)
        self.db = self.client.aquaculture
        self.logger = logging.getLogger("aqua_database")

    async def validate_connection(self) -> bool:
        try:
            await self.client.admin.command('ping')
            return True
        except Exception as e:
            self.logger.error(f"Database connection error: {e}")
            return False

    async def save_feeding_record(self, record: DetailedFeedingRecommendation) -> str:
        """Save feeding recommendation to database."""
        try:
            result = await self.db.feeding_records.insert_one(record._dict_)
            return str(result.inserted_id)
        except Exception as e:
            self.logger.error(f"Error saving feeding record: {e}")
            raise

    async def get_pond_history(
        self,
        pond_id: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict]:
        """Retrieve feeding history for a pond."""
        try:
            cursor = self.db.feeding_records.find({
                "pond_id": pond_id,
                "date": {"$gte": start_date, "$lte": end_date}
            }).sort("date", -1)
            return await cursor.to_list(length=None)
        except Exception as e:
            self.logger.error(f"Error retrieving pond history: {e}")
            raise

    async def update_growth_data(self, pond_id: str, growth_data: Dict):
        """Update pond growth and biomass data."""
        try:
            await self.db.pond_data.update_one(
                {"pond_id": pond_id},
                {"$set": {
                    "last_sampling": growth_data,
                    "updated_at": datetime.now()
                }},
                upsert=True
            )
        except Exception as e:
            self.logger.error(f"Error updating growth data: {e}")
            raise

class AquaFeedAPI:
    def _init_(self, feed_system: AquaFeedSystem, database: AquaDatabase):
        self.feed_system = feed_system
        self.db = database
        self.logger = logging.getLogger("aqua_feed_api")

    async def generate_feeding_recommendation(self, request_data: Dict) -> Dict:
        """Generate feeding recommendation with validation."""
        try:
            # Validate database connection
            if not await self.db.validate_connection():
                raise HTTPException(500, "Database connection failed")

            # Generate recommendation
            recommendation = self.feed_system.calculate_feeding_details(
                pond_id=request_data['pond_id'],
                biomass_kg=request_data['biomass'],
                avg_size_g=request_data['avg_size'],
                days_of_culture=request_data['doc'],
                water_params=request_data['water_params'],
                lunar_phase=request_data['lunar_phase']
            )

            # Save recommendation
            record_id = await self.db.save_feeding_record(recommendation)

            return {
                "status": "success",
                "record_id": record_id,
                "recommendation": recommendation._dict_
            }

        except Exception as e:
            self.logger.error(f"Error generating recommendation: {e}")
            raise HTTPException(500, str(e))

# FastAPI implementation
app = FastAPI(title="Aquaculture Feeding Management System")

@app.post("/api/feeding/recommend")
async def get_feeding_recommendation(
    request: Dict,
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    try:
        # Initialize components
        database = AquaDatabase(os.getenv("MONGO_URI"))
        feed_system = AquaFeedSystem()
        api = AquaFeedAPI(feed_system, database)

        # Generate recommendation
        result = await api.generate_feeding_recommendation(request)
        return result

    except Exception as e:
        logging.exception("Recommendation generation failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/feeding/history/{pond_id}")
async def get_pond_history(
    pond_id: str,
    start_date: str,
    end_date: str,
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    try:
        database = AquaDatabase(os.getenv("MONGO_URI"))
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        history = await database.get_pond_history(pond_id, start, end)
        
        return {
            "status": "success",
            "pond_id": pond_id,
            "history": history
        }
    except Exception as e:
        logging.exception("History retrieval failed")
        raise HTTPException(status_code=500, detail=str(e))

# Monitoring endpoints
@app.get("/api/system/health")
async def check_system_health():
    """Check system health including database connection."""
    try:
        database = AquaDatabase(os.getenv("MONGO_URI"))
        db_status = await database.validate_connection()
        
        return {
            "status": "healthy" if db_status else "unhealthy",
            "database_connection": "ok" if db_status else "failed",
            "timestamp": datetime.now()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail="System health check failed")
