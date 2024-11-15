# UPCHECK
# Aquaculture Management System

## Setup

1. Clone the repository
2. Create and activate a virtual environment
3. Install dependencies: `pip install -r requirements.txt`
4. Create `.env` file with required environment variables
5. Run the application: `python main.py`

## Docker Setup

1. Build and run with Docker Compose:
   ```
   docker-compose up --build
   ```

## API Endpoints

- POST /analyze_pond: Analyze pond parameters
- GET /analysis_history/{location}: Get historical analysis
- GET /health: System health check

## Configuration

Edit `config.yaml` to modify:
- Measurement ranges
- Growth factors
- API settings

## Monitoring

Prometheus metrics available at :8000

## Testing

Run tests: `pytest tests/`
