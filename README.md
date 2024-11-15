# UPCHECK
# Aquaculture Management System
![WhatsApp Image 2024-11-12 at 19 08 16_26676f8b](https://github.com/user-attachments/assets/862f6ea5-f565-4488-842a-9724b639854e)

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
