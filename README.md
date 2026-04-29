# Smart Contract Deployment Optimizer

ML-enhanced smart contract analysis system for optimal deployment strategy.

## Prerequisites

- Python 3.11+
- Node.js 18+
- Docker (optional)
- API keys: Etherscan, BscScan, PolygonScan

## Installation

### Backend
```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` with your API keys.

### Frontend
```bash
cd frontend
npm install
```

## Initial Setup
```bash
python scripts/initial_setup.py
```

This collects 10000 blocks per network and trains ML models.

## Running

### Development

Backend:
```bash
cd backend
python -m uvicorn api.main:app --reload
```

Frontend:
```bash
cd frontend
npm start
```

### Docker
```bash
docker-compose up --build
```

## API Documentation

Interactive docs: http://localhost:8000/docs

## Model Updates
```bash
python scripts/update_models.py
```

Schedule via cron for continuous improvement.

## Architecture

- Backend: FastAPI + scikit-learn
- Frontend: React
- Data: CSV storage
- Models: joblib persistence

## License

Academic research use only.
