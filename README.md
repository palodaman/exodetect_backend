# 🚀 ExoDetect - AI-Powered Exoplanet Detection System

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![Next.js](https://img.shields.io/badge/Next.js-15.5+-black.svg)](https://nextjs.org)
[![Railway](https://img.shields.io/badge/Deploy-Railway-purple.svg)](https://railway.app)

A state-of-the-art machine learning system for detecting and classifying exoplanets from light curve data, achieving **99%+ accuracy** on NASA's Kepler data.

## ✨ Features

- 🤖 **Multiple ML Models**: XGBoost, LightGBM, Random Forest, and more
- 📊 **Light Curve Analysis**: Automated transit detection using Box Least Squares
- 🌍 **Multi-Source Data**: Integrates KOI, TOI, and K2 datasets from NASA
- ⚡ **Real-time Predictions**: FastAPI backend with <1s inference time
- 🎨 **Modern UI**: Next.js frontend with interactive visualizations
- 🚢 **Production Ready**: Dockerized with Railway deployment configs

## 📂 Project Structure

```
nasa-exodetect/
├── src/
│   ├── api/           # FastAPI server
│   ├── ml/            # ML models and predictors
│   ├── training/      # Model training scripts
│   └── utils/         # Data loaders and utilities
├── models/            # Trained model files
├── data/              # Dataset storage
├── exodetect/         # Next.js frontend
├── deployment/        # Docker and deployment configs
├── docs/              # Documentation
└── scripts/           # Utility scripts
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+ (for frontend)
- 2GB+ RAM

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/nasa-exodetect.git
cd nasa-exodetect
```

2. **Install Python dependencies**
```bash
pip install -r requirements.txt -r requirements-api.txt
```

3. **Train models** (or use pre-trained)
```bash
python src/training/train_model.py
```

4. **Start the API server**
```bash
python -m uvicorn src.api.api_server:app --reload
```

5. **Start the frontend** (in new terminal)
```bash
cd exodetect
npm install
npm run dev
```

Visit http://localhost:3000 🎉

## 🔬 Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|-------|----------|-----------|---------|----------|---------|
| **XGBoost** | 99.01% | 98.53% | 99.47% | 99.00% | 99.89% |
| **LightGBM** | 98.85% | 98.43% | 99.26% | 98.84% | 99.89% |
| **Random Forest** | 98.95% | 98.63% | 99.26% | 98.95% | 99.89% |
| **Gradient Boosting** | 99.06% | 98.53% | 99.58% | 99.05% | 99.85% |

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/models` | GET | List available models |
| `/api/predict/light-curve` | POST | Predict from raw light curve |
| `/api/predict/features` | POST | Predict from features |
| `/api/predict/upload` | POST | Upload CSV/FITS file |
| `/api/predict/batch` | POST | Batch predictions |

### Example Request

```python
import requests

response = requests.post(
    "http://localhost:8000/api/predict/features",
    json={
        "period": 10.5,      # days
        "duration": 3.2,     # hours
        "depth": 500,        # ppm
        "snr": 15.0
    }
)

print(response.json())
# Output: {"model_label": "Likely Candidate", "probability": 0.875, ...}
```

## 🚢 Deployment

### Railway (Recommended)

1. **Push to GitHub**
```bash
git push origin main
```

2. **Deploy on Railway**
- Connect GitHub repo at [railway.app](https://railway.app)
- Railway auto-detects Dockerfile
- Set environment variables if needed

3. **Update frontend**
```bash
echo "NEXT_PUBLIC_API_URL=https://your-api.railway.app" > exodetect/.env.local
```

See [deployment guide](docs/railway-deployment.md) for detailed instructions.

### Docker

```bash
# Build and run with Docker Compose
docker-compose up --build

# API at http://localhost:8000
```

## 🧪 Testing

```bash
# Test ML pipeline
python src/ml/exoplanet_predictor.py

# Test API
curl http://localhost:8000/health
```

## 📊 Data Sources

- **KOI**: [Kepler Objects of Interest](https://exoplanetarchive.ipac.caltech.edu)
- **TOI**: [TESS Objects of Interest](https://exoplanetarchive.ipac.caltech.edu)
- **K2**: [K2 Candidates](https://exoplanetarchive.ipac.caltech.edu)

## 🛠️ Tech Stack

- **Backend**: FastAPI, Python 3.10
- **ML**: XGBoost, LightGBM, scikit-learn
- **Frontend**: Next.js 15, React 18, TypeScript
- **Database**: In-memory (models stored as pickle files)
- **Deployment**: Docker, Railway
- **Data Processing**: pandas, numpy, astropy

## 📝 License

MIT License - See [LICENSE](LICENSE) file

## 🤝 Contributing

Contributions welcome! Please read our [contributing guidelines](CONTRIBUTING.md).

## 👥 Team

Built for NASA Space Apps Challenge 2024

## 🙏 Acknowledgments

- NASA Exoplanet Archive for datasets
- Kepler and TESS missions for light curve data
- scikit-learn and XGBoost communities

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**⭐ Star this repo if you find it useful!**