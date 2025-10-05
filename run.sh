#!/bin/bash

# ExoDetect - Quick Start Script

echo "🚀 ExoDetect - Starting Services"
echo "================================"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.10+"
    exit 1
fi

# Check if models exist
if [ ! -d "models" ] && [ ! -d "models_enhanced" ]; then
    echo "⚠️  No models found. Would you like to train them now? (y/n)"
    read -r response
    if [[ "$response" == "y" ]]; then
        echo "Training models (this will take 5-10 minutes)..."
        python src/training/train_model.py
    else
        echo "Skipping model training. API will fail without models."
    fi
fi

# Start API server
echo ""
echo "📡 Starting API server..."
echo "API will be available at: http://localhost:8000"
echo "API docs at: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop"
echo ""

python -m uvicorn src.api.api_server:app --host 0.0.0.0 --port 8000 --reload