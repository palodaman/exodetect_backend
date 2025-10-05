#!/bin/bash

echo "🚀 Preparing ExoDetect for Railway Deployment"

# Check if models exist, if not train them
if [ ! -d "models" ] && [ ! -d "models_enhanced" ]; then
    echo "⚠️  No models found. Training models first..."
    echo "This will take 5-10 minutes..."

    # Train basic models
    python train_model.py

    # Optional: Train enhanced models
    # python train_enhanced_model.py

    echo "✅ Models trained successfully!"
else
    echo "✅ Models found!"
fi

# Create a minimal test to ensure everything works
echo "🧪 Testing the API locally..."
python -c "
from exoplanet_predictor import ExoplanetPredictor
print('Testing model loading...')
predictor = ExoplanetPredictor()
print('✅ Models load successfully!')
"

# Check file sizes
echo ""
echo "📦 Checking deployment size..."
if [ -d "models" ]; then
    du -sh models/
fi
if [ -d "models_enhanced" ]; then
    du -sh models_enhanced/
fi

echo ""
echo "✅ Ready for Railway deployment!"
echo ""
echo "Next steps:"
echo "1. Create a GitHub repository"
echo "2. Push your code:"
echo "   git init"
echo "   git add ."
echo "   git commit -m 'Initial commit'"
echo "   git remote add origin YOUR_GITHUB_URL"
echo "   git push -u origin main"
echo ""
echo "3. Go to https://railway.app"
echo "4. Create new project → Deploy from GitHub repo"
echo "5. Railway will automatically detect Dockerfile and deploy!"
echo ""
echo "Your API will be available at: https://YOUR-PROJECT.railway.app"