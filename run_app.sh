#!/bin/bash
# Quick start script for the Low Birth Weight Prediction App

echo "=========================================="
echo "Low Birth Weight Risk Prediction App"
echo "=========================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Check if models exist
if [ ! -f "outputs/models/Random_Forest.pkl" ]; then
    echo ""
    echo "⚠️  Models not found. Training models first..."
    echo "This may take a few minutes..."
    python main.py
    echo ""
fi

# Install/update dependencies
echo "Checking dependencies..."
pip install -q -r requirements.txt

# Run the app
echo ""
echo "🚀 Starting Streamlit app..."
echo "The app will open in your browser at http://localhost:8501"
echo ""
streamlit run app.py


