# Quick Start Guide

## 🚀 Get Started in 3 Steps

### Step 1: Install Dependencies

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Note for macOS users:** Install OpenMP for XGBoost:
```bash
brew install libomp
```

### Step 2: Train Models (First Time Only)

```bash
python main.py
```

This will:
- Load the dataset
- Train 5 ML models
- Generate visualizations
- Save models to `outputs/models/`

⏱️ Takes ~2-3 minutes

### Step 3: Launch Web Application

```bash
streamlit run app.py
```

Or use the quick start script:
```bash
./run_app.sh
```

The app will automatically open in your browser at `http://localhost:8501`

## 📱 Using the Web Application

### Navigation Pages

1. **🏠 Home** - Overview dashboard with model summary
2. **📊 Model Dashboard** - Interactive model performance comparison
3. **🔮 Make Prediction** - Input patient data and get risk predictions
4. **📈 Visualizations** - Browse all generated plots and charts
5. **ℹ️ About** - Project documentation and references

### Making a Prediction

1. Go to **🔮 Make Prediction** page
2. Enter patient information:
   - Maternal age, weight, race
   - Smoking status, hypertension
   - Uterine irritability, physician visits
   - Previous premature labors
3. Select a model (Random Forest recommended)
4. Click **🔮 Predict Risk**
5. View results with risk percentage and interpretation

## 🐳 Docker Deployment

### Build and Run

```bash
# Build image
docker build -t lbw-prediction-app .

# Run container
docker run -p 8501:8501 lbw-prediction-app
```

### Using Docker Compose

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## ☁️ Cloud Deployment

### Streamlit Cloud (Easiest)

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repository
4. Deploy!

### Other Platforms

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed instructions on:
- AWS (ECS, EKS, EC2)
- Azure (Container Instances, AKS)
- Google Cloud (Cloud Run, GKE)
- Heroku

## 📁 Project Structure

```
.
├── app.py                 # Streamlit web application ⭐
├── main.py                # Model training script
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker configuration
├── docker-compose.yml    # Docker Compose config
├── run_app.sh            # Quick start script
├── outputs/              # Generated files
│   ├── models/          # Trained models (.pkl)
│   └── *.png            # Visualizations
└── .streamlit/          # Streamlit config
    └── config.toml
```

## 🎯 Key Features

✅ **Interactive Predictions** - Real-time risk assessment  
✅ **Model Comparison** - Compare 5 different ML models  
✅ **Visualizations** - 10+ interactive charts and plots  
✅ **SHAP Explanations** - Understand model decisions  
✅ **Production Ready** - Docker, cloud deployment support  

## 🆘 Troubleshooting

### Models not loading?
- Ensure `main.py` has been run successfully
- Check `outputs/models/` contains .pkl files

### Port 8501 already in use?
```bash
streamlit run app.py --server.port=8502
```

### Docker build fails?
- Check Dockerfile syntax
- Ensure all dependencies in requirements.txt
- For macOS: Install libomp via Homebrew

### Import errors?
```bash
pip install -r requirements.txt --upgrade
```

## 📚 Next Steps

- Read [README.md](README.md) for detailed documentation
- Check [DEPLOYMENT.md](DEPLOYMENT.md) for production deployment
- Explore the code in `app.py` to customize the interface

## 💡 Tips

- **Best Model**: Random Forest (ROC-AUC: 0.8319)
- **For Screening**: Use Logistic Regression with SMOTE (higher recall)
- **For Accuracy**: Use Random Forest (best overall performance)
- **Visualizations**: All plots are interactive and can be downloaded

---

**Need Help?** Check the documentation or review the code comments in `app.py` and `main.py`.

