# Deployment Guide - Low Birth Weight Risk Prediction System

This guide provides instructions for deploying the application in production environments.

## Prerequisites

- Python 3.11+ or Docker
- Trained models (run `python main.py` first)
- Required dependencies installed

## Option 1: Local Development

### Setup

1. **Create virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Train models (if not already done):**
```bash
python main.py
```

4. **Run the application:**
```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`

## Option 2: Docker Deployment

### Build and Run

1. **Build Docker image:**
```bash
docker build -t lbw-prediction-app .
```

2. **Run container:**
```bash
docker run -p 8501:8501 lbw-prediction-app
```

Or use docker-compose:
```bash
docker-compose up -d
```

### Docker Compose (Recommended)

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## Option 3: Cloud Deployment

### Streamlit Cloud

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Deploy!

### AWS/Azure/GCP

1. **Build Docker image:**
```bash
docker build -t lbw-prediction-app .
```

2. **Push to container registry:**
```bash
# Example for AWS ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account>.dkr.ecr.us-east-1.amazonaws.com
docker tag lbw-prediction-app:latest <account>.dkr.ecr.us-east-1.amazonaws.com/lbw-prediction-app:latest
docker push <account>.dkr.ecr.us-east-1.amazonaws.com/lbw-prediction-app:latest
```

3. **Deploy using your cloud provider's container service:**
   - AWS: ECS or EKS
   - Azure: Container Instances or AKS
   - GCP: Cloud Run or GKE

### Heroku

1. **Create Procfile:**
```
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

2. **Deploy:**
```bash
heroku create lbw-prediction-app
git push heroku main
```

## Production Considerations

### Security

1. **Add authentication** (Streamlit Cloud supports this)
2. **Use HTTPS** in production
3. **Restrict access** to authorized users
4. **Sanitize inputs** (already handled by scikit-learn)

### Performance

1. **Enable caching** (already implemented with `@st.cache_data`)
2. **Use production-grade WSGI server** for high traffic:
   ```bash
   pip install gunicorn
   gunicorn -w 4 -k uvicorn.workers.UvicornWorker app:app
   ```

3. **Load balancing** for multiple instances

### Monitoring

1. **Health checks** (Docker healthcheck included)
2. **Logging** - Streamlit logs to console
3. **Metrics** - Add monitoring tools (Prometheus, Datadog, etc.)

### Environment Variables

Set these for production:

```bash
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
```

## File Structure

```
.
├── app.py                 # Streamlit application
├── main.py                # Model training script
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker configuration
├── docker-compose.yml    # Docker Compose configuration
├── outputs/              # Generated models and visualizations
│   ├── models/          # Trained model files (.pkl)
│   └── *.png           # Visualization images
└── DEPLOYMENT.md        # This file
```

## Troubleshooting

### Models not loading
- Ensure `main.py` has been run to generate models
- Check that `outputs/models/` directory exists and contains .pkl files

### Port already in use
- Change port: `streamlit run app.py --server.port=8502`
- Or kill process using port 8501

### Docker build fails
- Check Dockerfile syntax
- Ensure all dependencies are in requirements.txt
- Check for system dependencies (libomp for XGBoost)

### Memory issues
- Reduce number of models loaded
- Use model selection instead of loading all models
- Increase Docker memory limits

## Support

For issues or questions, please refer to:
- Streamlit documentation: https://docs.streamlit.io
- Project README.md

