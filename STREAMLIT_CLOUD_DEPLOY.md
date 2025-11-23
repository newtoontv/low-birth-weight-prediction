# Simple Streamlit Cloud Deployment Guide

## 🚀 Quick Preview (Local)

### Step 1: Activate Virtual Environment
```bash
cd "/Users/emmanuel.ogunlakin/Downloads/Uni/Computational intelligence"
source venv/bin/activate
```

### Step 2: Run the App
```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`

**That's it!** No Docker needed.

---

## ☁️ Deploy to Streamlit Cloud (Free & Easy)

### Prerequisites
- GitHub account (free)
- Code pushed to a GitHub repository

### Step-by-Step Deployment

#### 1. Create GitHub Repository

```bash
# Initialize git (if not already done)
git init
git add .
git commit -m "Initial commit - Low Birth Weight Prediction App"

# Create repository on GitHub (via web interface), then:
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git branch -M main
git push -u origin main
```

#### 2. Deploy on Streamlit Cloud

1. Go to **[https://share.streamlit.io](https://share.streamlit.io)**
2. Click **"Sign in"** and authorize with GitHub
3. Click **"New app"**
4. Select your repository
5. Set **Main file path** to: `app.py`
6. Click **"Deploy"**

**That's it!** Your app will be live in ~2 minutes.

### Important Notes for Streamlit Cloud

✅ **Already configured:**
- `requirements.txt` - All dependencies listed
- `app.py` - Main application file
- Models will be loaded from `outputs/models/` directory

⚠️ **Make sure:**
- All model files (`.pkl`) are committed to GitHub
- All visualization images are in `outputs/` directory
- `outputs/results_summary.csv` exists

### File Structure for Streamlit Cloud

Your repository should have:
```
your-repo/
├── app.py              ← Main app (required)
├── main.py             ← Training script
├── requirements.txt    ← Dependencies (required)
├── outputs/
│   ├── models/        ← Model files (.pkl)
│   │   ├── Logistic_Regression.pkl
│   │   ├── Random_Forest.pkl
│   │   └── ...
│   ├── *.png          ← Visualization images
│   └── results_summary.csv
└── README.md
```

### Troubleshooting

**App won't load?**
- Check that all model files are in the repository
- Verify `requirements.txt` has all dependencies
- Check Streamlit Cloud logs for errors

**Models not found?**
- Make sure `outputs/models/` directory is committed
- Run `git add outputs/` to include all outputs

**Dependencies error?**
- Check `requirements.txt` is up to date
- Streamlit Cloud will install automatically

---

## 📝 Quick Checklist

Before deploying:
- [ ] Models trained (`python main.py` completed)
- [ ] All files committed to git
- [ ] Pushed to GitHub
- [ ] `requirements.txt` includes all dependencies
- [ ] `app.py` is the main file

---

## 🎉 That's It!

No Docker, no complicated setup. Just:
1. Push to GitHub
2. Deploy on Streamlit Cloud
3. Share your link!

Your app will be live at: `https://YOUR-APP-NAME.streamlit.app`

