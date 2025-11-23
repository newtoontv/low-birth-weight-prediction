# Simple Deployment Guide - No Docker Needed! 🚀

## 📱 Step 1: Preview Locally (Simple!)

```bash
# Navigate to project folder
cd "/Users/emmanuel.ogunlakin/Downloads/Uni/Computational intelligence"

# Activate virtual environment
source venv/bin/activate

# Run the app
streamlit run app.py
```

**That's it!** Your browser will open automatically at `http://localhost:8501`

---

## ☁️ Step 2: Deploy to Streamlit Cloud (Free & Easy)

Streamlit Cloud is **FREE** and **NO DOCKER** needed! Just follow these 3 steps:

### Step 1: Push to GitHub

```bash
# Initialize git (if not done)
git init

# Add all files
git add .

# Commit
git commit -m "Low Birth Weight Prediction App"

# Create a new repository on GitHub.com, then:
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git branch -M main
git push -u origin main
```

### Step 2: Deploy on Streamlit Cloud

1. Go to **[https://share.streamlit.io](https://share.streamlit.io)**
2. Click **"Sign in"** → Authorize with GitHub
3. Click **"New app"**
4. Select your repository
5. **Main file path:** `app.py`
6. Click **"Deploy"**

**Done!** Your app will be live in 2 minutes! 🎉

### Step 3: Share Your App

Your app URL will be: `https://YOUR-APP-NAME.streamlit.app`

---

## ✅ What You Need

Make sure these files are in your GitHub repo:
- ✅ `app.py` (main app)
- ✅ `requirements.txt` (dependencies)
- ✅ `outputs/models/*.pkl` (trained models)
- ✅ `outputs/*.png` (visualizations)
- ✅ `outputs/results_summary.csv` (results)

**All of these should already exist!** Just commit and push.

---

## 🆘 Quick Troubleshooting

**App won't start locally?**
```bash
# Make sure dependencies are installed
pip install -r requirements.txt
```

**Streamlit Cloud deployment fails?**
- Check that all model files are in the repo
- Verify `requirements.txt` is complete
- Check the logs on Streamlit Cloud dashboard

**Models not loading?**
- Make sure `outputs/models/` folder is committed to git
- Run: `git add outputs/` before committing

---

## 🎯 That's It!

No Docker, no complicated setup. Just:
1. **Preview locally:** `streamlit run app.py`
2. **Push to GitHub**
3. **Deploy on Streamlit Cloud**

Simple and free! 🚀

