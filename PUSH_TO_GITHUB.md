# Push to GitHub - Quick Guide

## Option 1: If you already have a GitHub repository

Replace `YOUR_USERNAME` and `YOUR_REPO_NAME` with your actual GitHub username and repository name:

```bash
cd "/Users/emmanuel.ogunlakin/Downloads/Uni/Computational intelligence"

# Add remote (replace with your repo URL)
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# Push the new branch
git push -u origin streamlit-app
```

## Option 2: Create a new GitHub repository first

1. Go to [github.com](https://github.com) and sign in
2. Click the "+" icon → "New repository"
3. Name it (e.g., "low-birth-weight-prediction")
4. **Don't** initialize with README, .gitignore, or license
5. Click "Create repository"
6. Copy the repository URL (e.g., `https://github.com/YOUR_USERNAME/low-birth-weight-prediction.git`)

Then run:

```bash
cd "/Users/emmanuel.ogunlakin/Downloads/Uni/Computational intelligence"

# Add remote (use your actual repo URL)
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# Push the new branch
git push -u origin streamlit-app
```

## After pushing

Your code is now on the `streamlit-app` branch. You can:
- View it on GitHub
- Deploy to Streamlit Cloud using this branch
- Create a Pull Request to merge into main if needed


