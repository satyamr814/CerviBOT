# Git Setup and Publishing Guide

## Step-by-Step Guide to Publish Your Code to GitHub

### Prerequisites
- Git installed on your computer
- GitHub account (create one at https://github.com if you don't have one)

---

## Option 1: Using GitHub Website (Easier for Beginners)

### Step 1: Create a New Repository on GitHub
1. Go to https://github.com and sign in
2. Click the **"+"** icon in the top right corner
3. Select **"New repository"**
4. Fill in:
   - **Repository name**: `cervibot` (or any name you prefer)
   - **Description**: "Cervical Cancer Risk Prediction API"
   - **Visibility**: Choose Public or Private
   - **DO NOT** initialize with README, .gitignore, or license (we already have these)
5. Click **"Create repository"**

### Step 2: Copy the Repository URL
After creating, GitHub will show you a page with commands. Copy the repository URL (it will look like):
- `https://github.com/your-username/cervibot.git` (HTTPS)
- OR `git@github.com:your-username/cervibot.git` (SSH)

### Step 3: Run These Commands in Your Terminal

Open PowerShell or Command Prompt in your project directory and run:

```powershell
# Make sure you're in the project directory
cd "C:\Users\Satyam Raj\Desktop\cervibot\cerviBOT"

# Commit your staged files
git commit -m "Initial commit: CerviBOT deployment ready"

# Add the remote repository (replace with YOUR repository URL)
git remote add origin https://github.com/your-username/cervibot.git

# Push to GitHub
git branch -M main
git push -u origin main
```

---

## Option 2: Using GitHub CLI (If Installed)

If you have GitHub CLI (`gh`) installed:

```powershell
# Create repository and push in one command
gh repo create cervibot --public --source=. --remote=origin --push
```

---

## Option 3: Complete Manual Setup

### Step 1: Commit Your Changes
```powershell
cd "C:\Users\Satyam Raj\Desktop\cervibot\cerviBOT"
git commit -m "Initial commit: CerviBOT deployment ready"
```

### Step 2: Create Repository on GitHub
- Follow Step 1 from Option 1 above

### Step 3: Connect and Push
```powershell
# Add remote (replace with your actual repository URL)
git remote add origin https://github.com/your-username/cervibot.git

# Rename branch to main (if needed)
git branch -M main

# Push your code
git push -u origin main
```

---

## After Publishing

### View Your Repository
- Go to: `https://github.com/your-username/cervibot`
- Your code will be visible there!

### Future Updates
When you make changes to your code:

```powershell
# Stage all changes
git add .

# Commit with a message
git commit -m "Description of your changes"

# Push to GitHub
git push
```

---

## Troubleshooting

### Authentication Issues
If you get authentication errors:

**For HTTPS:**
- GitHub no longer accepts passwords. Use a Personal Access Token:
  1. Go to GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
  2. Generate new token with `repo` permissions
  3. Use the token as your password when pushing

**For SSH (Recommended):**
```powershell
# Generate SSH key (if you don't have one)
ssh-keygen -t ed25519 -C "your_email@example.com"

# Add SSH key to GitHub
# Copy the public key: cat ~/.ssh/id_ed25519.pub
# Add it at: GitHub → Settings → SSH and GPG keys

# Use SSH URL instead:
git remote set-url origin git@github.com:your-username/cervibot.git
```

### Large File Issues
If your model file is too large (>100MB), GitHub won't accept it. Options:
- Use Git LFS: `git lfs install && git lfs track "*.pkl"`
- Or host the model separately and download it during deployment

---

## Quick Reference Commands

```powershell
# Check status
git status

# Stage all files
git add .

# Commit
git commit -m "Your commit message"

# Push to GitHub
git push

# View remotes
git remote -v

# Change remote URL
git remote set-url origin NEW_URL
```

