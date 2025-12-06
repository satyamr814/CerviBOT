# Free Deployment Guide - Render (100% Free!)

## Why Render?
✅ **Completely FREE** - No credit card required  
✅ **750 hours/month** - Enough for 24/7 hosting  
✅ **Auto-deploy from GitHub**  
✅ **HTTPS included**  
✅ **Perfect for FastAPI apps**  

**Note:** Free tier services spin down after 15 minutes of inactivity, but wake up automatically when accessed (takes ~30 seconds).

---

## Step-by-Step Deployment on Render

### Step 1: Sign Up for Render (Free)

1. Go to **https://render.com**
2. Click **"Get Started for Free"** or **"Sign Up"**
3. Choose **"Sign up with GitHub"**
4. Authorize Render to access your GitHub account
5. **No credit card required!** ✅

---

### Step 2: Create a New Web Service

1. Once logged in, you'll see the Render dashboard
2. Click **"New +"** button (top right)
3. Select **"Web Service"**

---

### Step 3: Connect Your GitHub Repository

1. Render will show "Connect a repository"
2. Click **"Connect account"** if you haven't already
3. Find and select **"CerviBOT"** from your repositories
4. Click **"Connect"**

---

### Step 4: Configure Your Service

Render will auto-detect your `render.yaml` file! But you can also configure manually:

**Basic Settings:**
- **Name:** `cervibot` (or any name you like)
- **Region:** Choose closest to you (e.g., `Oregon (US West)`)
- **Branch:** `main`
- **Root Directory:** Leave empty (or `cerviBOT` if files are in subfolder)

**Build & Deploy:**
- **Runtime:** `Python 3`
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `python app.py`

**Environment Variables:**
Click **"Advanced"** → **"Add Environment Variable"** and add:
- `HOST` = `0.0.0.0`
- `PORT` = `8000` (Render sets this automatically, but add it just in case)
- `RELOAD` = `false`

**OR** - Just use your `render.yaml` file (Render will auto-detect it!)

---

### Step 5: Choose Free Plan

1. Scroll down to **"Plan"** section
2. Select **"Free"** plan
3. Click **"Create Web Service"**

---

### Step 6: Wait for Deployment

1. Render will start building your app
2. You'll see build logs in real-time
3. This takes **3-7 minutes** for the first deployment
4. Watch for:
   ```
   Building...
   Installing dependencies...
   Starting service...
   ```

---

### Step 7: Get Your Free URL

1. Once deployment is complete, you'll see a green status
2. Render automatically generates a URL like:
   - `https://cervibot.onrender.com`
3. Click the URL to open your app
4. **Your chatbot is now live for FREE!** 🎉

---

### Step 8: Test Your Deployment

1. Open your Render URL in a browser
2. Test the health endpoint:
   - Go to: `https://your-app.onrender.com/health`
   - Should return: `{"status": "ok", "model_loaded": true}`
3. Test the frontend:
   - Go to: `https://your-app.onrender.com/`
   - Your chatbot interface should load!

---

## Important Notes About Free Tier

### Auto Spin-Down
- **Free services spin down after 15 minutes of inactivity**
- **First request after spin-down takes ~30 seconds** (wake-up time)
- **Subsequent requests are instant** while the service is awake
- This is normal and expected for free hosting!

### How to Keep It Awake (Optional)
If you want to prevent spin-down, you can:
1. Use a free uptime monitor like:
   - **UptimeRobot** (https://uptimerobot.com) - Free, monitors every 5 minutes
   - **Cron-job.org** - Free cron jobs
2. Set it to ping your health endpoint every 10-14 minutes
3. This keeps your service awake 24/7

---

## Troubleshooting

### Issue: Build Fails

**Check:**
- Go to **"Logs"** tab in Render dashboard
- Look for error messages

**Common fixes:**
- Model file not found → Ensure `backend/xgb_cervical_pipeline.pkl` is in your repo
- Dependencies error → Check `requirements.txt`
- Port error → Ensure app reads `PORT` from environment (already configured)

### Issue: Service Spins Down

**This is normal!** Free tier services spin down after 15 min of inactivity.

**Solution:**
- First request will take ~30 seconds (wake-up)
- Subsequent requests are fast
- Use UptimeRobot to keep it awake (optional)

### Issue: Can't Access After Deployment

**Check:**
- Service status is "Live" (green)
- URL is correct
- Try the health endpoint: `/health`
- Wait 30 seconds if it's spinning up

---

## Updating Your App

When you make code changes:

1. **Commit and push to GitHub:**
   ```powershell
   git add .
   git commit -m "Your update message"
   git push
   ```

2. **Render automatically detects the push** and redeploys
3. Watch the **"Events"** tab for deployment progress
4. Your app updates automatically! ✨

---

## Free Tier Limits

✅ **750 hours/month** - Enough for 24/7 hosting  
✅ **512 MB RAM** - Sufficient for your FastAPI app  
✅ **0.1 CPU** - Good for low-medium traffic  
✅ **100 GB bandwidth/month** - Plenty for testing  
✅ **Auto HTTPS** - SSL certificate included  
✅ **Auto deployments** - From GitHub  

**For your CerviBOT:** These limits are perfect for a demo/portfolio project!

---

## Alternative: Fly.io (Also Free!)

If you want another free option:

### Fly.io Free Tier:
- **3 shared-cpu VMs** (256 MB RAM each)
- **3 GB persistent storage**
- **160 GB outbound data transfer**
- **No spin-down** (always on!)

**Deployment:**
1. Sign up at https://fly.io
2. Install Fly CLI
3. Run: `fly launch`
4. Deploy: `fly deploy`

---

## Quick Reference

- **Render Dashboard:** https://dashboard.render.com
- **Your Repository:** https://github.com/satyamr814/CerviBOT
- **Render Docs:** https://render.com/docs

---

## Success Checklist

✅ Render account created (no credit card!)  
✅ Web service created and connected to GitHub  
✅ Free plan selected  
✅ Deployment successful  
✅ App accessible via Render URL  
✅ Health endpoint working  
✅ Frontend loads correctly  

**Congratulations! Your CerviBOT is now live for FREE! 🚀**

---

## Pro Tips

1. **Bookmark your Render URL** - Easy access to your chatbot
2. **Set up UptimeRobot** - Keep service awake 24/7 (optional)
3. **Monitor usage** - Check Render dashboard for resource usage
4. **Share your chatbot** - Your free URL is shareable!

---

## Cost Summary

| Platform | Cost | Spin-Down | Best For |
|----------|------|-----------|----------|
| **Render** | **FREE** | 15 min | Demos, portfolios |
| **Fly.io** | **FREE** | None | Always-on apps |
| Railway | $1+/month | None | Production apps |

**For your chatbot: Render is the best free option!** ✅

