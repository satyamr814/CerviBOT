# Deployment Guide for CerviBOT

This guide provides instructions for deploying the Cervical Cancer Risk Prediction API to various platforms.

## Prerequisites

- Python 3.10+
- Model file: `backend/xgb_cervical_pipeline.pkl` must be present
- All dependencies listed in `requirements.txt`

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
python app.py

# Access at http://localhost:8000
```

## Docker Deployment

### Build the Docker image

```bash
docker build -t cervibot:latest .
```

### Run the container

```bash
docker run -p 8000:8000 cervibot:latest
```

### Docker Compose (optional)

Create a `docker-compose.yml`:

```yaml
version: '3.8'
services:
  web:
    build: .
    ports:
      - "8000:8000"
    environment:
      - HOST=0.0.0.0
      - PORT=8000
```

Run with: `docker-compose up`

## Platform-Specific Deployments

### 1. Render (Recommended - FREE!)

**Best for:** Free hosting, easy setup, perfect for demos and portfolios

1. Sign up at [render.com](https://render.com) - **No credit card required!**
2. Create a new Web Service
3. Connect your GitHub repository
4. Render will automatically detect the `render.yaml` configuration
5. Select **Free** plan
6. Deploy!

The app will be available at `https://your-app-name.onrender.com`

**Note:** Free tier services spin down after 15 minutes of inactivity (wakes up automatically when accessed).

📖 **For detailed step-by-step instructions, see [RENDER_FREE_DEPLOYMENT.md](RENDER_FREE_DEPLOYMENT.md)**

### 2. Railway (Alternative)

1. Sign up at [railway.app](https://railway.app)
2. Create a new project
3. Connect your GitHub repository
4. Railway will automatically detect the `railway.json` configuration
5. Deploy!

The app will be available at `https://your-app-name.railway.app`

**Note:** Railway offers a free trial with $5 credits, then $1/month minimum.

### 3. Heroku

1. Install Heroku CLI
2. Login: `heroku login`
3. Create app: `heroku create your-app-name`
4. Deploy: `git push heroku main`

The app will be available at `https://your-app-name.herokuapp.com`

**Note**: Heroku requires a `Procfile` (already included)

### 4. AWS (EC2/ECS/Elastic Beanstalk)

#### EC2:
1. Launch an EC2 instance (Ubuntu)
2. Install Docker: `sudo apt-get install docker.io`
3. Clone repository and build image
4. Run container with port mapping

#### Elastic Beanstalk:
1. Install EB CLI: `pip install awsebcli`
2. Initialize: `eb init`
3. Create environment: `eb create`
4. Deploy: `eb deploy`

### 5. Google Cloud Platform (Cloud Run)

1. Install gcloud CLI
2. Build and push to Container Registry:
   ```bash
   gcloud builds submit --tag gcr.io/PROJECT-ID/cervibot
   ```
3. Deploy to Cloud Run:
   ```bash
   gcloud run deploy cervibot --image gcr.io/PROJECT-ID/cervibot --platform managed
   ```

### 6. Azure (App Service)

1. Install Azure CLI
2. Create resource group and app service:
   ```bash
   az group create --name cervibot-rg --location eastus
   az webapp create --resource-group cervibot-rg --plan cervibot-plan --name your-app-name
   ```
3. Configure deployment from GitHub or use Azure Container Registry

## Environment Variables

You can configure the following environment variables:

- `HOST`: Server host (default: `127.0.0.1` for dev, `0.0.0.0` for production)
- `PORT`: Server port (default: `8000`)
- `RELOAD`: Enable auto-reload (default: `true` for dev, `false` for production)

## Health Check

After deployment, verify the service is running:

```bash
curl https://your-app-url/health
```

Expected response:
```json
{
  "status": "ok",
  "model_loaded": true,
  "model_path": "..."
}
```

## API Endpoints

- `GET /` - Frontend interface
- `GET /health` - Health check
- `POST /predict` - Make predictions
- `POST /upload-model` - Upload a new model (if needed)

## Troubleshooting

### Model not loading
- Ensure `backend/xgb_cervical_pipeline.pkl` exists in the deployment
- Check file permissions
- Verify the model path in logs

### Port issues
- Most platforms set `PORT` automatically via environment variable
- Ensure your app reads `PORT` from environment (already configured)

### CORS issues
- The app allows all origins (`*`) for development
- For production, update `allow_origins` in `app.py` to specific domains

## Security Considerations

1. **CORS**: Update `allow_origins` in production to specific domains
2. **Rate Limiting**: Consider adding rate limiting for production
3. **Authentication**: Add API keys or authentication if needed
4. **HTTPS**: Ensure your platform provides HTTPS (most do automatically)

## Monitoring

Consider adding:
- Application logging (already configured)
- Error tracking (Sentry, etc.)
- Performance monitoring
- Health check endpoints (already included)

