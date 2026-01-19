# Docker Setup Guide (For You)

## What We Built

A complete Docker-based demo that investor can run with ONE command.

---

## Files Created/Updated

### ✅ Updated:
1. **`docker-compose.yml`**
   - Added health checks
   - Network isolation
   - Frontend waits for backend
   - Port 3000 for easy access

2. **`backend/Dockerfile`**
   - Already perfect! (includes all data)
   - Health check configured
   - 270 PDFs embedded

3. **`frontend/Dockerfile`**
   - Already using nginx
   - Production optimized

4. **`backend/app/main.py`**
   - Added `/health` endpoint

### ✅ Created:
1. **`README_INVESTOR.md`**
   - Quick start guide
   - Troubleshooting
   - Professional presentation

---

## How to Test Locally

### Step 1: Build Images
```bash
# Navigate to project root
cd "c:\G\Maclens chatbot w api"

# Build all images (first time: ~5 minutes)
docker-compose build
```

### Step 2: Start Containers
```bash
# Start everything
docker-compose up

# Or run in background
docker-compose up -d
```

### Step 3: Verify
```bash
# Check container status
docker ps

# Should see:
# solomind_backend (healthy)
# solomind_frontend

# View logs
docker-compose logs -f
```

### Step 4: Test
- Open: http://localhost:3000
- Try chatting
- Toggle dark mode
- Click DPD Canada link

### Step 5: Stop
```bash
# Stop containers
docker-compose down

# Or keep data
docker-compose stop
```

---

## Package for Investor

### Option A: ZIP Archive (Simple)

```bash
# Stop containers
docker-compose down

# Create archive (Windows PowerShell)
Compress-Archive -Path "c:\G\Maclens chatbot w api" -DestinationPath "solomind-demo.zip"

# Upload to Google Drive/Dropbox
# Share link with investor
```

**Size:** ~3GB compressed

### Option B: Docker Hub (Professional)

```bash
# Tag images
docker tag solomind_backend:latest yourdockerhub/solomind-backend:v1.0
docker tag solomind_frontend:latest yourdockerhub/solomind-frontend:v1.0

# Push to Docker Hub
docker push yourdockerhub/solomind-backend:v1.0
docker push yourdockerhub/solomind-frontend:v1.0

# Investor only needs docker-compose.yml
```

**Then send them:**
- `docker-compose.yml`
- `README_INVESTOR.md`
- `.env` file (with backend/.env)

They just run:
```bash
docker-compose pull
docker-compose up
```

---

## What Investor Receives

### Minimal Package:
```
solomind-demo/
├── docker-compose.yml
├── README_INVESTOR.md  
├── backend/
│   ├── Dockerfile
│   ├── .env              # YOUR Azure keys!
│   ├── app/
│   ├── data/             # 270 PDFs, database
│   └── requirements.txt
└── frontend/
    ├── Dockerfile
    ├── nginx.conf
    ├── package.json
    └── src/
```

### Their Experience:
1. Extract ZIP
2. Install Docker Desktop (if needed)
3. Run: `docker-compose up`
4. Wait 60 seconds
5. Browse to http://localhost:3000
6. **WOW!** 🎉

---

## Important Notes

### Security
⚠️ **Your `.env` file contains Azure keys**  
- They can see them if they inspect
- Set Azure spending alert ($100/day)
- Can rotate keys after demo
- Monitor usage in Azure portal

### Data
✅ **All 270 PDFs included**  
- Embedded in Docker image
- No download needed
- Works offline (except AI calls)

### Costs
💰 **Azure API usage**  
- ~$0.50 per 100 queries
- GPT-4o: $0.03/1K tokens
- Embeddings: $0.0001/1K tokens
- Set spending limits!

---

## Troubleshooting

### "Cannot connect to Docker daemon"
```bash
# Start Docker Desktop
# Wait for it to fully start
docker ps  # Should work
```

### "Port 3000 already in use"
```bash
# Stop whatever is on port 3000
# Or change port in docker-compose.yml
ports: - "3001:80"
```

### "Build failed"
```bash
# Clean build
docker-compose down
docker system prune -a
docker-compose build --no-cache
```

### "Backend unhealthy"
```bash
# Check logs
docker-compose logs backend

# Common issues:
# - Missing .env file
# - Invalid Azure keys
# - Data folder missing
```

---

## Final Checklist

Before sending to investor:

- [ ] Test locally with `docker-compose up`
- [ ] Verify chat works
- [ ] Check dark mode
- [ ] Test DPD link
- [ ] Verify all 270 PDFs accessible
- [ ] Set Azure spending alert
- [ ] Create README_INVESTOR.md
- [ ] Package everything
- [ ] Test extracted package
- [ ] Share via secure method

---

## Cost Estimate

**For investor demo (1 week):**
- Azure API: $10-50 (depending on usage)
- Docker Hub: $0 (free tier)
- Your time: Already done! ✅

**ROI:** Potential $1M+ investment 🚀

---

## Next Steps

1. **Test locally** (you are here)
2. **Package for investor**
3. **Send with pitch deck**
4. **Monitor Azure usage**
5. **Rotate keys after demo**

**Ready to impress!** 🎯
