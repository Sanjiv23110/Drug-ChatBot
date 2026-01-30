# PowerShell Startup Script for Ingestion
# This script sets up the correct environment and runs the ingestion

# Set UTF-8 encoding to prevent UnicodeEncodeError
$env:PYTHONIOENCODING = 'utf-8'

Write-Host "🔧 Environment Setup" -ForegroundColor Cyan
Write-Host "  ✓ PYTHONIOENCODING set to utf-8" -ForegroundColor Green

# Check if Docker is running
Write-Host "`n🐳 Checking Docker..." -ForegroundColor Cyan
$dockerRunning = docker ps 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "  ✗ Docker is not running!" -ForegroundColor Red
    Write-Host "  Please start Docker Desktop and try again." -ForegroundColor Yellow
    exit 1
}
Write-Host "  ✓ Docker is running" -ForegroundColor Green

# Check if PostgreSQL container is running
Write-Host "`n🗄️  Checking PostgreSQL container..." -ForegroundColor Cyan
$pgContainer = docker ps --filter "name=medical-rag-postgres" --format "{{.Names}}"
if ($pgContainer -eq "medical-rag-postgres") {
    Write-Host "  ✓ PostgreSQL container is running" -ForegroundColor Green
} else {
    Write-Host "  ✗ PostgreSQL container not found!" -ForegroundColor Red
    Write-Host "  Run: docker start medical-rag-postgres" -ForegroundColor Yellow
    exit 1
}

# Check if virtual environment is activated
Write-Host "`n🐍 Checking Python environment..." -ForegroundColor Cyan
if ($env:VIRTUAL_ENV) {
    Write-Host "  ✓ Virtual environment is active: $env:VIRTUAL_ENV" -ForegroundColor Green
} else {
    Write-Host "  ! Virtual environment not activated" -ForegroundColor Yellow
    Write-Host "  Activating venv..." -ForegroundColor Cyan
    & ".\venv\Scripts\Activate.ps1"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  ✗ Failed to activate virtual environment" -ForegroundColor Red
        exit 1
    }
    Write-Host "  ✓ Virtual environment activated" -ForegroundColor Green
}

Write-Host "`n✅ Environment ready for ingestion!" -ForegroundColor Green
Write-Host "`nUsage:" -ForegroundColor Cyan
Write-Host "  Single PDF:  python run_ingest.py data/pdfs/00062205.pdf --skip-vision"
Write-Host "  All PDFs:    python run_ingest.py data/pdfs --skip-vision"
Write-Host "  Initialize:  python run_ingest.py --init-db"
Write-Host ""
