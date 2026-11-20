# ─────────────────────────────────────────────────────────────────────────────
# Enterprise NIDS — Windows Docker Helper Script
# ─────────────────────────────────────────────────────────────────────────────
# Usage:
#   .\run_docker.ps1 up      - Start all services (default)
#   .\run_docker.ps1 down    - Stop and clean up all services
#   .\run_docker.ps1 build   - Rebuild Python/API/Worker container images
#   .\run_docker.ps1 status  - Check running containers and health status
#   .\run_docker.ps1 logs    - Stream container logs
# ─────────────────────────────────────────────────────────────────────────────

param (
    [Parameter(Position=0)]
    [ValidateSet("up", "down", "build", "status", "logs")]
    [string]$Action = "up"
)

$DockerDesktopPath = "C:\Program Files\Docker\Docker\Docker Desktop.exe"

# ── Function: Check if Docker is installed ────────────────────────────────────
function Check-DockerInstalled {
    $dockerPath = Get-Command docker -ErrorAction SilentlyContinue
    if (-not $dockerPath) {
        Write-Host ""
        Write-Host "[ERROR] Docker command-line tool ('docker') was not found in your system PATH." -ForegroundColor Red
        Write-Host "If you haven't installed Docker Desktop yet, please refer to the guide:" -ForegroundColor Yellow
        Write-Host "  -> .\docs\docker_guide.md" -ForegroundColor Yellow
        Write-Host ""
        return $false
    }
    return $true
}

# ── Function: Check and start Docker Daemon ──────────────────────────────────
function Ensure-DockerRunning {
    Write-Host "[*] Checking if Docker daemon is running..." -ForegroundColor Cyan
    
    # Try running a simple docker command to test daemon connection
    docker ps -q > $null 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[+] Docker daemon is running and responsive." -ForegroundColor Green
        return $true
    }

    Write-Host "[!] Docker daemon is not running." -ForegroundColor Yellow

    if (Test-Path $DockerDesktopPath) {
        Write-Host "[*] Launching Docker Desktop from: $DockerDesktopPath" -ForegroundColor Cyan
        Start-Process $DockerDesktopPath
        
        Write-Host "[*] Waiting for Docker daemon to initialize (this may take up to 45 seconds)..." -ForegroundColor Cyan
        $timeout = 45 # seconds
        $elapsed = 0
        while ($elapsed -lt $timeout) {
            Start-Sleep -Seconds 3
            $elapsed += 3
            docker ps -q > $null 2>&1
            if ($LASTEXITCODE -eq 0) {
                Write-Host "[+] Docker daemon is now running!" -ForegroundColor Green
                return $true
            }
            Write-Host "  ... waiting ($elapsed seconds elapsed) ..." -ForegroundColor Gray
        }
        Write-Host "[ERROR] Docker Desktop failed to start or initialize in a timely manner." -ForegroundColor Red
        Write-Host "Please start Docker Desktop manually and run this script again." -ForegroundColor Yellow
        return $false
    } else {
        Write-Host "[ERROR] Docker Desktop was not found at the expected path: $DockerDesktopPath" -ForegroundColor Red
        Write-Host "Please start Docker Desktop manually or install it if it is missing." -ForegroundColor Yellow
        return $false
    }
}

# ── Main Script Logic ─────────────────────────────────────────────────────────

if (-not (Check-DockerInstalled)) {
    exit 1
}

if ($Action -ne "down" -and -not (Ensure-DockerRunning)) {
    exit 1
}

# Ensure .env file exists, copy from .env.example if missing
if (-not (Test-Path ".env")) {
    Write-Host "[*] Creating .env file from .env.example..." -ForegroundColor Cyan
    Copy-Item ".env.example" ".env"
    Write-Host "[+] Created local .env config." -ForegroundColor Green
}

switch ($Action) {
    "up" {
        Write-Host "[*] Starting NIDS infrastructure & application services..." -ForegroundColor Cyan
        docker compose up -d
        if ($LASTEXITCODE -eq 0) {
            Write-Host ""
            Write-Host "================────────────────────────────────==========" -ForegroundColor Green
            Write-Host " Enterprise NIDS stack started successfully!" -ForegroundColor Green
            Write-Host "================────────────────────────────────==========" -ForegroundColor Green
            Write-Host " -> FastAPI API URL  : http://localhost:8000" -ForegroundColor Cyan
            Write-Host " -> API Swagger Docs : http://localhost:8000/docs" -ForegroundColor Cyan
            Write-Host " -> Grafana Dashboard: http://localhost:3001 (Default credentials: admin/admin)" -ForegroundColor Cyan
            Write-Host " -> TimescaleDB Port : 5432" -ForegroundColor Cyan
            Write-Host " -> Redis Port       : 6379" -ForegroundColor Cyan
            Write-Host "================────────────────────────────────==========" -ForegroundColor Green
            Write-Host "To monitor logs, run: .\run_docker.ps1 logs" -ForegroundColor Yellow
        } else {
            Write-Host "[ERROR] Failed to start Docker services." -ForegroundColor Red
        }
    }
    "down" {
        Write-Host "[*] Stopping NIDS container services..." -ForegroundColor Cyan
        docker compose down
        if ($LASTEXITCODE -eq 0) {
            Write-Host "[+] Containers stopped and networks removed." -ForegroundColor Green
        }
    }
    "build" {
        Write-Host "[*] Rebuilding container images..." -ForegroundColor Cyan
        docker compose build
        if ($LASTEXITCODE -eq 0) {
            Write-Host "[+] Build completed successfully." -ForegroundColor Green
        } else {
            Write-Host "[ERROR] Build failed." -ForegroundColor Red
        }
    }
    "status" {
        Write-Host "[*] Showing NIDS container status..." -ForegroundColor Cyan
        docker compose ps
    }
    "logs" {
        Write-Host "[*] Streaming NIDS service logs (Press Ctrl+C to stop)..." -ForegroundColor Cyan
        docker compose logs -f
    }
}
