# Enterprise NIDS — Docker on Windows Setup Guide

This guide describes how to install, configure, and troubleshoot Docker Desktop on Windows for the Enterprise NIDS project.

---

## 1. Installation of Docker Desktop

To run the multi-container stack, you must install **Docker Desktop for Windows**.

### Steps:
1. **Download the Installer**:
   - Go to [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop/) and download the installer.
2. **Run the Installer**:
   - Double-click `Docker Desktop Installer.exe` to run the installer.
   - Ensure the **Use WSL 2 instead of Hyper-V (recommended)** option is checked on the Configuration page.
3. **Restart Your Computer**:
   - Follow the prompts to finish the installer and restart your machine.
4. **Complete Setup**:
   - Start **Docker Desktop** from the Start Menu.
   - Accept the Docker subscription service agreement.

---

## 2. WSL 2 Backend Integration (Crucial for ML Performance)

Docker Desktop uses Windows Subsystem for Linux 2 (WSL 2) as its virtualization engine, which offers superior performance for machine learning pipelines compared to Hyper-V.

### Enable WSL 2 Integration:
1. Open **Docker Desktop**.
2. Click the **Settings** gear icon in the top right.
3. Go to **General** and ensure **Use the WSL 2 based engine** is checked.
4. Go to **Resources** > **WSL Integration**.
5. Enable integration with your default WSL distro if you are using one (e.g., Ubuntu).
6. Click **Apply & Restart**.

---

## 3. Configuring WSL 2 Resource Limits (Prevent RAM Exhaustion)

By default, WSL 2 can consume up to 50% (or more) of your host machine's total RAM, which may cause Windows to slow down or crash, especially when loading ML models (PyTorch, LightGBM) inside containers.

It is **highly recommended** to limit WSL 2 memory.

### Setup Steps:
1. Press `Win + R`, type `%USERPROFILE%`, and press Enter. This opens your user profile folder (e.g. `C:\Users\YourName`).
2. Create a new text file named `.wslconfig` (ensure it does not end with `.txt`).
3. Paste the following configuration, adjusting values according to your host RAM:

```ini
[wsl2]
# Limit memory to half of your host's memory (e.g., 8GB if you have 16GB, or 16GB if you have 32GB)
memory=8GB

# Limit processors to 4 cores
processors=4

# Set swap space to 2GB
swap=2GB

# Turn off page reporting to reclaim memory immediately
pageReporting=true
```

4. Save the file.
5. Restart WSL by opening a PowerShell command prompt and running:
   ```powershell
   wsl --shutdown
   ```
6. Restart Docker Desktop.

---

## 4. Running the Stack with the PowerShell Helper

A PowerShell script `run_docker.ps1` is provided in the project root to manage the container lifecycle.

### Usage Instructions:
Open a PowerShell terminal in the project root directory and run the following commands:

* **Start the stack** (Auto-creates `.env` if missing, starts Redis, TimescaleDB, Grafana, FastAPI, and Celery workers):
  ```powershell
  .\run_docker.ps1 up
  ```
  Once up, the services are mapped as follows:
  - **FastAPI API**: [http://localhost:8000](http://localhost:8000)
  - **Interactive Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
  - **Grafana Dashboards**: [http://localhost:3001](http://localhost:3001) (Credentials: `admin/admin`)
  - **TimescaleDB Database**: Port `5432` (Credentials in `.env`)
  - **Redis Cache**: Port `6379`

* **Stream container logs** to watch API requests and Celery tasks processing:
  ```powershell
  .\run_docker.ps1 logs
  ```

* **Check container health status**:
  ```powershell
  .\run_docker.ps1 status
  ```

* **Stop the stack**:
  ```powershell
  .\run_docker.ps1 down
  ```

* **Rebuild Python app/worker images** (run this whenever you change dependencies or configuration):
  ```powershell
  .\run_docker.ps1 build
  ```

---

## 5. Troubleshooting Common Issues

### Issue 1: `docker : The term 'docker' is not recognized`
* **Cause**: Docker Desktop is not installed, or its installation path was not added to your system environment variables.
* **Fix**: Reinstall Docker Desktop, and restart your PowerShell window. Ensure "Docker Desktop" is checked in your system startup programs.

### Issue 2: Port conflicts on 5432 (Postgres) or 6379 (Redis)
* **Cause**: You have a local PostgreSQL or Redis service running on your host machine outside of Docker.
* **Fix**:
  - Stop local PostgreSQL: Open services (`services.msc`), find `postgresql-x64-...`, click **Stop**.
  - Stop local Redis (if running as a service): Find `Redis` in services, click **Stop**.
  - *Alternative*: Modify the external ports in `docker-compose.yml` (e.g. change `"5432:5432"` to `"5435:5432"` for TimescaleDB).

### Issue 3: Container crashes with Out Of Memory (OOM)
* **Cause**: PyTorch or SHAP worker consumed more memory than allocated to WSL 2.
* **Fix**: Increase the memory allocation in your `%USERPROFILE%\.wslconfig` (e.g., set `memory=12GB` or `16GB`) and restart WSL.
