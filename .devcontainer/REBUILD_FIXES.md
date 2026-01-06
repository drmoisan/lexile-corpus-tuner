# Dev Container Rebuild Error Fixes

## Fix Applied (2024-12-24)

### Issue
Exit code 1 error when rebuilding container with WSL Wayland socket mount error:
```
--mount type=bind,source=\\wsl.localhost\Ubuntu\mnt\wslg\runtime-dir\wayland-0...
```

### Root Cause
The `desktop-lite` feature attempts to mount a WSL Wayland socket that may not exist or isn't accessible on Windows hosts, especially when:
- Running Docker Desktop with WSL2 backend
- The Wayland display server socket path doesn't exist or has permission issues
- The feature tries to auto-detect and mount GUI-related sockets

### Solution Applied
Removed the `desktop-lite` feature from `devcontainer.json`:

```json
// REMOVED:
"ghcr.io/devcontainers/features/desktop-lite:1": {
  "password": "vscode",
  "webPort": "6080",
  "vncPort": "5901"
},
"forwardPorts": [6080, 5901],
```

### Why This Works
- The desktop-lite feature is designed for GUI applications running in the container
- This project doesn't require a VNC desktop environment
- Removing it eliminates the problematic Wayland socket mount
- CLI tools (PowerShell, Python, git, etc.) work fine without it

### If You Need GUI Applications
If you later need to run GUI apps in the container:

1. **Use X11 forwarding instead**:
   ```json
   "runArgs": [
     "--env=DISPLAY=host.docker.internal:0"
   ]
   ```

2. **Use Windows X Server**: Install VcXsrv or Xming on Windows host

3. **Alternative**: Run GUI apps on host, use container for CLI only

---

## Additional Troubleshooting Steps

If rebuilding still fails after removing desktop-lite:

### 1. Check Docker Desktop Status
```powershell
# Ensure Docker Desktop is running
docker info
```

### 2. Verify WSL2 Integration
**Docker Desktop → Settings → Resources → WSL Integration**
- Enable integration with your WSL distros
- Ensure "Use the WSL 2 based engine" is enabled

### 3. Clean Docker System
```powershell
# WARNING: Removes all unused Docker data
docker system prune -a --volumes
```

### 4. Check Bind Mount Paths
Ensure these directories exist:
```powershell
# Main workspace (should exist)
Test-Path "d:\repos\lexile-corpus-tuner"

# Artifacts directory
Test-Path "d:\repos\lexile-artifacts"
```

Create if missing:
```powershell
New-Item -ItemType Directory -Path "d:\repos\lexile-artifacts" -Force
```

### 5. Rebuild Without Cache
In VS Code:
- `F1` → **Dev Containers: Rebuild Container Without Cache**

### 6. Check Docker Logs
```powershell
# View recent Docker logs
docker events --since 10m
```

### 7. Verify Docker Resources
**Docker Desktop → Settings → Resources**
- Memory: At least 4GB (8GB recommended)
- CPUs: At least 2 (4 recommended)
- Disk: At least 20GB free

### 8. Check Windows Firewall
- Docker Desktop may need firewall exceptions
- Check Windows Security → Firewall & network protection

### 9. Restart Docker Desktop
Sometimes a simple restart helps:
- Right-click Docker Desktop tray icon → Quit Docker Desktop
- Wait 10 seconds
- Restart Docker Desktop

---

## Alternative: Use Remote-WSL Instead

If dev containers continue to have issues, consider using WSL directly:

1. **Move workspace to WSL**:
   ```bash
   # In WSL terminal
   cd ~
   git clone <your-repo-url>
   ```

2. **Open in VS Code**:
   ```bash
   code .
   ```

3. **Install Remote-WSL extension**
   - Extension ID: `ms-vscode-remote.remote-wsl`

This avoids Docker entirely while still providing a Linux environment.

---

## Reporting Issues

If none of these solutions work, collect this information:

```powershell
# Docker version
docker version

# Docker Desktop settings
docker info

# WSL version
wsl --version

# VS Code version
code --version

# Dev Container extension version
code --list-extensions --show-versions | Select-String "remote-containers"
```

Include the full error log from:
- View → Output → Select "Dev Containers" from dropdown
- Copy the entire log, especially lines before "Exit code 1"
