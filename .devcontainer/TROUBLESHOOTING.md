# Dev Container Troubleshooting Guide

## Common Issues and Solutions

### 1. Container Won't Build

#### Symptoms
- Build fails with errors in Docker logs
- "Failed to create container" message

#### Solutions

**Check Docker Desktop Status**
```bash
# Ensure Docker Desktop is running
# Windows: Check system tray for Docker icon
# Should show "Docker Desktop is running"
```

**Check Resources**
- Docker Desktop → Settings → Resources
- Minimum: 4GB RAM, 2 CPUs
- Recommended: 8GB RAM, 4 CPUs

**Clear Docker Cache**
```bash
# Outside container, in PowerShell:
docker system prune -a --volumes
# WARNING: This removes all unused Docker data
```

**Rebuild Without Cache**
```
F1 → Dev Containers: Rebuild Container Without Cache
```

---

### 2. Poetry Install Fails

#### Symptoms
- "poetry install" errors during post-create
- Missing Python packages

#### Solutions

**Check poetry.lock**
```bash
# Inside container:
poetry lock --check
```

**Clear Poetry Cache**
```bash
poetry cache clear pypi --all
poetry install --no-cache
```

**Verify Python Version**
```bash
python --version  # Should be 3.13.x
poetry env info
```

---

### 3. PowerShell Modules Not Found

#### Symptoms
- "Module PSScriptAnalyzer not found"
- "Module Pester not found"

#### Solutions

**Verify Installation**
```bash
pwsh -Command "Get-Module -ListAvailable PSScriptAnalyzer, Pester"
```

**Reinstall Modules**
```bash
sudo pwsh -Command "
    Set-PSRepository -Name PSGallery -InstallationPolicy Trusted;
    Install-Module -Name PSScriptAnalyzer -Scope AllUsers -Force;
    Install-Module -Name Pester -MinimumVersion 5.0.0 -Scope AllUsers -Force;
"
```

**Import PoshQC Module**
```bash
pwsh -Command "Import-Module -Force ./scripts/powershell/PoshQC"
```

---

### 4. VS Code Extensions Not Loading

#### Symptoms
- Extensions list shows "Not Installed"
- Pylance, Black, Ruff, PowerShell not working

#### Solutions

**Reinstall Extensions**
```
F1 → Dev Containers: Rebuild Container
```

**Manual Install**
1. Open Extensions panel (`Ctrl+Shift+X`)
2. Search for extension
3. Click "Install in Dev Container"

**Check Extension Compatibility**
- Some extensions don't support Linux containers
- Check extension details for platform support

---

### 5. Git Configuration Issues

#### Symptoms
- Git asks for name/email on every commit
- Git operations fail

#### Solutions

**Configure Git in Container**
```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

**Use SSH Keys**
```bash
# Add to devcontainer.json mounts:
"mounts": [
  "source=${localEnv:HOME}/.ssh,target=/home/vscode/.ssh,type=bind,consistency=cached"
]
```

---

### 6. Performance Issues (Windows)

#### Symptoms
- Slow file operations
- High CPU usage
- Lag in VS Code

#### Solutions

**Use WSL 2 Backend**
1. Docker Desktop → Settings → General
2. Enable "Use the WSL 2 based engine"
3. Restart Docker Desktop

**Move Workspace to WSL**
```bash
# In WSL terminal:
cd ~
git clone your-repo
code .
```

**Optimize File Watching**
Add to devcontainer.json:
```json
"settings": {
  "files.watcherExclude": {
    "**/.venv/**": true,
    "**/node_modules/**": true
  }
}
```

---

### 7. Port Forwarding Issues

#### Symptoms
- Cannot access services running in container
- Localhost connections fail

#### Solutions

**Check Forwarded Ports**
- VS Code → PORTS panel (bottom)
- Should auto-forward when service starts

**Manual Port Forward**
Add to devcontainer.json:
```json
"forwardPorts": [8000, 3000]
```

**Firewall Check**
- Ensure Windows Firewall allows Docker
- Docker Desktop should configure this automatically

---

### 8. Volume Mount Problems

#### Symptoms
- Changes not reflected in container
- `.venv` folder issues

#### Solutions

**Verify Mounts**
```bash
# Inside container:
df -h
mount | grep workspace
```

**Reset .venv**
```bash
# Outside container (host):
rm -rf .venv

# Reopen in container - will recreate .venv
```

**Check File Permissions**
```bash
# Inside container:
ls -la /workspace/.venv
# Should be owned by vscode:vscode
```

---

### 9. Terminal Not Working

#### Symptoms
- Terminal won't open
- Commands not found
- Wrong shell

#### Solutions

**Reset Default Shell**
```
F1 → Terminal: Select Default Profile
Choose: pwsh or bash
```

**New Terminal**
```
Ctrl+Shift+`
```

**Check Shell Path**
```bash
echo $SHELL
which pwsh
which bash
```

---

### 10. Tasks Not Running

#### Symptoms
- "Task not found" errors
- Tasks fail in container

#### Solutions

**Verify Task Paths**
- Tasks in `.vscode/tasks.json` should use `${workspaceFolder}`
- Not absolute paths

**Run Task Manually**
```bash
# Test the command directly:
poetry run pytest
pwsh scripts/dev-tools/fix-all.ps1
```

**Check Working Directory**
```bash
pwd  # Should be /workspace
```

---

## Advanced Troubleshooting

### Container Logs
```
F1 → Dev Containers: Show Container Log
```

### Attach Shell to Running Container
```bash
# Outside container:
docker ps  # Find container ID
docker exec -it <container-id> pwsh
```

### Inspect Container
```bash
docker inspect lexile-corpus-tuner-dev
```

### Start Container Manually
```bash
docker start lexile-corpus-tuner-dev
docker attach lexile-corpus-tuner-dev
```

---

## Getting Help

### Useful Commands

**Container Status**
```bash
docker ps -a
```

**Container Resources**
```bash
docker stats lexile-corpus-tuner-dev
```

**Docker Logs**
```bash
docker logs lexile-corpus-tuner-dev
```

### Documentation Links

- [VS Code Dev Containers Docs](https://code.visualstudio.com/docs/devcontainers/containers)
- [Docker Desktop Troubleshooting](https://docs.docker.com/desktop/troubleshoot/)
- [Dev Container Specification](https://containers.dev/)

### Repository Issues

If you encounter a bug specific to this dev container setup:
1. Check existing issues: `gh issue list --label devcontainer`
2. Create new issue: `gh issue create --label devcontainer`
3. Include container logs and error messages

---

## Nuclear Options

### Complete Reset

**Remove Container**
```bash
docker rm -f lexile-corpus-tuner-dev
```

**Remove Image**
```bash
docker images | grep lexile-corpus-tuner
docker rmi <image-id>
```

**Rebuild Everything**
```
F1 → Dev Containers: Rebuild Container Without Cache
```

### Clean Docker Installation

**Remove All Docker Data**
```bash
docker system prune -a --volumes
# WARNING: Removes ALL Docker containers, images, volumes
```

**Reset Docker Desktop**
- Docker Desktop → Troubleshoot → Reset to Factory Defaults
- WARNING: Nuclear option - last resort only
