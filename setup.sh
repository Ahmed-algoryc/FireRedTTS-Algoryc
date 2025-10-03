#!/usr/bin/env bash
set -euo pipefail

# FireRedTTS2 setup script
# - Creates Python venv
# - Installs build deps and requirements
# - Prepares repo structure
# - Ready to run the main server (main.py)

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

# Colors
GREEN="\033[0;32m"
YELLOW="\033[1;33m"
RED="\033[0;31m"
NC="\033[0m"

msg() { echo -e "${GREEN}==>${NC} $*"; }
warn() { echo -e "${YELLOW}==>${NC} $*"; }
err() { echo -e "${RED}==>${NC} $*"; }

# 1) Optional system deps (safe on Ubuntu/Debian; skip on failure)
msg "Installing optional system packages (ffmpeg, git-lfs)..."
if command -v apt >/dev/null 2>&1; then
  sudo apt update -y || true
  sudo apt install -y ffmpeg git-lfs || true
  git lfs install || true
else
  warn "apt not found; skipping system package install"
fi

# 2) Python venv
PYTHON_BIN=${PYTHON_BIN:-python3}
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  err "python3 not found; please install Python 3.10+"
  exit 1
fi

if [ ! -d "venv" ]; then
  msg "Creating virtual environment..."
  "$PYTHON_BIN" -m venv venv
fi

msg "Activating virtual environment..."
# shellcheck disable=SC1091
source venv/bin/activate

msg "Upgrading pip..."
pip install -U pip wheel setuptools

# 3) Install project and requirements
if [ -f requirements.txt ]; then
  msg "Installing requirements.txt..."
  pip install -r requirements.txt || warn "Some requirements may already be satisfied"
fi

msg "Installing project in editable mode..."
pip install -e .

# 4) Extra runtime deps used by the server/UI
msg "Installing extra runtime dependencies..."
pip install aiohttp aiohttp-cors websockets openai torchaudio --upgrade
# Whisper (smaller model) for auto-transcription
pip install -U openai-whisper || true

# 5) Create expected directories
msg "Ensuring templates/ and voice_profiles/ exist..."
mkdir -p templates voice_profiles/audio tests/output

# 6) Final instructions
cat <<EOT

${GREEN}Setup complete.${NC}

Next steps:
1) Put FireRedTTS2 weights in pretrained_models/FireRedTTS2 (see README.md)
2) Set your OpenAI key in the environment before running (if needed):
   export OPENAI_API_KEY="YOUR_KEY"
3) Run the server:
   source venv/bin/activate && python main.py

Repo layout expectations:
- templates/voice_chat.html (UI)
- main.py (RunPod-compatible server)
- voice_profiles/ (cloned profiles)
- pretrained_models/FireRedTTS2/ (model weights)

EOT
