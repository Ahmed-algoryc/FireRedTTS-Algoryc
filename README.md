# FireRedTTS2 - Realtime Chat 

This repo hosts a working setup for FireRedTTS2 with:
- Full TTS generation (stable, non-streaming) via WebSocket UI
- Voice cloning with PKL profiles (prompt WAV optional)
- RunPod-compatible single-port server

## 1) Prerequisites
- Ubuntu 20.04+ (or compatible Linux)
- Python 3.10–3.12
- NVIDIA GPU + CUDA drivers (for PyTorch CUDA)

## 2) Setup
```bash
# Clone your project
cd /root/Projects
# (assuming repo already present as FireRedTTS2)
    cd FireRedTTS2

# Create venv
python3 -m venv venv
source venv/bin/activate

# Install project (editable) and deps
    pip install -e .
# Install extras you use
pip install aiohttp aiohttp-cors whisper openai torchaudio websockets

# Download pretrained FireRedTTS2 weights
mkdir -p pretrained_models/FireRedTTS2
# Place llm_pretrain.pt, llm_posttrain.pt, codec.pt, config files, and Qwen2.5 tokenizer there
```

Expected layout:
```
pretrained_models/FireRedTTS2/
  ├─ config_llm.json
  ├─ llm_pretrain.pt
  ├─ llm_posttrain.pt
  ├─ config_codec.json
  ├─ codec.pt
  └─ Qwen2.5-1.5B/  (tokenizer folder)
```

## 3) Run the server
```bash
source venv/bin/activate
python runpod_server.py
```
- URL: http://<host>:8080/voice_chat.html
- WebSocket endpoint: /ws

## 4) Using the UI
- Select a voice from the dropdown
- Type a message and Send
- You’ll see the GPT reply and hear full TTS audio

## 5) Voice cloning
- In the UI, provide a name and upload a short sample (2–8s works well)
- If transcription is left empty, Whisper auto-transcribes and stores it
- Profile is saved in `voice_profiles/<name>.pkl`
  - If a WAV was uploaded, it’s saved to `voice_profiles/audio/<name>.wav`
  - PKL-first loading is supported; if `audio_tokens` exist they will be used even if WAV is missing

## 6) Project structure (key files)
```
FireRedTTS2/
  ├─ runpod_server.py        # main entry (RunPod-ready HTTP+WS server)
  ├─ templates/              # HTML templates (voice_chat.html)
  ├─ tests/                  # test scripts (moved here)
  ├─ voice_profiles/         # cloned profiles (PKL) and audio cache
  ├─ pretrained_models/      # model weights + tokenizer
  ├─ fireredtts2/            # model package (unchanged)
  └─ README.md
```

## 7) Notes on streaming vs full TTS
- Full TTS (non-streaming) is the most stable path and matches quality of simple tests
- If you need streaming, use sentence-level chunks with a small playback buffer; avoid token-drip

## 8) Troubleshooting
- If audio cuts off: increase generation length in `fireredtts2/fireredtts2.py` (generate_single)
- If KV cache assertions: trim prompt audio to 2–4s and keep prompt_text ≤80 chars
- If no audio: ensure profiles load (see server logs) and that paths aren’t under /tmp

## 9) Development
- WAV files and venv are excluded by `.gitignore`
- Tests live under `tests/`
- Templates under `templates/`

## 10) Push to repo
```bash
git add .
git commit -m "Setup RunPod server, cloning, full TTS; structure & README"
git push origin <your-branch>
```
