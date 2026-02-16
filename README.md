# English Morning Call ☀️

Real-time English conversation practice with Gemini Live API.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python server.py
```

Open `http://<your-ip>:8080` on your phone.

## Notes
- Audio: Input PCM 16-bit mono 16kHz, Output PCM 16-bit mono 24kHz
- iOS Safari requires user gesture to start AudioContext
- For HTTPS (required for mic on non-localhost), use a reverse proxy or ngrok
