# HermesUSPG

Código de la Jetson Orin Nano del robot Hermes (copia del 2026-10-31).

- `hermesJetson/`: voz, visión, gestos y red hacia los brazos.
- `requirements.txt`: dependencias del `venv` de la Jetson (arm64).
- `voices/`: configuración de la voz Piper. El modelo `es_ES-davefx-medium.onnx` (60 MB) no está en el repo; descárgalo de https://huggingface.co/rhasspy/piper-voices/tree/main/es/es_ES/davefx/medium

La API key de Groq va en la variable de entorno `GROQ_API_KEY`; no la escribas en el código.
