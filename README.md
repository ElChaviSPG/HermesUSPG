# HermesUSPG

Código de la Jetson Orin Nano del robot Hermes (copia del 2026-10-31).

- `hermesJetson/`: voz, visión, gestos y red hacia los brazos.
- `requirements.txt`: dependencias del `venv` de la Jetson (arm64, Python 3.10).
- `voices/`: configuración de la voz Piper. El modelo `es_ES-davefx-medium.onnx` (60 MB) no está en el repo; descárgalo de https://huggingface.co/rhasspy/piper-voices/tree/main/es/es_ES/davefx/medium

La API key de Groq va en la variable de entorno `GROQ_API_KEY`; no la escribas en el código.

## Ramas y responsables

Cada parte del robot tiene su rama. Trabaja en la tuya y abre un pull request hacia `main`; `main` no recibe cambios directos.

| Rama | Parte | Archivos principales | Responsable (GitHub) |
|---|---|---|---|
| `voz` | Voz | `groqChat.py`, `voices/` | por definir |
| `audio` | Audio | `bocinas.py` | por definir |
| `mano` | Mano | `hermes_gesture.py` | por definir |
| `brazo` | Brazo | `hermes_net.py` | por definir |
| `ia` | IA y visión | `vision.py`, `mirror.py` | por definir |
| `ojos` | Ojos | (en la Raspberry Pi, pendiente) | por definir |
| `pantalla` | Pantalla | (pendiente) | por definir |
