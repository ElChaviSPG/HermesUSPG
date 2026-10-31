import os
import warnings
import logging

# ─── Silenciar advertencias ───────────────────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"]     = "3"
os.environ["GRPC_VERBOSITY"]       = "ERROR"
os.environ["PYTHONWARNINGS"]       = "ignore"
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)
logging.getLogger("mediapipe").setLevel(logging.ERROR)

import cv2
import time
import threading
import base64
import numpy as np
import mediapipe as mp
from jtop import jtop
from piper import PiperVoice
import wave
import subprocess
import tempfile
from PIL import Image, ImageDraw, ImageFont
from groq import Groq

# [PARCHE-GESTOS] capa de red UDP + motor de gestos de dedos
import hermes_net as net
import hermes_gesture as gesture
# [PARCHE-PRESENTA] saludo protocolario con cola de audio
import hermes_presenta as presenta

# ─── Configuración ────────────────────────────────────────────────────────────
CAM_W         = 1280
CAM_H         = 720
VOICE_MODEL   = "/home/jetson/voices/es_ES-davefx-medium.onnx"
MIC_NAME      = "C270"
MIC_CHANNELS  = 1
MIC_RATE      = 16000
CHAT_MAX      = 6

# ─── Auto-detección de resolución de pantalla ─────────────────────────────────
import subprocess as _sp_for_screen

try:
    _out = _sp_for_screen.check_output(["xrandr"], stderr=_sp_for_screen.DEVNULL).decode()
    SCREEN_W, SCREEN_H = 1280, 720
    for _line in _out.splitlines():
        if "*" in _line:
            for _p in _line.split():
                if "x" in _p and _p[0].isdigit():
                    _w, _h = _p.split("x")
                    SCREEN_W = int(_w)
                    SCREEN_H = int(_h.split("+")[0] if "+" in _h else _h)
                    break
            break
except Exception:
    SCREEN_W, SCREEN_H = 1280, 720

print(f"Pantalla detectada: {SCREEN_W}x{SCREEN_H}")

# ─── Factor de escala de UI ───────────────────────────────────────────────────
# Base: diseño pensado para 1280x720. Todo se escala proporcionalmente.
UI_SCALE = min(SCREEN_W / 1280, SCREEN_H / 720)

# Boost adicional si la pantalla es chica (≤1024) — letras más grandes en 7"
if SCREEN_W <= 1024:
    UI_SCALE *= 1.15

print(f"UI_SCALE = {UI_SCALE:.2f}")

def S(x):
    """Escala un valor numérico (int o float) según UI_SCALE."""
    return int(round(x * UI_SCALE))

CHAT_W = S(380)

# ─── Groq ─────────────────────────────────────────────────────────────────────
# NOTA (ago 2026): llama-3.3-70b-versatile y meta-llama/llama-4-scout-17b-16e-
# instruct fueron deprecados por Groq (apagados el 16 ago 2026). Reemplazados
# por los modelos recomendados en su pagina de deprecaciones:
#   - LLM texto     -> openai/gpt-oss-120b
#   - Vision (multimodal) -> qwen/qwen3.6-27b (Groq lo sirve como preview,
#     es el unico modelo multimodal vigente en su catalogo por ahora)
# whisper-large-v3-turbo sigue vigente, no se toca.
GROQ_API_KEY        = os.environ.get("GROQ_API_KEY", "TU_API_KEY_AQUI")
GROQ_STT_MODEL      = "whisper-large-v3-turbo"
GROQ_LLM_MODEL      = "openai/gpt-oss-120b"
GROQ_VISION_MODEL   = "qwen/qwen3.6-27b"
groq_client         = Groq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = (
    "Eres Hermes, un humanoide inteligente creado por Creabot. "
    "Existes para ayudar, enseñar y conversar de manera natural y cercana. "
    "Eres carismático, cálido y directo, vas al punto sin rodeos. "
    "Usas humor ligero y sutil, eres paciente y nunca haces sentir a nadie torpe. "
    "Hablas simple, como a un niño curioso de 8 años, salvo que la persona sea técnica. "
    "Siempre respondes en español natural, sin sonar traducido ni robótico. "
    "Nunca uses listas, viñetas ni formato, solo texto conversacional. "
    "Nunca reveles qué modelo de lenguaje eres ni menciones marcas de IA. "
    "Si te preguntan de qué estás hecho, dices que eres Hermes, un humanoide de Creabot, y cambias el tema. "
    "Nunca digas que eres IA, asistente virtual o chatbot. Nunca rompas el personaje. "
    "No inventes datos, si no sabes algo lo dices con honestidad. "
    "IMPORTANTE: responde siempre en máximo 2 oraciones cortas, nunca más."
)

VISION_PROMPT = (
    "Analiza esta secuencia de imágenes tomadas por una cámara de robot. "
    "Responde SOLO con una línea corta en español describiendo: "
    "cuántas personas hay, género aproximado, edad aproximada (niño/joven/adulto/mayor), "
    "y expresión facial (feliz/serio/confundido/neutro). "
    "Si no hay personas visibles, responde: 'No se ven personas.' "
    "Ejemplo: '2 personas: mujer joven ~25 años sonriendo, hombre adulto ~40 años serio.' "
    "Sé breve y directo, sin explicaciones adicionales."
)

# ─── Captura de fotos durante grabación ───────────────────────────────────────
_vision_frames       = []
_vision_frames_lock  = threading.Lock()
_vision_context      = ""
_vision_context_lock = threading.Lock()

_latest_frame      = None
_latest_frame_lock = threading.Lock()

VISION_PHOTOS = 3

def _capture_vision_frames(duration):
    """
    Captura fotos cada 1.5 segundos durante la grabación.
    Si el usuario corta antes, igual hay al menos 1-2 fotos.
    """
    global _vision_frames
    with _vision_frames_lock:
        _vision_frames = []

    # Primera foto a los 0.5s (rápido), luego cada 1.5s
    captured = 0
    next_capture = 0.5
    t0 = time.time()

    while recording and captured < 5:
        elapsed = time.time() - t0
        if elapsed >= next_capture:
            with _latest_frame_lock:
                frame = _latest_frame
            if frame is not None:
                with _vision_frames_lock:
                    _vision_frames.append(frame.copy())
                captured += 1
                print(f">>> Foto {captured} capturada (t={elapsed:.1f}s)")
            next_capture += 1.5
        time.sleep(0.1)
    print(f">>> Captura terminada: {captured} fotos")

def _frame_to_base64(frame):
    """Reduce a 640px antes de codificar — ahorra ~75% de bandwidth a Groq."""
    h, w = frame.shape[:2]
    if w > 640:
        scale = 640 / w
        frame = cv2.resize(frame, (640, int(h * scale)),
                           interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return base64.b64encode(buf).decode("utf-8")

def _analyze_vision(frames):
    print(f">>> Analizando {len(frames)} fotos con Groq Vision...")
    if not frames:
        print(">>> No hay fotos para analizar")
        return "No se pudo capturar imagen."
    try:
        # Guardar las fotos a disco para inspección manual (debug)
        for i, f in enumerate(frames):
            cv2.imwrite(f"/tmp/hermes_vision_{i}.jpg", f)
        print(f">>> Fotos guardadas en /tmp/hermes_vision_*.jpg")

        content = [{"type": "text", "text": VISION_PROMPT}]
        for frame in frames:
            b64 = _frame_to_base64(frame)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
            })
        print(f">>> Enviando a Groq Vision ({GROQ_VISION_MODEL})...")
        response = groq_client.chat.completions.create(
            model=GROQ_VISION_MODEL,
            messages=[{"role": "user", "content": content}],
            max_tokens=100,
            temperature=0.2,
        )
        result = response.choices[0].message.content.strip()
        print(f">>> Visión respondió: {result}")
        return result
    except Exception as e:
        print(f">>> Error visión: {e}")
        import traceback
        traceback.print_exc()
        return ""

# ─── Fuentes UTF-8 ────────────────────────────────────────────────────────────
FONT_PATH      = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_PATH_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
_font_cache    = {}

def get_font(size, bold=False):
    # Aplicar escala UI automáticamente
    real_size = max(8, int(round(size * UI_SCALE)))
    key = (real_size, bold)
    if key not in _font_cache:
        _font_cache[key] = ImageFont.truetype(
            FONT_PATH_BOLD if bold else FONT_PATH, real_size)
    return _font_cache[key]

# ─── Auto-descubrimiento de audio ─────────────────────────────────────────────
def find_mic_device(name_hint=MIC_NAME):
    try:
        import re
        out = subprocess.check_output(["arecord", "-l"],
                                      stderr=subprocess.DEVNULL).decode()
        for line in out.splitlines():
            if name_hint.lower() in line.lower() and line.startswith("card"):
                card_m = re.search(r"card\s+(\d+)", line)
                dev_m  = re.search(r"device\s+(\d+)", line)
                if card_m and dev_m:
                    device = f"hw:{card_m.group(1)},{dev_m.group(1)}"
                    print(f"Micrófono: {device}")
                    return device
    except Exception as e:
        print(f"Error buscando micrófono: {e}")
    return None

def find_audio_output(name_hint="USB PnP"):
    try:
        import re
        out = subprocess.check_output(["aplay", "-l"],
                                      stderr=subprocess.DEVNULL).decode()
        for line in out.splitlines():
            if name_hint.lower() in line.lower() and line.startswith("card"):
                card_m = re.search(r"card\s+(\d+)", line)
                dev_m  = re.search(r"device\s+(\d+)", line)
                if card_m and dev_m:
                    device = f"plughw:{card_m.group(1)},{dev_m.group(1)}"
                    print(f"Audio salida: {device}")
                    return "default"
    except Exception as e:
        print(f"Error buscando audio salida: {e}")
    return "default"

MIC_DEVICE = find_mic_device() or "hw:0,0"
AUDIO_OUT  = find_audio_output()
print(f"MIC_DEVICE={MIC_DEVICE}  AUDIO_OUT={AUDIO_OUT}")

# ─── TTS (Piper local) ────────────────────────────────────────────────────────
tts_lock    = threading.Lock()
tts_playing = False
voice       = None

def speak(text):
    def _run():
        global tts_playing
        with tts_lock:
            tts_playing = True
        try:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.close()
            with wave.open(tmp.name, "wb") as wav_file:
                voice.synthesize_wav(text, wav_file)
            subprocess.run(["aplay", "-D", AUDIO_OUT, tmp.name],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        finally:
            try:
                os.unlink(tmp.name)
            except Exception:
                pass
            with tts_lock:
                tts_playing = False
    threading.Thread(target=_run, daemon=True).start()

# ─── Chat ─────────────────────────────────────────────────────────────────────
chat_lock    = threading.Lock()
chat_history = []
chat_status  = ""
chat_dirty   = True

def add_chat(role, text):
    global chat_dirty
    with chat_lock:
        chat_history.append({"role": role, "text": text})
        if len(chat_history) > CHAT_MAX * 2:
            chat_history.pop(0)
        chat_dirty = True

def set_status(s):
    global chat_status, chat_dirty
    if s != chat_status:
        chat_status = s
        chat_dirty  = True

# ─── Grabación + STT + Visión + LLM ──────────────────────────────────────────
recording       = False
rec_start_time  = None
rec_proc        = None
rec_tmpfile     = None
_vision_thread  = None

def start_recording():
    global recording, rec_proc, rec_tmpfile, rec_start_time, _vision_thread
    tmp         = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    rec_tmpfile = tmp.name
    tmp.close()
    rec_proc = subprocess.Popen([
        "arecord", "-D", MIC_DEVICE,
        "-f", "S16_LE", "-r", str(MIC_RATE), "-c", str(MIC_CHANNELS),
        rec_tmpfile
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    recording      = True
    rec_start_time = time.time()
    set_status("grabando")
    print(">>> Grabando...")

    _vision_thread = threading.Thread(
        target=_capture_vision_frames, args=(30,), daemon=True)
    _vision_thread.start()

def stop_and_process():
    global recording, rec_proc, rec_tmpfile, rec_start_time
    if not recording:
        return
    if rec_proc:
        rec_proc.terminate()
        rec_proc.wait()
        rec_proc = None
    recording      = False
    rec_start_time = None
    tmp_path       = rec_tmpfile
    rec_tmpfile    = None
    print(">>> Grabación detenida, procesando...")

    with _vision_frames_lock:
        frames_snapshot = list(_vision_frames)

    def _process():
        global _vision_context, chat_dirty
        try:
            if os.path.getsize(tmp_path) < 16000:
                print(">>> Audio muy corto, descartando")
                set_status("")
                return

            # STT y Visión EN PARALELO — ambos van a Groq, sin dependencia entre sí
            vision_result = {"text": ""}

            def _vision_task():
                vision_result["text"] = _analyze_vision(frames_snapshot)

            vision_thread = threading.Thread(target=_vision_task, daemon=True)
            vision_thread.start()

            set_status("transcribiendo")
            with open(tmp_path, "rb") as audio_file:
                transcription = groq_client.audio.transcriptions.create(
                    file=("audio.wav", audio_file, "audio/wav"),
                    model=GROQ_STT_MODEL,
                    language="es",
                    response_format="text",
                )
            text = transcription.strip() if isinstance(transcription, str) \
                   else transcription.text.strip()
            print(f">>> STT: {text}")

            if not text:
                set_status("")
                return

            add_chat("user", text)

            # Esperar a que termine la visión (probablemente ya terminó)
            set_status("analizando")
            vision_thread.join(timeout=8.0)
            visual_context = vision_result["text"]

            with _vision_context_lock:
                _vision_context = visual_context
            chat_dirty = True

            set_status("pensando")

            if visual_context and visual_context != "No se pudo capturar imagen." \
                    and visual_context.strip() != "":
                system_with_vision = (
                    SYSTEM_PROMPT +
                    f"\n\nEN ESTE MOMENTO TUS OJOS VEN: {visual_context}\n"
                    "Esta es información REAL de tu visión actual. "
                    "Si te preguntan qué ves, descríbelo con naturalidad. "
                    "Adapta tu tono según la edad y expresión de la persona."
                )
                print(f">>> System con visión inyectada")
            else:
                system_with_vision = SYSTEM_PROMPT
                print(f">>> Sin contexto visual, usando system base")

            with chat_lock:
                messages = [
                    {"role": "system", "content": system_with_vision}
                ] + [{"role": m["role"], "content": m["text"]}
                     for m in chat_history]

            completion = groq_client.chat.completions.create(
                model=GROQ_LLM_MODEL,
                messages=messages,
                max_tokens=200,
                temperature=0.6,
                top_p=0.85,
                frequency_penalty=0.1,
            )
            answer = completion.choices[0].message.content.strip()
            print(f">>> Hermes: {answer}")

            add_chat("assistant", answer)
            set_status("")
            speak(answer)

        except Exception as e:
            print(f">>> Error: {e}")
            add_chat("assistant", f"[Error: {e}]")
            set_status("")
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    threading.Thread(target=_process, daemon=True).start()

# ─── Stats Jetson (hilo 1 Hz) ─────────────────────────────────────────────────
sys_stats     = {"cpu": 0.0, "gpu": 0.0, "ram": 0.0, "ram_total": 1.0,
                 "temp_cpu": 0.0, "gpu_temp": 0.0, "extra_temps": {}}
_stats_lock   = threading.Lock()
_stats_jetson = None
stats_dirty   = True

def _stats_worker():
    global stats_dirty
    while True:
        try:
            if _stats_jetson is not None:
                j        = _stats_jetson
                cpu_list = j.cpu.get("cpu", [])
                vals     = [c.get("val", 0) for c in cpu_list
                            if c.get("val") is not None]
                cpu  = sum(vals) / len(vals) if vals else 0.0
                gpu  = j.gpu.get("gpu", {}).get("status", {}).get("load", 0.0)
                ram  = j.memory.get("RAM", {})
                used, total = ram.get("used", 0), ram.get("tot", 1)
                temps = j.temperature
                with _stats_lock:
                    sys_stats["cpu"]         = cpu
                    sys_stats["gpu"]         = gpu
                    sys_stats["ram"]         = (used / total) * 100 if total else 0.0
                    sys_stats["ram_total"]   = total / 1024
                    sys_stats["temp_cpu"]    = temps.get("CPU", {}).get("temp", 0.0)
                    sys_stats["gpu_temp"]    = temps.get("GPU", {}).get("temp", 0.0)
                    sys_stats["extra_temps"] = {k: v.get("temp", 0.0)
                                                for k, v in temps.items()}
                stats_dirty = True
        except Exception:
            pass
        time.sleep(1.0)

threading.Thread(target=_stats_worker, daemon=True).start()

# ─── MediaPipe ────────────────────────────────────────────────────────────────
mp_hands = mp.solutions.hands
mp_face  = mp.solutions.face_detection

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6,
)
face_detector = mp_face.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.6,
)

MP_SCALE   = 0.4
FACE_EVERY = 10

mp_input_lock   = threading.Lock()
mp_result_lock  = threading.Lock()
mp_input_frame  = None
mp_results      = {"hand": None, "face": None}
mp_face_counter = 0

def _mp_worker():
    global mp_input_frame, mp_face_counter
    while True:
        with mp_input_lock:
            frame_rgb      = mp_input_frame
            mp_input_frame = None
        if frame_rgb is None:
            time.sleep(0.002)
            continue
        hand_res = hands.process(frame_rgb)
        mp_face_counter += 1
        face_res = None
        if mp_face_counter >= FACE_EVERY:
            mp_face_counter = 0
            face_res = face_detector.process(frame_rgb)
        with mp_result_lock:
            mp_results["hand"] = hand_res
            if face_res is not None:
                mp_results["face"] = face_res

threading.Thread(target=_mp_worker, daemon=True).start()

# ─── Detección palma ──────────────────────────────────────────────────────────
FINGERTIPS = [8, 12, 16, 20]
FINGER_PIP = [6, 10, 14, 18]

def is_palm_open(landmarks):
    extended = sum(
        1 for tip, pip in zip(FINGERTIPS, FINGER_PIP)
        if landmarks[tip].y < landmarks[pip].y
    )
    return extended >= 3

# ─── Color helpers ────────────────────────────────────────────────────────────
def color_for_pct(pct):
    return (0, 220, 80) if pct < 60 else (0, 200, 255) if pct < 85 else (0, 60, 255)

def color_for_temp(t):
    return (0, 220, 80) if t < 50 else (0, 200, 255) if t < 70 else (0, 60, 255)

# ══════════════════════════════════════════════════════════════════════════════
# PANELES PRE-RENDERIZADOS — clave de la optimización
# Los paneles se renderizan una vez con PIL y se cachean como BGRA.
# Solo se re-renderizan cuando cambia su contenido (chat_dirty / stats_dirty).
# El loop principal solo hace un blit con alpha, operación rápida.
# ══════════════════════════════════════════════════════════════════════════════

_chat_panel_cache = None
_chat_panel_lock  = threading.Lock()

def _wrap_text(text, max_chars):
    words = text.split()
    lines, line = [], ""
    for word in words:
        if len(line) + len(word) + 1 <= max_chars:
            line += ("" if not line else " ") + word
        else:
            if line:
                lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines

def _render_chat_panel():
    global _chat_panel_cache
    h = SCREEN_H
    img = Image.new("RGBA", (CHAT_W, h), (10, 10, 10, 204))
    draw = ImageDraw.Draw(img)

    # Constantes escaladas
    HEADER_H = S(46)
    FOOTER_H = S(34)
    PAD      = S(10)

    draw.line([(0, 0), (0, h)], fill=(120, 200, 0, 255), width=2)
    draw.line([(0, HEADER_H), (CHAT_W, HEADER_H)],
              fill=(40, 40, 40, 255), width=1)
    draw.line([(0, h - FOOTER_H), (CHAT_W, h - FOOTER_H)],
              fill=(40, 40, 40, 255), width=1)

    f_title = get_font(18, True)
    draw.text((PAD, S(10)), "HERMES CHAT", font=f_title, fill=(120, 255, 0, 255))

    if chat_status:
        colors_map = {
            "grabando":       (255,   0,   0, 255),
            "transcribiendo": (255, 200,   0, 255),
            "analizando":     (255, 100, 200, 255),
            "pensando":       (  0, 200, 255, 255),
        }
        col = colors_map.get(chat_status, (200, 200, 200, 255))
        f_st = get_font(14, False)
        draw.text((PAD, HEADER_H + S(6)), f"● {chat_status.upper()}",
                  font=f_st, fill=col)

    with chat_lock:
        messages = list(chat_history)

    # Estimar ancho del char a la fuente escalada para wrap
    f_msg     = get_font(13, False)
    char_w    = max(6, int(7 * UI_SCALE))
    max_ch    = (CHAT_W - 2 * PAD) // char_w
    line_h    = S(19)

    y = HEADER_H + S(38)
    for msg in messages[-CHAT_MAX:]:
        is_user = msg["role"] == "user"
        prefix  = "Tú: " if is_user else "Hermes: "
        color   = (150, 220, 255, 255) if is_user else (100, 255, 150, 255)
        lines   = _wrap_text(prefix + msg["text"], max_ch)
        for line in lines[:4]:
            if y > h - FOOTER_H - S(40):
                break
            draw.text((PAD, y), line, font=f_msg, fill=color)
            y += line_h
        y += S(5)

    with _vision_context_lock:
        vc = _vision_context
    if vc:
        f_vc     = get_font(12, False)
        vc_lines = _wrap_text(f"[ojo] {vc}", max_ch)
        line_h_v = S(16)
        vy = h - FOOTER_H - len(vc_lines) * line_h_v - S(4)
        for line in vc_lines:
            draw.text((PAD - S(2), vy), line, font=f_vc, fill=(255, 140, 180, 255))
            vy += line_h_v

    rec_color = (255, 0, 0, 255) if recording else (120, 120, 120, 255)
    f_hint = get_font(12, False)
    draw.text((PAD - S(2), h - FOOTER_H + S(8)),
              "Abierta/Z=grabar  Puño/X=enviar  T=saludo",
              font=f_hint, fill=rec_color)

    arr = np.array(img)
    arr = arr[:, :, [2, 1, 0, 3]]
    with _chat_panel_lock:
        _chat_panel_cache = np.ascontiguousarray(arr)

_stats_panel_cache = None
_stats_panel_lock  = threading.Lock()
STATS_PW = S(240)
STATS_PH = S(265)

def _render_stats_panel():
    global _stats_panel_cache
    with _stats_lock:
        stats = dict(sys_stats)

    img = Image.new("RGBA", (STATS_PW, STATS_PH), (10, 10, 10, 178))
    draw = ImageDraw.Draw(img)
    draw.rectangle([(0, 0), (STATS_PW - 1, STATS_PH - 1)],
                   outline=(60, 60, 60, 255), width=1)

    PAD     = S(10)
    BAR_H   = S(11)
    BAR_W   = STATS_PW - 2 * PAD
    LINE_H  = S(20)

    def draw_bar(pct, yo):
        col = color_for_pct(pct)
        col_pil = (col[2], col[1], col[0], 255)
        bx, by, bw, bh = PAD, yo, BAR_W, BAR_H
        draw.rectangle([(bx, by), (bx + bw, by + bh)], fill=(50, 50, 50, 255))
        fill_w = int(bw * min(pct / 100, 1))
        if fill_w > 0:
            draw.rectangle([(bx, by), (bx + fill_w, by + bh)], fill=col_pil)
        return col_pil

    # Posiciones Y de las barras
    y_cpu_bar = S(30)
    y_gpu_bar = S(68)
    y_ram_bar = S(106)
    y_sep     = S(132)
    y_temps   = S(140)

    col_cpu = draw_bar(stats["cpu"], y_cpu_bar)
    col_gpu = draw_bar(stats["gpu"], y_gpu_bar)
    draw_bar(stats["ram"], y_ram_bar)
    draw.line([(S(5), y_sep), (STATS_PW - S(5), y_sep)],
              fill=(40, 40, 40, 255), width=1)

    f_lbl  = get_font(12, False)
    f_bold = get_font(12, True)

    ts_cpu = f" {stats['temp_cpu']:.0f}°C" if stats["temp_cpu"] else ""
    ts_gpu = f" {stats['gpu_temp']:.0f}°C" if stats["gpu_temp"] else ""
    draw.text((PAD, S(14)), f"CPU{ts_cpu}: {stats['cpu']:.0f}%",
              font=f_lbl, fill=col_cpu)
    draw.text((PAD, S(52)), f"GPU{ts_gpu}: {stats['gpu']:.0f}%",
              font=f_lbl, fill=col_gpu)
    draw.text((PAD, S(90)), f"RAM: {stats['ram']:.0f}%",
              font=f_lbl, fill=(200, 200, 200, 255))
    ru = (stats["ram"] / 100) * stats["ram_total"]
    draw.text((PAD, S(118)), f"  {ru:.1f}/{stats['ram_total']:.1f} GB",
              font=f_lbl, fill=(150, 150, 150, 255))
    draw.text((PAD, y_temps), "Temperaturas",
              font=f_bold, fill=(200, 200, 200, 255))

    ty = y_temps + S(18)
    for key, val in stats.get("extra_temps", {}).items():
        if ty > STATS_PH - S(16):
            break
        col = color_for_temp(val)
        draw.text((PAD, ty), f"  {key}: {val:.0f}°C",
                  font=f_lbl, fill=(col[2], col[1], col[0], 255))
        ty += S(18)

    arr = np.array(img)
    arr = arr[:, :, [2, 1, 0, 3]]
    with _stats_panel_lock:
        _stats_panel_cache = np.ascontiguousarray(arr)

# ─── Blit BGRA con alpha — versión optimizada ─────────────────────────────────
# Pre-computa el alpha float32 una sola vez por panel y lo cachea.
_blit_cache = {}  # id(src_bgra) -> (alpha_f32, src_rgb_f32)

def blit_bgra(dst, src_bgra, x, y):
    """Pega BGRA sobre BGR con alpha. Cache de float32 para evitar recálculo."""
    src_id = id(src_bgra)
    cache = _blit_cache.get(src_id)
    if cache is None:
        alpha = src_bgra[:, :, 3:4].astype(np.float32) * (1.0 / 255.0)
        src_rgb_f = src_bgra[:, :, :3].astype(np.float32) * alpha
        inv = 1.0 - alpha
        cache = (inv, src_rgb_f)
        # Limitar tamaño de cache (chat + stats = 2 paneles)
        if len(_blit_cache) > 4:
            _blit_cache.clear()
        _blit_cache[src_id] = cache
    inv, src_rgb_f = cache

    sh, sw = src_bgra.shape[:2]
    dh, dw = dst.shape[:2]
    x0 = max(0, x); y0 = max(0, y)
    x1 = min(dw, x + sw); y1 = min(dh, y + sh)
    if x0 >= x1 or y0 >= y1:
        return
    sx0 = x0 - x; sy0 = y0 - y
    sx1 = x1 - x; sy1 = y1 - y

    dst_slice = dst[y0:y1, x0:x1]
    inv_s     = inv[sy0:sy1, sx0:sx1]
    src_s     = src_rgb_f[sy0:sy1, sx0:sx1]

    # dst = dst * (1 - alpha) + src * alpha  — todo in-place
    tmp = dst_slice.astype(np.float32)
    np.multiply(tmp, inv_s, out=tmp)
    np.add(tmp, src_s, out=tmp)
    np.clip(tmp, 0, 255, out=tmp)
    dst[y0:y1, x0:x1] = tmp.astype(np.uint8)

def _invalidate_blit_cache():
    """Llamar cuando se re-renderiza un panel (cambia su id)."""
    _blit_cache.clear()

# ─── Dibujo directo con cv2 ───────────────────────────────────────────────────
def draw_hand_skeleton_fast(frame, landmarks, w, h):
    line_t = max(1, S(2))
    pt_r   = max(2, S(4))
    for conn in mp_hands.HAND_CONNECTIONS:
        a, b = conn
        pa = (int(landmarks[a].x * w), int(landmarks[a].y * h))
        pb = (int(landmarks[b].x * w), int(landmarks[b].y * h))
        cv2.line(frame, pa, pb, (180, 180, 180), line_t, cv2.LINE_AA)
    for lm in landmarks:
        cv2.circle(frame, (int(lm.x * w), int(lm.y * h)),
                   pt_r, (255, 255, 255), -1, cv2.LINE_AA)

def draw_faces_fast(frame, detections, w, h, scale=1.0):
    for det in detections:
        bbox  = det.location_data.relative_bounding_box
        x1    = max(0, int(bbox.xmin * w * scale))
        y1    = max(0, int(bbox.ymin * h * scale))
        x2    = min(w, x1 + int(bbox.width  * w * scale))
        y2    = min(h, y1 + int(bbox.height * h * scale))
        score = det.score[0]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 255), max(1, S(2)))
        cv2.putText(frame, f"Cara {score:.2f}", (x1, max(15, y1 - S(8))),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5 * UI_SCALE, (0, 220, 255),
                    max(1, S(1)), cv2.LINE_AA)

def draw_palm_state(frame, open_hand, w):
    sw    = w - CHAT_W
    color = (0, 255, 120) if open_hand else (0, 80, 255)
    label = "ABIERTA" if open_hand else "CERRADA"
    box_w = S(160); box_h = S(38)
    cv2.rectangle(frame, (sw - box_w, S(8)), (sw - S(8), S(8) + box_h),
                  (0, 0, 0), -1)
    cv2.putText(frame, label, (sw - box_w + S(5), S(8) + box_h - S(8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8 * UI_SCALE, color,
                max(1, S(2)), cv2.LINE_AA)

def draw_rec_indicator(frame):
    cy = S(95)
    if int(time.time() * 2) % 2 == 0:
        cv2.circle(frame, (S(20), cy), S(8), (0, 0, 255), -1, cv2.LINE_AA)
    cv2.putText(frame, "REC", (S(34), cy + S(5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55 * UI_SCALE, (0, 0, 255),
                max(1, S(2)), cv2.LINE_AA)

# ─── Menú ─────────────────────────────────────────────────────────────────────
MENU_ITEMS    = ["1. Guardar captura", "2. Limpiar chat", "3. Salir"]
menu_visible  = False
menu_selected = 0

def draw_menu(frame, selected):
    h, w   = frame.shape[:2]
    mw, mh = S(420), S(240)
    x1, y1 = (w - CHAT_W - mw) // 2, (h - mh) // 2
    sub = frame[y1:y1 + mh, x1:x1 + mw]
    sub[:] = (sub * 0.22 + 15 * 0.78).astype(np.uint8)
    cv2.rectangle(frame, (x1, y1), (x1 + mw, y1 + mh),
                  (0, 255, 120), max(1, S(2)))
    cv2.putText(frame, "MENU", (x1 + S(175), y1 + S(38)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85 * UI_SCALE,
                (120, 255, 0), max(1, S(2)), cv2.LINE_AA)
    for i, item in enumerate(MENU_ITEMS):
        col = (120, 255, 0) if i == selected else (200, 200, 200)
        pfx = ">" if i == selected else " "
        cv2.putText(frame, f"{pfx} {item}",
                    (x1 + S(20), y1 + S(95) + i * S(45)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65 * UI_SCALE, col,
                    max(1, S(2)) if i == selected else max(1, S(1)),
                    cv2.LINE_AA)
    cv2.putText(frame, "flechas=nav  S=ok  Q=cerrar",
                (x1 + S(90), y1 + mh - S(14)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45 * UI_SCALE,
                (130, 130, 130), max(1, S(1)), cv2.LINE_AA)

# ─── Inicialización ───────────────────────────────────────────────────────────
print("Cargando modelo de voz Piper...")
voice = PiperVoice.load(VOICE_MODEL)
print("Piper listo.")

_render_chat_panel()
_render_stats_panel()

# [PARCHE-GESTOS] ------------------------------------------------------
# El motor de gestos SONDEA el estado en vez de que el estado lo
# empuje. Cualquier camino de salida (incluida una excepcion) devuelve
# el estado a idle y la mano relaja sola, sin acordarse de nada.
# Prioridad: hablar manda sobre grabar, y grabar sobre pensar.
def _gesture_state():
    # [PARCHE-PRESENTA] presenta.is_running() cubre las pausas entre frases
    if tts_playing or presenta.is_running():
        return "speaking"
    if recording:
        return "recording"
    if chat_status in ("transcribiendo", "analizando", "pensando"):
        return "thinking"
    return "idle"

net.start()
gesture.start(_gesture_state)

# ─── Cámara ───────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
cap.set(cv2.CAP_PROP_FPS,          30)
cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

WIN = "Hermes Vision"
cv2.namedWindow(WIN, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

print("Corriendo. H=toggle mano  M=menú  Q=salir")

fps          = 0.0
frame_count  = 0
t_start      = time.time()
prev_open    = False
hand_enabled = True

# Pre-cálculo de offsets (todo en coordenadas de PANTALLA)
STATS_X = SCREEN_W - CHAT_W - STATS_PW - S(20)
STATS_Y = SCREEN_H - STATS_PH - S(10)
CHAT_X  = SCREEN_W - CHAT_W

# Resolución pre-calculada para MediaPipe (sobre el frame de cámara original)
MP_W = int(CAM_W * MP_SCALE)
MP_H = int(CAM_H * MP_SCALE)

# Factor de escala para overlays cv2 (FONT_HERSHEY)
CV_SCALE = UI_SCALE
CV_THICK = max(1, int(round(2 * UI_SCALE)))

with jtop() as jetson:
    _stats_jetson = jetson

    while True:
        ret, raw_frame = cap.read()
        if not ret:
            break

        # Flip horizontal — efecto espejo
        raw_frame = cv2.flip(raw_frame, 1)

        # Compartir referencia del frame de cámara (sin copy) — para captura de visión
        with _latest_frame_lock:
            _latest_frame = raw_frame

        # MediaPipe trabaja sobre el frame original (sin escalar a SCREEN aún)
        with mp_input_lock:
            worker_free = (mp_input_frame is None)

        if worker_free:
            small = cv2.resize(raw_frame, (MP_W, MP_H), interpolation=cv2.INTER_NEAREST)
            rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            rgb_small.flags.writeable = False
            with mp_input_lock:
                mp_input_frame = rgb_small

        with mp_result_lock:
            hand_res = mp_results["hand"]
            face_res = mp_results["face"]

        # ── Resize del frame al tamaño de pantalla — TODO lo demás trabaja aquí ──
        if SCREEN_W == CAM_W and SCREEN_H == CAM_H:
            frame = raw_frame
        else:
            frame = cv2.resize(raw_frame, (SCREEN_W, SCREEN_H),
                               interpolation=cv2.INTER_LINEAR)

        # Las coordenadas de MP están en [0,1] así que multiplicar por SCREEN_W/H ya las pone bien
        face_detected = False
        if face_res and face_res.detections:
            face_detected = True
            draw_faces_fast(frame, face_res.detections, SCREEN_W, SCREEN_H, scale=1.0)

        # Manos
        hand_open = False
        hand_fist = False

        if hand_enabled and hand_res and hand_res.multi_hand_landmarks:
            for hand_lm in hand_res.multi_hand_landmarks:
                lm = hand_lm.landmark
                draw_hand_skeleton_fast(frame, lm, SCREEN_W, SCREEN_H)
                open_now  = is_palm_open(lm)
                hand_open = hand_open or open_now
                hand_fist = hand_fist or (not open_now)
            draw_palm_state(frame, hand_open, SCREEN_W)

        # Lógica de grabación
        if hand_enabled:
            if hand_open and not prev_open and not recording and chat_status == "":
                start_recording()

            rec_elapsed = (time.time() - rec_start_time) if rec_start_time else 0

            if hand_fist and not hand_open and recording and rec_elapsed >= 1.5:
                stop_and_process()

            if recording and rec_start_time and time.time() - rec_start_time > 30:
                stop_and_process()

        if recording:
            draw_rec_indicator(frame)

        prev_open = hand_open

        # Re-render de paneles solo si están sucios
        if chat_dirty:
            _render_chat_panel()
            chat_dirty = False
            _invalidate_blit_cache()

        if stats_dirty:
            _render_stats_panel()
            stats_dirty = False
            _invalidate_blit_cache()

        # Blit de paneles cacheados
        with _stats_panel_lock:
            if _stats_panel_cache is not None:
                blit_bgra(frame, _stats_panel_cache, STATS_X, STATS_Y)
        with _chat_panel_lock:
            if _chat_panel_cache is not None:
                blit_bgra(frame, _chat_panel_cache, CHAT_X, 0)

        if menu_visible:
            draw_menu(frame, menu_selected)

        # FPS + indicadores
        frame_count += 1
        elapsed = time.time() - t_start
        if elapsed >= 1.0:
            fps, frame_count, t_start = frame_count / elapsed, 0, time.time()

        cv2.rectangle(frame, (S(8), S(8)), (S(150), S(42)), (0, 0, 0), -1)
        cv2.putText(frame, f"FPS: {fps:.1f}", (S(12), S(32)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9 * UI_SCALE,
                    (0, 255, 80), max(1, S(2)), cv2.LINE_AA)

        cara_col = (0, 220, 255) if face_detected else (80, 80, 80)
        cv2.putText(frame, "CARA: " + ("SI" if face_detected else "NO"),
                    (S(12), S(62)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6 * UI_SCALE, cara_col, max(1, S(2)), cv2.LINE_AA)

        color_h = (0, 255, 120) if hand_enabled else (0, 60, 200)
        label_h = "MANO: ON [H]" if hand_enabled else "MANO: OFF [H]"
        cv2.rectangle(frame, (S(8), S(72)), (S(190), S(100)), (0, 0, 0), -1)
        cv2.putText(frame, label_h, (S(12), S(93)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6 * UI_SCALE,
                    color_h, max(1, S(2)), cv2.LINE_AA)

        # Mostrar directo, sin resize ni canvas
        cv2.imshow(WIN, frame)

        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            if menu_visible:
                menu_visible = False
            else:
                break
        elif key == ord("h"):
            hand_enabled = not hand_enabled
            estado = "ON" if hand_enabled else "OFF"
            print(f"Detección de mano: {estado}")
            if not hand_enabled and recording:
                stop_and_process()
        elif key == ord("m"):
            menu_visible = not menu_visible
        # [PARCHE-GESTOS] grabacion por teclado
        # Activas SIEMPRE, no solo con H apagado: si la deteccion de
        # mano esta encendida pero falla, Z/X siguen siendo la salida.
        elif key == ord("z"):
            if not recording and chat_status == "":
                start_recording()
            else:
                print(">>> Z ignorado (ya grabando o procesando)")
        elif key == ord("x"):
            if recording:
                stop_and_process()
            else:
                print(">>> X ignorado (no hay grabacion activa)")
        # [PARCHE-PRESENTA] T = presentacion, T de nuevo = cortarla
        elif key == ord("t"):
            if presenta.is_running():
                presenta.stop()
                print(">>> Presentacion cortada")
            elif not recording and chat_status == "":
                presenta.start(voice, AUDIO_OUT,
                               on_line=lambda t: add_chat("assistant", t))
            else:
                print(">>> T ignorado (grabando o procesando)")
        elif key == ord("s") and menu_visible:
            if menu_selected == 0:
                fname = f"captura_{int(time.time())}.jpg"
                cv2.imwrite(fname, frame)
                print(f"Captura guardada: {fname}")
            elif menu_selected == 1:
                with chat_lock:
                    chat_history.clear()
                with _vision_context_lock:
                    _vision_context = ""
                chat_dirty = True
                print("Chat limpiado")
            elif menu_selected == 2:
                break
            menu_visible = False
        elif key == 82 and menu_visible:
            menu_selected = (menu_selected - 1) % len(MENU_ITEMS)
        elif key == 84 and menu_visible:
            menu_selected = (menu_selected + 1) % len(MENU_ITEMS)

# ─── Limpieza ─────────────────────────────────────────────────────────────────
print("Cerrando Hermes Vision...")

# [PARCHE-GESTOS] parar gestos y red
# No se rampea a neutro: hermes_net rampa a 0 y el firmware tiene
# SAFE_PCT=0. Al salir, las manos se cierran. Es la posicion segura.
try:
    # [PARCHE-PRESENTA]
    presenta.stop()
    gesture.stop()
    net.stop()
except Exception:
    pass

# 1. Matar arecord si está activo (terminate puede no bastar)
if rec_proc:
    try:
        rec_proc.terminate()
        rec_proc.wait(timeout=1.0)
    except Exception:
        try:
            rec_proc.kill()
        except Exception:
            pass

# 2. Matar cualquier arecord/aplay huérfano por si acaso
try:
    subprocess.run(["pkill", "-9", "arecord"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                   timeout=2.0)
    subprocess.run(["pkill", "-9", "aplay"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                   timeout=2.0)
except Exception:
    pass

# 3. Liberar cámara
try:
    cap.release()
except Exception:
    pass

# 4. Cerrar MediaPipe
try:
    hands.close()
except Exception:
    pass
try:
    face_detector.close()
except Exception:
    pass

# 5. Cerrar ventana OpenCV (con timeout — VNC a veces no responde)
try:
    cv2.destroyAllWindows()
    # Forzar procesado de eventos pendientes
    for _ in range(5):
        cv2.waitKey(1)
except Exception:
    pass

print("Hermes Vision cerrado.")

# 6. Salida agresiva — no esperamos a hilos daemon ni a jtop
import os as _os
_os.kill(_os.getpid(), 9) if False else _os._exit(0)
