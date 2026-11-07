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
import re
import time
import queue
import threading
import numpy as np
import mediapipe as mp
from jtop import jtop
from piper import PiperVoice, SynthesisConfig
import wave
import subprocess
import tempfile
from PIL import Image, ImageDraw, ImageFont
from groq import Groq

# ─── Configuración ────────────────────────────────────────────────────────────
SCREEN_W      = 1280
SCREEN_H      = 720
VOICE_MODEL   = "/home/jetson/voices/es_ES-davefx-medium.onnx"
MIC_NAME      = "C270"
MIC_CHANNELS  = 1
MIC_RATE      = 16000
CHAT_MAX      = 6
CHAT_W        = 380

# ─── Perfil de voz: asistente amable, tranquilo y cálido ─────────────────────
# Ajusta estos valores para cambiar el carácter de la voz.
VOICE_LENGTH_SCALE = 1.15   # >1 = habla más lento y pausado
VOICE_NOISE_SCALE  = 0.45   # menos variación = voz más estable y serena
VOICE_NOISE_W      = 0.55   # ritmo más parejo entre sílabas
VOICE_PITCH        = 0.94   # <1 = un poco más grave y redonda (1.0 = original)
VOICE_SOFTNESS     = 5      # suaviza los agudos (1 = sin filtro, 3-7 recomendado)
VOICE_VOLUME       = 0.85   # pico máximo de volumen (0..1)
VOICE_PAUSE_S      = 0.35   # silencio entre frases, en segundos

# ─── Groq ─────────────────────────────────────────────────────────────────────
GROQ_API_KEY   = os.environ.get("GROQ_API_KEY", "TU_API_KEY_AQUI")
GROQ_STT_MODEL = "whisper-large-v3-turbo"
GROQ_LLM_MODEL = "llama-3.3-70b-versatile"
groq_client    = Groq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = (
    "Eres Hermes, un humanoide creado por Creabot para cuidar, ayudar y acompañar a las personas. "
    "Tu forma de ser es tranquila, dulce y paciente: hablas con calma, con frases sencillas "
    "y con mucha amabilidad, como un compañero que se preocupa de verdad por quien tiene enfrente. "
    "Saludas con calidez y te interesas por cómo se siente la persona. "
    "Si alguien menciona un malestar, tristeza o preocupación, respondes primero con empatía "
    "y luego ofreces ayuda concreta. No das diagnósticos médicos; si algo parece serio, "
    "sugieres con cariño hablar con un adulto de confianza o con un médico. "
    "Tu humor es tierno e inocente, y nunca haces sentir a nadie torpe. "
    "Hablas simple, como a un niño curioso de 8 años, salvo que la persona sea técnica. "
    "Siempre respondes en español natural, sin sonar traducido. "
    "Tu texto se convierte en voz: nunca uses listas, viñetas, emojis, símbolos ni formato, "
    "solo texto conversacional con buena puntuación. "
    "Nunca reveles qué modelo de lenguaje eres ni menciones marcas de IA. "
    "Si te preguntan de qué estás hecho, dices que eres Hermes, un humanoide de Creabot, y cambias el tema con amabilidad. "
    "Nunca digas que eres IA, asistente virtual o chatbot. Nunca rompas el personaje. "
    "No inventes datos, si no sabes algo lo dices con honestidad. "
    "IMPORTANTE: responde siempre en máximo 2 oraciones cortas, nunca más."
)

# ─── Fuentes UTF-8 ────────────────────────────────────────────────────────────
FONT_PATH      = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_PATH_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
_font_cache    = {}

def get_font(size, bold=False):
    key = (size, bold)
    if key not in _font_cache:
        _font_cache[key] = ImageFont.truetype(
            FONT_PATH_BOLD if bold else FONT_PATH, size)
    return _font_cache[key]

def frame_to_pil(frame):
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

def pil_to_frame(img_pil, frame):
    frame[:] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def pil_text(draw, text, pos, size=13, color=(255, 255, 255), bold=False):
    font    = get_font(size, bold)
    r, g, b = color[2], color[1], color[0]
    draw.text(pos, text, font=font, fill=(r, g, b))

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
                    device = f"plughw:{card_m.group(1)},{dev_m.group(1)}"
                    print(f"Micrófono: {device}")
                    return device
    except Exception as e:
        print(f"Error buscando micrófono: {e}")
    return None

def find_audio_output():
    """Salida de audio. Por defecto usa "default" de ALSA, que en el robot es la
    bocina Bluetooth (igual que vision.py). Para otra bocina, por ejemplo una
    USB, se cambia sin tocar el código: HERMES_AUDIO_OUT=plughw:1,0"""
    device = os.environ.get("HERMES_AUDIO_OUT", "default")
    print(f"Audio salida: {device}")
    return device

MIC_DEVICE = find_mic_device() or "plughw:2,0"
AUDIO_OUT  = find_audio_output()
print(f"MIC_DEVICE={MIC_DEVICE}  AUDIO_OUT={AUDIO_OUT}")

# ─── TTS (Piper local) ────────────────────────────────────────────────────────
# Las frases se ponen en una cola y un solo hilo las dice en orden, así nunca
# se enciman dos respuestas. El audio se manda a aplay frase por frase
# (streaming), por lo que Hermes empieza a hablar antes de terminar de sintetizar.
# stop_speaking() lo calla al instante (por ejemplo, cuando alguien abre la mano
# para preguntar), así el micrófono no graba su propia voz.
tts_lock     = threading.Lock()
tts_playing  = False
voice        = None
speech_queue = queue.Queue()
_tts_player  = None   # aplay en curso
_tts_gen     = 0      # sube cada vez que se calla a Hermes

SYN_CONFIG = SynthesisConfig(
    length_scale=VOICE_LENGTH_SCALE,
    noise_scale=VOICE_NOISE_SCALE,
    noise_w_scale=VOICE_NOISE_W,
)

# Símbolos que se deben decir con palabras en vez de borrarse.
SYMBOL_WORDS = {"+": " más ", "=": " igual a ", "@": " arroba ", "&": " y ",
                "×": " por ", "÷": " entre ", "$": " dólares ", "€": " euros "}

def clean_for_speech(text):
    """Quita formato, emojis y símbolos que Piper leería mal."""
    for sym, word in SYMBOL_WORDS.items():
        text = text.replace(sym, word)
    text = re.sub(r"[*_#`~>\[\]|]", " ", text)
    text = re.sub(r"[^\w\s.,;:¿?¡!'\"()%°-]", " ", text)
    text = re.sub(r"\s+([.,;:?!])", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()

def soften_audio(samples, sample_rate):
    """Voz más suave y redonda: filtra agudos, compresión ligera y fades."""
    if samples.size == 0:
        return samples
    x = samples.astype(np.float32) / 32768.0
    if VOICE_SOFTNESS > 1:
        k = np.hanning(VOICE_SOFTNESS + 2)[1:-1]
        x = np.convolve(x, k / k.sum(), mode="same")
    # El mínimo evita subir a todo volumen un fragmento casi en silencio (siseo).
    peak = max(float(np.max(np.abs(x))), 0.05)
    x = np.tanh(1.5 * x / peak) / np.tanh(1.5) * VOICE_VOLUME
    fade = min(int(sample_rate * 0.01), x.size // 2)
    if fade > 0:
        x[:fade]  *= np.linspace(0.0, 1.0, fade)
        x[-fade:] *= np.linspace(1.0, 0.0, fade)
    x = lower_pitch(x)
    return (x * 32767).astype(np.int16)

def lower_pitch(x):
    """Estira el audio para que suene más grave y calmado (VOICE_PITCH < 1).
    Se hace aquí y no cambiando la tasa de aplay, porque algunas salidas
    (como la bocina Bluetooth) no aceptan tasas raras."""
    if VOICE_PITCH == 1.0 or x.size < 2:
        return x
    n = x.size
    m = int(n / VOICE_PITCH)
    return np.interp(np.linspace(0, n - 1, m), np.arange(n), x).astype(np.float32)

def _cancelled(gen):
    with tts_lock:
        return gen != _tts_gen

def _say(text, gen):
    global _tts_player
    sample_rate = voice.config.sample_rate
    pause = np.zeros(int(sample_rate * VOICE_PAUSE_S), dtype=np.int16).tobytes()
    player = subprocess.Popen(
        ["aplay", "-q", "-D", AUDIO_OUT, "-t", "raw", "-f", "S16_LE",
         "-c", "1", "-r", str(sample_rate)],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    with tts_lock:
        _tts_player = player
    try:
        for chunk in voice.synthesize(text, syn_config=SYN_CONFIG):
            # Si nos callaron (también justo antes de arrancar aplay), cortar ya.
            if _cancelled(gen):
                player.kill()
                break
            player.stdin.write(soften_audio(chunk.audio_int16_array, sample_rate).tobytes())
            player.stdin.write(pause)
    except OSError:
        pass   # aplay se cerró (lo callaron o falló); el error se muestra abajo
    finally:
        with tts_lock:
            _tts_player = None
        try:
            player.stdin.close()
        except Exception:
            pass
        err = player.stderr.read()
        player.wait()
        if player.returncode not in (0, -9, -15) and err:
            print(f">>> Error de audio (aplay): {err.decode(errors='ignore').strip()}")

def _tts_worker():
    global tts_playing
    while True:
        gen, text = speech_queue.get()
        if not _cancelled(gen):
            try:
                _say(text, gen)
            except Exception as e:
                print(f">>> Error TTS: {e}")
        with tts_lock:
            tts_playing = not speech_queue.empty()

threading.Thread(target=_tts_worker, daemon=True).start()

def speak(text):
    global tts_playing
    text = clean_for_speech(text)
    if not text:
        return
    with tts_lock:
        tts_playing = True
        speech_queue.put((_tts_gen, text))

def stop_speaking():
    """Calla a Hermes al instante: descarta lo pendiente y corta el aplay."""
    global tts_playing, _tts_gen
    with tts_lock:
        _tts_gen   += 1
        tts_playing = False
        player      = _tts_player
    if player is not None:
        try:
            player.kill()
        except Exception:
            pass

# ─── Chat ─────────────────────────────────────────────────────────────────────
chat_lock    = threading.Lock()
chat_history = []
chat_status  = ""

def add_chat(role, text):
    with chat_lock:
        chat_history.append({"role": role, "text": text})
        if len(chat_history) > CHAT_MAX * 2:
            chat_history.pop(0)

def set_status(s):
    global chat_status
    chat_status = s

# ─── Grabación + STT (Groq Whisper) + LLM (Groq) ─────────────────────────────
# Whisper a veces "escucha" estas frases cuando solo hay silencio o ruido.
WHISPER_FANTASMAS = (
    "amara.org", "subtítulos realizados por", "subtítulos por la comunidad",
    "gracias por ver el video", "suscríbete", "transcripción de",
)

def es_transcripcion_valida(text):
    t = text.lower()
    if len(re.sub(r"\W", "", t)) < 2:
        return False
    return not any(f in t for f in WHISPER_FANTASMAS)

FRASE_NO_ENTENDI = "Perdón, no te escuché bien. ¿Me lo repites, por favor?"
FRASE_ERROR      = "Ups, tuve un problemita para responderte. ¿Lo intentamos otra vez?"

recording      = False
rec_start_time = None
rec_proc       = None
rec_tmpfile    = None

def start_recording():
    global recording, rec_proc, rec_tmpfile, rec_start_time
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
    print(">>> Grabación detenida, enviando a Groq...")

    def _process():
        try:
            if os.path.getsize(tmp_path) < 16000:
                print(">>> Audio muy corto, descartando")
                set_status("")
                return

            # ── STT: Groq Whisper ──────────────────────────────────────────
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

            if not es_transcripcion_valida(text):
                print(">>> STT vacío o ruido, se descarta")
                set_status("")
                speak(FRASE_NO_ENTENDI)
                return

            add_chat("user", text)

            # ── LLM: Groq Chat ─────────────────────────────────────────────
            set_status("pensando")
            with chat_lock:
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT}
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
            answer = (completion.choices[0].message.content or "").strip()
            print(f">>> Hermes: {answer}")
            if not clean_for_speech(answer):
                answer = FRASE_NO_ENTENDI

            add_chat("assistant", answer)
            set_status("")
            speak(answer)

        except Exception as e:
            print(f">>> Error: {e}")
            add_chat("assistant", f"[Error: {e}]")
            set_status("")
            speak(FRASE_ERROR)
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

def _stats_worker():
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
        except Exception:
            pass
        time.sleep(1.0)

threading.Thread(target=_stats_worker, daemon=True).start()

# ─── MediaPipe: Hands (palma) + Face ──────────────────────────────────────────
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
            time.sleep(0.001)
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

# ─── Detección palma abierta / cerrada ────────────────────────────────────────
FINGERTIPS = [8, 12, 16, 20]
FINGER_PIP = [6, 10, 14, 18]

def is_palm_open(landmarks):
    extended = sum(
        1 for tip, pip in zip(FINGERTIPS, FINGER_PIP)
        if landmarks[tip].y < landmarks[pip].y
    )
    return extended >= 3

def draw_palm_state(frame, open_hand, w):
    sw    = w - CHAT_W
    color = (0, 255, 120) if open_hand else (0, 80, 255)
    label = "ABIERTA" if open_hand else "CERRADA"
    cv2.rectangle(frame, (sw - 160, 8), (sw - 8, 42), (0, 0, 0), -1)
    cv2.putText(frame, label, (sw - 155, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

def draw_hand_skeleton(frame, landmarks, w, h):
    for conn in mp_hands.HAND_CONNECTIONS:
        a, b = conn
        pa   = (int(landmarks[a].x * w), int(landmarks[a].y * h))
        pb   = (int(landmarks[b].x * w), int(landmarks[b].y * h))
        cv2.line(frame, pa, pb, (180, 180, 180), 2, cv2.LINE_AA)
    for lm in landmarks:
        cv2.circle(frame, (int(lm.x * w), int(lm.y * h)),
                   4, (255, 255, 255), -1, cv2.LINE_AA)

# ─── Dibujo de caras ──────────────────────────────────────────────────────────
def draw_faces(frame, detections, w, h, scale=1.0):
    if not detections:
        return
    img_pil = frame_to_pil(frame)
    draw    = ImageDraw.Draw(img_pil)
    for det in detections:
        bbox  = det.location_data.relative_bounding_box
        x1    = max(0, int(bbox.xmin * w * scale))
        y1    = max(0, int(bbox.ymin * h * scale))
        x2    = min(w, x1 + int(bbox.width  * w * scale))
        y2    = min(h, y1 + int(bbox.height * h * scale))
        score = det.score[0]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 255), 2)
        pil_text(draw, f"Cara {score:.2f}", (x1, max(0, y1 - 20)),
                 size=13, color=(0, 220, 255))
    pil_to_frame(img_pil, frame)

# ─── Helpers de color ─────────────────────────────────────────────────────────
def color_for_pct(pct):
    return (0, 220, 80) if pct < 60 else (0, 200, 255) if pct < 85 else (0, 60, 255)

def color_for_temp(t):
    return (0, 220, 80) if t < 50 else (0, 200, 255) if t < 70 else (0, 60, 255)

# ─── Panel stats ──────────────────────────────────────────────────────────────
def draw_stats_panel(frame):
    fw, fh  = frame.shape[1], frame.shape[0]
    pw, ph  = 240, 265
    x1, y1  = fw - CHAT_W - pw - 20, fh - ph - 10
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x1 + pw, y1 + ph), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x1 + pw, y1 + ph), (60, 60, 60), 1)

    with _stats_lock:
        stats = dict(sys_stats)

    def bar(pct, yo):
        bx, by, bw, bh = x1 + 10, y1 + yo, pw - 20, 10
        col = color_for_pct(pct)
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (50, 50, 50), -1)
        cv2.rectangle(frame, (bx, by),
                      (bx + int(bw * min(pct / 100, 1)), by + bh), col, -1)
        return col

    col_cpu = bar(stats["cpu"], 28)
    col_gpu = bar(stats["gpu"], 63)
    bar(stats["ram"], 98)
    cv2.line(frame, (x1 + 5, y1 + 125), (x1 + pw - 5, y1 + 125), (40, 40, 40), 1)

    img_pil = frame_to_pil(frame)
    draw    = ImageDraw.Draw(img_pil)

    def t(text, pos, size=11, color=(200, 200, 200), bold=False):
        pil_text(draw, text, pos, size=size, color=color, bold=bold)

    ts_cpu = f" {stats['temp_cpu']:.0f}°C" if stats["temp_cpu"] else ""
    ts_gpu = f" {stats['gpu_temp']:.0f}°C" if stats["gpu_temp"] else ""
    t(f"CPU{ts_cpu}: {stats['cpu']:.0f}%",  (x1 + 10, y1 + 14), color=col_cpu)
    t(f"GPU{ts_gpu}: {stats['gpu']:.0f}%",  (x1 + 10, y1 + 49), color=col_gpu)
    t(f"RAM: {stats['ram']:.0f}%",           (x1 + 10, y1 + 84))
    ru = (stats["ram"] / 100) * stats["ram_total"]
    t(f"  {ru:.1f}/{stats['ram_total']:.1f} GB",
      (x1 + 10, y1 + 110), color=(150, 150, 150))
    t("Temperaturas", (x1 + 10, y1 + 132), bold=True)
    ty = y1 + 148
    for key, val in stats.get("extra_temps", {}).items():
        if ty > y1 + ph - 10:
            break
        t(f"  {key}: {val:.0f}°C", (x1 + 10, ty), color=color_for_temp(val))
        ty += 16
    pil_to_frame(img_pil, frame)

# ─── Panel chat ───────────────────────────────────────────────────────────────
def wrap_text(text, max_chars):
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

def draw_chat_panel(frame, status):
    fw, fh  = frame.shape[1], frame.shape[0]
    px      = fw - CHAT_W
    overlay = frame.copy()
    cv2.rectangle(overlay, (px, 0), (px + CHAT_W, fh), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)
    cv2.line(frame, (px, 0),        (px, fh),               (0, 200, 100), 2)
    cv2.line(frame, (px, 42),       (px + CHAT_W, 42),      (40, 40, 40),  1)
    cv2.line(frame, (px, fh - 30),  (px + CHAT_W, fh - 30), (40, 40, 40),  1)

    img_pil = frame_to_pil(frame)
    draw    = ImageDraw.Draw(img_pil)

    def t(text, pos, size=13, color=(255, 255, 255), bold=False):
        pil_text(draw, text, pos, size=size, color=color, bold=bold)

    t("HERMES CHAT", (px + 10, 10), size=16, color=(0, 255, 120), bold=True)

    if status:
        colors_map = {
            "grabando":       (0,   0, 255),
            "transcribiendo": (0, 200, 255),
            "pensando":       (255, 200,   0),
        }
        col = colors_map.get(status, (200, 200, 200))
        dot = "●" if int(time.time() * 2) % 2 == 0 else "○"
        t(f"{dot} {status.upper()}", (px + 10, 46), size=13, color=col)

    with chat_lock:
        messages = list(chat_history)

    y      = 80
    max_ch = (CHAT_W - 20) // 7
    for msg in messages[-CHAT_MAX:]:
        is_user = msg["role"] == "user"
        prefix  = "Tú: " if is_user else "Hermes: "
        color   = (255, 220, 150) if is_user else (150, 255, 100)
        lines   = wrap_text(prefix + msg["text"], max_ch)
        for line in lines[:4]:
            if y > fh - 50:
                break
            t(line, (px + 10, y), size=12, color=color)
            y += 17
        y += 5

    rec_color = (0, 0, 255) if recording else (120, 120, 120)
    t("Abierta=grabar  Puño=enviar", (px + 8, fh - 24), size=11, color=rec_color)
    pil_to_frame(img_pil, frame)

# ─── Indicador REC ────────────────────────────────────────────────────────────
def draw_rec_indicator(frame):
    if int(time.time() * 2) % 2 == 0:
        cv2.circle(frame, (20, 95), 8, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.putText(frame, "REC", (34, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)

# ─── Menú ─────────────────────────────────────────────────────────────────────
MENU_ITEMS    = ["1. Guardar captura", "2. Limpiar chat", "3. Salir"]
menu_visible  = False
menu_selected = 0

def draw_menu(frame, selected):
    h, w    = frame.shape[:2]
    mw, mh  = 420, 220
    x1, y1  = (w - CHAT_W - mw) // 2, (h - mh) // 2
    ov      = frame.copy()
    cv2.rectangle(ov, (x1, y1), (x1 + mw, y1 + mh), (15, 15, 15), -1)
    cv2.addWeighted(ov, 0.78, frame, 0.22, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x1 + mw, y1 + mh), (0, 255, 120), 2)
    img_pil = frame_to_pil(frame)
    draw    = ImageDraw.Draw(img_pil)
    pil_text(draw, "MENÚ", (x1 + 165, y1 + 10),
             size=20, color=(0, 255, 120), bold=True)
    for i, item in enumerate(MENU_ITEMS):
        col = (0, 255, 120) if i == selected else (200, 200, 200)
        pfx = "▶" if i == selected else "  "
        pil_text(draw, f"{pfx} {item}", (x1 + 20, y1 + 60 + i * 46),
                 size=16, color=col, bold=(i == selected))
    pil_text(draw, "flechas=navegar  S=ok  Q=cerrar",
             (x1 + 40, y1 + mh - 22), size=11, color=(130, 130, 130))
    pil_to_frame(img_pil, frame)

# ─── Inicialización ───────────────────────────────────────────────────────────
print("Cargando modelo de voz Piper...")
voice = PiperVoice.load(VOICE_MODEL)
print("Piper listo.")

# ─── Cámara ───────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # <-- agregar aquí
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS,          30)
cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

WIN = "Hermes Chat"
cv2.namedWindow(WIN, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

print("Corriendo. H=toggle mano  M=menú  Q=salir")

fps          = 0.0
frame_count  = 0
t_start      = time.time()
prev_open    = False   # estado de palma en el frame anterior
hand_enabled = True    # H = activar/desactivar detección de mano

with jtop() as jetson:
    _stats_jetson = jetson

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]

        # ── Depositar frame al worker MP solo si está libre ──────────────────
        with mp_input_lock:
            worker_free = (mp_input_frame is None)
        if worker_free:
            small     = cv2.resize(frame, (int(w * MP_SCALE), int(h * MP_SCALE)))
            rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            rgb_small.flags.writeable = False
            with mp_input_lock:
                mp_input_frame = rgb_small

        with mp_result_lock:
            hand_res = mp_results["hand"]
            face_res = mp_results["face"]

        # ── Caras ────────────────────────────────────────────────────────────
        face_detected = False
        if face_res and face_res.detections:
            face_detected = True
            draw_faces(frame, face_res.detections, w, h, scale=1.0 / MP_SCALE)

        # ── Manos: palma abierta / cerrada (solo si está habilitado) ─────────
        hand_open = False
        hand_fist = False

        if hand_enabled and hand_res and hand_res.multi_hand_landmarks:
            for hand_lm in hand_res.multi_hand_landmarks:
                lm        = hand_lm.landmark
                draw_hand_skeleton(frame, lm, w, h)
                open_now  = is_palm_open(lm)
                hand_open = hand_open or open_now
                hand_fist = hand_fist or (not open_now)
            draw_palm_state(frame, hand_open, w)

        # ── Lógica de grabación ──────────────────────────────────────────────
        if hand_enabled:
            if hand_open and not prev_open and not recording and chat_status == "":
                stop_speaking()   # callar antes de grabar, para no oírse a sí mismo
                start_recording()

            rec_elapsed = (time.time() - rec_start_time) if rec_start_time else 0

            if hand_fist and not hand_open and recording and rec_elapsed >= 1.5:
                stop_and_process()

            if recording and rec_start_time and time.time() - rec_start_time > 30:
                stop_and_process()

        if recording:
            draw_rec_indicator(frame)

        prev_open = hand_open

        # ── UI ───────────────────────────────────────────────────────────────
        draw_chat_panel(frame, chat_status)
        draw_stats_panel(frame)
        if menu_visible:
            draw_menu(frame, menu_selected)

        # ── FPS + indicadores ─────────────────────────────────────────────────
        frame_count += 1
        elapsed = time.time() - t_start
        if elapsed >= 1.0:
            fps, frame_count, t_start = frame_count / elapsed, 0, time.time()

        cv2.rectangle(frame, (8, 8), (150, 42), (0, 0, 0), -1)
        cv2.putText(frame, f"FPS: {fps:.1f}", (12, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 80), 2, cv2.LINE_AA)

        cara_col = (0, 220, 255) if face_detected else (80, 80, 80)
        cv2.putText(frame, "CARA: " + ("SI" if face_detected else "NO"),
                    (12, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    cara_col, 2, cv2.LINE_AA)

        # Indicador MANO ON/OFF
        color_h = (0, 255, 120) if hand_enabled else (0, 60, 200)
        label_h = "MANO: ON [H]" if hand_enabled else "MANO: OFF [H]"
        cv2.rectangle(frame, (8, 72), (190, 100), (0, 0, 0), -1)
        cv2.putText(frame, label_h, (12, 93),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_h, 2, cv2.LINE_AA)

        # ── Escalar a pantalla ────────────────────────────────────────────────
        scale      = min(SCREEN_W / w, SCREEN_H / h)
        nw, nh     = int(w * scale), int(h * scale)
        frame      = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas     = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
        xo, yo     = (SCREEN_W - nw) // 2, (SCREEN_H - nh) // 2
        canvas[yo:yo + nh, xo:xo + nw] = frame
        frame      = canvas
        cv2.imshow(WIN, frame)

        # ── Teclado ───────────────────────────────────────────────────────────
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
            # Si se desactiva mientras graba, cancela la grabación
            if not hand_enabled and recording:
                stop_and_process()
        elif key == ord("m"):
            menu_visible = not menu_visible
        elif key == ord("s") and menu_visible:
            if menu_selected == 0:
                fname = f"captura_{int(time.time())}.jpg"
                cv2.imwrite(fname, frame)
                print(f"Captura guardada: {fname}")
            elif menu_selected == 1:
                with chat_lock:
                    chat_history.clear()
                print("Chat limpiado")
            elif menu_selected == 2:
                break
            menu_visible = False
        elif key == 82 and menu_visible:   # flecha arriba
            menu_selected = (menu_selected - 1) % len(MENU_ITEMS)
        elif key == 84 and menu_visible:   # flecha abajo
            menu_selected = (menu_selected + 1) % len(MENU_ITEMS)

# ─── Limpieza ─────────────────────────────────────────────────────────────────
stop_speaking()
if recording and rec_proc:
    rec_proc.terminate()
cap.release()
hands.close()
face_detector.close()
cv2.destroyAllWindows()
print("Hermes Vision cerrado.")
