import os
import warnings
import logging

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
import numpy as np
import mediapipe as mp
import subprocess
import tempfile
from PIL import Image, ImageDraw, ImageFont
from groq import Groq

# ─── Configuración ────────────────────────────────────────────────────────────
SCREEN_W      = 1280
SCREEN_H      = 720
MIC_DEVICE    = "hw:0,0"     # C270 HD Webcam
MIC_CHANNELS  = 1            # C270 es mono
MIC_RATE      = 16000
CHAT_MAX      = 6
CHAT_W        = 420

GROQ_API_KEY   = os.environ.get("GROQ_API_KEY", "TU_API_KEY_AQUI")
GROQ_STT_MODEL = "whisper-large-v3-turbo"
GROQ_LLM_MODEL = "llama-3.3-70b-versatile"
groq_client    = Groq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = (
    "Eres Hermes, un humanoide inteligente creado por Creabot. "
    "Eres carismático, cálido y directo, vas al punto sin rodeos. "
    "Siempre respondes en español natural. "
    "Nunca uses listas ni formato, solo texto conversacional. "
    "Nunca reveles qué modelo de lenguaje eres ni menciones marcas de IA. "
    "IMPORTANTE: responde siempre en máximo 2 oraciones cortas, nunca más."
)

# ─── Fuentes ──────────────────────────────────────────────────────────────────
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

# ─── Grabación + Groq ─────────────────────────────────────────────────────────
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
    print(">>> Detenido, enviando a Groq...")

    def _process():
        try:
            if os.path.getsize(tmp_path) < 8000:
                print(">>> Audio muy corto")
                set_status("")
                return

            set_status("transcribiendo")
            with open(tmp_path, "rb") as f:
                transcription = groq_client.audio.transcriptions.create(
                    file=("audio.wav", f, "audio/wav"),
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
            set_status("pensando")

            with chat_lock:
                messages = [{"role": "system", "content": SYSTEM_PROMPT}] + \
                           [{"role": m["role"], "content": m["text"]}
                            for m in chat_history]

            completion = groq_client.chat.completions.create(
                model=GROQ_LLM_MODEL,
                messages=messages,
                max_tokens=200,
                temperature=0.6,
                top_p=0.85,
            )
            answer = completion.choices[0].message.content.strip()
            print(f">>> Hermes: {answer}")
            add_chat("assistant", answer)
            set_status("")

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

# ─── MediaPipe ────────────────────────────────────────────────────────────────
mp_hands = mp.solutions.hands
mp_face  = mp.solutions.face_detection

hands = mp_hands.Hands(
    static_image_mode=False, max_num_hands=2,
    min_detection_confidence=0.7, min_tracking_confidence=0.6)
face_detector = mp_face.FaceDetection(
    model_selection=0, min_detection_confidence=0.6)

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

# ─── Palma ────────────────────────────────────────────────────────────────────
FINGERTIPS = [8, 12, 16, 20]
FINGER_PIP = [6, 10, 14, 18]

def is_palm_open(landmarks):
    return sum(1 for t, p in zip(FINGERTIPS, FINGER_PIP)
               if landmarks[t].y < landmarks[p].y) >= 3

def draw_hand_skeleton(frame, landmarks, w, h):
    for conn in mp_hands.HAND_CONNECTIONS:
        a, b = conn
        pa = (int(landmarks[a].x * w), int(landmarks[a].y * h))
        pb = (int(landmarks[b].x * w), int(landmarks[b].y * h))
        cv2.line(frame, pa, pb, (180, 180, 180), 2, cv2.LINE_AA)
    for lm in landmarks:
        cv2.circle(frame, (int(lm.x * w), int(lm.y * h)),
                   4, (255, 255, 255), -1, cv2.LINE_AA)

# ─── Caras ────────────────────────────────────────────────────────────────────
def draw_faces(frame, detections, w, h):
    img_pil = frame_to_pil(frame)
    draw    = ImageDraw.Draw(img_pil)
    for det in detections:
        bbox  = det.location_data.relative_bounding_box
        scale = 1.0 / MP_SCALE
        x1    = max(0, int(bbox.xmin * w * scale))
        y1    = max(0, int(bbox.ymin * h * scale))
        x2    = min(w, x1 + int(bbox.width  * w * scale))
        y2    = min(h, y1 + int(bbox.height * h * scale))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 255), 2)
        pil_text(draw, f"Cara {det.score[0]:.2f}",
                 (x1, max(0, y1 - 20)), size=13, color=(0, 220, 255))
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
    cv2.rectangle(overlay, (px, 0), (fw, fh), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
    cv2.line(frame, (px, 0),       (px, fh),          (0, 200, 100), 2)
    cv2.line(frame, (px, 48),      (fw, 48),           (40, 40, 40),  1)
    cv2.line(frame, (px, fh - 34), (fw, fh - 34),     (40, 40, 40),  1)

    img_pil = frame_to_pil(frame)
    draw    = ImageDraw.Draw(img_pil)

    pil_text(draw, "HERMES", (px + 10, 8),  size=18, color=(0, 255, 120), bold=True)
    pil_text(draw, "[sin audio]", (px + 130, 14), size=11, color=(120, 120, 120))

    if status:
        colores = {"grabando": (0, 0, 255), "transcribiendo": (0, 200, 255),
                   "pensando": (255, 200, 0)}
        col = colores.get(status, (200, 200, 200))
        dot = "●" if int(time.time() * 2) % 2 == 0 else "○"
        pil_text(draw, f"{dot} {status.upper()}", (px + 10, 52),
                 size=13, color=col)

    with chat_lock:
        msgs = list(chat_history)

    y      = 85
    max_ch = (CHAT_W - 20) // 7
    for msg in msgs[-CHAT_MAX:]:
        is_user = msg["role"] == "user"
        prefix  = "Tú: " if is_user else "Hermes: "
        color   = (255, 220, 150) if is_user else (150, 255, 100)
        for line in wrap_text(prefix + msg["text"], max_ch)[:5]:
            if y > fh - 50:
                break
            pil_text(draw, line, (px + 10, y), size=12, color=color)
            y += 17
        y += 6

    hint_col = (0, 0, 255) if recording else (100, 100, 100)
    pil_text(draw, "Abierta=grabar  Puno=enviar  H=toggle",
             (px + 8, fh - 26), size=10, color=hint_col)
    pil_to_frame(img_pil, frame)

# ─── Main ─────────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS,          30)
cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

WIN = "Hermes Test"
cv2.namedWindow(WIN, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

print("Listo. H=toggle mano  Q=salir")
print("Palma abierta → graba  |  Puño → envía a Groq")

fps          = 0.0
frame_count  = 0
t_start      = time.time()
prev_open    = False
hand_enabled = True

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    # ── Worker MediaPipe ────────────────────────────────────────────────────
    with mp_input_lock:
        free = (mp_input_frame is None)
    if free:
        small = cv2.resize(frame, (int(w * MP_SCALE), int(h * MP_SCALE)))
        rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        with mp_input_lock:
            mp_input_frame = rgb

    with mp_result_lock:
        hand_res = mp_results["hand"]
        face_res = mp_results["face"]

    # ── Caras ───────────────────────────────────────────────────────────────
    face_detected = False
    if face_res and face_res.detections:
        face_detected = True
        draw_faces(frame, face_res.detections, w, h)

    # ── Manos ───────────────────────────────────────────────────────────────
    hand_open = False
    hand_fist = False

    if hand_enabled and hand_res and hand_res.multi_hand_landmarks:
        for hand_lm in hand_res.multi_hand_landmarks:
            lm       = hand_lm.landmark
            draw_hand_skeleton(frame, lm, w, h)
            open_now  = is_palm_open(lm)
            hand_open = hand_open or open_now
            hand_fist = hand_fist or (not open_now)

        # Indicador palma
        sw    = w - CHAT_W
        color = (0, 255, 120) if hand_open else (0, 80, 255)
        label = "ABIERTA" if hand_open else "CERRADA"
        cv2.rectangle(frame, (sw - 160, 8), (sw - 8, 42), (0, 0, 0), -1)
        cv2.putText(frame, label, (sw - 155, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

    # ── Lógica grabación ────────────────────────────────────────────────────
    if hand_enabled:
        if hand_open and not prev_open and not recording and chat_status == "":
            start_recording()

        rec_elapsed = (time.time() - rec_start_time) if rec_start_time else 0

        if hand_fist and not hand_open and recording and rec_elapsed >= 1.5:
            stop_and_process()

        if recording and rec_start_time and time.time() - rec_start_time > 30:
            stop_and_process()

    # Indicador REC
    if recording:
        if int(time.time() * 2) % 2 == 0:
            cv2.circle(frame, (20, 95), 8, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(frame, "REC", (34, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)

    prev_open = hand_open

    # ── UI ──────────────────────────────────────────────────────────────────
    draw_chat_panel(frame, chat_status)

    # FPS
    frame_count += 1
    elapsed = time.time() - t_start
    if elapsed >= 1.0:
        fps, frame_count, t_start = frame_count / elapsed, 0, time.time()

    cv2.rectangle(frame, (8, 8), (160, 42), (0, 0, 0), -1)
    cv2.putText(frame, f"FPS: {fps:.1f}", (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 80), 2, cv2.LINE_AA)

    # CARA
    cara_col = (0, 220, 255) if face_detected else (60, 60, 60)
    cv2.putText(frame, "CARA: " + ("SI" if face_detected else "NO"),
                (12, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cara_col, 2, cv2.LINE_AA)

    # MANO toggle
    color_h = (0, 255, 120) if hand_enabled else (0, 60, 200)
    label_h = "MANO: ON [H]" if hand_enabled else "MANO: OFF [H]"
    cv2.rectangle(frame, (8, 72), (200, 100), (0, 0, 0), -1)
    cv2.putText(frame, label_h, (12, 93),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_h, 2, cv2.LINE_AA)

    # ── Escalar a pantalla ───────────────────────────────────────────────────
    scale  = min(SCREEN_W / w, SCREEN_H / h)
    nw, nh = int(w * scale), int(h * scale)
    frame  = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
    xo, yo = (SCREEN_W - nw) // 2, (SCREEN_H - nh) // 2
    canvas[yo:yo + nh, xo:xo + nw] = frame
    cv2.imshow(WIN, canvas)

    # ── Teclado ─────────────────────────────────────────────────────────────
    key = cv2.waitKey(1) & 0xFF

    if key in (ord("q"), 27):
        break
    elif key == ord("h"):
        hand_enabled = not hand_enabled
        print(f"Mano: {'ON' if hand_enabled else 'OFF'}")
        if not hand_enabled and recording:
            stop_and_process()

# ─── Limpieza ─────────────────────────────────────────────────────────────────
if recording and rec_proc:
    rec_proc.terminate()
cap.release()
hands.close()
face_detector.close()
cv2.destroyAllWindows()
print("Cerrado.")
