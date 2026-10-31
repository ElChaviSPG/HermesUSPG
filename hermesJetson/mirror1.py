import os, warnings, logging

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"]     = "3"
os.environ["GRPC_VERBOSITY"]       = "ERROR"
os.environ["PYTHONWARNINGS"]       = "ignore"
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)
logging.getLogger("mediapipe").setLevel(logging.ERROR)

import cv2, time, threading, math, glob
import numpy as np
import mediapipe as mp
import serial
from jtop import jtop

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════════════════
CAM_W       = 1024
CAM_H       = 768
SERIAL_BAUD = 115200

MP_W = 320
MP_H = 180

# ══════════════════════════════════════════════════════════════════════════════
# COLORES (BGR)
# ══════════════════════════════════════════════════════════════════════════════
C_LEFT   = (100, 220, 255)
C_RIGHT  = (255, 160,  80)
C_WHITE  = (255, 255, 255)
C_GRAY   = (150, 150, 150)
C_GREEN  = (  0, 220,  80)
C_RED    = (  0,  60, 255)
C_YELLOW = (  0, 200, 255)
C_ORANGE = (  0, 140, 255)

FINGER_COLORS = {
    "pulgar":  ( 50, 100, 255),
    "indice":  (100, 255,  50),
    "medio":   (255, 150,  50),
    "anular":  (200,  50, 255),
    "menique": ( 50, 220, 255),
}
FINGER_NAMES = ["pulgar", "indice", "medio", "anular", "menique"]

# ══════════════════════════════════════════════════════════════════════════════
# SERIAL
# ══════════════════════════════════════════════════════════════════════════════
arduinos    = {"LEFT": None, "RIGHT": None}
serial_ok   = {"LEFT": False, "RIGHT": False}
esp32_codos = None
codos_ok    = False
serial_lock = threading.Lock()

_serial_queue  = {"LEFT": None, "RIGHT": None, "CODOS": None}
_serial_q_lock = threading.Lock()

# ══════════════════════════════════════════════════════════════════════════════
# PAUSA SERIAL (tecla H)
# ══════════════════════════════════════════════════════════════════════════════
serial_paused = False

PAUSE_STEPS = 20
PAUSE_DELAY = 0.04   # ~25 Hz → ramp de ~0.8 s

def _send_direct(ser, msg):
    """Envío directo en hilos de control, sin pasar por la cola."""
    try:
        with serial_lock:
            ser.write(msg)
    except Exception:
        pass

def _ramp_to_zero():
    """Baja codos y dedos gradualmente a 0 y luego pausa el envío."""
    start_izq = codo_pct["LEFT"]
    start_der = codo_pct["RIGHT"]
    start_fingers = {
        side: {n: finger_pct[side][n] for n in FINGER_NAMES}
        for side in ("LEFT", "RIGHT")
    }
    for step in range(PAUSE_STEPS + 1):
        t = step / PAUSE_STEPS  # 0.0 → 1.0
        # Codos
        if esp32_codos and codos_ok:
            vi = int(start_izq * (1.0 - t))
            vd = int(start_der * (1.0 - t))
            _send_direct(esp32_codos, f"{vi},{vd}\n".encode())
        # Dedos
        for side in ("LEFT", "RIGHT"):
            if serial_ok.get(side) and arduinos[side]:
                vals = ",".join(
                    str(int(start_fingers[side][n] * (1.0 - t)))
                    for n in FINGER_NAMES
                )
                _send_direct(arduinos[side], (vals + "\n").encode())
        time.sleep(PAUSE_DELAY)
    print("Serial PAUSADO — Hermes en posición 0.")

def _ramp_from_zero():
    """Resetea EMA a 0 para que el arranque sea orgánico desde cero."""
    for side in ("LEFT", "RIGHT"):
        codo_ema[side]  = 0.0
        codo_pct[side]  = 0
        for n in FINGER_NAMES:
            finger_ema[side][n] = 0.0
            finger_pct[side][n] = 0
    print("Serial REANUDADO — EMA reseteado a 0.")

# ══════════════════════════════════════════════════════════════════════════════
# WORKER SERIAL
# ══════════════════════════════════════════════════════════════════════════════
def _serial_worker():
    """Hilo dedicado al envío serial — nunca bloquea el loop de video."""
    while True:
        with _serial_q_lock:
            msgs = dict(_serial_queue)
            _serial_queue["LEFT"]  = None
            _serial_queue["RIGHT"] = None
            _serial_queue["CODOS"] = None

        for key, msg in msgs.items():
            if msg is None:
                continue
            try:
                if key in ("LEFT", "RIGHT"):
                    if serial_ok.get(key) and arduinos[key]:
                        with serial_lock:
                            arduinos[key].write(msg)
                elif key == "CODOS":
                    if codos_ok and esp32_codos:
                        with serial_lock:
                            esp32_codos.write(msg)
            except Exception as e:
                print(f"Error serial {key}: {e}")
                if key in ("LEFT", "RIGHT"):
                    serial_ok[key] = False
                else:
                    globals()["codos_ok"] = False
        time.sleep(0.02)

threading.Thread(target=_serial_worker, daemon=True).start()

def queue_fingers(side, pcts):
    if serial_paused:
        return
    msg = (",".join(str(int(p)) for p in pcts) + "\n").encode()
    with _serial_q_lock:
        _serial_queue[side] = msg

def queue_codos(pct_izq, pct_der):
    if serial_paused:
        return
    msg = f"{int(pct_izq)},{int(pct_der)}\n".encode()
    with _serial_q_lock:
        _serial_queue["CODOS"] = msg

# ══════════════════════════════════════════════════════════════════════════════
# DESCUBRIMIENTO DE PUERTOS
# ══════════════════════════════════════════════════════════════════════════════
def _query_who(port, baud=SERIAL_BAUD, timeout=3.0):
    try:
        ser = serial.Serial(port, baud, timeout=0.5)
        time.sleep(2)
        ser.reset_input_buffer()
        ser.write(b"#WHO\n")
        deadline = time.time() + timeout
        while time.time() < deadline:
            line = ser.readline().decode(errors="ignore").strip()
            if line in ("HERMES_LEFT", "HERMES_RIGHT", "HERMES_CODOS"):
                return line, ser
            if line:
                print(f"  {port} descartado: {line!r}")
        ser.close()
    except Exception as e:
        print(f"  {port} error: {e}")
    return None, None

def discover_arduinos():
    global esp32_codos, codos_ok
    ports = sorted(glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*"))
    if not ports:
        print("Serial: sin puertos, corriendo sin hardware")
        return
    print(f"Serial: escaneando {ports} en paralelo...")
    results = {}
    results_lock = threading.Lock()

    def probe(port):
        who, ser = _query_who(port)
        with results_lock:
            results[port] = (who, ser)

    threads = [threading.Thread(target=probe, args=(p,), daemon=True) for p in ports]
    for t in threads: t.start()
    for t in threads: t.join(timeout=6.0)

    for port, (who, ser) in results.items():
        if who in ("HERMES_LEFT", "HERMES_RIGHT"):
            side = who.split("_")[1]
            arduinos[side]  = ser
            serial_ok[side] = True
            print(f"  {side} -> {port}")
        elif who == "HERMES_CODOS":
            esp32_codos = ser
            codos_ok    = True
            print(f"  CODOS -> {port}")
        else:
            if ser:
                try: ser.close()
                except: pass

    for side in ("LEFT", "RIGHT"):
        print(f"  {side}: {'OK' if serial_ok[side] else 'no conectado'}")
    print(f"  CODOS: {'OK' if codos_ok else 'no conectado'}")

# ══════════════════════════════════════════════════════════════════════════════
# MEDIAPIPE — DOS WORKERS PARALELOS
# ══════════════════════════════════════════════════════════════════════════════
mp_hands_mod = mp.solutions.hands
mp_pose_mod  = mp.solutions.pose

pose_detector = mp_pose_mod.Pose(
    static_image_mode=False,
    model_complexity=0,
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5,
)
hands_detector = mp_hands_mod.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6,
)

_PL = mp_pose_mod.PoseLandmark
ELBOW_PTS = {
    "LEFT":  (_PL.LEFT_SHOULDER, _PL.LEFT_ELBOW,  _PL.LEFT_WRIST),
    "RIGHT": (_PL.RIGHT_SHOULDER,_PL.RIGHT_ELBOW, _PL.RIGHT_WRIST),
}

_pose_input_lock  = threading.Lock()
_pose_input       = None
_pose_result_lock = threading.Lock()
_pose_result      = None

_hand_input_lock  = threading.Lock()
_hand_input       = None
_hand_result_lock = threading.Lock()
_hand_result      = None

def _pose_worker():
    global _pose_input, _pose_result
    while True:
        with _pose_input_lock:
            frame = _pose_input
            _pose_input = None
        if frame is None:
            time.sleep(0.001)
            continue
        res = pose_detector.process(frame)
        with _pose_result_lock:
            _pose_result = res

def _hand_worker():
    global _hand_input, _hand_result
    while True:
        with _hand_input_lock:
            frame = _hand_input
            _hand_input = None
        if frame is None:
            time.sleep(0.001)
            continue
        res = hands_detector.process(frame)
        with _hand_result_lock:
            _hand_result = res

threading.Thread(target=_pose_worker, daemon=True).start()
threading.Thread(target=_hand_worker, daemon=True).start()

# ══════════════════════════════════════════════════════════════════════════════
# EMA + DEADZONE
# ══════════════════════════════════════════════════════════════════════════════
EMA_ALPHA = 0.25
DEAD_LOW  = 6
DEAD_HIGH = 94

def _ema(prev, raw):
    s = prev + EMA_ALPHA * (raw - prev)
    return 0.0 if s < DEAD_LOW else (100.0 if s > DEAD_HIGH else s)

# ══════════════════════════════════════════════════════════════════════════════
# ÁNGULOS
# ══════════════════════════════════════════════════════════════════════════════
def angle_between(a, b, c):
    v1 = np.array([a.x-b.x, a.y-b.y, a.z-b.z])
    v2 = np.array([c.x-b.x, c.y-b.y, c.z-b.z])
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 180.0
    return math.degrees(math.acos(np.clip(np.dot(v1,v2)/(n1*n2), -1.0, 1.0)))

# ══════════════════════════════════════════════════════════════════════════════
# CODOS
# ══════════════════════════════════════════════════════════════════════════════
calib_codo = {
    "LEFT":  {"low": 165.0, "high": 85.0},
    "RIGHT": {"low": 165.0, "high": 85.0},
}
_cal_codo      = {"active": False, "side": None, "endpoint": None, "samples": []}
CALIB_CODO_N   = 25
codo_ema       = {"LEFT": 0.0, "RIGHT": 0.0}
codo_pct       = {"LEFT": 0,   "RIGHT": 0}

def codo_angle_to_pct(side, angle):
    low, high = calib_codo[side]["low"], calib_codo[side]["high"]
    if abs(low - high) < 5: return 0.0
    return float(np.clip((low - angle) / (low - high) * 100.0, 0.0, 100.0))

def update_codo(side, angle):
    raw = codo_angle_to_pct(side, angle)
    codo_ema[side] = _ema(codo_ema[side], raw)
    codo_pct[side] = int(codo_ema[side])

def start_calib_codo(side, endpoint):
    _cal_codo.update({"active": True, "side": side, "endpoint": endpoint, "samples": []})
    print(f"Calib codo {'IZQ' if side=='LEFT' else 'DER'} {'ESTIRADO' if endpoint=='low' else 'DOBLADO'}...")

def feed_calib_codo(angle, side):
    if not _cal_codo["active"] or _cal_codo["side"] != side: return
    _cal_codo["samples"].append(angle)
    if len(_cal_codo["samples"]) >= CALIB_CODO_N:
        avg = sum(_cal_codo["samples"]) / len(_cal_codo["samples"])
        calib_codo[side][_cal_codo["endpoint"]] = avg
        print(f"OK codo {'IZQ' if side=='LEFT' else 'DER'} {'EST' if _cal_codo['endpoint']=='low' else 'DOB'}: {avg:.1f}")
        _cal_codo["active"] = False
        _cal_codo["samples"] = []

# ══════════════════════════════════════════════════════════════════════════════
# DEDOS
# ══════════════════════════════════════════════════════════════════════════════
FINGER_ANGLE_PTS = {
    "pulgar":  (1,  2,  3,  4),
    "indice":  (5,  6,  7,  8),
    "medio":   (9,  10, 11, 12),
    "anular":  (13, 14, 15, 16),
    "menique": (17, 18, 19, 20),
}
calib_low  = {"pulgar":70.0,  "indice":75.0, "medio":75.0, "anular":75.0, "menique":75.0}
calib_high = {"pulgar":155.0, "indice":165.0,"medio":165.0,"anular":165.0,"menique":165.0}
calib_smp_low  = {n: [] for n in FINGER_NAMES}
calib_smp_high = {n: [] for n in FINGER_NAMES}
calib_mode     = None
calib_frames   = 0
CALIB_HAND_N   = 20

finger_ema = {"LEFT":{n:0.0 for n in FINGER_NAMES}, "RIGHT":{n:0.0 for n in FINGER_NAMES}}
finger_pct = {"LEFT":{n:0   for n in FINGER_NAMES}, "RIGHT":{n:0   for n in FINGER_NAMES}}

def finger_raw_angle(lm, name):
    mcp, pip, dip, tip = FINGER_ANGLE_PTS[name]
    return angle_between(lm[mcp], lm[pip], lm[dip]) * 0.65 + \
           angle_between(lm[pip], lm[dip], lm[tip]) * 0.35

def finger_openness(lm, name):
    ang = finger_raw_angle(lm, name)
    low, high = calib_low[name], calib_high[name]
    if high <= low: return 0.0
    return float(np.clip((ang - low) / (high - low) * 100.0, 0.0, 100.0))

def update_fingers(side, lm):
    for name in FINGER_NAMES:
        raw = finger_openness(lm, name)
        finger_ema[side][name] = _ema(finger_ema[side][name], raw)
        finger_pct[side][name] = int(finger_ema[side][name])

def collect_calib(lm, mode):
    target = calib_smp_low if mode == "low" else calib_smp_high
    for name in FINGER_NAMES:
        target[name].append(finger_raw_angle(lm, name))

def apply_calib(mode):
    src = calib_smp_low if mode == "low" else calib_smp_high
    dst = calib_low     if mode == "low" else calib_high
    for name in FINGER_NAMES:
        if src[name]:
            dst[name] = sum(src[name]) / len(src[name])
            src[name].clear()
    print(f"Calib {'CERRADO' if mode=='low' else 'ABIERTO'}: { {n:f'{dst[n]:.1f}' for n in FINGER_NAMES} }")

# ══════════════════════════════════════════════════════════════════════════════
# DIBUJO
# ══════════════════════════════════════════════════════════════════════════════
FINGERTIPS = {"pulgar":4, "indice":8, "medio":12, "anular":16, "menique":20}
FINGER_PIP = {"pulgar":3, "indice":6, "medio":10, "anular":14, "menique":18}

LEFT_JOINTS  = {_PL.LEFT_SHOULDER,  _PL.LEFT_HIP,  _PL.LEFT_KNEE,  _PL.LEFT_ANKLE}
RIGHT_JOINTS = {_PL.RIGHT_SHOULDER, _PL.RIGHT_HIP, _PL.RIGHT_KNEE, _PL.RIGHT_ANKLE}

POSE_BONES = [
    (_PL.LEFT_SHOULDER,  _PL.RIGHT_SHOULDER, C_WHITE),
    (_PL.LEFT_SHOULDER,  _PL.LEFT_HIP,       C_LEFT),
    (_PL.RIGHT_SHOULDER, _PL.RIGHT_HIP,      C_RIGHT),
    (_PL.LEFT_HIP,       _PL.RIGHT_HIP,      C_WHITE),
    (_PL.LEFT_SHOULDER,  _PL.LEFT_ELBOW,     C_LEFT),
    (_PL.LEFT_ELBOW,     _PL.LEFT_WRIST,     C_LEFT),
    (_PL.RIGHT_SHOULDER, _PL.RIGHT_ELBOW,    C_RIGHT),
    (_PL.RIGHT_ELBOW,    _PL.RIGHT_WRIST,    C_RIGHT),
    (_PL.LEFT_HIP,       _PL.LEFT_KNEE,      C_LEFT),
    (_PL.LEFT_KNEE,      _PL.LEFT_ANKLE,     C_LEFT),
    (_PL.RIGHT_HIP,      _PL.RIGHT_KNEE,     C_RIGHT),
    (_PL.RIGHT_KNEE,     _PL.RIGHT_ANKLE,    C_RIGHT),
]

def draw_pose(frame, lm, w, h):
    def px(i): return int(lm[i].x*w), int(lm[i].y*h)
    def vis(i): return lm[i].visibility

    for a, b, col in POSE_BONES:
        if min(vis(a), vis(b)) < 0.5: continue
        cv2.line(frame, px(a), px(b), col, 2, cv2.LINE_AA)

    for idx in LEFT_JOINTS | RIGHT_JOINTS:
        if vis(idx) < 0.5: continue
        col = C_LEFT if idx in LEFT_JOINTS else C_RIGHT
        cv2.circle(frame, px(idx), 6, col,    -1, cv2.LINE_AA)
        cv2.circle(frame, px(idx), 6, C_WHITE,  1, cv2.LINE_AA)

    for side, el_idx in (("LEFT", _PL.LEFT_ELBOW), ("RIGHT", _PL.RIGHT_ELBOW)):
        if vis(el_idx) < 0.5: continue
        pct  = codo_pct[side]
        col  = C_RED if pct < 30 else C_YELLOW if pct < 70 else C_GREEN
        base = C_LEFT if side == "LEFT" else C_RIGHT
        ep   = px(el_idx)
        cv2.circle(frame, ep, 12, col,  -1, cv2.LINE_AA)
        cv2.circle(frame, ep, 12, base,  2, cv2.LINE_AA)
        cv2.putText(frame, f"{'IZQ' if side=='LEFT' else 'DER'} {pct}%",
                    (ep[0]+16, ep[1]-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, base, 1, cv2.LINE_AA)

def draw_hand(frame, lm, w, h, side):
    col_base = C_LEFT if side == "LEFT" else C_RIGHT
    for conn in mp_hands_mod.HAND_CONNECTIONS:
        a, b = conn
        pa = (int(lm[a].x*w), int(lm[a].y*h))
        pb = (int(lm[b].x*w), int(lm[b].y*h))
        cv2.line(frame, pa, pb, C_GRAY, 1, cv2.LINE_AA)
    for name, tip_idx in FINGERTIPS.items():
        col    = FINGER_COLORS[name]
        tip    = lm[tip_idx]
        pip    = lm[FINGER_PIP[name]]
        tip_px = (int(tip.x*w), int(tip.y*h))
        pip_px = (int(pip.x*w), int(pip.y*h))
        cv2.circle(frame, pip_px, 5, col, -1, cv2.LINE_AA)
        cv2.circle(frame, tip_px, 8, col, -1, cv2.LINE_AA)
        cv2.circle(frame, tip_px, 8, C_WHITE, 1, cv2.LINE_AA)
        pct = finger_pct[side][name]
        cv2.putText(frame, f"{pct}", (tip_px[0]+10, tip_px[1]+4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)
    wr = lm[0]
    cv2.circle(frame, (int(wr.x*w), int(wr.y*h)), 8, col_base, 2, cv2.LINE_AA)

def draw_hud(frame, fps):
    fh = frame.shape[0]

    # FPS
    cv2.rectangle(frame, (6, 6), (130, 38), (0,0,0), -1)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, C_GREEN, 2, cv2.LINE_AA)

    # Estado serial
    for i, side in enumerate(("LEFT","RIGHT")):
        col = C_GREEN if serial_ok[side] else C_RED
        cv2.putText(frame, f"{side[0]}:{'OK' if serial_ok[side] else '--'}",
                    (140+i*70, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv2.LINE_AA)
    cv2.putText(frame, f"C:{'OK' if codos_ok else '--'}",
                (290, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                C_GREEN if codos_ok else C_RED, 1, cv2.LINE_AA)

    # Indicador de pausa serial
    if serial_paused:
        cv2.rectangle(frame, (350, 8), (620, 36), (0, 0, 0), -1)
        cv2.rectangle(frame, (350, 8), (620, 36), C_ORANGE, 2)
        cv2.putText(frame, "SERIAL PAUSADO  [H=reanudar]",
                    (358, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.52, C_ORANGE, 1, cv2.LINE_AA)

    # Calibración dedos
    if calib_mode is not None:
        label = "Calib CERRADO..." if calib_mode=="low" else "Calib ABIERTO..."
        col   = C_RED if calib_mode=="low" else C_GREEN
        pct   = int(min(calib_frames/CALIB_HAND_N, 1.0)*200)
        cv2.putText(frame, label, (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
        cv2.rectangle(frame, (10,68),(210,76),(40,40,40),-1)
        cv2.rectangle(frame, (10,68),(10+pct,76), col,-1)

    # Calibración codos
    if _cal_codo["active"]:
        s_ = _cal_codo["side"]; e_ = _cal_codo["endpoint"]
        label = f"Calib codo {'IZQ' if s_=='LEFT' else 'DER'} {'EST.' if e_=='low' else 'DOB.'}..."
        col   = C_RED if e_=="low" else C_GREEN
        pct   = int(min(len(_cal_codo["samples"])/CALIB_CODO_N,1.0)*200)
        cv2.putText(frame, label, (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
        cv2.rectangle(frame, (10,98),(210,106),(40,40,40),-1)
        cv2.rectangle(frame, (10,98),(10+pct,106),col,-1)

    # Hints de teclado
    hints = [
        "C=ded.cerrado  B=ded.abierto",
        "Z/X=codo IZQ  N/F=codo DER",
        "H=pausar/reanudar serial    Q=salir",
    ]
    for i, txt in enumerate(hints):
        cv2.putText(frame, txt, (10, fh-14-(len(hints)-1-i)*18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100,180,100), 1, cv2.LINE_AA)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
discover_arduinos()

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG')) 
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
cap.set(cv2.CAP_PROP_FPS,          30)
cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

WIN = "Hermes Vision"
cv2.namedWindow(WIN, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

print("Corriendo | C=cerrado B=abierto | Z/X=codo IZQ | N/F=codo DER | H=pausar serial | Q=salir")

fps          = 0.0
frame_count  = 0
t_start      = time.time()
calib_mode   = None
calib_frames = 0
hand_found   = {"LEFT": False, "RIGHT": False}

with jtop() as jetson:
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        if not ret:
            break

        h, w = frame.shape[:2]

        small_bgr = cv2.resize(frame, (MP_W, MP_H), interpolation=cv2.INTER_LINEAR)
        small_rgb = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB)
        small_rgb.flags.writeable = False

        with _pose_input_lock:
            if _pose_input is None:
                _pose_input = small_rgb

        with _hand_input_lock:
            if _hand_input is None:
                _hand_input = small_rgb

        with _pose_result_lock:
            pose_res = _pose_result

        with _hand_result_lock:
            hand_res = _hand_result

        # ── Pose ───────────────────────────────────────────────
        if pose_res and pose_res.pose_landmarks:
            lm = pose_res.pose_landmarks.landmark
            for side, (sh_i, el_i, wr_i) in ELBOW_PTS.items():
                sh, el, wr = lm[sh_i], lm[el_i], lm[wr_i]
                if min(sh.visibility, el.visibility, wr.visibility) < 0.5:
                    continue
                angle = angle_between(sh, el, wr)
                update_codo(side, angle)
                feed_calib_codo(angle, side)
            draw_pose(frame, lm, w, h)
            queue_codos(codo_pct["LEFT"], codo_pct["RIGHT"])

        # ── Hands ──────────────────────────────────────────────
        for s in ("LEFT", "RIGHT"):
            hand_found[s] = False

        if hand_res and hand_res.multi_hand_landmarks and hand_res.multi_handedness:
            for hand_lm, handedness in zip(
                    hand_res.multi_hand_landmarks, hand_res.multi_handedness):
                cam_side = handedness.classification[0].label
                side     = "RIGHT" if cam_side == "Left" else "LEFT"
                hand_found[side] = True

                update_fingers(side, hand_lm.landmark)
                draw_hand(frame, hand_lm.landmark, w, h, side)

                if calib_mode is not None:
                    collect_calib(hand_lm.landmark, calib_mode)

                queue_fingers(side, [finger_pct[side][n] for n in FINGER_NAMES])

        # ── Calibración dedos ──────────────────────────────────
        if calib_mode is not None and any(hand_found.values()):
            calib_frames += 1
            if calib_frames >= CALIB_HAND_N:
                apply_calib(calib_mode)
                calib_mode, calib_frames = None, 0

        # ── HUD + FPS ──────────────────────────────────────────
        frame_count += 1
        elapsed = time.time() - t_start
        if elapsed >= 1.0:
            fps, frame_count, t_start = frame_count/elapsed, 0, time.time()

        draw_hud(frame, fps)
        cv2.imshow(WIN, frame)

        # ── Teclado ────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key in (ord('c'), ord('C')):
            calib_mode, calib_frames = "low", 0
            for n in FINGER_NAMES: calib_smp_low[n].clear()
            print("Calibrando CERRADO...")
        elif key in (ord('b'), ord('B')):
            calib_mode, calib_frames = "high", 0
            for n in FINGER_NAMES: calib_smp_high[n].clear()
            print("Calibrando ABIERTO...")
        elif key in (ord('z'), ord('Z')): start_calib_codo("LEFT",  "low")
        elif key in (ord('x'), ord('X')): start_calib_codo("LEFT",  "high")
        elif key in (ord('n'), ord('N')): start_calib_codo("RIGHT", "low")
        elif key in (ord('f'), ord('F')): start_calib_codo("RIGHT", "high")
        elif key in (ord('h'), ord('H')):
            serial_paused = not serial_paused
            if serial_paused:
                # Pausa: ramp a 0 en hilo separado para no bloquear video
                threading.Thread(target=_ramp_to_zero, daemon=True).start()
            else:
                # Reanuda: resetea EMA para arrancar desde 0
                _ramp_from_zero()

# ══════════════════════════════════════════════════════════════════════════════
# APAGADO CONTROLADO
# ══════════════════════════════════════════════════════════════════════════════
cap.release()
cv2.destroyAllWindows()

SHUTDOWN_STEPS = 20
SHUTDOWN_DELAY = 0.04

print("Apagando... bajando codos a 0")

if esp32_codos and codos_ok:
    start_izq = codo_pct["LEFT"]
    start_der = codo_pct["RIGHT"]
    for step in range(SHUTDOWN_STEPS + 1):
        t = step / SHUTDOWN_STEPS
        val_izq = int(start_izq * (1.0 - t))
        val_der = int(start_der * (1.0 - t))
        msg = f"{val_izq},{val_der}\n".encode()
        _send_direct(esp32_codos, msg)
        time.sleep(SHUTDOWN_DELAY)
    print(f"  Codos: {start_izq}%,{start_der}% → 0%,0% OK")

print("  Bajando dedos a 0")
zeros = "0,0,0,0,0\n".encode()
for side in ("LEFT", "RIGHT"):
    if serial_ok.get(side) and arduinos[side]:
        for _ in range(3):
            _send_direct(arduinos[side], zeros)
            time.sleep(0.02)

time.sleep(0.1)

for side in ("LEFT", "RIGHT"):
    if arduinos[side]:
        try: arduinos[side].close()
        except: pass
if esp32_codos:
    try: esp32_codos.close()
    except: pass

hands_detector.close()
pose_detector.close()
print("Apagado completo.")
