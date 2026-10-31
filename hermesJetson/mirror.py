#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════════════════════════
# HERMES VISION — Jetson Orin Nano
#
# Requisitos por boot (fuera de este script):
#   sudo nvpmodel -m 0      → modo MAXN
#   sudo jetson_clocks      → relojes fijos al máximo
#
# Cambios respecto a la versión Raspberry Pi:
#   - Captura por GStreamer/NVMM opcional (CSI) con fallback a V4L2
#   - Telemetría Jetson (temp/potencia/GPU) en el HUD vía jtop
#   - EMA independiente del framerate (constante de tiempo fija)
#   - Ángulos de codo desde pose_world_landmarks (métricos, sin distorsión)
#   - Resolución de MediaPipe derivada del aspecto real de la cámara
# ═══════════════════════════════════════════════════════════════════════════

import os, warnings, logging

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"]     = "3"
os.environ["GRPC_VERBOSITY"]       = "ERROR"
os.environ["PYTHONWARNINGS"]       = "ignore"
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)
logging.getLogger("mediapipe").setLevel(logging.ERROR)

import cv2, time, threading, math, glob, contextlib
import numpy as np
import mediapipe as mp

import hermes_net as net

try:
    from jtop import jtop
    HAS_JTOP = True
except Exception:
    HAS_JTOP = False

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════════════
USE_CSI     = False        # True = cámara CSI (IMX219) por nvarguscamerasrc
CSI_SENSOR  = 0
CAM_W       = 1024
CAM_H       = 768
CAM_FPS     = 30
SERIAL_BAUD = 115200

# Ancho de entrada a MediaPipe. El alto se calcula del aspecto REAL del frame
# para no deformar la geometría (ver nota en angle_between).
MP_TARGET_W = 320

# Hilos que OpenCV puede usar. El Orin Nano tiene 6 cores Cortex-A78; dejarle
# menos a OpenCV reduce la contención con los dos workers de MediaPipe.
OPENCV_THREADS = 2

# Antigüedad máxima de un resultado de MediaPipe antes de considerarlo muerto.
# Debe ser menor que el watchdog del ESP32 (300 ms) para que el motor se
# detenga por watchdog en lugar de repetir una pose congelada.
STALE_S = 0.25

cv2.setNumThreads(OPENCV_THREADS)

# ═══════════════════════════════════════════════════════════════════════════
# COLORES (BGR)
# ═══════════════════════════════════════════════════════════════════════════
C_LEFT   = (100, 220, 255)
C_RIGHT  = (255, 160,  80)
C_WHITE  = (255, 255, 255)
C_GRAY   = (150, 150, 150)
C_GREEN  = (  0, 220,  80)
C_RED    = (  0,  60, 255)
C_YELLOW = (  0, 200, 255)
C_ORANGE = (  0, 140, 255)
C_CYAN   = (255, 220, 120)

FINGER_COLORS = {
    "pulgar":  ( 50, 100, 255),
    "indice":  (100, 255,  50),
    "medio":   (255, 150,  50),
    "anular":  (200,  50, 255),
    "menique": ( 50, 220, 255),
}
FINGER_NAMES = ["pulgar", "indice", "medio", "anular", "menique"]
SIDES        = ("LEFT", "RIGHT")

# ═══════════════════════════════════════════════════════════════════════════
# ENLACE CON LOS ACTUADORES
# Todo el transporte vive ahora en hermes_net (UDP sobre WiFi). Aquí solo
# queda el reseteo del suavizado, que es lógica de visión, no de red.
# ═══════════════════════════════════════════════════════════════════════════
def _reset_smoothing():
    """Resetea el estado del EMA para que el rearranque salga desde cero."""
    for side in SIDES:
        codo_ema[side] = 0.0
        codo_pct[side] = 0
        for n in FINGER_NAMES:
            finger_ema[side][n] = 0.0
            finger_pct[side][n] = 0
    print("Enlace REANUDADO — suavizado reseteado a 0.")


# ═══════════════════════════════════════════════════════════════════════════
# MEDIAPIPE — dos workers paralelos, cada uno procesa solo el frame más nuevo
# ═══════════════════════════════════════════════════════════════════════════
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

# model_complexity en Hands existe desde mediapipe 0.8.10; si la versión del
# Jetson es anterior, el kwarg revienta y hay que caer al constructor viejo.
try:
    hands_detector = mp_hands_mod.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=0,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    )
except TypeError:
    hands_detector = mp_hands_mod.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    )

_PL = mp_pose_mod.PoseLandmark
ELBOW_PTS = {
    "LEFT":  (_PL.LEFT_SHOULDER,  _PL.LEFT_ELBOW,  _PL.LEFT_WRIST),
    "RIGHT": (_PL.RIGHT_SHOULDER, _PL.RIGHT_ELBOW, _PL.RIGHT_WRIST),
}

_pose_in_lock,  _pose_in  = threading.Lock(), None
_pose_out_lock, _pose_out = threading.Lock(), (None, 0.0)
_hand_in_lock,  _hand_in  = threading.Lock(), None
_hand_out_lock, _hand_out = threading.Lock(), (None, 0.0)

_mp_counts = {"pose": 0, "hand": 0}
_mp_hz     = {"pose": 0.0, "hand": 0.0}


def _pose_worker():
    global _pose_in, _pose_out
    while True:
        with _pose_in_lock:
            frame, _pose_in = _pose_in, None
        if frame is None:
            time.sleep(0.001)
            continue
        res = pose_detector.process(frame)
        with _pose_out_lock:
            _pose_out = (res, time.time())
        _mp_counts["pose"] += 1


def _hand_worker():
    global _hand_in, _hand_out
    while True:
        with _hand_in_lock:
            frame, _hand_in = _hand_in, None
        if frame is None:
            time.sleep(0.001)
            continue
        res = hands_detector.process(frame)
        with _hand_out_lock:
            _hand_out = (res, time.time())
        _mp_counts["hand"] += 1


# ═══════════════════════════════════════════════════════════════════════════
# SUAVIZADO — EMA independiente del framerate
#
# El filtro discreto  s ← s + α(r − s)  con α fijo cambia de comportamiento
# cuando cambian los FPS: a 30 Hz suaviza el doble de rápido que a 15 Hz.
# La forma correcta es derivar α del tiempo transcurrido:
#
#     α(Δt) = 1 − e^(−Δt/τ)
#
# con τ fijo (constante de tiempo en segundos). τ = 0.12 reproduce el tacto
# del α = 0.25 original a 30 FPS, pero ahora se mantiene igual bajo carga.
#
# El estado del EMA se guarda SIN recortar. Recortarlo (como hacía la versión
# anterior) atrapa el filtro: con α = 0.25 y un piso en 6, escapar de 0 exigía
# una lectura cruda mayor a 6/α = 24, o sea que todo movimiento por debajo del
# 24 % quedaba congelado. La zona muerta se aplica solo a la SALIDA, y además
# reescalada para que no haya un escalón al cruzar el umbral.
# ═══════════════════════════════════════════════════════════════════════════
TAU       = 0.12
DEAD_LOW  = 6.0
DEAD_HIGH = 94.0


def _ema(prev, raw, dt):
    dt = max(min(dt, 0.25), 1e-3)          # acota saltos tras un frame perdido
    a  = 1.0 - math.exp(-dt / TAU)
    return prev + a * (raw - prev)


def _shape(s):
    """Mapea el estado interno a 0–100 aplicando la zona muerta reescalada."""
    return float(np.clip((s - DEAD_LOW) / (DEAD_HIGH - DEAD_LOW) * 100.0, 0.0, 100.0))


# ═══════════════════════════════════════════════════════════════════════════
# ÁNGULOS
#
# Nota geométrica: en la versión anterior los ángulos se calculaban sobre
# landmarks normalizados de una imagen 4:3 comprimida a 16:9. Esa compresión
# por k = 0.75 en el eje vertical distorsiona todo ángulo según
#
#     θ_medido = atan(k · tan θ_real)
#
# — un codo a 45° reales se leía como 36.9°. Ahora hay dos defensas: el frame
# de entrada conserva el aspecto de la cámara, y los codos se miden sobre
# pose_world_landmarks (coordenadas métricas 3D centradas en la cadera,
# independientes de la proyección de la cámara).
# ═══════════════════════════════════════════════════════════════════════════
def angle_between(a, b, c):
    v1 = np.array([a.x - b.x, a.y - b.y, a.z - b.z])
    v2 = np.array([c.x - b.x, c.y - b.y, c.z - b.z])
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 180.0
    return math.degrees(math.acos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)))


def angle_2d(a, b, c):
    """Variante plana, para landmarks de imagen donde z no es fiable."""
    v1 = np.array([a.x - b.x, a.y - b.y])
    v2 = np.array([c.x - b.x, c.y - b.y])
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 180.0
    return math.degrees(math.acos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)))


# ═══════════════════════════════════════════════════════════════════════════
# CODOS
# ═══════════════════════════════════════════════════════════════════════════
calib_codo = {
    "LEFT":  {"low": 165.0, "high": 85.0},
    "RIGHT": {"low": 165.0, "high": 85.0},
}
_cal_codo    = {"active": False, "side": None, "endpoint": None, "samples": []}
CALIB_CODO_N = 25
codo_ema     = {"LEFT": 0.0, "RIGHT": 0.0}
codo_pct     = {"LEFT": 0,   "RIGHT": 0}


def codo_angle_to_pct(side, angle):
    low, high = calib_codo[side]["low"], calib_codo[side]["high"]
    if abs(low - high) < 5:
        return 0.0
    return float(np.clip((low - angle) / (low - high) * 100.0, 0.0, 100.0))


def update_codo(side, angle, dt):
    raw = codo_angle_to_pct(side, angle)
    codo_ema[side] = _ema(codo_ema[side], raw, dt)
    codo_pct[side] = int(_shape(codo_ema[side]))


def start_calib_codo(side, endpoint):
    _cal_codo.update({"active": True, "side": side, "endpoint": endpoint, "samples": []})
    print(f"Calib codo {'IZQ' if side == 'LEFT' else 'DER'} "
          f"{'ESTIRADO' if endpoint == 'low' else 'DOBLADO'}...")


def feed_calib_codo(angle, side):
    if not _cal_codo["active"] or _cal_codo["side"] != side:
        return
    _cal_codo["samples"].append(angle)
    if len(_cal_codo["samples"]) >= CALIB_CODO_N:
        avg = sum(_cal_codo["samples"]) / len(_cal_codo["samples"])
        calib_codo[side][_cal_codo["endpoint"]] = avg
        print(f"OK codo {'IZQ' if side == 'LEFT' else 'DER'} "
              f"{'EST' if _cal_codo['endpoint'] == 'low' else 'DOB'}: {avg:.1f}")
        _cal_codo["active"]  = False
        _cal_codo["samples"] = []


# ═══════════════════════════════════════════════════════════════════════════
# DEDOS — calibración SEPARADA por mano
#
# Antes calib_low/calib_high eran globales: con las dos manos en cuadro
# durante la calibración, las muestras de ambas caían en el mismo promedio y
# ninguna quedaba bien calibrada.
# ═══════════════════════════════════════════════════════════════════════════
FINGER_ANGLE_PTS = {
    "pulgar":  (1,  2,  3,  4),
    "indice":  (5,  6,  7,  8),
    "medio":   (9,  10, 11, 12),
    "anular":  (13, 14, 15, 16),
    "menique": (17, 18, 19, 20),
}
_DEF_LOW  = {"pulgar": 70.0,  "indice": 75.0,  "medio": 75.0,  "anular": 75.0,  "menique": 75.0}
_DEF_HIGH = {"pulgar": 155.0, "indice": 165.0, "medio": 165.0, "anular": 165.0, "menique": 165.0}

calib_low  = {s: dict(_DEF_LOW)  for s in SIDES}
calib_high = {s: dict(_DEF_HIGH) for s in SIDES}
calib_smp  = {s: {"low":  {n: [] for n in FINGER_NAMES},
                  "high": {n: [] for n in FINGER_NAMES}} for s in SIDES}

calib_mode   = None
calib_frames = {s: 0 for s in SIDES}
CALIB_HAND_N = 20

finger_ema = {s: {n: 0.0 for n in FINGER_NAMES} for s in SIDES}
finger_pct = {s: {n: 0   for n in FINGER_NAMES} for s in SIDES}


def finger_raw_angle(lm, name):
    mcp, pip, dip, tip = FINGER_ANGLE_PTS[name]
    return angle_2d(lm[mcp], lm[pip], lm[dip]) * 0.65 + \
           angle_2d(lm[pip], lm[dip], lm[tip]) * 0.35


def finger_openness(side, lm, name):
    ang  = finger_raw_angle(lm, name)
    low  = calib_low[side][name]
    high = calib_high[side][name]
    if high <= low:
        return 0.0
    return float(np.clip((ang - low) / (high - low) * 100.0, 0.0, 100.0))


def update_fingers(side, lm, dt):
    for name in FINGER_NAMES:
        raw = finger_openness(side, lm, name)
        finger_ema[side][name] = _ema(finger_ema[side][name], raw, dt)
        finger_pct[side][name] = int(_shape(finger_ema[side][name]))


def collect_calib(side, lm, mode):
    for name in FINGER_NAMES:
        calib_smp[side][mode][name].append(finger_raw_angle(lm, name))


def apply_calib(side, mode):
    src = calib_smp[side][mode]
    dst = calib_low[side] if mode == "low" else calib_high[side]
    for name in FINGER_NAMES:
        if src[name]:
            dst[name] = sum(src[name]) / len(src[name])
            src[name].clear()
    etq = "CERRADO" if mode == "low" else "ABIERTO"
    print(f"Calib {etq} {side}: { {n: f'{dst[n]:.1f}' for n in FINGER_NAMES} }")


# ═══════════════════════════════════════════════════════════════════════════
# DIBUJO
# ═══════════════════════════════════════════════════════════════════════════
FINGERTIPS = {"pulgar": 4, "indice": 8,  "medio": 12, "anular": 16, "menique": 20}
FINGER_PIP = {"pulgar": 3, "indice": 6,  "medio": 10, "anular": 14, "menique": 18}

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
    def px(i):  return int(lm[i].x * w), int(lm[i].y * h)
    def vis(i): return lm[i].visibility

    for a, b, col in POSE_BONES:
        if min(vis(a), vis(b)) < 0.5:
            continue
        cv2.line(frame, px(a), px(b), col, 2, cv2.LINE_AA)

    for idx in LEFT_JOINTS | RIGHT_JOINTS:
        if vis(idx) < 0.5:
            continue
        col = C_LEFT if idx in LEFT_JOINTS else C_RIGHT
        cv2.circle(frame, px(idx), 6, col,     -1, cv2.LINE_AA)
        cv2.circle(frame, px(idx), 6, C_WHITE,  1, cv2.LINE_AA)

    for side, el_idx in (("LEFT", _PL.LEFT_ELBOW), ("RIGHT", _PL.RIGHT_ELBOW)):
        if vis(el_idx) < 0.5:
            continue
        pct  = codo_pct[side]
        col  = C_RED if pct < 30 else C_YELLOW if pct < 70 else C_GREEN
        base = C_LEFT if side == "LEFT" else C_RIGHT
        ep   = px(el_idx)
        cv2.circle(frame, ep, 12, col,  -1, cv2.LINE_AA)
        cv2.circle(frame, ep, 12, base,  2, cv2.LINE_AA)
        cv2.putText(frame, f"{'IZQ' if side == 'LEFT' else 'DER'} {pct}%",
                    (ep[0] + 16, ep[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    base, 1, cv2.LINE_AA)


def draw_hand(frame, lm, w, h, side):
    col_base = C_LEFT if side == "LEFT" else C_RIGHT
    for a, b in mp_hands_mod.HAND_CONNECTIONS:
        pa = (int(lm[a].x * w), int(lm[a].y * h))
        pb = (int(lm[b].x * w), int(lm[b].y * h))
        cv2.line(frame, pa, pb, C_GRAY, 1, cv2.LINE_AA)

    for name, tip_idx in FINGERTIPS.items():
        col    = FINGER_COLORS[name]
        tip    = lm[tip_idx]
        pip    = lm[FINGER_PIP[name]]
        tip_px = (int(tip.x * w), int(tip.y * h))
        pip_px = (int(pip.x * w), int(pip.y * h))
        cv2.circle(frame, pip_px, 5, col,     -1, cv2.LINE_AA)
        cv2.circle(frame, tip_px, 8, col,     -1, cv2.LINE_AA)
        cv2.circle(frame, tip_px, 8, C_WHITE,  1, cv2.LINE_AA)
        cv2.putText(frame, f"{finger_pct[side][name]}",
                    (tip_px[0] + 10, tip_px[1] + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)

    wr = lm[0]
    cv2.circle(frame, (int(wr.x * w), int(wr.y * h)), 8, col_base, 2, cv2.LINE_AA)


def draw_hud(frame, fps, jstats):
    fh = frame.shape[0]

    cv2.rectangle(frame, (6, 6), (130, 38), (0, 0, 0), -1)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, C_GREEN, 2, cv2.LINE_AA)

    cv2.putText(frame, f"pose {_mp_hz['pose']:.0f}Hz  hand {_mp_hz['hand']:.0f}Hz",
                (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_CYAN, 1, cv2.LINE_AA)

    for i, role in enumerate(("LEFT", "RIGHT", "CODOS")):
        ok  = net.link_ok(role)
        col = C_GREEN if ok else C_RED
        cv2.putText(frame, f"{role[0]}:{'OK' if ok else '--'}",
                    (140 + i * 55, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv2.LINE_AA)
    cv2.putText(frame, f"UDP {net.stats()['tx_hz']:.0f}Hz", (10, 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_CYAN, 1, cv2.LINE_AA)

    # Telemetría del Orin
    if jstats:
        txt = "  ".join(jstats)
        cv2.putText(frame, txt, (frame.shape[1] - 420, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_CYAN, 1, cv2.LINE_AA)

    if net.is_paused():
        cv2.rectangle(frame, (350, 8), (620, 36), (0, 0, 0), -1)
        cv2.rectangle(frame, (350, 8), (620, 36), C_ORANGE, 2)
        cv2.putText(frame, "ENLACE PAUSADO  [H=reanudar]", (358, 29),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, C_ORANGE, 1, cv2.LINE_AA)

    if calib_mode is not None:
        label = "Calib CERRADO..." if calib_mode == "low" else "Calib ABIERTO..."
        col   = C_RED if calib_mode == "low" else C_GREEN
        done  = max(calib_frames.values())
        pct   = int(min(done / CALIB_HAND_N, 1.0) * 200)
        cv2.putText(frame, label, (10, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
        cv2.rectangle(frame, (10, 106), (210, 114), (40, 40, 40), -1)
        cv2.rectangle(frame, (10, 106), (10 + pct, 114), col, -1)

    if _cal_codo["active"]:
        s_, e_ = _cal_codo["side"], _cal_codo["endpoint"]
        label = (f"Calib codo {'IZQ' if s_ == 'LEFT' else 'DER'} "
                 f"{'EST.' if e_ == 'low' else 'DOB.'}...")
        col = C_RED if e_ == "low" else C_GREEN
        pct = int(min(len(_cal_codo["samples"]) / CALIB_CODO_N, 1.0) * 200)
        cv2.putText(frame, label, (10, 134), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
        cv2.rectangle(frame, (10, 142), (210, 150), (40, 40, 40), -1)
        cv2.rectangle(frame, (10, 142), (10 + pct, 150), col, -1)

    hints = [
        "C=ded.cerrado  B=ded.abierto",
        "Z/X=codo IZQ  N/F=codo DER",
        "H=pausar/reanudar enlace   Q=salir",
    ]
    for i, txt in enumerate(hints):
        cv2.putText(frame, txt, (10, fh - 14 - (len(hints) - 1 - i) * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 180, 100), 1, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════════════════
# CÁMARA
# ═══════════════════════════════════════════════════════════════════════════
def csi_pipeline(w, h, fps, sensor_id=0):
    """
    Pipeline CSI con buffers NVMM: la captura y el escalado ocurren en
    hardware, sin copias a memoria de CPU hasta el último videoconvert.
    """
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM),width={w},height={h},framerate={fps}/1 ! "
        f"nvvidconv flip-method=0 ! video/x-raw,format=BGRx ! "
        f"videoconvert ! video/x-raw,format=BGR ! "
        f"appsink drop=true max-buffers=1 sync=false"
    )


def open_camera():
    if USE_CSI:
        cap = cv2.VideoCapture(csi_pipeline(CAM_W, CAM_H, CAM_FPS, CSI_SENSOR),
                               cv2.CAP_GSTREAMER)
        if cap.isOpened():
            print("Cámara: CSI vía nvarguscamerasrc")
            return cap
        print("Cámara: CSI falló, cayendo a V4L2")

    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_FPS,          CAM_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
    print("Cámara: USB vía V4L2 (MJPG)")
    return cap


def read_jetson(jetson):
    """Extrae temp/potencia/GPU de jtop de forma tolerante a la versión."""
    out = []
    try:
        s = jetson.stats
        gpu = s.get("GPU")
        if gpu is not None:
            out.append(f"GPU {float(gpu):.0f}%")
        temp = next((v for k, v in s.items()
                     if k.startswith("Temp") and ("CPU" in k or "tj" in k.lower())), None)
        if temp is not None:
            out.append(f"{float(temp):.0f}C")
        pwr = next((v for k, v in s.items()
                    if k.startswith("Power") and "TOT" in k.upper()), None)
        if pwr is not None:
            out.append(f"{float(pwr) / 1000.0:.1f}W")
    except Exception:
        pass
    return out


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    global calib_mode, _pose_in, _hand_in

    net.start()
    threading.Thread(target=_pose_worker, daemon=True).start()
    threading.Thread(target=_hand_worker, daemon=True).start()

    cap = open_camera()
    if not cap.isOpened():
        print("No se pudo abrir la cámara.")
        return

    WIN = "Hermes Vision"
    cv2.namedWindow(WIN, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print("Corriendo | C=cerrado B=abierto | Z/X=codo IZQ | N/F=codo DER | "
          "H=pausar enlace | Q=salir")

    fps         = 0.0
    frame_count = 0
    t_start     = time.time()
    t_prev      = time.time()
    mp_h        = None                       # alto de entrada a MediaPipe
    jstats      = []
    t_jetson    = 0.0
    hand_found  = {s: False for s in SIDES}

    jctx = jtop() if HAS_JTOP else contextlib.nullcontext()

    with jctx as jetson:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:     # validar ANTES de tocar el frame
                print("Frame perdido, reintentando...")
                time.sleep(0.05)
                continue

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            now = time.time()
            dt  = now - t_prev
            t_prev = now

            # Alto de MediaPipe derivado del aspecto real (par, para el resize)
            if mp_h is None:
                mp_h = max(2, int(round(MP_TARGET_W * h / w / 2)) * 2)
                print(f"Entrada MediaPipe: {MP_TARGET_W}x{mp_h} "
                      f"(aspecto de cámara {w}x{h})")

            small = cv2.resize(frame, (MP_TARGET_W, mp_h), interpolation=cv2.INTER_AREA)
            small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            small.flags.writeable = False

            with _pose_in_lock:
                if _pose_in is None:
                    _pose_in = small
            with _hand_in_lock:
                if _hand_in is None:
                    _hand_in = small

            with _pose_out_lock:
                pose_res, pose_t = _pose_out
            with _hand_out_lock:
                hand_res, hand_t = _hand_out

            pose_fresh = pose_res is not None and (now - pose_t) < STALE_S
            hand_fresh = hand_res is not None and (now - hand_t) < STALE_S

            # ── Pose / codos ───────────────────────────────────────────────
            if pose_fresh and pose_res.pose_landmarks:
                lm = pose_res.pose_landmarks.landmark
                wlm = (pose_res.pose_world_landmarks.landmark
                       if pose_res.pose_world_landmarks else None)

                for side, (sh_i, el_i, wr_i) in ELBOW_PTS.items():
                    if min(lm[sh_i].visibility, lm[el_i].visibility,
                           lm[wr_i].visibility) < 0.5:
                        continue
                    if wlm is not None:
                        angle = angle_between(wlm[sh_i], wlm[el_i], wlm[wr_i])
                    else:
                        angle = angle_2d(lm[sh_i], lm[el_i], lm[wr_i])
                    update_codo(side, angle, dt)
                    feed_calib_codo(angle, side)

                draw_pose(frame, lm, w, h)
                net.set_codos(codo_pct["LEFT"], codo_pct["RIGHT"])

            # ── Manos / dedos ──────────────────────────────────────────────
            for s in SIDES:
                hand_found[s] = False

            if hand_fresh and hand_res.multi_hand_landmarks and hand_res.multi_handedness:
                for hand_lm, handedness in zip(hand_res.multi_hand_landmarks,
                                               hand_res.multi_handedness):
                    # Con el frame espejado, la etiqueta de MediaPipe se invierte.
                    cam_side = handedness.classification[0].label
                    side     = "RIGHT" if cam_side == "Left" else "LEFT"
                    hand_found[side] = True

                    update_fingers(side, hand_lm.landmark, dt)
                    draw_hand(frame, hand_lm.landmark, w, h, side)

                    if calib_mode is not None and calib_frames[side] < CALIB_HAND_N:
                        collect_calib(side, hand_lm.landmark, calib_mode)
                        calib_frames[side] += 1
                        if calib_frames[side] >= CALIB_HAND_N:
                            apply_calib(side, calib_mode)

                    net.set_fingers(side, [finger_pct[side][n] for n in FINGER_NAMES])

            # Cierra la calibración cuando toda mano visible terminó
            if calib_mode is not None:
                activas = [s for s in SIDES if hand_found[s]]
                if activas and all(calib_frames[s] >= CALIB_HAND_N for s in activas):
                    calib_mode = None
                    calib_frames = {s: 0 for s in SIDES}

            # ── Métricas ───────────────────────────────────────────────────
            frame_count += 1
            elapsed = now - t_start
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                _mp_hz["pose"] = _mp_counts["pose"] / elapsed
                _mp_hz["hand"] = _mp_counts["hand"] / elapsed
                _mp_counts["pose"] = _mp_counts["hand"] = 0
                frame_count, t_start = 0, now

            if HAS_JTOP and jetson is not None and now - t_jetson > 1.0:
                jstats   = read_jetson(jetson)
                t_jetson = now

            draw_hud(frame, fps, jstats)
            cv2.imshow(WIN, frame)

            # ── Teclado ────────────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                break
            elif key in (ord('c'), ord('C')):
                calib_mode   = "low"
                calib_frames = {s: 0 for s in SIDES}
                for s in SIDES:
                    for n in FINGER_NAMES:
                        calib_smp[s]["low"][n].clear()
                print("Calibrando CERRADO...")
            elif key in (ord('b'), ord('B')):
                calib_mode   = "high"
                calib_frames = {s: 0 for s in SIDES}
                for s in SIDES:
                    for n in FINGER_NAMES:
                        calib_smp[s]["high"][n].clear()
                print("Calibrando ABIERTO...")
            elif key in (ord('z'), ord('Z')): start_calib_codo("LEFT",  "low")
            elif key in (ord('x'), ord('X')): start_calib_codo("LEFT",  "high")
            elif key in (ord('n'), ord('N')): start_calib_codo("RIGHT", "low")
            elif key in (ord('f'), ord('F')): start_calib_codo("RIGHT", "high")
            elif key in (ord('h'), ord('H')):
                if net.is_paused():
                    _reset_smoothing()
                    net.resume()
                else:
                    net.pause()

    # ── Apagado controlado ─────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()

    print("Apagando... rampa a 0 (codos y dedos)")
    net.ramp_to_zero_blocking()    # misma rampa que la tecla H, pero espera
    net.stop()

    hands_detector.close()
    pose_detector.close()
    print("Apagado completo.")


if __name__ == "__main__":
    main()
