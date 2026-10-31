#!/usr/bin/env python3
# ===========================================================================
# HERMES GESTURE - motor de gestos de dedos sobre hermes_net
#
# QUE HACE
#   Un hilo independiente a 40 Hz que lee en que estado esta Hermes
#   (escuchando / procesando / hablando / quieto) y escribe una onda
#   viajera sobre los 5 dedos de cada mano via net.set_fingers().
#   No toca los codos.
#
# POR QUE UN OBSERVADOR Y NO HOOKS
#   El estado ya vive centralizado en hermes_vision (recording, chat_status,
#   tts_playing). Si el gesto se disparara con llamadas repartidas por el
#   codigo, cada camino de error nuevo tendria que acordarse de volver a
#   neutro. Aqui el gesto SONDEA el estado: si el estado vuelve a idle por
#   cualquier motivo, incluida una excepcion, la mano vuelve sola.
#
# LA ONDA
#   pct_i(t) = 100 - A * (1 - cos(theta - i*PHI)) / 2
#
#   - theta se INTEGRA (theta += 2*pi*dt/T) en vez de calcularse como
#     2*pi*t/T. Asi el periodo T puede cambiar de un ciclo al otro sin que
#     la fase de un salto: al cambiar de estado la mano acelera, no brinca.
#   - El coseno desplazado hace que el maximo de pct sea exactamente 100.
#     La mano nunca intenta pasarse del neutro, solo se curva hacia adentro.
#   - PHI = pi/5 desfasa cada dedo: eso es lo que se lee como onda y no
#     como los cinco dedos moviendose en bloque.
#   - La mano DERECHA usa el indice invertido (4-i), asi la onda corre en
#     sentido opuesto en cada mano. Con el mismo sentido en ambas el gesto
#     se lee como una copia; con sentidos opuestos se lee como intencion.
#
# LA AMPLITUD
#   A se relaja exponencialmente hacia su objetivo:
#       A += (A_obj - A) * (1 - exp(-dt/TAU_A))
#   Se suaviza SOLO la amplitud, nunca la salida. Filtrar la salida seria
#   un pasabajos sobre la propia onda: a T=0.8s y tau=0.12s la atenuaria
#   ~27% y le metaria retardo de fase. Modulando la amplitud, la onda sale
#   con su forma intacta y las transiciones de estado igual son suaves.
#   La forma exponencial (no lineal) es independiente del frame rate: si
#   el hilo se atrasa, dt crece y el resultado es el mismo.
#
# NEUTRO = 100 (dedos extendidos)
#   Ojo: hermes_net rampa a 0 al parar y el firmware usa SAFE_PCT = 0 en
#   timeout HARD. O sea que si se corta el enlace o se cierra el programa,
#   las manos se CIERRAN, no quedan extendidas. Es deliberado: el neutro
#   estetico y la posicion segura no tienen por que coincidir.
#
# Archivo en ASCII puro para que sobreviva a cualquier portapapeles.
# ===========================================================================

import math
import threading
import time

import hermes_net as net

# --- Constantes -----------------------------------------------------------
FINGER_COUNT = 5
RATE_HZ      = 40.0        # por debajo de los 50 Hz de TX: no compite
NEUTRAL      = 100.0       # dedos extendidos
PHI          = math.pi / 5.0
TAU_A        = 0.30        # constante de relajacion de la amplitud [s]
DT_MAX       = 0.25        # techo de dt: tras un freeze no da un latigazo

# estado -> (amplitud en %, periodo en s)
#   idle       quieto, dedos extendidos
#   recording  respiracion lenta y corta: "te estoy escuchando"
#   thinking   mas rapido y mas marcado: inquietud contenida
#   speaking   amplio y agil: acompana el habla
PROFILES = {
    "idle":      (0.0,  2.0),
    "recording": (12.0, 3.0),
    "thinking":  (25.0, 1.2),
    "speaking":  (45.0, 0.8),
}

# --- Estado interno -------------------------------------------------------
_running  = False
_thread   = None
_state_fn = None
_enabled  = True
_last     = {"mode": "idle", "amp": 0.0}


def _pct(theta, amp, i):
    v = NEUTRAL - amp * (1.0 - math.cos(theta - i * PHI)) * 0.5
    if v < 0.0:
        return 0.0
    if v > 100.0:
        return 100.0
    return v


def _worker():
    theta  = 0.0
    amp    = 0.0
    t_prev = time.perf_counter()
    period = 1.0 / RATE_HZ
    next_t = t_prev

    while _running:
        now = time.perf_counter()
        dt  = now - t_prev
        if dt <= 0.0:
            dt = 1e-3
        elif dt > DT_MAX:
            dt = DT_MAX
        t_prev = now

        if _enabled and _state_fn is not None:
            try:
                mode = _state_fn()
            except Exception:
                mode = "idle"
        else:
            mode = "idle"

        a_obj, T = PROFILES.get(mode, PROFILES["idle"])

        theta += 2.0 * math.pi * dt / T
        if theta > 2.0 * math.pi:
            theta -= 2.0 * math.pi
        amp += (a_obj - amp) * (1.0 - math.exp(-dt / TAU_A))

        if a_obj == 0.0 and amp < 0.05:
            # Ya converjio: mandar el neutro exacto en vez de 99.97.
            left  = [NEUTRAL] * FINGER_COUNT
            right = [NEUTRAL] * FINGER_COUNT
            amp   = 0.0
        else:
            left  = [_pct(theta, amp, i)             for i in range(FINGER_COUNT)]
            right = [_pct(theta, amp, FINGER_COUNT - 1 - i)
                     for i in range(FINGER_COUNT)]

        net.set_fingers("LEFT",  left)
        net.set_fingers("RIGHT", right)

        _last["mode"] = mode
        _last["amp"]  = amp

        # Reloj absoluto: sin deriva acumulada aunque un ciclo se atrase.
        next_t += period
        delay = next_t - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        else:
            next_t = time.perf_counter()


# ===========================================================================
# API publica
# ===========================================================================
def start(state_fn=None):
    """
    Arranca el motor. state_fn es un callable sin argumentos que devuelve
    una de las claves de PROFILES. Idempotente.
    """
    global _running, _thread, _state_fn
    if _running:
        return
    _state_fn = state_fn
    _running  = True
    _thread   = threading.Thread(target=_worker, daemon=True)
    _thread.start()
    print("[gesto] motor activo a %.0f Hz, neutro=%.0f%%" % (RATE_HZ, NEUTRAL))


def stop():
    global _running
    _running = False


def set_enabled(flag):
    """Apagar el gesto sin matar el hilo: la mano relaja a neutro."""
    global _enabled
    _enabled = bool(flag)


def is_enabled():
    return _enabled


def status():
    """Para depurar: (modo_actual, amplitud_actual)."""
    return _last["mode"], _last["amp"]
