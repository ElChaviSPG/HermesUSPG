#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════════════════════════
# HERMES NET — capa de transporte UDP (reemplaza al serial)
#
# TOPOLOGÍA
#   Jetson y los tres ESP32 en el mismo router dedicado.
#   El ESP32 anuncia su presencia por broadcast; la Jetson aprende su IP.
#   Ninguno de los dos necesita IPs fijas ni configuración previa.
#
#       ESP32  --HELLO (broadcast 255.255.255.255:5005)-->  Jetson
#       Jetson --CONTROL (unicast <ip_esp32>:5006)------->  ESP32
#       ESP32  --TELEMETRÍA (unicast <ip_jetson>:5005)--->  Jetson
#
# POR QUÉ UDP Y NO WEBSOCKET
#   WebSocket es TCP, y TCP retransmite. Un paquete de posición perdido
#   detiene la línea hasta reentregarlo, y luego descarga una ráfaga de
#   posiciones viejas: el brazo se congela y después salta. En un stream de
#   control eso está al revés. Aquí cada datagrama lleva el ESTADO COMPLETO
#   (nunca deltas), así que es idempotente: perder uno no importa porque el
#   siguiente llega 20 ms después y lo reemplaza.
#
# TASA FIJA, INDEPENDIENTE DEL VIDEO
#   El envío corre en su propio hilo a TX_HZ. La visión solo actualiza un
#   estado compartido. Si la visión tiene un hipo, los motores siguen
#   recibiendo un flujo constante en lugar de quedarse sin comandos.
# ═══════════════════════════════════════════════════════════════════════════

import socket, threading, time, math

# ── Configuración ──────────────────────────────────────────────────────────
PORT_DISCOVERY = 5005      # la Jetson escucha aquí (HELLO + telemetría)
PORT_CONTROL   = 5006      # los ESP32 escuchan aquí (comandos)
TX_HZ          = 50        # tasa de envío de comandos
PEER_TIMEOUT_S = 5.0       # sin noticias de un ESP32 → se marca caído

ROLES        = ("LEFT", "RIGHT")
FINGER_COUNT = 5


def _norm_role(s):
    """
    El firmware se anuncia como 'HERMES_RIGHT' (herencia del protocolo #WHO
    del serial), pero internamente los roles son 'RIGHT'. Aceptar ambas
    formas evita tener que reflashear las placas para cambiar una cadena.
    """
    s = s.strip().upper()
    if s.startswith("HERMES_"):
        s = s[7:]
    return s if s in ROLES else None

RAMP_S = 0.8               # duración de la rampa a cero (tecla H / apagado)

# ── Estado de peers ────────────────────────────────────────────────────────
# role -> {"ip", "port", "last_seen", "rssi", "seq_echo", "fw"}
_peers      = {}
_peers_lock = threading.Lock()

# ── Estado de control ──────────────────────────────────────────────────────
# _target: lo que pide la visión.  _out: lo que realmente se transmite.
# En modo "live" son lo mismo; durante la rampa y la pausa divergen.
_target = {
    "codo":   {"LEFT": 0, "RIGHT": 0},
    "finger": {"LEFT": [0] * FINGER_COUNT, "RIGHT": [0] * FINGER_COUNT},
}
_state_lock = threading.Lock()

_mode        = "live"      # "live" | "ramping" | "paused"
_ramp_t0     = 0.0
_ramp_start  = None
_mode_lock   = threading.Lock()

_seq      = 0
_tx_sock  = None
_running  = False
_stats    = {"tx": 0, "rx": 0, "tx_hz": 0.0}
_unknown_seen = set()


# ═══════════════════════════════════════════════════════════════════════════
# API pública — refleja la del antiguo módulo serial
# ═══════════════════════════════════════════════════════════════════════════
def set_codos(pct_izq, pct_der):
    with _state_lock:
        _target["codo"]["LEFT"]  = int(pct_izq)
        _target["codo"]["RIGHT"] = int(pct_der)


def set_fingers(side, pcts):
    with _state_lock:
        _target["finger"][side] = [int(p) for p in pcts]


def link_ok(role):
    with _peers_lock:
        p = _peers.get(role)
        return bool(p) and (time.time() - p["last_seen"]) < PEER_TIMEOUT_S


def peer_info(role):
    """Devuelve (ip, rssi, edad_en_ms) o None."""
    with _peers_lock:
        p = _peers.get(role)
        if not p:
            return None
        return p["ip"], p["rssi"], int((time.time() - p["last_seen"]) * 1000)


def pause():
    """Arranca la rampa a cero. No bloquea."""
    global _mode, _ramp_t0, _ramp_start
    with _mode_lock:
        _ramp_start = _snapshot_out()
        _ramp_t0    = time.time()
        _mode       = "ramping"


def resume():
    global _mode
    with _mode_lock:
        _mode = "live"


def is_paused():
    return _mode != "live"


def ramp_to_zero_blocking(timeout=RAMP_S + 0.4):
    """Para el apagado: lanza la rampa y espera a que termine de transmitirse."""
    pause()
    t0 = time.time()
    while time.time() - t0 < timeout:
        if _mode == "paused":
            time.sleep(0.15)          # unos ciclos más de ceros en el aire
            return
        time.sleep(0.02)


def stats():
    return dict(_stats)


# ═══════════════════════════════════════════════════════════════════════════
# Internos
# ═══════════════════════════════════════════════════════════════════════════
def _snapshot_out():
    with _state_lock:
        return {
            "codo":   dict(_target["codo"]),
            "finger": {s: list(_target["finger"][s]) for s in ("LEFT", "RIGHT")},
        }


def _resolve_out():
    """Calcula el payload a transmitir según el modo actual."""
    global _mode
    with _mode_lock:
        mode = _mode
        if mode == "ramping":
            t = (time.time() - _ramp_t0) / RAMP_S
            if t >= 1.0:
                _mode = "paused"
                mode  = "paused"
            else:
                k = 1.0 - t
                return {
                    "codo":   {s: int(_ramp_start["codo"][s] * k) for s in ("LEFT", "RIGHT")},
                    "finger": {s: [int(v * k) for v in _ramp_start["finger"][s]]
                               for s in ("LEFT", "RIGHT")},
                }

    if mode == "paused":
        return {"codo":   {"LEFT": 0, "RIGHT": 0},
                "finger": {"LEFT": [0] * FINGER_COUNT, "RIGHT": [0] * FINGER_COUNT}}

    return _snapshot_out()


def _build_packet(seq, out):
    """
    Un solo payload con TODO el estado, enviado por unicast a los tres peers.
    Cada ESP32 lee su porción e ignora el resto.

    Que los tres compartan paquete y número de secuencia significa que aplican
    el mismo instante de la pose: antes, con colas independientes, codos y
    dedos podían desincronizarse y el gesto salía incoherente.

        C,<seq>,<codoL>,<codoR>,<L0..L4>,<R0..R4>

    ~45 bytes. A 50 Hz por tres destinos son 6.7 KB/s: irrelevante.
    """
    c = out["codo"]
    l = out["finger"]["LEFT"]
    r = out["finger"]["RIGHT"]
    vals = [seq, c["LEFT"], c["RIGHT"]] + l + r
    return ("C," + ",".join(str(v) for v in vals) + "\n").encode()


def _tx_worker():
    global _seq
    period = 1.0 / TX_HZ
    next_t = time.perf_counter()
    count, t_mark = 0, time.time()

    while _running:
        out = _resolve_out()
        _seq = (_seq + 1) & 0xFFFF
        pkt  = _build_packet(_seq, out)

        with _peers_lock:
            targets = [(p["ip"], p["port"]) for p in _peers.values()]

        for addr in targets:
            try:
                _tx_sock.sendto(pkt, addr)
            except Exception:
                pass

        count += 1
        _stats["tx"] += 1
        now = time.time()
        if now - t_mark >= 1.0:
            _stats["tx_hz"] = count / (now - t_mark)
            count, t_mark = 0, now

        # Reloj absoluto: sin deriva acumulada aunque un ciclo se atrase.
        next_t += period
        delay = next_t - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        else:
            next_t = time.perf_counter()


def _rx_worker(sock):
    while _running:
        try:
            data, addr = sock.recvfrom(256)
        except socket.timeout:
            continue
        except Exception:
            time.sleep(0.1)
            continue

        _stats["rx"] += 1
        try:
            parts = data.decode(errors="ignore").strip().split(",")
        except Exception:
            continue
        if not parts:
            continue

        kind = parts[0]
        handled = False

        # HELLO: H,<rol>,<fw>
        if kind == "H" and len(parts) >= 2:
            role = _norm_role(parts[1])
            if role:
                with _peers_lock:
                    known = role in _peers
                    _peers[role] = {
                        "ip": addr[0], "port": PORT_CONTROL,
                        "last_seen": time.time(), "rssi": 0,
                        "seq_echo": 0, "fw": parts[2] if len(parts) > 2 else "?",
                    }
                handled = True
                if not known:
                    print(f"[net] {role} descubierto en {addr[0]}")

        # TELEMETRÍA: T,<rol>,<seq_eco>,<rssi>
        elif kind == "T" and len(parts) >= 4:
            role = _norm_role(parts[1])
            if role:
                with _peers_lock:
                    p = _peers.get(role)
                    if p is None:
                        p = {"ip": addr[0], "port": PORT_CONTROL, "fw": "?"}
                        _peers[role] = p
                        print(f"[net] {role} descubierto en {addr[0]} (vía telemetría)")
                    p["ip"]        = addr[0]
                    p["last_seen"] = time.time()
                    try:
                        p["seq_echo"] = int(parts[2])
                        p["rssi"]     = int(parts[3])
                    except ValueError:
                        pass
                handled = True


        if not handled:
            # Un paquete que llega y se descarta en silencio es la peor forma
            # de fallo: parece que no llegó nada. Se avisa una vez por origen.
            key = (addr[0], data[:16])
            if key not in _unknown_seen:
                _unknown_seen.add(key)
                print(f"[net] paquete no reconocido de {addr[0]}: "
                      f"{data[:48]!r} — revisa el formato")


def start():
    """Arranca los hilos de red. Idempotente."""
    global _tx_sock, _running
    if _running:
        return
    _running = True

    _tx_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    _tx_sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    rx.bind(("0.0.0.0", PORT_DISCOVERY))
    rx.settimeout(0.5)

    threading.Thread(target=_rx_worker, args=(rx,), daemon=True).start()
    threading.Thread(target=_tx_worker, daemon=True).start()
    print(f"[net] escuchando en :{PORT_DISCOVERY}, "
          f"transmitiendo a :{PORT_CONTROL} @ {TX_HZ} Hz")


def stop():
    global _running
    _running = False
