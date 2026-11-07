"""
Test de bocinas - Hermes
Reproduce sonidos y la voz de Hermes para verificar el audio de salida.

Uso:
    python bocinas.py            # todos los tests
    python bocinas.py voz        # solo uno: tonos, escala, estereo, sonidos, voz
    python bocinas.py voz 70     # con volumen del sistema al 70 %
"""
import subprocess
import numpy as np
import os
import re
import shutil
import sys

SAMPLE_RATE = 44100
VOICE_MODEL = "/home/jetson/voices/es_ES-davefx-medium.onnx"

# Mismo perfil de voz amable que usa groqChat.py (rama voz).
VOICE_LENGTH_SCALE = 1.15
VOICE_NOISE_SCALE  = 0.45
VOICE_NOISE_W      = 0.55
VOICE_PITCH        = 0.94

# ─── Auto-descubrimiento de salida de audio ───────────────────────────────────
def find_audio_output(name_hint="USB PnP"):
    try:
        out = subprocess.check_output(["aplay", "-l"],
                                      stderr=subprocess.DEVNULL).decode()
        print("Dispositivos de salida disponibles:")
        print(out)
        for line in out.splitlines():
            if name_hint.lower() in line.lower() and line.startswith("card"):
                card_m = re.search(r"card\s+(\d+)", line)
                dev_m  = re.search(r"device\s+(\d+)", line)
                if card_m and dev_m:
                    device = f"plughw:{card_m.group(1)},{dev_m.group(1)}"
                    print(f"→ Usando: {device}")
                    return device
    except Exception as e:
        print(f"Error buscando salida: {e}")
    print(f"AVISO: no encontré la bocina '{name_hint}'. Usando plughw:0,0, "
          "que puede ser el HDMI u otra salida y no sonar nada.")
    return "plughw:0,0"

AUDIO_OUT = find_audio_output()

# ─── Volumen del sistema (ALSA) ───────────────────────────────────────────────
def set_volume(percent):
    """Ajusta el volumen de la tarjeta de AUDIO_OUT y la quita de silencio.
    Prueba los nombres de control más comunes porque cambian según la tarjeta."""
    if not shutil.which("amixer"):
        print("amixer no instalado. Instala con: sudo apt install alsa-utils")
        return False
    percent = max(0, min(100, percent))
    card_m  = re.search(r":(\d+)", AUDIO_OUT)
    card    = card_m.group(1) if card_m else "0"
    for control in ("PCM", "Speaker", "Master", "Headphone"):
        # Algunos controles no tienen interruptor de silencio; por eso se
        # intenta primero con "unmute" y luego sin él.
        for extra in (["unmute"], []):
            r = subprocess.run(["amixer", "-c", card, "sset", control,
                                f"{percent}%"] + extra,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if r.returncode == 0:
                print(f"Volumen: {control} al {percent}% (tarjeta {card})")
                return True
    print("No se pudo ajustar el volumen con amixer.")
    return False

# ─── Reproducción (sin archivos temporales) ───────────────────────────────────
class AudioError(Exception):
    pass

def play(samples, sample_rate=SAMPLE_RATE, channels=1):
    """Manda audio int16 directo a aplay por stdin. Si falla, lanza AudioError
    con el mensaje de aplay (antes fallaba en silencio y el test "pasaba")."""
    r = subprocess.run(
        ["aplay", "-q", "-D", AUDIO_OUT, "-t", "raw", "-f", "S16_LE",
         "-c", str(channels), "-r", str(int(sample_rate))],
        input=np.ascontiguousarray(samples).tobytes(),
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if r.returncode != 0:
        msg = r.stderr.decode(errors="ignore").strip()
        if "busy" in msg.lower():
            msg += ("\n  La bocina está ocupada: cierra vision.py u otro programa "
                    "que esté usando el audio y vuelve a probar.")
        raise AudioError(msg or f"aplay terminó con código {r.returncode}")

def to_int16(x):
    return (np.clip(x, -1.0, 1.0) * 32767).astype(np.int16)

# ─── Generadores de sonido ────────────────────────────────────────────────────
def envelope(n, attack=0.01, release=0.08, sample_rate=SAMPLE_RATE):
    """Envolvente suave: subida corta y caída exponencial, evita clicks."""
    env = np.ones(n, dtype=np.float32)
    a   = min(int(sample_rate * attack), n)
    r   = min(int(sample_rate * release), n - a)
    if a:
        env[:a] = np.linspace(0, 1, a)
    if r:
        env[-r:] *= np.exp(-np.linspace(0, 5, r))
    return env

def generate_tone(freq_hz, duration_s, volume=0.5, warm=False):
    """Tono puro, o con armónicos suaves (warm=True) para un timbre más cálido."""
    n = int(SAMPLE_RATE * duration_s)
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    x = np.sin(2 * np.pi * freq_hz * t)
    if warm:
        x += 0.30 * np.sin(2 * np.pi * freq_hz * 2 * t)
        x += 0.10 * np.sin(2 * np.pi * freq_hz * 3 * t)
        x /= 1.40
    return x * envelope(n, release=duration_s * 0.6 if warm else 0.02) * volume

def chime(notes, volume=0.4):
    """Varias notas cálidas que se encadenan un poco (sonido tipo campanita)."""
    out = np.zeros(0, dtype=np.float32)
    for freq, dur in notes:
        tone    = generate_tone(freq, dur + 0.15, volume, warm=True)
        overlap = int(SAMPLE_RATE * 0.15)
        if out.size >= overlap:
            out[-overlap:] += tone[:overlap]
            out = np.concatenate([out, tone[overlap:]])
        else:
            out = np.concatenate([out, tone])
    return out

# Sonidos amigables para que Hermes avise lo que está haciendo.
SONIDOS = {
    "encendido":  [(392.00, 0.18), (523.25, 0.18), (659.25, 0.35)],  # Sol-Do-Mi
    "escuchando": [(523.25, 0.12), (783.99, 0.22)],                  # Do-Sol
    "listo":      [(783.99, 0.12), (523.25, 0.25)],                  # Sol-Do
    "apagado":    [(659.25, 0.18), (523.25, 0.18), (392.00, 0.40)],  # Mi-Do-Sol
}

def play_sound(nombre):
    play(to_int16(chime(SONIDOS[nombre])))

# ─── Tests ────────────────────────────────────────────────────────────────────
def test_tonos():
    print("\n[1] Tonos puros (440Hz, 880Hz, 1320Hz)")
    for freq in [440, 880, 1320]:
        print(f"  → {freq} Hz")
        play(to_int16(generate_tone(freq, 0.5)))

def test_escala():
    print("\n[2] Escala musical (Do mayor)")
    notas = [
        ("Do", 261.63), ("Re", 293.66), ("Mi", 329.63),
        ("Fa", 349.23), ("Sol", 392.00), ("La", 440.00),
        ("Si", 493.88), ("Do", 523.25),
    ]
    for nombre, freq in notas:
        print(f"  → {nombre} ({freq:.0f} Hz)")
        play(to_int16(generate_tone(freq, 0.3, warm=True)))

def test_estereo():
    print("\n[3] Test estéreo (izquierdo → derecho)")
    for canal, nombre in [(0, "IZQUIERDO"), (1, "DERECHO")]:
        print(f"  → Canal {nombre}")
        mono   = to_int16(generate_tone(523.25, 1.0))
        stereo = np.zeros((len(mono), 2), dtype=np.int16)
        stereo[:, canal] = mono
        play(stereo, channels=2)

def test_sonidos():
    print("\n[4] Sonidos de Hermes")
    for nombre in SONIDOS:
        print(f"  → {nombre}")
        play_sound(nombre)
        play(np.zeros(int(SAMPLE_RATE * 0.4), dtype=np.int16))

def test_voz():
    print("\n[5] Voz amable de Hermes")
    texto = ("Hola, soy Hermes. Las bocinas funcionan correctamente. "
             "¿Cómo te sientes hoy?")
    try:
        from piper import PiperVoice, SynthesisConfig
        if not os.path.exists(VOICE_MODEL):
            raise FileNotFoundError(VOICE_MODEL)
        voice = PiperVoice.load(VOICE_MODEL)
        cfg   = SynthesisConfig(length_scale=VOICE_LENGTH_SCALE,
                                noise_scale=VOICE_NOISE_SCALE,
                                noise_w_scale=VOICE_NOISE_W)
        audio = np.concatenate([c.audio_int16_array
                                for c in voice.synthesize(texto, syn_config=cfg)])
        audio = (audio * 0.85).astype(np.int16)   # Piper sale al máximo; evita saturar
        play(audio, sample_rate=voice.config.sample_rate * VOICE_PITCH)
        return
    except AudioError:
        raise   # falla la bocina, no Piper: no tiene caso probar con espeak
    except Exception as e:
        print(f"  Piper no disponible ({e}); probando con espeak.")
    try:
        # Más lento y grave que el default para que suene tranquilo.
        subprocess.run(["espeak", "-v", "es", "-s", "130", "-p", "35", texto],
                       check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        print("  espeak no instalado. Instala con: sudo apt install espeak")
    except Exception as e:
        print(f"  Error: {e}")

TESTS = {
    "tonos":    test_tonos,
    "escala":   test_escala,
    "estereo":  test_estereo,
    "sonidos":  test_sonidos,
    "voz":      test_voz,
}

# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = sys.argv[1:]
    for a in args:
        if a.isdigit():
            set_volume(int(a))
    elegidos = [a for a in args if not a.isdigit()]
    for a in elegidos:
        if a not in TESTS:
            sys.exit(f"Test desconocido: {a}. Opciones: {', '.join(TESTS)}")

    if not shutil.which("aplay"):
        sys.exit("aplay no instalado. Instala con: sudo apt install alsa-utils")
    print(f"Salida de audio: {AUDIO_OUT}\n")
    print("Presiona Ctrl+C para detener en cualquier momento.\n")

    try:
        for nombre in (elegidos or TESTS):
            TESTS[nombre]()
        print("\n Test completo.")
    except KeyboardInterrupt:
        print("\nTest interrumpido.")
    except AudioError as e:
        sys.exit(f"\nNo se pudo reproducir audio en {AUDIO_OUT}:\n  {e}")
