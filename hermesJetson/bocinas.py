"""
Test de bocinas - Hermes
Reproduce varios sonidos para verificar audio de salida.
"""
import subprocess
import numpy as np
import wave
import tempfile
import os
import re

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
    return "plughw:0,0"

AUDIO_OUT = find_audio_output()

# ─── Generador de tonos ───────────────────────────────────────────────────────
def generate_tone(freq_hz, duration_s, sample_rate=44100, volume=0.5):
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    wave_data = np.sin(2 * np.pi * freq_hz * t) * volume
    # Fade in/out para evitar clicks
    fade = int(sample_rate * 0.02)
    wave_data[:fade]  *= np.linspace(0, 1, fade)
    wave_data[-fade:] *= np.linspace(1, 0, fade)
    return (wave_data * 32767).astype(np.int16)

def save_wav(samples, path, sample_rate=44100):
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())

def play(path):
    subprocess.run(["aplay", "-D", AUDIO_OUT, path],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# ─── Tests ────────────────────────────────────────────────────────────────────
def test_tonos():
    print("\n[1] Tonos puros (440Hz, 880Hz, 1320Hz)")
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    for freq in [440, 880, 1320]:
        print(f"  → {freq} Hz")
        samples = generate_tone(freq, 0.5)
        save_wav(samples, tmp.name)
        play(tmp.name)
    os.unlink(tmp.name)

def test_escala():
    print("\n[2] Escala musical (Do mayor)")
    notas = [
        ("Do", 261.63), ("Re", 293.66), ("Mi", 329.63),
        ("Fa", 349.23), ("Sol", 392.00), ("La", 440.00),
        ("Si", 493.88), ("Do", 523.25),
    ]
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    for nombre, freq in notas:
        print(f"  → {nombre} ({freq:.0f} Hz)")
        samples = generate_tone(freq, 0.3)
        save_wav(samples, tmp.name)
        play(tmp.name)
    os.unlink(tmp.name)

def test_estereo():
    print("\n[3] Test estéreo (izquierdo → derecho)")
    sample_rate = 44100
    duration = 1.0
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    for canal, nombre in [(0, "IZQUIERDO"), (1, "DERECHO")]:
        print(f"  → Canal {nombre}")
        mono = generate_tone(523.25, duration)
        stereo = np.zeros((len(mono), 2), dtype=np.int16)
        stereo[:, canal] = mono
        with wave.open(tmp.name, "wb") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(stereo.tobytes())
        play(tmp.name)
    os.unlink(tmp.name)

def test_speak():
    print("\n[4] Voz con espeak (si está instalado)")
    try:
        subprocess.run(
            ["espeak", "-v", "es", "-s", "150",
             "Hola, soy Hermes. Las bocinas funcionan correctamente."],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        print("  espeak no instalado. Instala con: sudo apt install espeak")
    except Exception as e:
        print(f"  Error: {e}")

# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Salida de audio: {AUDIO_OUT}\n")
    print("Presiona Ctrl+C para detener en cualquier momento.\n")

    try:
        test_tonos()
        test_escala()
        test_estereo()
        test_speak()
        print("\n Test completo.")
    except KeyboardInterrupt:
        print("\nTest interrumpido.")
