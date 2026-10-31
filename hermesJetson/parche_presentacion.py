#!/usr/bin/env python3
# ===========================================================================
# PARCHE 2 - tecla T: presentacion hablada
#
#   python3 parche_presentacion.py
#
# REQUIERE que parche_gestos.py ya se haya corrido: se engancha sobre las
# lineas que ese parche inserto (el import de gesture, _gesture_state, el
# bloque de teclas y el cierre). Si no encuentra esas anclas, aborta.
#
# Idempotente. Deja copia en hermes_vision.py.bak2
# ===========================================================================

import io
import os
import shutil
import sys

TARGET = sys.argv[1] if len(sys.argv) > 1 else "vision.py"

MARK1 = "# [PARCHE-GESTOS]"
MARK2 = "# [PARCHE-PRESENTA]"

# --- 1. import -------------------------------------------------------------
A1_OLD = "import hermes_gesture as gesture\n"
A1_NEW = (
    "import hermes_gesture as gesture\n"
    "" + MARK2 + " saludo protocolario con cola de audio\n"
    "import hermes_presenta as presenta\n"
)

# --- 2. el gesto tambien se activa durante la presentacion -----------------
# Sin esto la mano caeria a neutro en los silencios ENTRE frases: tts_playing
# solo lo mueve speak(), y la presentacion no pasa por speak().
A2_OLD = (
    "def _gesture_state():\n"
    "    if tts_playing:\n"
    "        return \"speaking\"\n"
)
A2_NEW = (
    "def _gesture_state():\n"
    "    " + MARK2 + " presenta.is_running() cubre las pausas entre frases\n"
    "    if tts_playing or presenta.is_running():\n"
    "        return \"speaking\"\n"
)

# --- 3. tecla T ------------------------------------------------------------
A3_OLD = (
    "        elif key == ord(\"x\"):\n"
    "            if recording:\n"
    "                stop_and_process()\n"
    "            else:\n"
    "                print(\">>> X ignorado (no hay grabacion activa)\")\n"
)
A3_NEW = (
    "        elif key == ord(\"x\"):\n"
    "            if recording:\n"
    "                stop_and_process()\n"
    "            else:\n"
    "                print(\">>> X ignorado (no hay grabacion activa)\")\n"
    "        " + MARK2 + " T = presentacion, T de nuevo = cortarla\n"
    "        elif key == ord(\"t\"):\n"
    "            if presenta.is_running():\n"
    "                presenta.stop()\n"
    "                print(\">>> Presentacion cortada\")\n"
    "            elif not recording and chat_status == \"\":\n"
    "                presenta.start(voice, AUDIO_OUT,\n"
    "                               on_line=lambda t: add_chat(\"assistant\", t))\n"
    "            else:\n"
    "                print(\">>> T ignorado (grabando o procesando)\")\n"
)

# --- 4. limpieza -----------------------------------------------------------
A4_OLD = (
    "try:\n"
    "    gesture.stop()\n"
    "    net.stop()\n"
    "except Exception:\n"
    "    pass\n"
)
A4_NEW = (
    "try:\n"
    "    " + MARK2 + "\n"
    "    presenta.stop()\n"
    "    gesture.stop()\n"
    "    net.stop()\n"
    "except Exception:\n"
    "    pass\n"
)

# --- 5. ayuda en pantalla --------------------------------------------------
A5_OLD = "\"Abierta/Z=grabar  Pu\u00f1o/X=enviar\""
A5_NEW = "\"Abierta/Z=grabar  Pu\u00f1o/X=enviar  T=saludo\""

EDITS = [
    ("import",          A1_OLD, A1_NEW),
    ("gesto",           A2_OLD, A2_NEW),
    ("tecla T",         A3_OLD, A3_NEW),
    ("limpieza",        A4_OLD, A4_NEW),
    ("texto de ayuda",  A5_OLD, A5_NEW),
]


def main():
    if not os.path.exists(TARGET):
        print("ERROR: no existe %s" % TARGET)
        return 1

    with io.open(TARGET, "r", encoding="utf-8") as f:
        src = f.read()

    if MARK1 not in src:
        print("ERROR: corre primero parche_gestos.py. Este parche se engancha")
        print("       sobre las lineas que aquel inserta.")
        return 1

    if MARK2 in src:
        print("Ya parcheado (marca %s). No toco nada." % MARK2)
        return 0

    faltan = []
    for name, old, _new in EDITS:
        n = src.count(old)
        if n != 1:
            faltan.append("%s (ancla encontrada %d veces, esperaba 1)" % (name, n))
    if faltan:
        print("ERROR: no se pudo parchear. Anclas problematicas:")
        for f_ in faltan:
            print("  - %s" % f_)
        print("El archivo no se modifico.")
        return 1

    for name, old, new in EDITS:
        src = src.replace(old, new, 1)

    shutil.copy2(TARGET, TARGET + ".bak2")
    with io.open(TARGET, "w", encoding="utf-8") as f:
        f.write(src)

    print("OK: %s parcheado (%d ediciones). Copia previa en %s.bak2"
          % (TARGET, len(EDITS), TARGET))
    return 0


if __name__ == "__main__":
    sys.exit(main())
