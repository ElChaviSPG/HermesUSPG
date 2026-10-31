#!/usr/bin/env python3
# ===========================================================================
# PARCHE - engancha hermes_gesture + teclas Z/X en hermes_vision.py
#
#   python3 parche_gestos.py            -> parchea ./hermes_vision.py
#   python3 parche_gestos.py otro.py    -> parchea otro archivo
#
# Es IDEMPOTENTE: si ya esta parcheado avisa y no toca nada. Antes de
# escribir deja una copia en hermes_vision.py.bak.
#
# POR QUE UN SCRIPT Y NO SED
#   Las inserciones son bloques de varias lineas y el archivo destino es
#   UTF-8 con acentos. sed sobre UTF-8 desde CMD ya te rompio archivos
#   antes. Este script abre y escribe declarando encoding="utf-8"
#   explicitamente, asi que los acentos no dependen del locale del shell.
#   El script en si es ASCII puro (los pocos caracteres no-ASCII van
#   escapados como \uXXXX) para que sobreviva al transporte.
# ===========================================================================

import io
import os
import shutil
import sys

TARGET = sys.argv[1] if len(sys.argv) > 1 else "vision.py"

MARK = "# [PARCHE-GESTOS]"

# --- 1. imports ------------------------------------------------------------
A1_OLD = "from groq import Groq\n"
A1_NEW = (
    "from groq import Groq\n"
    "\n"
    "" + MARK + " capa de red UDP + motor de gestos de dedos\n"
    "import hermes_net as net\n"
    "import hermes_gesture as gesture\n"
)

# --- 2. arranque de red y gestos ------------------------------------------
A2_OLD = "_render_chat_panel()\n_render_stats_panel()\n"
A2_NEW = (
    "_render_chat_panel()\n"
    "_render_stats_panel()\n"
    "\n"
    "" + MARK + " ------------------------------------------------------\n"
    "# El motor de gestos SONDEA el estado en vez de que el estado lo\n"
    "# empuje. Cualquier camino de salida (incluida una excepcion) devuelve\n"
    "# el estado a idle y la mano relaja sola, sin acordarse de nada.\n"
    "# Prioridad: hablar manda sobre grabar, y grabar sobre pensar.\n"
    "def _gesture_state():\n"
    "    if tts_playing:\n"
    "        return \"speaking\"\n"
    "    if recording:\n"
    "        return \"recording\"\n"
    "    if chat_status in (\"transcribiendo\", \"analizando\", \"pensando\"):\n"
    "        return \"thinking\"\n"
    "    return \"idle\"\n"
    "\n"
    "net.start()\n"
    "gesture.start(_gesture_state)\n"
)

# --- 3. teclas Z / X -------------------------------------------------------
A3_OLD = (
    "        elif key == ord(\"m\"):\n"
    "            menu_visible = not menu_visible\n"
)
A3_NEW = (
    "        elif key == ord(\"m\"):\n"
    "            menu_visible = not menu_visible\n"
    "        " + MARK + " grabacion por teclado\n"
    "        # Activas SIEMPRE, no solo con H apagado: si la deteccion de\n"
    "        # mano esta encendida pero falla, Z/X siguen siendo la salida.\n"
    "        elif key == ord(\"z\"):\n"
    "            if not recording and chat_status == \"\":\n"
    "                start_recording()\n"
    "            else:\n"
    "                print(\">>> Z ignorado (ya grabando o procesando)\")\n"
    "        elif key == ord(\"x\"):\n"
    "            if recording:\n"
    "                stop_and_process()\n"
    "            else:\n"
    "                print(\">>> X ignorado (no hay grabacion activa)\")\n"
)

# --- 4. limpieza -----------------------------------------------------------
A4_OLD = "print(\"Cerrando Hermes Vision...\")\n"
A4_NEW = (
    "print(\"Cerrando Hermes Vision...\")\n"
    "\n"
    "" + MARK + " parar gestos y red\n"
    "# No se rampea a neutro: hermes_net rampa a 0 y el firmware tiene\n"
    "# SAFE_PCT=0. Al salir, las manos se cierran. Es la posicion segura.\n"
    "try:\n"
    "    gesture.stop()\n"
    "    net.stop()\n"
    "except Exception:\n"
    "    pass\n"
)

# --- 5. texto de ayuda en el panel de chat --------------------------------
A5_OLD = "\"Abierta=grabar  Pu\u00f1o=enviar\""
A5_NEW = "\"Abierta/Z=grabar  Pu\u00f1o/X=enviar\""

EDITS = [
    ("imports",        A1_OLD, A1_NEW),
    ("arranque red",   A2_OLD, A2_NEW),
    ("teclas Z/X",     A3_OLD, A3_NEW),
    ("limpieza",       A4_OLD, A4_NEW),
    ("texto de ayuda", A5_OLD, A5_NEW),
]


def main():
    if not os.path.exists(TARGET):
        print("ERROR: no existe %s" % TARGET)
        return 1

    with io.open(TARGET, "r", encoding="utf-8") as f:
        src = f.read()

    if MARK in src:
        print("Ya parcheado (encontre la marca %s). No toco nada." % MARK)
        return 0

    # Verificar TODAS las anclas antes de escribir nada. Un parche a medias
    # es peor que uno que no corre.
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

    shutil.copy2(TARGET, TARGET + ".bak")
    with io.open(TARGET, "w", encoding="utf-8") as f:
        f.write(src)

    print("OK: %s parcheado (%d ediciones). Copia previa en %s.bak"
          % (TARGET, len(EDITS), TARGET))
    return 0


if __name__ == "__main__":
    sys.exit(main())
