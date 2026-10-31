#!/usr/bin/env python3
# ===========================================================================
# HERMES PRESENTA - saludo protocolario (tecla T)
#
# POR QUE UN MODULO Y NO UNA FUNCION MAS EN hermes_vision
#   speak() de hermes_vision es fire-and-forget: lanza un hilo y vuelve.
#   Llamarla cinco veces seguidas arranca cinco aplay simultaneos y las
#   frases se pisan. Un discurso necesita COLA, no paralelismo. Este modulo
#   sintetiza y reproduce en un solo hilo, en orden, esperando a que cada
#   aplay termine antes de empezar el siguiente.
#
#   Ademas recibe voice y audio_out por parametro en vez de importarlos:
#   hermes_vision ya importa este modulo, asi que importarlo de vuelta seria
#   un ciclo.
#
# ACENTOS
#   El archivo es ASCII puro para sobrevivir al transporte, pero Piper
#   fonemiza con espeak-ng y la tilde SI cambia la pronunciacion: sin ella
#   "aqui" se lee como esdrujula. Por eso el texto va con escapes \uXXXX.
#   Si preferis editarlo comodo, crea un archivo presentacion.txt en UTF-8
#   al lado de este modulo: si existe, tiene prioridad sobre el texto de
#   abajo. Una linea por frase, lineas vacias y las que empiezan con # se
#   ignoran. El marcador {saludo} se reemplaza segun la hora.
# ===========================================================================

import io
import os
import subprocess
import tempfile
import threading
import time
import wave

TXT_FILE  = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "presentacion.txt")
PAUSA_S   = 0.35        # silencio entre frases

# Cada elemento es una frase que se reproduce completa antes de la siguiente.
# Frases cortas a proposito: dan respiracion natural y le dan al motor de
# gestos tiempo de articular la onda en vez de un bloque continuo.
LINEAS = [

    "Buenos dias, Doctora Karin Herrera, Bienvenida al Stand de",
    "Creabot, es un placer tenerla con nosotros, en que puedo ayudarla",
#    "{saludo} a todas y a todos. Mi nombre es Hermes, "
#    "y soy un humanoide creado por Creabot, aqu\u00ed en Guatemala.",

#    "Es un honor enorme darle la bienvenida a la doctora Karin Herrera, "
#    "vicepresidenta de la Rep\u00fablica de Guatemala.",

#    "Y por supuesto, bienvenidas y bienvenidos todos los invitados "
#    "que hoy nos acompa\u00f1an. Gracias por estar aqu\u00ed.",

#    "Todo lo que ven en m\u00ed, mis manos, mi voz y mis ojos, "
#    "lo hicieron manos guatemaltecas. Estoy muy contento de conocerlos.",
]

# --- Estado interno -------------------------------------------------------
_running = False
_thread  = None
_proc    = None
_lock    = threading.Lock()


def _saludo():
    h = time.localtime().tm_hour
    if h < 12:
        return "Muy buenos d\u00edas"
    if h < 19:
        return "Muy buenas tardes"
    return "Muy buenas noches"


def lineas():
    """Texto final, ya resuelto. Lee presentacion.txt si existe."""
    src = LINEAS
    if os.path.exists(TXT_FILE):
        try:
            with io.open(TXT_FILE, "r", encoding="utf-8") as f:
                ext = [ln.strip() for ln in f
                       if ln.strip() and not ln.strip().startswith("#")]
            if ext:
                src = ext
        except Exception as e:
            print("[presenta] no pude leer %s: %s" % (TXT_FILE, e))
    s = _saludo()
    return [ln.replace("{saludo}", s) for ln in src]


def _worker(voice, audio_out, on_line, on_done):
    global _running, _proc
    try:
        for texto in lineas():
            if not _running:
                break
            if on_line:
                try:
                    on_line(texto)
                except Exception:
                    pass

            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.close()
            try:
                with wave.open(tmp.name, "wb") as wav_file:
                    voice.synthesize_wav(texto, wav_file)
                if not _running:
                    break
                # Popen + wait, no run(): asi stop() puede matar el proceso
                # a mitad de frase en vez de esperar a que termine sola.
                p = subprocess.Popen(
                    ["aplay", "-D", audio_out, tmp.name],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                with _lock:
                    _proc = p
                p.wait()
                with _lock:
                    _proc = None
            except Exception as e:
                # Fallo AISLADO POR FRASE. Si el audio de una linea revienta,
                # el discurso sigue con la siguiente: en vivo, perder una
                # frase es un tropiezo, perder el saludo entero es el fin.
                print("[presenta] frase omitida: %s" % e)
                with _lock:
                    _proc = None
            finally:
                try:
                    os.unlink(tmp.name)
                except Exception:
                    pass

            if _running:
                time.sleep(PAUSA_S)
    except Exception as e:
        print("[presenta] error: %s" % e)
    finally:
        _running = False
        with _lock:
            _proc = None
        if on_done:
            try:
                on_done()
            except Exception:
                pass
        print("[presenta] presentacion terminada")


# ===========================================================================
# API publica
# ===========================================================================
def start(voice, audio_out, on_line=None, on_done=None):
    """
    Arranca la presentacion. Devuelve True si arranco, False si ya corria.
    on_line(texto) se llama antes de reproducir cada frase (para el chat).
    """
    global _running, _thread
    if _running:
        return False
    _running = True
    _thread = threading.Thread(target=_worker,
                               args=(voice, audio_out, on_line, on_done),
                               daemon=True)
    _thread.start()
    print("[presenta] iniciando presentacion (%d frases)" % len(lineas()))
    return True


def stop():
    """Corta la presentacion. Mata el aplay en curso si lo hay."""
    global _running
    _running = False
    with _lock:
        p = _proc
    if p is not None:
        try:
            p.terminate()
        except Exception:
            pass


def is_running():
    return _running
