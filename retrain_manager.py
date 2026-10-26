import json, os, subprocess, sys, threading, time
import pandas as pd
import config, model_registry

import logging
log = logging.getLogger("ai-server")


def _p(msg):
    print(f"[retrain] {msg}", flush=True)
    log.info("[retrain] %s", msg)

STATE = config.DATA_DIR / "retrain_state.json"
EVERY = int(os.environ.get("AI_RETRAIN_EVERY", "200"))
MASTER = config.MODELS_DIR / "V11" / "data" / "datos_modelo.parquet"
_L = threading.RLock()

# Por defecto NO se reinicia el servidor tras reentrenar: se activa en caliente para no
# cortar peticiones MCP en vuelo. Restaurar el comportamiento antiguo con
# AI_RESTART_AFTER_RETRAIN=1.
RESTART = os.environ.get("AI_RESTART_AFTER_RETRAIN", "0").strip().lower() not in ("0", "false", "no")


def _restart():
    _p("reiniciando servidor para cargar la nueva versión")
    time.sleep(2)  # deja terminar la petición en curso
    cmd = [sys.executable, str(config.BASE_DIR / "main.py")]
    if os.name == "nt":
        subprocess.Popen(cmd, cwd=config.BASE_DIR, creationflags=subprocess.CREATE_NEW_CONSOLE)
        os._exit(0)
    os.execv(sys.executable, cmd)


def load():
    try: return json.loads(STATE.read_text(encoding="utf-8"))
    except Exception: return {}


def save(s):
    t = STATE.with_suffix(".tmp"); t.write_text(json.dumps(s), encoding="utf-8"); os.replace(t, STATE)


def update(**kw):
    with _L:
        s = load(); s.update(kw); save(s); return s


def n_rows():
    return len(pd.read_parquet(MASTER, columns=["UNICODE"])) if MASTER.exists() else 0


def init_baseline():
    with _L:
        if "rows_at_last_train" not in load(): update(rows_at_last_train=n_rows())


def _alive(pid):
    if not pid: return False
    if os.name == "nt":
        import ctypes
        k = ctypes.windll.kernel32
        h = k.OpenProcess(0x1000, False, int(pid))
        if not h: return False
        c = ctypes.c_ulong(); k.GetExitCodeProcess(h, ctypes.byref(c)); k.CloseHandle(h)
        return c.value == 259
    try: os.kill(int(pid), 0); return True
    except OSError: return False


def _running(s):
    if s.get("status") != "running": return False
    if _alive(s.get("pid")): return True
    _p(f"proceso {s.get('pid')} muerto con status=running -> error")
    update(status="error", last_error="el proceso de reentreno terminó inesperadamente")
    return False


def status():
    s = load()
    return {"pending": max(0, n_rows() - s.get("rows_at_last_train", 0)), "threshold": EVERY,
            "status": s.get("status", "idle"), "last_error": s.get("last_error"),
            "last_version": s.get("new_version"), "promoted": s.get("promoted")}


def maybe_start(force=False):
    with _L:
        s = load()
        n, base = n_rows(), s.get("rows_at_last_train", 0)
        _p(f"check: filas={n} base={base} pendientes={n - base} umbral={EVERY} "
           f"status={s.get('status')} pid={s.get('pid')} force={force}")
        if _running(s):
            _p("no lanzo: ya hay un reentreno en curso"); return False
        if not force and n - base < EVERY:
            _p("no lanzo: pendientes < umbral"); return False
        (config.DATA_DIR / "logs").mkdir(exist_ok=True)
        lf = config.DATA_DIR / "logs" / "retrain.log"
        out = open(lf, "a", encoding="utf-8")
        p = subprocess.Popen([sys.executable, str(config.BASE_DIR / "retrain.py")],
                             cwd=config.BASE_DIR, stdout=out, stderr=subprocess.STDOUT)
        update(status="running", pid=p.pid, started=time.time(), last_error=None)
        _p(f"LANZADO pid={p.pid} log={lf}")
        return True


def finalize():
    with _L:
        s = load()
        _running(s)
        s = load()
        if s.get("status") == "done" and not s.get("activated"):
            update(activated=True)  # antes de reiniciar, para no entrar en bucle
            _p(f"{s['new_version']} lista (promoted={s.get('promoted')})")
            if RESTART:
                _restart()
            else:
                model_registry.activate(s["new_version"])
                if s.get("promoted"):
                    config.MODEL_NAME = s["new_version"]


def start_watcher():
    def loop():
        while True:
            try: finalize()
            except Exception: 
                log.exception("[retrain] watcher")
                import traceback; traceback.print_exc()
            time.sleep(10)
    threading.Thread(target=loop, daemon=True).start()