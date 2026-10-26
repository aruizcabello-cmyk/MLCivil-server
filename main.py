"""Servidor HTTP central de inferencia y comprobacion estructural.

Cada modelo es un paquete autocontenido en `models/<NAME>/` (lo instala el Makefile).
Se cargan todos los modelos instalados y conviven en memoria; cada peticion se enruta al
modelo indicado. Los resultados son SIEMPRE las 63 columnas de CODAME (`Tabla_MOMENT_M1`).
Sobre esa tabla (editada por el revisor) se puede ejecutar la comprobacion AISC 360-10.

Endpoints:
  GET  /health                  Estado del servidor
  GET  /models                  Modelos instalados y cargados
  GET  /model                   Contrato del modelo por defecto
  GET  /schema?model=...        Inputs del modelo + columnas de resultado CODAME
  GET  /profiles?names=...      Geometria de secciones (para el visualizador 2D)
  POST /model/activate          Carga/recarga un modelo por nombre
  POST /predict                 {"model": "...", "rows": [...]} -> columnas CODAME + comprobacion ("checks")
  POST /scenarios               {"row": {...}} -> E1 (entrada), E2/E3 = max. carga que cumple sin/con cantonera
  POST /check                   Comprobacion AISC 360-10 sobre filas CODAME
  POST /validate                Guarda filas validadas (sin reentreno)
  GET  /dataset/summary         Resumen del dataset acumulado
  MCP  /mcp                     Servidor MCP para integracion con Microsoft Copilot
"""

import asyncio
import contextlib
import json
import logging
import os
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import pandas as pd

import catalogo
import config
import model_registry
import escenarios
import predictor
import refinar
import storage

import dataset_valid, retrain_manager

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

# Logs: un archivo por sesion (cada arranque del servidor), sin salida por pantalla.
_LOG_DIR = config.DATA_DIR / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = os.environ.setdefault(
    "AI_LOG_FILE", str(_LOG_DIR / f"server_{datetime.now():%Y-%m-%d_%H-%M-%S}.log")
)
_handlers = [logging.FileHandler(LOG_FILE, encoding="utf-8")]
if os.environ.get("AI_LOG_CONSOLE", "").strip().lower() in ("1", "true", "yes"):
    _handlers.append(logging.StreamHandler())
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=_handlers,
)
log = logging.getLogger("ai-server")


# =============================================================================
# Runner de trabajo pesado (no bloquear el event loop de FastAPI/MCP)
# =============================================================================
class _Runner:
    """Ejecuta funciones sincronas en una hebra dedicada.

    Si una peticion supera su timeout, se reemplaza el executor para que las siguientes
    no queden detras de una tarea colgada (la hebra antigua puede seguir viva; se deja
    terminar y se descarta).
    """

    def __init__(self, name: str, max_workers: int = 1) -> None:
        self._name = name
        self._max_workers = max(1, int(max_workers))
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=self._max_workers, thread_name_prefix=name
        )

    def submit(self, fn, *args, **kwargs):
        with self._lock:
            return self._executor.submit(fn, *args, **kwargs)

    def reset(self) -> None:
        with self._lock:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = ThreadPoolExecutor(
                max_workers=self._max_workers, thread_name_prefix=self._name
            )

    def shutdown(self) -> None:
        with self._lock:
            self._executor.shutdown(wait=False, cancel_futures=True)


_PREDICT_RUNNER = _Runner("mcp-predict")
_CHECK_RUNNER = _Runner("mcp-check")
_SCENARIO_RUNNER = _Runner("scenarios", max_workers=config.SCENARIO_WORKERS)


class ServerBusyError(RuntimeError):
    """La cola de escenarios esta llena o se agoto el tiempo de espera en cola."""


class _ScenarioGate:
    """Control de admision para escenarios: limita concurrencia y tamano de cola.

    Si hay `max_workers` escenarios ejecutandose y `max_queue` esperando, rechaza de
    inmediato con `ServerBusyError`. Si un escenario en cola supera `queue_timeout` sin
    conseguir worker, tambien falla con `ServerBusyError` antes de agotar el timeout del
    cliente MCP (`AI_CLIENT_TIMEOUT_S`).
    """

    def __init__(self, max_workers: int, max_queue: int, queue_timeout: float) -> None:
        self.max_workers = max(1, int(max_workers))
        self.max_queue = max(0, int(max_queue))
        self.queue_timeout = max(0.0, float(queue_timeout))
        self._lock = threading.Lock()
        self._active = 0
        self._waiting = 0
        self._waiters: Deque[asyncio.Future] = deque()

    def status(self) -> Dict[str, int]:
        with self._lock:
            return {
                "active": self._active,
                "waiting": self._waiting,
                "max_workers": self.max_workers,
                "max_queue": self.max_queue,
            }

    async def acquire(self) -> float:
        loop = asyncio.get_running_loop()
        with self._lock:
            if self._active < self.max_workers and not self._waiters:
                self._active += 1
                return 0.0
            if self._waiting >= self.max_queue:
                raise ServerBusyError(
                    f"Servidor ocupado ({self._active}/{self.max_workers} en ejecucion, "
                    f"{self._waiting}/{self.max_queue} en cola). "
                    f"Maximo {self.max_workers} peticiones concurrentes; reintenta en unos segundos."
                )
            self._waiting += 1
            fut: asyncio.Future = loop.create_future()
            self._waiters.append(fut)

        t0 = time.monotonic()
        try:
            if self.queue_timeout > 0:
                # `asyncio.wait` no cancela el future al expirar: evita liberar por error
                # una plaza que no se habia concedido.
                done, _ = await asyncio.wait({fut}, timeout=self.queue_timeout)
                if not done:
                    with self._lock:
                        if not fut.done():
                            try:
                                self._waiters.remove(fut)
                            except ValueError:
                                pass
                            self._waiting = max(0, self._waiting - 1)
                            raise ServerBusyError(
                                f"Servidor ocupado tras esperar {self.queue_timeout:g}s en cola "
                                f"({self._active}/{self.max_workers} en ejecucion). "
                                f"Reintenta en unos segundos."
                            ) from None
                    # Nos concedieron una plaza justo al expirar: devolverla.
                    self.release()
                    raise ServerBusyError(
                        f"Servidor ocupado tras esperar {self.queue_timeout:g}s en cola. "
                        f"Reintenta en unos segundos."
                    ) from None
            else:
                await fut
            return time.monotonic() - t0
        except BaseException as exc:
            if isinstance(exc, ServerBusyError):
                raise
            with self._lock:
                if fut.done() and not fut.cancelled():
                    self._active = max(0, self._active - 1)
                    self._wake_next_locked()
                else:
                    try:
                        self._waiters.remove(fut)
                    except ValueError:
                        pass
                    self._waiting = max(0, self._waiting - 1)
            raise

    def release(self) -> None:
        with self._lock:
            self._active = max(0, self._active - 1)
            self._wake_next_locked()

    def _wake_next_locked(self) -> None:
        while self._waiters and self._active < self.max_workers:
            fut = self._waiters.popleft()
            self._waiting = max(0, self._waiting - 1)
            if not fut.done():
                self._active += 1
                fut.get_loop().call_soon_threadsafe(fut.set_result, True)
                return


_SCENARIO_GATE = _ScenarioGate(
    max_workers=config.SCENARIO_WORKERS,
    max_queue=config.SCENARIO_MAX_QUEUE,
    queue_timeout=config.SCENARIO_QUEUE_TIMEOUT_S,
)


async def _run_sync(runner: _Runner, fn, args: tuple, timeout: Optional[float],
                    label: str, kwargs: Optional[Dict[str, Any]] = None):
    """Ejecuta `fn(*args, **kwargs)` fuera del event loop con timeout y reinicio del runner."""
    future = runner.submit(fn, *args, **(kwargs or {}))
    if timeout is None or timeout <= 0:
        return await asyncio.wrap_future(future)
    try:
        return await asyncio.wait_for(asyncio.wrap_future(future), timeout=timeout)
    except asyncio.TimeoutError:
        runner.reset()
        log.error("%s: timeout tras %.0fs; runner reiniciado", label, timeout)
        raise TimeoutError(f"{label}: tiempo de espera agotado ({timeout:.0f}s)") from None


async def _run_scenario(name: str, row: Dict[str, Any], label: str, **kwargs) -> tuple[Dict[str, Any], float]:
    """Pasa por el control de admision y ejecuta `escenarios.run` dentro del presupuesto."""
    queue_wait = await _SCENARIO_GATE.acquire()
    try:
        result = await _run_sync(
            _SCENARIO_RUNNER,
            escenarios.run,
            (name, row),
            config.SCENARIO_TIMEOUT_S,
            label,
            {"timeout_s": config.SCENARIO_TIMEOUT_S, **kwargs},
        )
        return result, queue_wait
    finally:
        _SCENARIO_GATE.release()


def _parse_mcp_row(row: Any) -> Dict[str, Any]:
    """Acepta objeto o string JSON (Copilot a veces manda las filas serializadas)."""
    if isinstance(row, dict):
        return row
    if isinstance(row, str):
        for _ in range(2):  # tolera doble serializacion
            try:
                row = json.loads(row)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"fila no es un JSON valido: {exc}") from exc
            if isinstance(row, dict):
                return row
    raise ValueError("cada fila debe ser un objeto o un string JSON con la conexion")


def _parse_mcp_rows(rows: Any) -> List[Dict[str, Any]]:
    """Normaliza `rows` a lista de conexiones (lista, objeto unico o string JSON)."""
    if isinstance(rows, str):
        for _ in range(2):  # tolera doble serializacion
            try:
                rows = json.loads(rows)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"rows no es un JSON valido: {exc}") from exc
            if not isinstance(rows, str):
                break
    if isinstance(rows, dict):
        rows = [rows]
    if not isinstance(rows, (list, tuple)):
        raise ValueError("rows debe ser una lista de conexiones")
    if not rows:
        raise ValueError("rows esta vacio")
    return [_parse_mcp_row(r) for r in rows]


def _uid(row: Dict[str, Any]) -> str:
    return str((row or {}).get("UNICODE") or (row or {}).get("Unicode") or "?")


def _compact_result(name: str, result: Dict[str, Any]) -> Dict[str, Any]:
    """Quita `rows_raw`/`historial` y deja los checks compactos para el tunel MCP."""
    compact_checks = [
        {
            k: c.get(k)
            for k in ("Unicode", "Escenario", "cumple", "ratio_pesimo", "descripcion_ratio_pesimo")
            if k in c
        }
        for c in (result.get("checks") or [])
    ]
    out = {k: v for k, v in result.items() if k not in ("rows_raw", "historial")}
    out["checks"] = compact_checks
    return {"model_name": name, "status": "OK", **out}


def _run_batch_sync(name: str, rows: List[Dict[str, Any]], options: Dict[str, Any],
                    timeout_s: float) -> List[Dict[str, Any]]:
    """Ejecuta filas secuencialmente con un unico presupuesto total (sin gate)."""
    deadline = time.monotonic() + max(1.0, timeout_s)
    out: List[Dict[str, Any]] = []
    for row in rows:
        uid = _uid(row)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            out.append({
                "unicode": uid, "status": "TIMEOUT_SERVIDOR", "http_status": 504,
                "mensaje": "sin tiempo restante para ejecutar esta conexion",
            })
            continue
        try:
            result = escenarios.run(name, row, timeout_s=remaining, **options)
            out.append({
                "unicode": uid, "status": "OK", "resultado": _compact_result(name, result),
            })
        except TimeoutError as exc:
            out.append({
                "unicode": uid, "status": "TIMEOUT_SERVIDOR", "http_status": 504,
                "mensaje": str(exc),
            })
        except predictor.InputContractError as exc:
            out.append({
                "unicode": uid, "status": "ERROR_ENTRADA", "http_status": 422,
                "mensaje": str(exc),
            })
        except ValueError as exc:
            out.append({
                "unicode": uid, "status": "ERROR_ENTRADA", "http_status": 422,
                "mensaje": str(exc),
            })
        except KeyError as exc:
            out.append({
                "unicode": uid, "status": "ERROR_MODELO", "http_status": 404,
                "mensaje": str(exc),
            })
        except Exception as exc:  # noqa: BLE001
            log.exception("[MCP] error generando escenarios (unicode=%s)", uid)
            out.append({
                "unicode": uid, "status": "ERROR_SERVIDOR", "http_status": 500,
                "mensaje": str(exc),
            })
    return out


async def _run_scenario_batch(name: str, rows: List[Dict[str, Any]], label: str,
                              **kwargs) -> tuple[List[Dict[str, Any]], float]:
    queue_wait = await _SCENARIO_GATE.acquire()
    try:
        results = await _run_sync(
            _SCENARIO_RUNNER,
            _run_batch_sync,
            (name, rows, kwargs, config.SCENARIO_TIMEOUT_S),
            config.SCENARIO_TIMEOUT_S + 2.0,
            label,
        )
        return results, queue_wait
    finally:
        _SCENARIO_GATE.release()


def _mcp_status(model: str, status: str, http_status: int, message: str,
                retry_after_s: Optional[int] = None) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "model_name": model,
        "status": status,
        "http_status": http_status,
        "mensaje": message,
    }
    if retry_after_s is not None:
        out["retry_after_s"] = retry_after_s
    return out


# =============================================================================
# Servidor MCP (Model Context Protocol) para Microsoft Copilot
# =============================================================================
mcp = FastMCP(
    "ServidorConexionesEstructurales",
    instructions=(
        f"Servidor de diseno y comprobacion AISC 360-10 de conexiones estructurales. "
        f"disenar_escenarios_carga acepta una lista `rows` de hasta "
        f"{config.SCENARIO_BATCH_ROWS} conexiones por llamada (se procesan las "
        f"{config.SCENARIO_BATCH_ROWS} primeras y el resto se descarta). Usa SIEMPRE una sola "
        f"llamada con varias filas para varias conexiones; no lances una llamada por conexion. "
        f"Como maximo {config.SCENARIO_WORKERS} llamadas en paralelo (mas "
        f"{config.SCENARIO_MAX_QUEUE} en cola). Si recibes SERVIDOR_OCUPADO, reintenta las "
        f"pendientes en el siguiente turno."
    ),
    stateless_http=True,
    json_response=config.MCP_JSON_RESPONSE,
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=False  # o allowed_hosts=["0tpn3dld-8080.uks1.devtunnels.ms", "localhost:*"]
    ),
)

@mcp.tool()
async def predecir_conexion(rows: List[Dict[str, Any]], model: str = "") -> Dict[str, Any]:
    """Pre-diseña conexiones y devuelve las 63 columnas CODAME con sus comprobaciones.

    Args:
        rows: Lista de diccionarios con las propiedades geométricas y de carga.
        model: Nombre opcional del modelo a usar (si falta, usa el modelo configurado por defecto).
    """
    name = model or config.MODEL_NAME
    log.info("[MCP] /predict: modelo=%s, %d filas", name, len(rows))
    predictions, checks = await _run_sync(
        _PREDICT_RUNNER,
        predictor.predict_and_check,
        (name, rows),
        config.PREDICT_TIMEOUT_S,
        "predecir_conexion",
    )
    return {"model_name": name, "rows": predictions, "checks": checks}


@mcp.tool()
async def comprobar_aisc(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Ejecuta la comprobación estructural AISC 360-10 sobre filas CODAME.

    Args:
        rows: Lista de filas con los parámetros del diseño CODAME a verificar.
    """
    log.info("[MCP] /check: comprobando %d filas", len(rows))
    results = await _run_sync(
        _CHECK_RUNNER,
        predictor.check_rows,
        (rows,),
        config.CHECK_TIMEOUT_S,
        "comprobar_aisc",
    )
    return {"status": "ok", "rows": results}


def _consultar_perfiles(names: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for raw in names.split(","):
        name = raw.strip()
        if not name:
            continue
        geom = catalogo.get_profile(name)
        if geom is None:
            result[name] = None
        else:
            def g(k):
                if k not in geom:
                    return None
                v = geom.get(k)
                return None if pd.isna(v) else v
            bf = g("bf_mm")
            if bf in (None, 0, 0.0):
                bf = g("b_mm")
            result[name] = {
                "type": g("type"), "d_mm": g("d_mm"), "bf_mm": bf,
                "tw_mm": g("tw_mm"), "tf_mm": g("tf_mm"), "r1_mm": g("r1_mm"),
            }
    return {"profiles": result}


@mcp.tool()
async def consultar_perfiles(names: str) -> Dict[str, Any]:
    """Consulta la geometría de secciones/perfiles de acero por nombre para el visualizador.

    Args:
        names: Nombres de los perfiles separados por comas (ej. "HEB200, IPE300").
    """
    return await asyncio.to_thread(_consultar_perfiles, names)


@mcp.tool()
async def listar_modelos() -> Dict[str, Any]:
    """Consulta los modelos de IA instalados y cargados en el servidor."""
    return {
        "available_models": model_registry.list_models(),
        "loaded_models": model_registry.loaded_models(),
        "default_model": config.MODEL_NAME,
    }


# =============================================================================
# Escenarios de carga (MCP)
# =============================================================================
def _batch_summary(results: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Resumen compacto por conexion y lista de errores (para el contexto MCP)."""
    resumen: List[Dict[str, Any]] = []
    errores: List[Dict[str, Any]] = []
    for item in results:
        uid = item.get("unicode", "?")
        status = item.get("status", "ERROR")
        if status != "OK":
            err = {
                "unicode": uid, "status": status,
                "http_status": item.get("http_status"), "mensaje": item.get("mensaje"),
            }
            errores.append(err)
            resumen.append({
                "unicode": uid, "status": status, "E1": "-", "E2": "-", "E3": "-",
                "ratio_pesimo": None,
            })
            continue
        result = item.get("resultado") or {}
        escenarios = {e.get("id"): e for e in (result.get("escenarios") or [])}
        row: Dict[str, Any] = {"unicode": uid, "status": "OK", "parada": result.get("parada")}
        peor = 0.0
        for clave in ("E1", "E2", "E3"):
            esc = escenarios.get(clave) or {}
            if not esc.get("encontrado"):
                row[clave] = "NO ENCONTRADO"
            elif esc.get("cumple"):
                row[clave] = "OK"
            else:
                row[clave] = "NO CUMPLE"
            try:
                peor = max(peor, float(esc.get("ratio_pesimo") or 0.0))
            except (TypeError, ValueError):
                pass
        row["ratio_pesimo"] = round(peor, 4) if peor else None
        resumen.append(row)
    return resumen, errores


def _summary_markdown(resumen: List[Dict[str, Any]]) -> str:
    lineas = [
        "| UNICODE | Estado | E1 | E2 | E3 | Ratio pésimo |",
        "|---|---|---|---|---|---|",
    ]
    for r in resumen:
        lineas.append(
            "| {u} | {s} | {e1} | {e2} | {e3} | {rp} |".format(
                u=r.get("unicode", ""), s=r.get("status", ""),
                e1=r.get("E1", ""), e2=r.get("E2", ""), e3=r.get("E3", ""),
                rp="" if r.get("ratio_pesimo") is None else r.get("ratio_pesimo"),
            )
        )
    return "\n".join(lineas)


@mcp.tool()
async def disenar_escenarios_carga(
    rows: Any,
    model: str = "",
    incremento: float = 0.05,
    max_pasos: int = 40,
) -> Dict[str, Any]:
    """Disena varias conexiones en una sola llamada (maximo configurable, 50 por defecto).

    Pasa SIEMPRE una lista en `rows` (no una llamada por conexion). Se procesan como maximo
    `AI_SCENARIO_BATCH_ROWS` conexiones; el resto se descarta y se indica en
    `n_descartadas`/`descartadas`. No lances una llamada por conexion.

    Devuelve el detalle en `lote` (una entrada por conexion, con `resultado.rows`,
    `resultado.checks` compactos, `escenarios` y `tabla_markdown`) y omite `historial`.
    Tambien incluye `resumen`, `errores`, contadores y una `tabla_markdown` resumen.

      E1: diseno de entrada (paso 0, sin tocar cargas).
      E2: mayor carga que aguanta SIN cantonera.
      E3: mayor carga que aguanta CON cantonera.
    E2 y E3 priorizan siempre el diseno que cumple la comprobacion AISC 360-10; si ninguno
    cumple, devuelven el de mayor carga (con aviso). Si no hay ninguna prediccion sin (E2) o con
    (E3) cantonera, el escenario sale vacio con `encontrado=false` y su motivo.

    Args:
        rows: Lista de conexiones (cada una con UNICODE, BEAM, COLUMN, ratio_moment,
            ratio_shear, ratio_axial y opcionalmente Normativas, grado_acero, clase_tornillo).
            Tambien acepta un unico objeto o un string JSON con la lista.
        model: Nombre opcional del modelo (por defecto, el configurado).
        incremento: Aumento de carga por paso (0.05 = 5 %).
        max_pasos: Maximo de pasos de aumento a explorar.
    """
    name = model or config.MODEL_NAME
    try:
        all_rows = _parse_mcp_rows(rows)
    except ValueError as exc:
        log.warning("[MCP] escenarios entrada invalida: %s", exc)
        return _mcp_status(name, "ERROR_ENTRADA", 422, str(exc))

    max_rows = config.SCENARIO_BATCH_ROWS
    selected = all_rows[:max_rows]
    discarded = all_rows[max_rows:]
    discarded_uids = [_uid(r) for r in discarded]
    log.info(
        "[MCP] escenarios: modelo=%s, filas=%d, procesadas=%d, descartadas=%d",
        name, len(all_rows), len(selected), len(discarded),
    )

    try:
        results, queue_wait = await _run_scenario_batch(
            name, selected, "disenar_escenarios_carga",
            incremento=incremento, max_pasos=max_pasos,
        )
    except ServerBusyError as exc:
        log.warning("[MCP] escenarios ocupado: %s", exc)
        return _mcp_status(name, "SERVIDOR_OCUPADO", 503, str(exc), 5)
    except TimeoutError as exc:
        log.error("[MCP] escenarios timeout: %s", exc)
        return _mcp_status(name, "TIMEOUT_SERVIDOR", 504, str(exc), 5)
    except Exception as exc:  # noqa: BLE001
        log.exception("[MCP] error generando escenarios")
        return _mcp_status(name, "ERROR_SERVIDOR", 500, str(exc))

    has_errors = any(r.get("status") != "OK" for r in results)
    if has_errors:
        status = "CON_ERRORES"
    elif discarded:
        status = "CON_DESCARTES"
    else:
        status = "OK"

    resumen, errores = _batch_summary(results)
    tabla = _summary_markdown(resumen)
    log.info(
        "[MCP] escenarios: %s (procesadas=%d, descartadas=%d, cola=%.1fs)",
        status, len(selected), len(discarded), queue_wait,
    )
    return {
        "model_name": name,
        "status": status,
        "n_solicitadas": len(all_rows),
        "n_procesadas": len(selected),
        "n_descartadas": len(discarded),
        "descartadas": discarded_uids,
        "lote": results,
        "resumen": resumen,
        "errores": errores,
        "tabla_markdown": tabla,
    }


# =============================================================================
# App FastAPI & Lifespan
# =============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Carga de modelos, batcher de inferencia, pool de refinado y watcher de reentreno
    try:
        model_registry.activate_configured()
        log.info("Modelos cargados: %s", model_registry.loaded_models())
    except Exception as exc:  # noqa: BLE001
        log.error("No se pudo cargar ningun modelo: %s", exc)
    try:
        predictor.start_inference()
    except Exception:  # noqa: BLE001
        log.exception("No se pudo iniciar el batcher de inferencia")
    try:
        refinar.start_backend()
        retrain_manager.start_watcher()
    except Exception as exc:  # noqa: BLE001
        log.error("No se pudo iniciar el refinado/reentreno: %s", exc)

    try:
        # Administrador de sesión de MCP sobre HTTP/SSE
        async with contextlib.AsyncExitStack() as stack:
            await stack.enter_async_context(mcp.session_manager.run())
            yield
    finally:
        _SCENARIO_RUNNER.shutdown()
        try:
            predictor.shutdown_inference()
        except Exception:  # noqa: BLE001
            log.exception("Error cerrando el batcher de inferencia")
        try:
            refinar.shutdown_backend()
        except Exception:  # noqa: BLE001
            log.exception("Error cerrando el backend de refinado")
        _PREDICT_RUNNER.shutdown()
        _CHECK_RUNNER.shutdown()


app = FastAPI(
    title="Servidor de inferencia y comprobacion - conexiones",
    version="1.0.0",
    description=(
        "Servidor de pre-diseno por IA y comprobacion estructural AISC 360-10. Los modelos "
        "viven en `models/<NAME>/` y conviven en memoria; cada peticion se enruta al modelo "
        "indicado en `model`. El contrato de entrada lo define su `core/adapter.py`; si una "
        "peticion no lo cumple se devuelve **HTTP 422**. El resultado son SIEMPRE las 63 "
        "columnas de CODAME (`Tabla_MOMENT_M1`), con independencia del modelo."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

mcp_app = mcp.streamable_http_app()
app.router.routes.extend(mcp_app.routes)

# =============================================================================
# Modelos Pydantic & Handlers
# =============================================================================
class RowsIn(BaseModel):
    model: str | None = Field(
        default=None, description="Modelo a usar (carpeta en models/). Si falta, el por defecto."
    )
    rows: List[Dict[str, Any]] = Field(
        ..., description="Filas de la tabla (inputs del modelo o filas CODAME, segun endpoint)."
    )


class ActivateIn(BaseModel):
    name: str = Field(..., description="Nombre de la carpeta del modelo, p. ej. 'V11'.")


@app.exception_handler(predictor.InputContractError)
async def _contract_error_handler(request: Request, exc: predictor.InputContractError):
    return JSONResponse(
        status_code=422,
        content={
            "detail": str(exc),
            "model": exc.model,
            "missing": exc.missing,
            "expected": exc.expected,
        },
    )


# =============================================================================
# Endpoints HTTP REST
# =============================================================================
@app.get("/health", tags=["estado"], summary="Estado del servidor")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "default_model": config.MODEL_NAME,
        **predictor.model_status(),
        "dataset_rows": storage.summary()["total_rows"],
        "client_timeout_s": config.CLIENT_TIMEOUT_S,
        "scenarios": _SCENARIO_GATE.status(),
    }


@app.get("/models", tags=["modelos"], summary="Lista los modelos instalados y cargados")
def list_models() -> Dict[str, Any]:
    return {
        "available_models": model_registry.list_models(),
        "loaded_models": model_registry.loaded_models(),
        "default_model": config.MODEL_NAME,
    }


@app.get(
    "/profiles",
    tags=["catalogo"],
    summary="Geometria de secciones por nombre (para el visualizador 2D)",
)
def profiles(names: str = Query(..., description="Nombres de perfil separados por comas")):
    """Devuelve {nombre: {cota: valor}} o null por cada perfil desconocido."""
    return _consultar_perfiles(names)


@app.get("/model", tags=["modelos"], summary="Contrato del modelo por defecto")
def default_model() -> Dict[str, Any]:
    try:
        bundle = model_registry.get(config.MODEL_NAME)
    except KeyError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return {"name": bundle["name"], **bundle["contract"]}


@app.get(
    "/schema",
    tags=["modelos"],
    summary="Inputs del modelo y columnas de resultado",
    response_description="Campos de entrada (tipo/descripcion/opciones) y las 63 columnas CODAME.",
    responses={404: {"description": "El modelo no esta disponible"}},
)
def schema(model: str | None = Query(default=None, description="Modelo (por defecto, el configurado)")):
    """Esquema de entrada del modelo indicado y columnas de resultado (fijas)."""
    name = model or config.MODEL_NAME
    try:
        return predictor.schema(name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.post("/model/activate", tags=["modelos"], summary="Carga/recarga un modelo")
def activate_model(payload: ActivateIn):
    try:
        bundle = model_registry.activate(payload.name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        log.exception("Error activando modelo %s", payload.name)
        raise HTTPException(status_code=500, detail=f"Error activando modelo: {exc}")
    log.info("Modelo cargado: %s", bundle["name"])
    return {"status": "ok", "name": bundle["name"], **bundle["contract"]}


@app.post(
    "/predict",
    tags=["inferencia"],
    summary="Pre-diseña y devuelve las 63 columnas CODAME",
    response_description="Una fila por entrada con las columnas de CODAME.",
    responses={
        422: {"description": "La peticion no cumple el contrato del modelo."},
        404: {"description": "El modelo indicado no esta disponible"},
    },
)
def predict(payload: RowsIn):
    if not payload.rows:
        raise HTTPException(status_code=400, detail="La peticion no trae filas (rows vacio)")
    name = payload.model or config.MODEL_NAME
    log.info("/predict: modelo=%s, %d filas", name, len(payload.rows))
    try:
        predictions, checks, rows_raw = predictor.predict_full(name, payload.rows)
    except predictor.InputContractError:
        raise
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        log.exception("Error prediciendo")
        raise HTTPException(status_code=500, detail=f"Error al predecir: {exc}")
    log.info("/predict: %d filas devueltas", len(predictions))
    return {"model_name": name, "rows": predictions, "checks": checks, "rows_raw": rows_raw}


@app.post(
    "/check",
    tags=["comprobacion"],
    summary="Ejecuta la comprobacion AISC 360-10 sobre filas CODAME",
    response_description="Una fila por entrada con cumple, ratio pesimo y detalle.",
)
@app.post("/comprobar", tags=["comprobacion"], include_in_schema=False)
def check(payload: RowsIn):
    if not payload.rows:
        raise HTTPException(status_code=400, detail="La peticion no trae filas (rows vacio)")
    log.info("/check: comprobando %d filas", len(payload.rows))
    try:
        results = predictor.check_rows(payload.rows)
    except Exception as exc:  # noqa: BLE001
        log.exception("Error en la comprobacion")
        raise HTTPException(status_code=500, detail=f"Error en la comprobacion: {exc}")
    return {"status": "ok", "rows": results}


@app.post("/dataset/append", tags=["datos"], summary="Guarda uniones validadas")
@app.post("/validate", tags=["datos"], include_in_schema=False)
def validate(payload: RowsIn):
    if not payload.rows:
        raise HTTPException(status_code=400, detail="rows vacio")
    retrain_manager.init_baseline()
    res = dataset_valid.append(payload.rows)

    print(f"[validate] recibidas={len(payload.rows)} added={res['added']} updated={res['updated']} "
          f"rechazadas={len(res['rejected'])} total={res['total_rows']}", flush=True)
    for r in res["rejected"][:10]: print("[validate] rechazada:", r, flush=True)

    if not res["added"] and not res["updated"]:
        raise HTTPException(status_code=422, detail="; ".join(res["rejected"][:5]) or "sin filas validas")
    started = retrain_manager.maybe_start() if res["added"] else False
    return {"status": "ok", **res, "retrain": retrain_manager.status(), "retrain_started": started}


@app.get("/retrain/status", tags=["datos"])
def retrain_status():
    return retrain_manager.status()


@app.post("/retrain", tags=["datos"])
def retrain_now():
    return {"started": retrain_manager.maybe_start(force=True), **retrain_manager.status()}


class ScenarioIn(BaseModel):
    model: str | None = Field(default=None, description="Modelo a usar. Si falta, el por defecto.")
    row: Dict[str, Any] = Field(..., description="UNA fila de entrada del modelo.")
    incremento: float = Field(default=0.05, gt=0, description="Aumento de carga por paso (0.05 = 5 %).")
    max_pasos: int = Field(default=40, ge=1, le=200)
    max_fallos_seguidos: int = Field(default=0, ge=0, description="0 = explorar siempre hasta el 100 %.")


@app.post(
    "/scenarios",
    tags=["inferencia"],
    summary="Predice una fila y busca 3 disenos subiendo las cargas",
    response_description="Tabla E1 (diseno de entrada), E2 (max. sin cantonera que cumple), E3 (max. con cantonera que cumple).",
)
async def scenarios(payload: ScenarioIn):
    name = payload.model or config.MODEL_NAME
    log.info("/scenarios: modelo=%s", name)
    try:
        result, queue_wait = await _run_scenario(
            name,
            payload.row,
            "scenarios",
            incremento=payload.incremento,
            max_pasos=payload.max_pasos,
            max_fallos_seguidos=payload.max_fallos_seguidos,
        )
    except predictor.InputContractError:
        raise
    except ServerBusyError as exc:
        log.warning("/scenarios ocupado: %s", exc)
        raise HTTPException(
            status_code=503, detail=str(exc), headers={"Retry-After": "5"}
        )
    except TimeoutError as exc:
        log.error("/scenarios timeout: %s", exc)
        raise HTTPException(status_code=504, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        log.exception("Error generando escenarios")
        raise HTTPException(status_code=500, detail=f"Error en escenarios: {exc}")
    log.info(
        "/scenarios: %s (cola=%.1fs, ejec=%.1fs)",
        result.get("parada"), queue_wait, result.get("tiempo_s", 0.0),
    )
    return {"model_name": name, **result}


@app.get("/dataset/summary", tags=["datos"], summary="Resumen del dataset acumulado")
def dataset_summary():
    return storage.summary()


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/web/conexiones_tr.html")


app.mount("/web", StaticFiles(directory=config.BASE_DIR / "web", html=True), name="web")

if __name__ == "__main__":
    import uvicorn

    # log_config=None: los logs de uvicorn van al mismo archivo, no a pantalla
    print(f"Servidor en http://{config.HOST}:{config.PORT}  |  logs: {LOG_FILE}", flush=True)
    uvicorn.run(app, host=config.HOST, port=config.PORT, log_config=None)