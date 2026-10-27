"""Servidor HTTP central de inferencia y comprobacion estructural.

Cada modelo es un paquete autocontenido en `models/<NAME>/` (lo instala el Makefile).
Se cargan todos los modelos instalados y conviven en memoria; cada peticion se enruta al
modelo indicado. Los resultados son SIEMPRE las columnas de CODAME (`Tabla_MOMENT_M1`).
Sobre esa tabla (editada por el revisor) se puede ejecutar la comprobacion AISC 360-10.

Endpoints:
  GET  /health                  Estado del servidor
  GET  /models                  Modelos instalados y cargados
  GET  /model                   Contrato del modelo por defecto
  GET  /schema?model=...        Inputs del modelo + columnas de resultado CODAME
  GET  /profiles?names=...      Geometria de secciones (para el visualizador 2D)
  POST /model/activate          Carga/recarga un modelo por nombre
  POST /predict                 {"model": "...", "rows": [...]} -> columnas CODAME + comprobacion ("checks")
  POST /scenarios                 {"row": {...}} -> E1 (entrada), E2/E3 = max. carga que cumple sin/con cantonera
  POST /scenarios/batch/jobs      {"client_id","rows": [...]} -> job asincrono (web)
  GET  /scenarios/batch/jobs/{id}?since=&limit= -> resultados incrementales del job
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
import checker
import codame_schema
import config
import model_registry
import escenarios
import predictor
import refinar
import storage

import dataset_valid, retrain_manager

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import CallToolResult, TextContent

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

_SCENARIO_EXEC: Optional[asyncio.Semaphore] = None
_SCENARIO_EXEC_LOOP: Optional[asyncio.AbstractEventLoop] = None


def _exec_semaphore() -> asyncio.Semaphore:
    """Semaforo de ejecucion compartido por escenarios (se recrea por event loop)."""
    global _SCENARIO_EXEC, _SCENARIO_EXEC_LOOP
    loop = asyncio.get_running_loop()
    if _SCENARIO_EXEC is None or _SCENARIO_EXEC_LOOP is not loop:
        _SCENARIO_EXEC = asyncio.Semaphore(config.SCENARIO_WORKERS)
        _SCENARIO_EXEC_LOOP = loop
    return _SCENARIO_EXEC


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
        async with _exec_semaphore():
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


def _as_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


_PERFIL_CAMPOS = ("d_mm", "tw_mm", "tf_mm")


def _positivo(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return number == number and number > 0.0


def _perfil_problema(nombre: str) -> Optional[str]:
    geom = catalogo.get_profile(nombre)
    if geom is None:
        return f"perfil '{nombre}' no encontrado en el catalogo Tekla"
    faltan = [k for k in _PERFIL_CAMPOS if not _positivo(geom.get(k))]
    if not (_positivo(geom.get("bf_mm")) or _positivo(geom.get("b_mm"))):
        faltan.append("bf_mm/b_mm")
    if faltan:
        return (
            f"perfil '{nombre}' sin geometria en el catalogo Tekla "
            f"({', '.join(faltan)} vacio o 0)"
        )
    return None


def _validar_perfiles(row: Dict[str, Any]) -> Optional[str]:
    """Valida BEAM/COLUMN antes de predecir. Devuelve el problema o None."""
    problemas: List[str] = []
    for col in ("BEAM", "COLUMN"):
        nombre = str(row.get(col) or "").strip()
        if not nombre:
            problemas.append(f"falta {col}")
            continue
        problema = _perfil_problema(nombre)
        if problema:
            problemas.append(problema)
    return "; ".join(problemas) if problemas else None


def _codame_row(row: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Filas CODAME canonicas (columnas CODAME en orden), sin columnas auxiliares."""
    return codame_schema.fill_missing(row or {})


def _compact_result(name: str, result: Dict[str, Any]) -> Dict[str, Any]:
    """Detalle MCP por conexion: columnas CODAME + comprobacion completa por escenario."""
    rows = {}
    for row in result.get("rows") or []:
        if not isinstance(row, dict):
            continue
        clave = str(row.get("Escenario") or "")[:2].upper()
        if clave in ("E1", "E2", "E3"):
            rows[clave] = row

    checks = {}
    for check in result.get("checks") or []:
        if not isinstance(check, dict):
            continue
        clave = str(check.get("Escenario") or "")[:2].upper()
        if clave in ("E1", "E2", "E3"):
            checks[clave] = check

    escenarios: List[Dict[str, Any]] = []
    for escenario in result.get("escenarios") or []:
        if not isinstance(escenario, dict):
            continue
        clave = str(escenario.get("id") or "").upper()
        item: Dict[str, Any] = {
            "escenario_id": clave,
            "escenario": escenario.get("escenario"),
            "encontrado": bool(escenario.get("encontrado")),
        }
        if not item["encontrado"]:
            item["motivo"] = {
                "codigo": escenario.get("codigo_motivo") or "SIN_DISENO",
                "mensaje": str(escenario.get("motivo") or ""),
            }
            escenarios.append(item)
            continue

        item.update({
            "paso": escenario.get("paso"),
            "factor_carga": escenario.get("factor_carga"),
            "cargas": {
                "ratio_moment": escenario.get("ratio_moment"),
                "ratio_shear": escenario.get("ratio_shear"),
                "ratio_axial": escenario.get("ratio_axial"),
            },
            "cumple": escenario.get("cumple"),
            "Lh": escenario.get("Lh"),
            "n_cantonera": escenario.get("n_cantonera"),
            "fuera_rango_entrenamiento": escenario.get("fuera_rango_entrenamiento") or [],
            "codame": _codame_row(rows.get(clave)),
        })
        check = checks.get(clave)
        item["comprobacion"] = (
            checker.detalle_comprobacion(check) if check is not None else None
        )
        escenarios.append(item)

    avisos = result.get("avisos_estructurados")
    if not avisos:
        avisos = [{"codigo": "AVISO", "mensaje": str(texto)}
                  for texto in (result.get("avisos") or [])]

    return {
        "parada": result.get("parada"),
        "tiempo_s": result.get("tiempo_s"),
        "avisos": avisos,
        "escenarios": escenarios,
    }


def _run_batch_sync(name: str, rows: List[Dict[str, Any]], options: Dict[str, Any],
                    timeout_s: float) -> List[Dict[str, Any]]:
    """Ejecuta filas secuencialmente con un unico presupuesto total (sin gate)."""
    deadline = time.monotonic() + max(1.0, timeout_s)
    out: List[Dict[str, Any]] = []
    for row in rows:
        uid = _uid(row)
        problema_perfil = _validar_perfiles(row)
        if problema_perfil:
            out.append({
                "unicode": uid, "status": "PERFIL_SIN_GEOMETRIA", "http_status": 422,
                "mensaje": problema_perfil,
            })
            continue
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
        except KeyError as exc:
            log.exception("KeyError en /scenarios (modelo=%s)", name)
            raise HTTPException(status_code=404, detail=f"KeyError: {exc}")
    return out


async def _run_scenario_batch(name: str, rows: List[Dict[str, Any]], label: str,
                              **kwargs) -> tuple[List[Dict[str, Any]], float]:
    queue_wait = await _SCENARIO_GATE.acquire()
    try:
        async with _exec_semaphore():
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


def _compact_web_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Resultado de /scenarios sin los bloques que la web no usa."""
    omitidos = {"historial", "tabla", "tabla_markdown", "parametros"}
    return {
        clave: valor
        for clave, valor in result.items()
        if clave not in omitidos
    }


async def _run_one_scenario(name: str, row: Dict[str, Any],
                            options: Dict[str, Any], timeout_s: float) -> Dict[str, Any]:
    uid = _uid(row)
    web_id = row.get("__web_id")

    def _error(status: str, http_status: int, mensaje: str) -> Dict[str, Any]:
        return {"unicode": uid, "web_id": web_id, "status": status,
                "http_status": http_status, "mensaje": mensaje}

    try:
        async with _exec_semaphore():
            result = await _run_sync(
                _SCENARIO_RUNNER,
                escenarios.run,
                (name, row),
                timeout_s + 2.0,
                "scenarios_batch",
                {"timeout_s": timeout_s, **options},
            )
        return {
            "unicode": uid,
            "web_id": web_id,
            "status": "OK",
            "resultado": _compact_web_result(result),
        }
    except TimeoutError as exc:
        return _error("TIMEOUT_SERVIDOR", 504, str(exc))
    except predictor.InputContractError as exc:
        return _error("ERROR_ENTRADA", 422, str(exc))
    except ValueError as exc:
        return _error("ERROR_ENTRADA", 422, str(exc))
    except KeyError as exc:
        return _error("ERROR_MODELO", 404, str(exc))
    except asyncio.CancelledError:
        return _error("ERROR_SERVIDOR", 500,
                      "fila cancelada por un timeout del lote")
    except Exception as exc:  # noqa: BLE001
        log.exception("[batch] error generando escenarios (unicode=%s)", uid)
        return _error("ERROR_SERVIDOR", 500, str(exc))


_BATCH_JOBS: Dict[str, Dict[str, Any]] = {}
_BATCH_JOB_TTL_S = 600


def _cleanup_jobs() -> None:
    ahora = time.time()
    for job_id, job in list(_BATCH_JOBS.items()):
        if ahora - job["created"] > _BATCH_JOB_TTL_S:
            _BATCH_JOBS.pop(job_id, None)


def _job_payload(job: Dict[str, Any], since: int, limit: int) -> Dict[str, Any]:
    resultados = job["resultados"]
    since = max(0, min(int(since), len(resultados)))
    limit = max(1, min(int(limit), 10))
    nuevos = resultados[since:since + limit]
    return {
        "job_id": job["id"],
        "estado": job["estado"],
        "total": job["total"],
        "done": len(resultados),
        "siguiente": since + len(nuevos),
        "resultados": nuevos,
        "error": job.get("error"),
    }


async def _run_job(job: Dict[str, Any], name: str, rows: List[Dict[str, Any]],
                   options: Dict[str, Any]) -> None:
    try:
        tareas = [
            asyncio.create_task(
                _run_one_scenario(
                    name, row, options, config.SCENARIO_BATCH_ROW_TIMEOUT_S
                )
            )
            for row in rows
        ]
        for futura in asyncio.as_completed(tareas):
            job["resultados"].append(await futura)
    except Exception as exc:  # noqa: BLE001
        job["error"] = str(exc)
        log.exception("[batch] job %s fallo", job["id"])
    finally:
        job["estado"] = "hecho"


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
        f"pendientes en el siguiente turno. La respuesta es compacta: `content` es un resumen "
        f"corto y `structuredContent` trae por escenario las columnas CODAME y la "
        f"comprobacion completa (sin tablas ni SVG). Los perfiles sin geometria se rechazan "
        f"por fila con PERFIL_SIN_GEOMETRIA."
    ),
    stateless_http=True,
    json_response=config.MCP_JSON_RESPONSE,
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=False  # o allowed_hosts=["0tpn3dld-8080.uks1.devtunnels.ms", "localhost:*"]
    ),
)

@mcp.tool()
async def predecir_conexion(rows: List[Dict[str, Any]], model: str = "") -> Dict[str, Any]:
    """Pre-diseña conexiones y devuelve las columnas CODAME con sus comprobaciones.

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
def _batch_counts(results: List[Dict[str, Any]]) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Recuentos del lote por escenario y por estado (conexion y comprobacion)."""
    por_escenario = {
        clave: {"encontrado": 0, "cumple": 0, "no_cumple": 0, "no_encontrado": 0}
        for clave in ("E1", "E2", "E3")
    }
    por_estado_conexion: Dict[str, int] = {}
    por_estado_escenario: Dict[str, int] = {}
    errores: List[Dict[str, Any]] = []

    for item in results:
        uid = item.get("unicode", "?")
        status = item.get("status", "ERROR")
        por_estado_conexion[status] = por_estado_conexion.get(status, 0) + 1
        if status != "OK":
            errores.append({
                "unicode": uid, "status": status,
                "http_status": item.get("http_status"), "mensaje": item.get("mensaje"),
            })
            continue

        result = item.get("resultado") or {}
        escenarios = {
            str(e.get("escenario_id") or "").upper(): e
            for e in (result.get("escenarios") or []) if isinstance(e, dict)
        }
        for clave in ("E1", "E2", "E3"):
            conteo = por_escenario[clave]
            escenario = escenarios.get(clave) or {}
            if not escenario.get("encontrado"):
                conteo["no_encontrado"] += 1
                continue
            conteo["encontrado"] += 1
            if escenario.get("cumple"):
                conteo["cumple"] += 1
            else:
                conteo["no_cumple"] += 1
            estado = ((escenario.get("comprobacion") or {}).get("estado")) or "SIN_DATOS"
            por_estado_escenario[estado] = por_estado_escenario.get(estado, 0) + 1

    return {
        "por_escenario": por_escenario,
        "por_estado_conexion": por_estado_conexion,
        "por_estado_escenario": por_estado_escenario,
    }, errores


def _resumen_texto(resumen: Dict[str, Any], conexiones: List[Dict[str, Any]]) -> str:
    """Markdown corto para el `content` del MCP; el detalle va en structuredContent."""
    lineas = ["**Resumen del lote**"]
    estados = ", ".join(f"{k}={v}" for k, v in resumen["por_estado_conexion"].items())
    if estados:
        lineas.append(f"- Conexiones: {estados}")
    for clave in ("E1", "E2", "E3"):
        c = resumen["por_escenario"][clave]
        lineas.append(
            f"- {clave}: encontrado={c['encontrado']} cumple={c['cumple']} "
            f"no_cumple={c['no_cumple']} no_encontrado={c['no_encontrado']}"
        )
    if resumen["por_estado_escenario"]:
        estados = ", ".join(f"{k}={v}" for k, v in resumen["por_estado_escenario"].items())
        lineas.append(f"- Comprobación: {estados}")
    lineas.append("")
    lineas.append("| UNICODE | Estado | E1 | E2 | E3 |")
    lineas.append("|---|---|---|---|---|")
    for conexion in conexiones:
        estados_esc: Dict[str, str] = {}
        for escenario in conexion.get("escenarios") or []:
            clave = str(escenario.get("escenario_id") or "")
            if not escenario.get("encontrado"):
                estados_esc[clave] = "NO"
            else:
                estados_esc[clave] = "OK" if escenario.get("cumple") else "NO CUMPLE"
        lineas.append(
            "| {u} | {s} | {e1} | {e2} | {e3} |".format(
                u=conexion.get("unicode", ""), s=conexion.get("status", ""),
                e1=estados_esc.get("E1", "-"), e2=estados_esc.get("E2", "-"),
                e3=estados_esc.get("E3", "-"),
            )
        )
    return "\n".join(lineas)


def _mcp_result(payload: Dict[str, Any], texto: Optional[str] = None) -> CallToolResult:
    """Respuesta MCP sin duplicar: texto corto + structuredContent completo."""
    if texto is None:
        texto = str(payload.get("mensaje") or payload.get("status") or "OK")
    return CallToolResult(
        content=[TextContent(type="text", text=texto)],
        structuredContent=payload,
        isError=False,
    )


@mcp.tool()
async def disenar_escenarios_carga(
    rows: Any,
    model: str = "",
    incremento: float = 0.05,
    max_pasos: int = 40,
) -> CallToolResult:
    """Disena varias conexiones en una sola llamada (maximo configurable, 25 por defecto).

    Pasa SIEMPRE una lista en `rows` (no una llamada por conexion). Se procesan como maximo
    `AI_SCENARIO_BATCH_ROWS` conexiones; el resto se descarta y se indica en
    `n_descartadas`/`descartadas`.

    Respuesta compacta y sin duplicar: `content` lleva un resumen corto y
    `structuredContent` el detalle. Por conexion devuelve `escenarios`, y cada escenario:
    `escenario_id` ("E1"/"E2"/"E3"), metadatos, `codame` (las columnas CODAME) y
    `comprobacion` (todos los ratios y comprobaciones geometricas con valor, limite y
    estado). No incluye tablas markdown ni SVG.

    Los perfiles sin geometria en el catalogo se rechazan por fila con
    `PERFIL_SIN_GEOMETRIA` (HTTP 422) antes de llamar al modelo.

    `resumen` trae recuentos por escenario y por estado; `avisos` y `motivo` usan codigos
    (p. ej. E1_NO_CUMPLE, E3_SIN_CANTONERA).

      E1: diseno de entrada (paso 0, sin tocar cargas).
      E2: mayor carga que aguanta SIN cantonera.
      E3: mayor carga que aguanta CON cantonera.
    E2 y E3 priorizan siempre el diseno que cumple la comprobacion AISC 360-10; si ninguno
    cumple, devuelven el de mayor carga (con aviso). Si no hay opcion sin/con cantonera, el
    escenario sale `encontrado=false` con su `motivo`.

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
        return _mcp_result(_mcp_status(name, "ERROR_ENTRADA", 422, str(exc)))

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
        return _mcp_result(_mcp_status(name, "SERVIDOR_OCUPADO", 503, str(exc), 5))
    except TimeoutError as exc:
        log.error("[MCP] escenarios timeout: %s", exc)
        return _mcp_result(_mcp_status(name, "TIMEOUT_SERVIDOR", 504, str(exc), 5))
    except Exception as exc:  # noqa: BLE001
        log.exception("[MCP] error generando escenarios")
        return _mcp_result(_mcp_status(name, "ERROR_SERVIDOR", 500, str(exc)))

    has_errors = any(r.get("status") != "OK" for r in results)
    if has_errors:
        status = "CON_ERRORES"
    elif discarded:
        status = "CON_DESCARTES"
    else:
        status = "OK"

    resumen, errores = _batch_counts(results)
    conexiones: List[Dict[str, Any]] = []
    for item in results:
        if item.get("status") == "OK":
            conexiones.append({
                "unicode": item.get("unicode"),
                "status": "OK",
                **(item.get("resultado") or {}),
            })
        else:
            conexiones.append({
                "unicode": item.get("unicode"),
                "status": item.get("status"),
                "http_status": item.get("http_status"),
                "mensaje": item.get("mensaje"),
                "escenarios": [],
            })

    payload = {
        "model_name": name,
        "status": status,
        "n_solicitadas": len(all_rows),
        "n_procesadas": len(selected),
        "n_descartadas": len(discarded),
        "descartadas": discarded_uids,
        "resumen": resumen,
        "conexiones": conexiones,
        "errores": errores,
    }
    log.info(
        "[MCP] escenarios: %s (procesadas=%d, descartadas=%d, cola=%.1fs)",
        status, len(selected), len(discarded), queue_wait,
    )
    return _mcp_result(payload, _resumen_texto(resumen, conexiones))


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
    response_description="Campos de entrada (tipo/descripcion/opciones) y las columnas CODAME.",
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
    summary="Pre-diseña y devuelve las columnas CODAME",
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


class ScenarioJobIn(BaseModel):
    client_id: str = Field(..., min_length=1, max_length=120)
    model: str | None = None
    rows: List[Dict[str, Any]]
    incremento: float = Field(default=0.05, gt=0)
    max_pasos: int = Field(default=40, ge=1, le=200)
    max_fallos_seguidos: int = Field(default=0, ge=0)


@app.post(
    "/scenarios/batch/jobs",
    tags=["inferencia"],
    summary="Crea un lote asincrono y devuelve su id",
)
async def create_scenarios_job(payload: ScenarioJobIn):
    _cleanup_jobs()
    if not payload.rows:
        raise HTTPException(status_code=400, detail="rows vacio")
    if len(payload.rows) > config.SCENARIO_BATCH_ROWS:
        raise HTTPException(
            status_code=413,
            detail=f"Maximo {config.SCENARIO_BATCH_ROWS} filas por lote",
        )
    existing = _BATCH_JOBS.get(payload.client_id)
    if existing is not None:
        return _job_payload(existing, 0, 1)
    name = payload.model or config.MODEL_NAME
    job = {
        "id": payload.client_id,
        "created": time.time(),
        "estado": "procesando",
        "total": len(payload.rows),
        "resultados": [],
    }
    _BATCH_JOBS[payload.client_id] = job
    job["task"] = asyncio.create_task(
        _run_job(
            job,
            name,
            payload.rows,
            {
                "incremento": payload.incremento,
                "max_pasos": payload.max_pasos,
                "max_fallos_seguidos": payload.max_fallos_seguidos,
            },
        )
    )
    log.info(
        "/scenarios/batch/jobs: job=%s modelo=%s %d filas",
        payload.client_id, name, len(payload.rows),
    )
    return _job_payload(job, 0, 1)


@app.get(
    "/scenarios/batch/jobs/{job_id}",
    tags=["inferencia"],
    summary="Consulta un lote asincrono",
)
async def get_scenarios_job(job_id: str,
                            since: int = Query(0, ge=0),
                            limit: int = Query(2, ge=1, le=10)):
    _cleanup_jobs()
    job = _BATCH_JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job no encontrado o expirado")
    return _job_payload(job, since, limit)


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