"""Batching dinamico de inferencia para una sola copia del modelo.

Las peticiones concurrentes (REST, MCP y escenarios) depositan sus features en una cola
unica. Un worker dedicado las agrupa hasta `max_rows` filas o durante `max_wait_ms` y
ejecuta una sola llamada a `adapter.predict`. Despues reparte el resultado por peticion
conservando el orden.

El batcher solo cubre la inferencia. La traduccion a CODAME y el refinado siguen
haciendose por peticion, fuera del worker, para no bloquear la GPU.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional

import pandas as pd

log = logging.getLogger("ai-server")


@dataclass
class _Request:
    bundle: dict
    features: pd.DataFrame
    future: Future


class InferenceBatcher:
    """Cola dinamica + worker unico de inferencia."""

    def __init__(self, max_rows: int = 64, max_wait_ms: int = 10, name: str = "inference") -> None:
        self.max_rows = max(1, int(max_rows))
        self.max_wait_ms = max(0, int(max_wait_ms))
        self._name = name
        self._requests: Deque[_Request] = deque()
        self._cond = threading.Condition()
        self._thread: Optional[threading.Thread] = None
        self._stop = False
        self._batches = 0
        self._rows = 0
        self._last_batch_rows = 0

    # ------------------------------------------------------------------ ciclo
    def start(self) -> None:
        with self._cond:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop = False
            self._thread = threading.Thread(
                target=self._loop, name=f"{self._name}-batcher", daemon=True
            )
            self._thread.start()

    def close(self) -> None:
        with self._cond:
            self._stop = True
            pending = list(self._requests)
            self._requests.clear()
            self._cond.notify_all()
        for req in pending:
            if req.future.set_running_or_notify_cancel():
                req.future.set_exception(RuntimeError("batcher detenido"))
        thread = self._thread
        if thread is not None:
            thread.join(timeout=5.0)

    def submit(self, bundle: dict, features: pd.DataFrame) -> Future:
        if features is None or len(features) == 0:
            raise ValueError("El batcher necesita al menos una fila")
        with self._cond:
            if self._stop or self._thread is None or not self._thread.is_alive():
                raise RuntimeError("batcher no iniciado")
            future: Future = Future()
            self._requests.append(_Request(bundle=bundle, features=features, future=future))
            self._cond.notify()
            return future

    def stats(self) -> Dict[str, int]:
        with self._cond:
            return {
                "batches": self._batches,
                "rows": self._rows,
                "last_batch_rows": self._last_batch_rows,
                "pending_requests": len(self._requests),
            }

    # ------------------------------------------------------------------ worker
    def _loop(self) -> None:
        while True:
            batch = self._take_batch()
            if batch is None:
                return
            self._process_batch(batch)

    def _take_batch(self) -> Optional[List[_Request]]:
        with self._cond:
            while not self._requests and not self._stop:
                self._cond.wait()
            if self._stop:
                pending = list(self._requests)
                self._requests.clear()
                for req in pending:
                    if req.future.set_running_or_notify_cancel():
                        req.future.set_exception(RuntimeError("batcher detenido"))
                return None

            batch = [self._requests.popleft()]
            rows = len(batch[0].features)
            deadline = time.monotonic() + (self.max_wait_ms / 1000.0)

            while rows < self.max_rows:
                if self._requests:
                    nxt = self._requests[0]
                    n = len(nxt.features)
                    if rows + n > self.max_rows:
                        break
                    self._requests.popleft()
                    batch.append(nxt)
                    rows += n
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._cond.wait(remaining)

            self._last_batch_rows = rows
            return batch

    def _process_batch(self, batch: List[_Request]) -> None:
        active: List[_Request] = []
        for req in batch:
            if req.future.set_running_or_notify_cancel():
                active.append(req)
        if not active:
            return

        # No mezclar bundles distintos (recarga de modelo entre medias).
        groups: Dict[int, List[_Request]] = {}
        for req in active:
            groups.setdefault(id(req.bundle), []).append(req)
        for group in groups.values():
            self._run_group(group)

    def _run_group(self, group: List[_Request]) -> None:
        lengths = [len(r.features) for r in group]
        try:
            features = pd.concat([r.features for r in group], ignore_index=True)
            design = self._predict(group[0].bundle, features)
            if len(design) != len(features):
                raise RuntimeError(
                    f"El modelo devolvio {len(design)} filas para {len(features)} entradas"
                )
            start = 0
            for req, n in zip(group, lengths):
                req.future.set_result(
                    design.iloc[start:start + n].reset_index(drop=True)
                )
                start += n
            with self._cond:
                self._batches += 1
                self._rows += len(features)
            log.debug(
                "batcher: %d peticiones, %d filas", len(group), len(features)
            )
        except Exception as exc:  # noqa: BLE001
            if len(group) > 1:
                # Aislar la peticion que rompe el lote: reintentar una a una.
                log.warning("batcher: fallo el lote de %d peticiones: %s", len(group), exc)
                for req in group:
                    self._run_group([req])
                return
            group[0].future.set_exception(exc)

    def _predict(self, bundle: dict, features: pd.DataFrame) -> pd.DataFrame:
        adapter = bundle["adapter"]
        if len(features) <= self.max_rows:
            return adapter.predict(features, bundle)
        parts: List[pd.DataFrame] = []
        for start in range(0, len(features), self.max_rows):
            chunk = features.iloc[start:start + self.max_rows].reset_index(drop=True)
            parts.append(adapter.predict(chunk, bundle))
        return pd.concat(parts, ignore_index=True)
