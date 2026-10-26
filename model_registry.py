"""Registro multi-modelo con carga simultanea y recarga en caliente.

Cada modelo vive en `models/<NAME>/` como paquete autocontenido:

    models/V11/
        core/adapter.py     <- contrato + preproceso + load + predict
        weights.pt
        preprocessor.joblib
        data/

Se cargan TODOS los modelos instalados y conviven en memoria. Para que los
preprocesadores serializados (`core.dataset.Preprocessor`) resuelvan, cada paquete se
importa con el nombre top-level `core` de forma secuencial; las referencias (modulos,
clases y objetos) quedan capturadas en cada bundle, de modo que cargar el siguiente no
afecta a los anteriores. Los adaptadores no deben hacer `import core.*` en tiempo de
ejecucion (solo en el import del modulo).
"""

import importlib
import importlib.util
import logging
import sys
import threading
from pathlib import Path

import config

log = logging.getLogger("ai-server")

MODELS_DIR: Path = config.MODELS_DIR

_LOCK = threading.RLock()
_BUNDLES: dict = {}


def list_models() -> list:
    """Nombres de los modelos instalados (carpetas con core/adapter.py)."""
    if not MODELS_DIR.exists():
        return []
    return sorted(
        p.name for p in MODELS_DIR.iterdir() if not p.name.startswith("_") and (p / "core" / "adapter.py").exists()
    )


def loaded_models() -> list:
    return sorted(_BUNDLES)


def _purge_core():
    for name in [n for n in list(sys.modules) if n == "core" or n.startswith("core.")]:
        del sys.modules[name]


def _register_core(model_dir: Path):
    core_dir = (model_dir / "core").resolve()
    spec = importlib.util.spec_from_file_location(
        "core", core_dir / "__init__.py", submodule_search_locations=[str(core_dir)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["core"] = module
    spec.loader.exec_module(module)
    return module


def _load_one(name: str) -> dict:
    """Carga el modelo `name` desde disco y lo registra. Devuelve su bundle."""
    model_dir = MODELS_DIR / name
    if not (model_dir / "core" / "adapter.py").exists():
        raise FileNotFoundError(
            f"No hay modelo instalado en {model_dir}. "
            f"Instalalo con 'make install-model MODEL={name}'. "
            f"Disponibles: {list_models()}"
        )

    with _LOCK:
        _purge_core()
        _register_core(model_dir)
        adapter = importlib.import_module("core.adapter")
        bundle = adapter.load()
        bundle.update(
            {
                "name": getattr(adapter, "MODEL_NAME", name),
                "adapter": adapter,
                "dir": str(model_dir),
                "contract": adapter.contract(),
            }
        )
        device = bundle.get("device")
        if device is None:
            log.info("Modelo %s cargado (dispositivo no especificado)", name)
        else:
            kind = getattr(device, "type", str(device)).lower()
            where = "GPU" if kind == "cuda" else "CPU"
            log.info("Modelo %s cargado en %s (%s)", name, where, device)
        _BUNDLES[name] = bundle
        return bundle


def load_all() -> dict:
    """Carga todos los modelos instalados. Devuelve {nombre: bundle}."""
    for name in list_models():
        try:
            _load_one(name)
        except Exception as exc:  # noqa: BLE001
            log.error("No se pudo cargar el modelo %s: %s", name, exc)
    return dict(_BUNDLES)


def activate(name: str) -> dict:
    """Carga o recarga el modelo `name` (util tras `make install-model`)."""
    return _load_one(name)


def get(name: str = None) -> dict:
    """Bundle del modelo pedido (o el configurado por defecto)."""
    name = name or config.MODEL_NAME
    if name not in _BUNDLES:
        if name in list_models():
            return _load_one(name)
        raise KeyError(
            f"Modelo '{name}' no disponible. Cargados: {loaded_models()}; "
            f"instalados: {list_models()}."
        )
    return _BUNDLES[name]


def activate_configured() -> dict:
    """Carga todos los modelos instalados (el configurado por defecto, como minimo)."""
    load_all()
    if config.MODEL_NAME not in _BUNDLES and _BUNDLES:
        return next(iter(_BUNDLES.values()))
    if config.MODEL_NAME not in _BUNDLES:
        raise RuntimeError(
            f"No hay modelos instalados en {MODELS_DIR}. "
            f"Ejecuta 'make install-model MODEL=V11'."
        )
    return _BUNDLES[config.MODEL_NAME]
