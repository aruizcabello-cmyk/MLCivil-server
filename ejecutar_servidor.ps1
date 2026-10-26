# 1. Establecer la variable para certificados del sistema
$env:UV_SYSTEM_CERTS = "1"

uv add -r requirements.txt

# 2. Iniciar el servidor Uvicorn en segundo plano
uv run uvicorn main:app --app-dir . --port 8080 --workers 1