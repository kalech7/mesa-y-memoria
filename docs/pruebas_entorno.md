# Pruebas del entorno y del simulador

Desde la raíz, ejecuta ambas suites con Python 3.12 y las versiones del archivo `uv.lock`. El grupo `research` instala en `.venv` las bibliotecas de pruebas, PyTorch para CPU, Transformers, PEFT y Safetensors. La instalación inicial necesita acceso a los índices de paquetes o una caché que contenga las versiones fijadas.

```bash
uv sync --locked --python 3.12 --group research
uv run --locked --python 3.12 --group research playwright install --only-shell chromium
uv run --locked --python 3.12 --group research python -m scripts.check_environment
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1 \
  uv run --locked --python 3.12 --group research python -m pytest -q -rs
```

Pytest descubre `tests/` y `research/tests/`. Para ejecutar solo un archivo o una prueba, añade su ruta o identificador al último comando. La opción `-rs` enumera las integraciones omitidas y la razón de cada omisión.

Playwright instala Chromium en la caché del usuario para comprobar el chat en un navegador real. En una imagen Linux mínima, la opción adicional `--with-deps` instala las bibliotecas del sistema que necesita Chromium.

Las pruebas de identificadores de tokens y logits reales de la etapa histórica utilizan `models/Qwen2.5-0.5B-Instruct`. Las comprobaciones contra trazas archivadas también necesitan sus corridas originales. La copia pública omite el modelo base y varias corridas históricas. Las pruebas de políticas y memoria con datos sintéticos se ejecutan en CPU. Los casos numéricos específicos de CUDA se ejecutan cuando el intérprete dispone de PyTorch para CUDA y una GPU compatible.

Para comprobar específicamente el controlador en un entorno CUDA configurado para cálculo determinista, establece la memoria de trabajo de cuBLAS antes de iniciar Python.

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m pytest -q research/tests/test_stable_controller_policy.py
```

## Comprobar el chat v6

Usa el intérprete con PyTorch para CUDA que ejecutará el servicio. Desde la raíz, comprueba el modelo base Qwen2.5-1.5B-Instruct, los adaptadores seleccionados, los artefactos sellados y el soporte de bfloat16 antes de abrir el chat.

```bash
python -m scripts.check_environment --demo-v6
```

El comando termina con código 1 si falta un requisito. El modelo de 1.5B tiene su propio tokenizador y huellas SHA-256 en el plan de la corrida. La simulación CPU y las dos suites pueden ejecutarse sin ese modelo. [Plan v6](../reports/tesis_timing_qwen_a100_v6/PLAN.json) · [Laboratorio en la A100](momento_aclaracion_v6.md)
