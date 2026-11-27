# Ejecutar el chat con los adaptadores de la tesis

El chat v6 carga un Qwen2.5-1.5B-Instruct en una GPU NVIDIA con CUDA y soporte para bfloat16. El repositorio incluye el adaptador común y los doce adaptadores LoRA que eligió la validación para cuatro métodos y tres semillas. El chat sirve dos de ellos, SFT de la semilla 101 y GRPO longitudinal de la 303. Ambos corresponden a la segunda ronda.

## Instalar y abrir el laboratorio

Desde la raíz de la copia pública, en un host Linux con un controlador compatible con CUDA 12.8, prepara un entorno de Python 3.12. Estas versiones de PyTorch, Transformers y PEFT corresponden a las de la corrida.

```bash
python3.12 -m venv .venv-gpu
source .venv-gpu/bin/activate
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r research/requirements.txt
python -c "import torch; assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()"
```

Descarga los aproximadamente 3,1 GB de la revisión pública utilizada para el modelo base. El programa comprueba los SHA-256 del modelo y el tokenizador al arrancar. [Archivos de Qwen en Hugging Face](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/tree/989aa7980e4cf806f80c7fef2b1adb7bc71aa306)

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    "Qwen/Qwen2.5-1.5B-Instruct",
    revision="989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
    local_dir="models/Qwen2.5-1.5B-Instruct",
)
PY
HF_HUB_OFFLINE=1 python -m web_demo.timing_service --port 8767
```

Abre el puerto 8767 en la pestaña **Puertos** de VS Code si trabajas por Remote - SSH. En el navegador, entra al chat o al simulador utilizando el puerto local que te muestre VS Code. [Chat de alimentación](http://127.0.0.1:8767/chat) · [Simulador](http://127.0.0.1:8767/)

## Procedencia de los archivos públicos

Las matrices LoRA se copiaron byte por byte. Se puede comprobar cada archivo de pesos con `sha256sum` y el campo `pesos_sin_modificaciones_sha256` del manifiesto. La carpeta de evidencia conserva las versiones históricas de los informes cuyas referencias de integridad cambiaron al preparar esta distribución. El manifiesto registra también la huella del plan de entrenamiento original, que contenía rutas privadas. [Manifiesto](../reports/tesis_timing_qwen_a100_v6/PROCEDENCIA.json) · [Evidencia original](../reports/tesis_timing_qwen_a100_v6/evidencia_original/) · [Licencia de Qwen](../reports/tesis_timing_qwen_a100_v6/LICENSE-Qwen.txt)

Para reproducir la conversión desde los archivos originales:

1. Comprueba que los archivos registrados en `LOCK.json` y las dos firmas de cada adaptador coincidan con el sello original. Copia los pesos de los adaptadores seleccionados y del padre común a las mismas rutas relativas al repositorio.
2. En cada `adapter_config.json`, cambia el valor de `base_model_name_or_path` por `Qwen/Qwen2.5-1.5B-Instruct`. Serializa con `json.dumps(config, ensure_ascii=False, indent=2)` y un salto de línea final en UTF-8. Registra la correspondencia entre cada SHA-256 de configuración original y el nuevo.
3. Procesa, en este orden, `CPU.json`, `VALIDATION.json`, `PLAN.json`, `TRAIN_PLAN.json`, `TRAIN_RESULT.json`, `LOCK.json`, `RESERVED_RESULTS.json`, `DEMO_PAR.json` y `PAIR_MARGIN_DIAGNOSTIC.json`. En `TRAIN_PLAN.json`, expresa `parent` y `model` usando respectivamente los valores relativos `parent` y `base` de `PLAN.json`. En el texto de cada archivo sustituye las huellas originales ya procesadas por sus equivalentes portátiles. Conserva el formato de los demás bytes y registra la correspondencia entre el SHA-256 original y el resultante antes de continuar.
4. Contrasta las huellas obtenidas con `PROCEDENCIA.json`. Los porcentajes de evaluación y las elecciones de método, semilla y ronda son los de la corrida original. El SHA-256 del sello original se encuentra en `informes_sha256["LOCK.json"]["original"]`.
