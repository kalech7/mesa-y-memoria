# Enmienda 001: backbone ajustado al hardware local

Fecha: 2026-10-23. Protocolo operativo: 1.1. Momento: antes de abrir test.

Autorización: el usuario pidió «usa otro modelo que se ajuste a mi pc». Esta instrucción sustituye la restricción de backbone 3B/fallback1.5B de la SPEC1.0; no modifica sus hipótesis ni habilita la revisión humana pendiente. La SPEC original y los artefactos de su cierre técnico fallido se conservan.

## Selección técnica

Hardware observado: NVIDIA RTX3050 Ti Laptop,4096MiB VRAM, y32837603328bytes RAM. Se selecciona `Qwen/Qwen2.5-0.5B-Instruct`, disponible localmente, con revisión de pesos y tokenizer `7ae557604adf67be50417f59c2c2f167def9a775`. El manifiesto de descarga SHA256 se verifica, además de arquitectura y escala. La elección se basa en recursos y compatibilidad; no se consultan resultados de test ni se declara que el modelo sea mejor en la tarea.

Configuración: BF16 sin cuantización, LoRA rango16/alpha32, q_proj/k_proj/v_proj/o_proj, gradient checkpointing, microbatch1, rollouts secuenciales, referencia B1 en CPU con idéntica precisión y adaptador. El límite de entrada por fase pasa de4096 a2048tokens; no se permite truncar silenciosamente. Los límites de salida SELECT64/RESPOND256/WRITE512 permanecen. Se ejecuta una sonda de capacidad con2048tokens de prefijo y512de salida para verificar backward en el peor límite de la fase WRITE.

Presupuesto posterior a SFT: máximo3600segundos de recurso asignado por continuación y semilla. Es un límite operativo de una hora, no una predicción de duración ni permiso para abrir test. Se mantiene el máximo120actualizaciones; el tiempo puede detener una rama antes. Debe publicarse consumo real, exceso de una actualización en curso y prefijo común B2-L/B2-I. La sonda técnica y sus contadores se conservan para contextualizar el presupuesto; no se estima potencia ni coste completo de la tesis a partir de una sonda corta.

No se instala software, no se descargan nuevos pesos y no se usan APIs de inferencia. La carga, generación, paso SFT y actualización GRPO ocurren localmente. Los retornos de la sonda son asignados para comprobar gradientes; no son una comparación científica.

## Evidencia y límites

La nueva prueba GPU se registra en `reports/g0_smoke/technical_smoke.json` y el lock actualizado en `configs/environment.lock.yaml`. Su resultado determina G0; este documento no inventa su aprobación. Las sondas anteriores0.5B de CPU siguen siendo artefactos históricos no elegibles para la versión1.0. Ningún resultado antiguo se reetiqueta como experimento final1.1.

La menor escala puede reducir seguimiento de instrucciones y acierto de la tarea. La capacidad de producir acciones válidas se medirá en desarrollo/validación; que el modelo quepa en GPU no implica que sea competente ni que RL mejore SFT. La auditoría humana, selección de checkpoints y test cerrado mantienen sus puertas.

Fuentes primarias verificadas: [tarjeta oficial](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) y [configuración oficial](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/raw/main/config.json), consultadas2026-10-23. La configuración oficial coincide con los artefactos locales fijados; no se usa la rama main como revisión de ejecución.
