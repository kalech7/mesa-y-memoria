# Entorno de la tesis y significado del resultado RL

Comprobación del 27 de noviembre de 2026. La copia de investigación está en `tesis_llm_local_001_limpio` y el repositorio público en `mesa-y-memoria`. La CPU local ejecuta pruebas y simulaciones con SQLite. La inferencia y el ajuste LoRA se realizan en la A100 de la universidad mediante SSH.

## Componentes utilizados

| Parte | Uso |
| --- | --- |
| Qwen2.5-1.5B-Instruct | Puntuar las dos decisiones de preguntar o esperar |
| LoRA y PEFT 0.14.0 | Ajustar adaptadores sin reentrenar todos los pesos de Qwen |
| PyTorch 2.7.1 y Transformers 4.49.0 | Entrenamiento e inferencia |
| CUDA 12.8 y NVIDIA A100 80 GB | Cómputo remoto |
| SQLite y lector compartido | Guardar citas, vigencia y preferencias, y comprobar consultas futuras |
| Pytest y Playwright | Verificar simulador, artefactos y comportamiento del chat |
| Git y GitHub Actions | Publicar código, conservar protocolos y ejecutar pruebas |

Se comprobó que el servicio de laboratorio del puerto 8766 y el chat del 8767 responden con HTTP 200. El comprobador local de dependencias pasó. En la A100, la comprobación de Qwen, CUDA bfloat16, adaptadores y selección v6 también pasó. El script de comprobación se sincronizó al servidor, donde faltaba. Las pruebas completas anteriores a v8 pasaron con 1.449 casos y 21 integraciones omitidas con motivo registrado. El histórico de integraciones GPU debe consultarse junto con las comprobaciones reales ejecutadas en A100.

## Qué se hizo con RL

El ajuste histórico forzó las dos acciones en cada estado de entrenamiento, valoró sus consecuencias en los mundos ponderados de SQLite y usó la continuación del modelo para atribuir el valor futuro. Aplicó una adaptación del objetivo GRPO con referencia KL, clipping y temperatura. Se comparó con SFT y dos controles de recompensa o continuación. Los adaptadores se seleccionaron en validación antes de abrir el reservado.

La ejecución produjo checkpoints y resultados verificables. En la prueba principal a gasto fijo, GRPO longitudinal alcanzó 62,04 % y SFT 62,21 %. La diferencia fue −0,17 puntos porcentuales y el intervalo incluyó cero. Esto describe el rendimiento del método entrenado bajo ese protocolo. El mejor resultado posterior en los 30 pares de Ecuador procedía de SFT con objetivo por pares. [Informe v6](../reports/tesis_timing_qwen_a100_v6/INFORME.md) · [Prueba de Ecuador](../reports/tesis_ecuador_30_v1/INFORME.md)

La auditoría de entrenamiento detectó un desequilibrio a favor de preguntar. La nueva comparación v8 prueba lotes con ambas agendas, presupuesto uno, ventajas normalizadas y dos formas de atribuir el valor de la continuación. Parte de pesos supervisados comunes y tiene un control supervisado con el mismo presupuesto de optimización. [Diagnóstico y configuración](rl_exploracion_recompensa_v8.md)

La decisión del autor es continuar con textos y revisión de IA y con su propia revisión, sin reclutar participantes. [Procedimiento](prueba_ecuador_solo_ia.md)
