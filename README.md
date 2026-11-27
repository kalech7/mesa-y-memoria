# Mesa y Memoria

**Clarificación ante memorias alimentarias incompatibles** es una tesis sobre cuándo conviene que un asistente pregunte si debe recordar una preferencia. Si alguien dice «hoy quiero sopa», la respuesta sirve para la cena de hoy. Para usarla en la próxima visita, el asistente necesita saber si era una excepción o un hábito. El laboratorio compara dos formas de entrenar esa decisión con Qwen y guarda en SQLite la evidencia de cada recuerdo.

## Resultado principal

En 160 historias simuladas de tres visitas, con diez preguntas disponibles por cada veinte historias, el acierto futuro fue de **62,21 % para SFT** y **62,04 % para GRPO longitudinal**. La comparación principal no mostró una mejora de GRPO. El informe incluye el intervalo de la diferencia, los resultados por método y las limitaciones del banco sintético. [Informe del experimento](reports/tesis_timing_qwen_a100_v6/INFORME.md)

## Probar en CPU

Desde la raíz del repositorio, con Python 3.13 y `uv`, este comando comprueba el banco de historias, el simulador, las métricas y el comportamiento del chat. Las pruebas usan conversaciones sintéticas y decisiones controladas.

```bash
uv run --extra dev python -m pytest -q \
  research/tests/test_tesis_timing_v6_bank.py \
  research/tests/test_tesis_timing_v6_env.py \
  research/tests/test_tesis_timing_v6_eval.py \
  tests/test_final_food_chat.py
```

## Usar el chat con los modelos entrenados

El backend y los modelos se ejecutan en el equipo con CUDA. Desde una Mac con acceso autorizado, conecta VS Code a ese equipo con la extensión **Remote - SSH** y reenvía el puerto **8767** desde la pestaña **Puertos**. Abre la dirección local que muestre VS Code y añade `/chat`. En la raíz del mismo puerto encontrarás la simulación guiada. El chat usa SQLite en `outputs/final_food_chats/` y guarda la referencia de cada conversación en el navegador.

Para instalar el laboratorio en otro servidor, consulta la guía de pesos y requisitos. [Preparación del modelo](docs/modelos-publicos.md)

## Explorar la tesis

| Para consultar | Recurso |
| --- | --- |
| Cómo se generan las historias pareadas y cuándo conviene aclarar | [Banco del experimento](research/tesis_timing_v6_bank.py) |
| Qué recuerda el asistente y cómo se evalúa cada visita | [Simulador](research/tesis_timing_v6_env.py) |
| Cómo se entrenan, seleccionan y evalúan los modelos | [Protocolo ejecutable](research/tesis_timing_v6_experiment.py) |
| Cómo responde el chat y guarda las preferencias | [Backend](web_demo/food_chat.py) |
| Resultados de la comparación anterior por perfiles | [Informe v5](reports/tesis_profiles_qwen_a100_v5/INFORME.md) |

La edición pública reúne código, pruebas, bancos sintéticos, protocolos e informes. Algunas referencias de los informes históricos apuntan a archivos de corridas conservados en el entorno de investigación. Los historiales de conversaciones pertenecen al servidor donde se generaron.
