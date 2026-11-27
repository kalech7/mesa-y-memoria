# Mesa y Memoria

**Clarificación ante memorias alimentarias incompatibles** es una tesis sobre cuándo conviene que un asistente pregunte si debe recordar una preferencia. Si alguien dice «hoy quiero sopa», la respuesta sirve para la cena de hoy. Para usarla en la próxima visita, el asistente necesita saber si era una excepción o un hábito. El laboratorio compara dos formas de entrenar esa decisión con Qwen y guarda en SQLite la evidencia de cada recuerdo.

## Resultado principal

En 160 historias simuladas de tres visitas, con diez preguntas disponibles por cada veinte historias, el acierto futuro fue de **62,21 % para SFT** y **62,04 % para GRPO longitudinal**. La comparación principal no mostró una mejora de GRPO. El informe incluye el intervalo de la diferencia, los resultados por método y las limitaciones del banco sintético. [Informe del experimento](reports/tesis_timing_qwen_a100_v6/INFORME.md)

En un segundo reservado de 80 pares, con platos y frases nuevos, comparamos dos ajustes supervisados del mismo Qwen. Al cambiar únicamente la agenda, el modelo con entrenamiento por pares eligió las dos visitas correctas en el **12,92 % de los pares**, frente al **8,33 %** con SFT convencional. La regla que lee la agenda resolvió los 80 pares. El informe incluye los seis adaptadores seleccionados, los resultados por semilla y la verificación del piloto ejecutado en una A100. [Piloto v7](reports/tesis_timing_pares_a100_v7/INFORME.md)


Los seis adaptadores se sometieron después a 30 pares nuevos de pedidos y agendas con referencias de Ecuador. SFT por pares acertó ambas visitas en el **27,78 %**, frente al **10,00 %** con SFT convencional. A igualdad de diez preguntas por veinte historias, la diferencia fue de **+0,56 puntos** de acierto futuro. El banco tiene revisión ciega por IA y una réplica exacta en A100. [Prueba de transferencia y datos](reports/tesis_ecuador_30_v1/INFORME.md)

En una comparación posterior de RL, desde el mismo inicio supervisado, la variante con respaldo Bellman pasó del **15,56 % al 21,11 %** de pares completos frente al control y del **63,28 % al 63,72 %** de acierto futuro. Se usaron otros 30 pares sintéticos, tres semillas y el mismo gasto de preguntas. Se publican también las variantes que rindieron peor, los quince adaptadores seleccionados y la reproducción exacta en A100. [Auditoría y resultados de RL](reports/tesis_rl_credit_a100_v8/INFORME.md)

## Probar en CPU

Desde la raíz del repositorio, con Python 3.13 y `uv`, este comando comprueba el banco de historias, el simulador, las métricas y el comportamiento del chat. Las pruebas usan conversaciones sintéticas y decisiones controladas.

```bash
uv run --extra dev python -m pytest -q \
  research/tests/test_tesis_timing_v6_bank.py \
  research/tests/test_tesis_timing_v6_env.py \
  research/tests/test_tesis_timing_v6_eval.py \
  tests/test_final_food_chat.py
```

Para ejecutar las dos suites completas con dependencias bloqueadas, sigue la [guía de pruebas](docs/pruebas_entorno.md). El caso «qué cena tengo guardada mañana» también se prueba sin signos ni tildes y comprueba que una excepción de hoy caduca para mañana.

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
| Qué aportan estos resultados frente a ATRBench, ActiveWebPref y HabitusMem, y qué experimento queda por hacer | [Revisión de la literatura y propuestas](REVISION_LITERATURA.md) |
| Qué valoraron los dos revisores de IA en las 100 historias | [Acuerdo y juicios individuales](reports/tesis_revision_ia_100_v1/INFORME.md) |
| Cómo distribuir los paquetes ciegos y conciliar las valoraciones de personas | [Revisión humana](docs/revision_humana_tesis.md) |

La edición pública reúne código, pruebas, bancos sintéticos, protocolos e informes. Algunas referencias de los informes históricos apuntan a archivos de corridas conservados en el entorno de investigación. Los historiales de conversaciones pertenecen al servidor donde se generaron.
