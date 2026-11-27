# ¿Ayuda enseñar a Qwen a distinguir dos agendas opuestas?

El 27 de noviembre de 2026 se completó un piloto de Qwen2.5-1.5B-Instruct con LoRA en una A100. Cuando el modelo pudo decidir libremente, añadir una pérdida que compara las dos agendas elevó el porcentaje medio de pares resueltos del **8,33 % al 12,92 %**. Dos semillas mejoraron y una empeoró. El planificador que lee la agenda resolvió todos los pares. [Resultados por semilla y paquete](RESERVED_RESULTS.json)

La diferencia primaria fue de **+4,58 puntos porcentuales**. Tras cerrar la evaluación, se reprodujeron las decisiones libres de los seis adaptadores y se calculó un **intervalo emparejado del 95 % de +0,83 a +8,75 puntos**. Cada uno de los 5.000 remuestreos tomó ocho paquetes completos con reemplazo y mantuvo los mismos pares, métodos y semillas dentro de cada paquete. Al excluir alternativamente una de las ocho redacciones de agenda, la diferencia quedó entre +3,33 y +5,24 puntos. El intervalo y la sensibilidad describen este banco sintético y las tres semillas observadas. [Indicadores por par y diagnóstico posterior](PRIMARY_PAIR_DIAGNOSTIC.json)

| Método | Pares completos de 80 en las semillas 101 / 202 / 303 | Acierto futuro con diez preguntas por veinte historias |
| --- | ---: | ---: |
| SFT de este piloto | 6 / 12 / 2 | 62,875 % |
| SFT con pérdida adicional por pares | 12 / 8 / 11 | 63,125 % |
| SFT v6 seleccionado previamente | 0 / 0 / 0 | 62,375 % |
| GRPO longitudinal v6 seleccionado previamente | 0 / 0 / 0 | 62,000 % |
| Planificador con agenda pública | 80 / 80 / 80 | 67,500 % |

Resolver un par exige preguntar al principio en una historia, esperar en la otra y preguntar en la visita siguiente. Las dos personas hacen los mismos pedidos, pero esperan necesitar comidas distintas durante la semana. La segunda columna mide las decisiones individuales libres. En la tercera se asignan diez preguntas a veinte historias según las puntuaciones del modelo. En esa comparación, SFT con pérdida por pares superó al SFT equivalente en **0,25 puntos porcentuales**. El intervalo percentil del 95 % fue de **+0,04 a +0,46 puntos** al remuestrear los ocho paquetes 5.000 veces manteniendo juntas las tres semillas observadas. Todos los métodos gastaron 0,5 preguntas por historia, con 100 % de acierto actual y 0 % de reutilizaciones sin respaldo. La memoria y la respuesta actual las gestiona el simulador compartido. [Evaluador](../../research/tesis_timing_v6_eval.py) · [Resultados](RESERVED_RESULTS.json)

## Comparación y alcance

Los dos métodos nuevos partieron del mismo adaptador. En los 30 pares de entrenamiento obtuvieron la decisión preferida de los resultados futuros de las dos ramas en SQLite, ponderando los ocho mundos simulados de cada historia. El SFT aprendió cuatro decisiones por par, dos en cada visita. El otro brazo agregó una pérdida logística sobre la diferencia entre las puntuaciones de preguntar al principio en las dos agendas. Ambos hicieron 90 actualizaciones por semilla y procesaron 720 decisiones y 141.012 tokens de entrada sin relleno. Compartieron los lotes, el orden, la tasa de aprendizaje y las opciones de selección. Los ajustes v6 se evaluaron aquí con sus pesos y umbrales históricos. [Método](../../research/tesis_timing_v7.py) · [Plan y huellas](PLAN.json)

La selección usó otros 30 pares de validación. Se eligió la época con más pares completos en decisiones libres y, en caso de empate, el mayor acierto futuro a igual gasto y la época más temprana. El umbral de asignación de cada modelo también se fijó en validación. El nuevo reservado contiene **80 pares, 160 historias y 1.280 mundos**. Usa otros platos, frases para ambas visitas y ocho redacciones de agenda. Se selló su huella antes del ajuste y se aplicaron las selecciones congeladas. Las historias siguen siendo sintéticas y comparten la mecánica del simulador v6. La mejora observada se refiere a la comparación supervisada por pares, y la mayoría de los pares sigue quedando sin resolver por Qwen. La generalización a conversaciones humanas queda abierta. [Banco v7](../../research/tesis_timing_v7_bank.py) · [Selección congelada](LOCK.json)

## Verificar y reproducir

Desde la raíz del proyecto, la verificación de CPU contrasta las fuentes, los pesos publicados, las métricas por semilla, los presupuestos por paquete y el intervalo con las actas originales. Las pruebas de regresión cubren la independencia de las particiones, las etiquetas derivadas de los resultados y el gradiente de la pérdida por pares.

```bash
python -m research.tesis_timing_v7_verify
python -m research.tesis_timing_v7_pairs
python -m pytest -q research/tests/test_tesis_timing_v7.py
```

Para repetir las predicciones de los seis adaptadores publicados en una GPU CUDA, instala las dependencias de investigación y coloca el modelo base en `models/Qwen2.5-1.5B-Instruct`. El verificador utiliza el adaptador padre incluido en el repositorio. Una segunda ejecución en la A100 reprodujo exactamente los resultados por paquete, las decisiones individuales agregadas y las métricas de los seis adaptadores portátiles. El entorno original y las huellas del modelo y de los archivos del tokenizador figuran en el plan. [Plan](PLAN.json)

```bash
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 python -m research.tesis_timing_v7_verify --replay
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 python -m research.tesis_timing_v7_pairs --replay
```

Para repetir el entrenamiento y la evaluación completa utiliza un nombre nuevo de corrida. Si se interrumpe, ejecuta de nuevo el mismo comando con su directorio. Cada brazo terminado se valida por sus huellas y cada brazo incompleto se vuelve a ajustar desde el padre. El entrenamiento requiere el modelo base y los adaptadores históricos seleccionados v6 en las rutas del proyecto.

```bash
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 python -m research.tesis_timing_v7 \
    --run reports/mi_replica_pares_v7
```

Los seis adaptadores publicados contienen los mismos tensores que los seleccionados en la A100. En las configuraciones portátiles el identificador público de Qwen sustituye la ruta local del modelo base. El manifiesto conserva las huellas de estos archivos y el acta de selección conserva las huellas originales. [Adaptadores portátiles](ADAPTADORES.json) · [Licencia de Qwen](LICENSE-Qwen.txt)
