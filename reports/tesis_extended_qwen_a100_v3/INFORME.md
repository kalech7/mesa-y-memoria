# Qwen1.5B con dos preguntas entre tres visitas

La corrida en la A100 comparó Qwen2.5-1.5B con SFT y GRPO por ramas. El adaptador heredado obtuvo **100 %** de acierto actual en validación y pasó el mínimo de 98 %. Fue el punto de partida común de los seis brazos entrenados. Los métodos usaron las mismas 240 decisiones públicas y las mismas ramas. Cada brazo hizo **120 actualizaciones y procesó 784.212 tokens** por semilla, con el mismo orden y padding dentro de cada pareja. Los resultados reservados se calcularon después de sellar los seis checkpoints seleccionados y los tres iniciales. [Plan](PLAN.json) · [Competencia inicial](shared-competence/PREFLIGHT.json) · [Ramas](BRANCHES.json) · [Sello de checkpoints](SELECTION_LOCK.json)

| Política | Futuro en validación | Utilidad en validación | Futuro en reservado | Utilidad en reservado |
| --- | ---: | ---: | ---: | ---: |
| Qwen inicial | 22,47 % | 1,3371 | 47,22 % | 1,1694 |
| Regla pública | 61,86 % | 1,7680 | 61,11 % | 1,3244 |
| Planificador público | 78,98 % | 2,0451 | 61,11 % | 1,3244 |
| SFT por ramas | 75,00 % | 2,0010 | 57,41 % | 1,2844 |
| GRPO por ramas | 72,50 % | 1,9704 | 57,41 % | 1,2844 |

El planificador de validación obtuvo **17,12 puntos** de acierto futuro más que el valor de un paso y pasó de 2,00 a 1,75 preguntas por historia. La diferencia de utilidad fue **+0,2771**, con intervalo emparejado del 95 % de **+0,2668 a +0,2874**. El cambio de planes observado en la segunda visita y las consultas futuras ligadas a la tercera crean el costo de oportunidad. [Diagnóstico CPU](CPU_MARGIN.json)

En validación SFT y GRPO empataron en las semillas 101 y 303. En 202, SFT obtuvo **7,51 puntos** más de acierto futuro. El promedio favoreció a SFT por **2,50 puntos**. Los intervalos de validación son condicionales a las tres semillas y a un banco que también se usó para elegir checkpoints. Ambos ajustes reservaron preguntas para la tercera visita; el acierto actual en validación fue de 100 % en los dos. [Comparaciones y contabilidad por semilla](RESULTS.json)

En el reservado, ambos ajustes pasaron de **47,22 % a 57,41 %** frente al adaptador común y empataron en cada semilla. La regla pública y el planificador obtuvieron **61,11 %**. El acierto actual estricto en las 12 historias fue de **69,44 %** para todas las políticas. En un replay con presupuesto completo y sin preguntas anteriores, el analizador habilitó **7 de 9** oportunidades de aclaración anotadas por el autor. Los dos casos omitidos muestran que la lista de acciones del analizador también limita el resultado. El corpus es de autoría independiente por IA y contiene 12 raíces con 16 mundos; se agruparon mundos y semillas por raíz al comparar métodos. [Resultados reservados](RESERVED_RESULTS.json) · [Cobertura del menú](../tesis_extended_reserve_review/MASK_AUDIT.json) · [Adjudicación](../tesis_extended_reserve_review/ADJUDICACION.md)

La ejecución terminó con código 0 en `tmux`. Se conservaron los adaptadores de cada época y los estados del optimizador. [Registro](EJECUCION.log) · [Dependencias](DEPENDENCIAS.txt) · [Protocolo y auditoría de memoria](../../docs/experimento_tres_visitas.md)
