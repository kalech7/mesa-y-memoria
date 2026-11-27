# Enmienda de integridad de particiones, previa a pilot_v2

El protocolo original se conserva sin cambios. La auditoría de `pilot_v1/data_audit.json` detectó 101 firmas latentes idénticas entre train y test y 62 entre train y validación. Usuarios y familias textuales distintos no eliminaban esa repetición semántica. `pilot_v1` se conserva completo y **no se utiliza para demostrar generalización a estados nuevos**.

La corrección asigna cada estado semántico a una única partición mediante SHA256 estable, excluyendo IDs, semillas, categoría, etiquetas de partición y redacción. El muestreo con rechazo conserva la cobertura de categorías. El runner impide entrenar si train/validación comparten firmas y detiene el test si encuentra firmas compartidas con entrenamiento/validación. Test y OOD pueden compartir estados entre sí: se reporta explícitamente su solapamiento; ambos siguen separados de entrenamiento y validación.

La nueva ejecución emplea semillas de datos nuevas (offset 5000: 5142, 5243, 5344, 5445). Se mantienen representación de política, políticas, demostrador supervisado, recompensas, umbral práctico, algoritmo y presupuesto. La corrección responde al requisito de separar estados; no se eligen casos según aciertos, errores ni retorno. Los hiperparámetros no se seleccionan con test.

La abstracción observacional del modelo es deliberadamente pequeña e invariante al nombre del alimento y a algunos valores de contexto. Estados nativos distintos pueden colapsar en el mismo estado de política. Se medirá ese solapamiento y no se afirmará generalización a abstracciones de decisión desconocidas si todas estuvieron en entrenamiento. La evidencia sigue siendo sintética, con fuertes asociaciones entre estructura y preguntas útiles.

Correcciones de contrato anteriores a la primera evaluación: M06 manual usaba `office` y se normalizó a `work`, sin cambiar semántica ni expectativas; una prueba de frontera usaba `amigos` y se adaptó al enum `friends`. No son ajustes derivados de errores del test. Se conservan los hashes finales y la anotación de estas correcciones.
