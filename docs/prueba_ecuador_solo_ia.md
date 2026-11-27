# Prueba con IA y revisión del autor

El estudio usa conversaciones sintéticas y revisión del autor de la tesis. El autor decidió trabajar sin reclutamiento de participantes. Los materiales preliminares para hablantes quedaron archivados como borrador sin utilizar en `outputs/borrador_hablantes_cancelado/`. Ninguna respuesta de esos formularios forma parte del banco de evaluación.

## Qué se guarda

Cada prueba conserva los textos originales de autoría IA, el modelo y fecha cuando están disponibles, las instrucciones, una revisión con las etiquetas ocultas, las observaciones del autor y la versión que finalmente se congela. Si varias instancias usan el mismo modelo, se describe como separación de tareas entre instancias. Se informa el origen sintético de los textos y del juicio. La naturalidad regional se considera una apreciación de IA y del autor.

La prueba existente de Ecuador tiene 30 pares escritos por dos agentes y revisados por una tercera instancia. Las tres rondas, incluidas las ambigüedades, están archivadas en `reports/tesis_ecuador_30_v1/`. Se ejecutaron los seis adaptadores en la A100 y se reprodujeron exactamente sus márgenes. Esos pares ya se han observado y sirven para diagnósticos posteriores.

## La siguiente prueba

1. Fijar por escrito las variantes de RL, la recompensa, el presupuesto y la selección. Ajustarlas con entrenamiento y validación.
2. Redactar un banco nuevo en español cotidiano con referencias de Ecuador, sin mostrar las predicciones a quien redacta o revisa. Conservar los mismos dos pedidos dentro del par e invertir únicamente la agenda. Evitar presentar un pedido puntual como preferencia permanente.
3. Reasignar códigos y orden de agendas antes de la revisión. Pedir a otra instancia que indique en qué visita preguntaría y cite la frase que lo justifica. Permitir «indeterminado». Registrar también si la persona del relato elige para sí y si el alcance del gusto queda abierto.
4. El autor revisa discrepancias y documenta qué cambió y por qué. Conservar todas las versiones. Una revisión posterior a corregir el texto tiene ese carácter y se informa junto con el acuerdo inicial.
5. Congelar los textos, los juicios, las decisiones del autor y los adaptadores seleccionados con SHA-256 antes de abrir predicciones. El simulador comprueba en SQLite que invertir la agenda invierte la visita óptima.
6. Ejecutar todas las variantes seleccionadas y los controles con el mismo banco, presupuesto y condiciones. Publicar resultados por semilla, los dos miembros del par, márgenes, errores e intervalos descriptivos. Informar por separado decisiones libres y acierto con diez preguntas por veinte historias.

Una mejora obtenida tras probar repetidamente el mismo reservado es exploratoria. El nuevo reservado se abre después de fijar las variantes y seleccionar checkpoints en validación. El objetivo es medir si RL mejora bajo estas condiciones sintéticas, con controles comparables.
