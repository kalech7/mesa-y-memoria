# Correcciones de validez antes del test final

La primera ejecución `reports/rl_longitudinal/final` se interrumpió durante la
preparación de supervisión; no llegó a validación ni a test. Se conservan el log,
las fuentes de datos y `VALIDITY_NOTE.md`.

La revisión independiente detectó y corrigió:

1. El control contrafactual protegido y el tratamiento NOOP sin guardas cambiaban
   dos factores. Ambos brazos causales usan ahora guardas deshabilitadas, con
   idéntica política; la referencia protegida se conserva por separado.
2. El contador de conflictos podía incluir un numerador fuera del denominador o
   medir después de una resolución. Se mide ahora la detección inicial en las
   exposiciones materialmente conflictivas anotadas, con máscara idéntica.
3. Las aclaraciones comunicadas pero perdidas por el parser deben incorporarse
   al denominador del evaluador. Se usa el hecho autorado sólo después de que la
   pregunta correspondiente produjo la respuesta, nunca anticipadamente.
4. LAST_WRITE_WINS conserva el conjunto de hechos de la misma fecha más reciente;
   no desempata una declaración simultánea usando el nombre del alimento.
5. Una pregunta anterior no impide una nueva ocasión o una ambigüedad reabierta
   por evidencia posterior. Las preguntas repetidas sin novedad siguen bloqueadas.
6. La penalización de propuesta insegura ya declarada se aplica aunque la guarda
   impida ejecutarla. No se cambia su coeficiente ni se recompensa copiar reglas.

Se añaden regresiones sobre estas propiedades. No se cambian umbrales, presupuestos,
hiperparámetros, particiones ni etiquetas para conseguir un resultado favorable.
La ejecución válida se guardará en `reports/rl_longitudinal/final_matched`.
