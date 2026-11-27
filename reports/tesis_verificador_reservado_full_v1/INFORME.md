# Verificador de escritura en el conjunto reservado

Se reprodujeron 201 pares de semilla y episodio por método, correspondientes a 50 historias raíz y 3 semillas. Los métodos pueden utilizar los mismos pesos seleccionados. Cada contraste conserva la pregunta, la réplica y la respuesta actual.

| Selector | Acierto futuro original | Veto | Reparación | Escrituras respaldadas perdidas con veto | Con reparación |
| --- | ---: | ---: | ---: | ---: | ---: |
| SFT | 49.7 % (98/201) | 60.7 % (122/201) | 93.3 % (181/201) | 0/1 | 0/1 |
| SFT-adicional-SELECT | 49.7 % (98/201) | 61.0 % (123/201) | 94.0 % (183/201) | 0/1 | 0/1 |
| GRPO-SELECT | 49.7 % (98/201) | 60.7 % (122/201) | 93.3 % (181/201) | 0/1 | 0/1 |

Los porcentajes son medias por familia. Una fracción de 0/0 indica que el original no ejecutó ninguna escritura respaldada. Los informes por selector detallan las escrituras rechazadas, las omisiones y las familias. Los resultados se limitan al banco sintético y a los checkpoints seleccionados.

- [SFT](SFT/INFORME.md)
- [SFT-adicional-SELECT](SFT-adicional-SELECT/INFORME.md)
- [GRPO-SELECT](GRPO-SELECT/INFORME.md)
