# Verificador de escritura en el conjunto reservado

Se reprodujeron 21 pares de semilla y episodio por método, correspondientes a 5 historias raíz y 3 semillas. Los métodos pueden utilizar los mismos pesos seleccionados. Cada contraste conserva la pregunta, la réplica y la respuesta actual.

| Selector | Acierto futuro original | Veto | Reparación | Escrituras respaldadas perdidas con veto | Con reparación |
| --- | ---: | ---: | ---: | ---: | ---: |
| SFT | 53.3 % (11/21) | 66.7 % (14/21) | 93.3 % (19/21) | 0/0 | 0/0 |
| SFT-adicional-SELECT | 43.3 % (10/21) | 70.0 % (15/21) | 100.0 % (21/21) | 0/0 | 0/0 |
| GRPO-SELECT | 53.3 % (11/21) | 66.7 % (14/21) | 93.3 % (19/21) | 0/0 | 0/0 |

Los porcentajes son medias por familia. Una fracción de 0/0 indica que el original no ejecutó ninguna escritura respaldada. Los informes por selector detallan las escrituras rechazadas, las omisiones y las familias. Los resultados se limitan al banco sintético y a los checkpoints seleccionados.

- [SFT](SFT/INFORME.md)
- [SFT-adicional-SELECT](SFT-adicional-SELECT/INFORME.md)
- [GRPO-SELECT](GRPO-SELECT/INFORME.md)
