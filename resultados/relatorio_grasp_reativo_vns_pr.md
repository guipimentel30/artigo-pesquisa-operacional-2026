# Relatório Auditável — Etapa 2: GRASP Reativo + Basic VNS + Path Relinking (Slide 76)

Comparação evolutiva nas mesmas instâncias padronizadas:
`GRASP Clássico (Artigo)` $\to$ `GRASP Reativo + Basic VNS` $\to$ `GRASP Reativo + Basic VNS + Path Relinking (Slide 76)`.

## Experimento 1 — Heiloo (Tabela 4)

| Demandas | Clássico Treino | RVNS Treino | RVNS+PR Treino | Clássico Eval | RVNS Eval | RVNS+PR Eval | Tempo PR (s) | PR (Cham/Sk/+SB) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 250 | 48.93% | 48.93% | 48.93% | 41.68% | 41.68% | 41.68% | 0.8628 | 95/0/+1 |
| 500 | 48.07% | 48.07% | 48.07% | 42.57% | 42.57% | 42.57% | 1.3185 | 95/0/+1 |
| 1000 | 46.23% | 46.24% | 46.24% | 43.24% | 43.23% | 43.23% | 3.0319 | 95/0/+1 |

## Experimento 2 — Realocação de AEDs Existentes (Tabela 5)

| Município | Clássico Treino | RVNS Treino | RVNS+PR Treino | Clássico Eval | RVNS Eval | RVNS+PR Eval | Tempo PR (s) | PR (Cham/Sk/+SB) | VNS (LS \| +k1/k2/k3) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Aalsmeer | 23.37% | 23.37% | 23.37% | 23.57% | 23.57% | 23.57% | 3.611 | 95/0/+0 | LS:509 | +10/26/21 |
| Uithoorn | 48.87% | 48.87% | 48.87% | 48.15% | 48.15% | 48.15% | 5.323 | 95/0/+0 | LS:454 | +0/11/16 |
| Heiloo | 46.79% | 46.79% | 46.79% | 43.99% | 43.99% | 43.99% | 6.344 | 95/0/+0 | LS:453 | +0/9/17 |
| Enkhuizen | 60.57% | 60.56% | 60.57% | 58.07% | 58.11% | 57.90% | 6.025 | 95/2/+5 | LS:427 | +1/3/12 |
| Zaanstad | 53.20% | 53.19% | 53.25% | 49.63% | 49.50% | 49.41% | 4.054 | 95/0/+8 | LS:407 | +2/3/5 |

## Experimento 3 — AEDs Adicionais (Tabela 6 — Treino % / Eval %)

| Município | N = 5 (Eval) | N = 10 (Eval) | N = 20 (Eval) | N = 40 (Eval) | Treino N = 40 (Clás / RVNS / RVNS+PR) |
| --- | ---: | ---: | ---: | ---: | --- |
| Aalsmeer | 36.14% | 45.98% | 56.80% | 66.62% | 69.51% / 69.49% / 69.53% |
| Uithoorn | 46.97% | 56.97% | 66.82% | 75.08% | 76.88% / 76.90% / 76.90% |
| Zaanstad | 35.90% | 41.54% | 48.10% | 55.64% | 58.72% / 58.74% / 58.74% |
