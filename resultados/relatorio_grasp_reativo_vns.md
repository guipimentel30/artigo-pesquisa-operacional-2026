# Relatório Auditável — Etapa 1: GRASP Reativo + Basic VNS

Comparação direta sobre as mesmas instâncias padronizadas do baseline (**GRASP Clássico**, Buter et al., 2024).
Vizinhanças Basic VNS (Slide 28–29): $N^{(1)}$ (`1-swap`), $N^{(2)}$ (`2-swap`) e $N^{(3)}$ (`3-swap`) com `BuscaLocal = 1-swap`.

## Experimento 1 — Heiloo (Qualidade, Tempo e Basic VNS)

| Demandas | Clássico Treino | RVNS Treino | Clássico Eval | RVNS Eval | t Clássico (s) | t RVNS (s) | VNS (LS \| +k1/+k2/+k3) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 250 | 48.93% | 48.93% | 41.68% | 41.68% | 0.4610 | 0.8184 | LS:422 | +0/7/8 |
| 500 | 48.07% | 48.07% | 42.57% | 42.57% | 0.8427 | 1.4009 | LS:432 | +1/7/11 |
| 1000 | 46.23% | 46.24% | 43.24% | 43.23% | 1.7623 | 2.4119 | LS:431 | +1/2/14 |

## Experimento 2 — Realocação de AEDs Existentes (Tabela 5)

| Município | Clássico Treino | RVNS Treino | Clássico Eval | RVNS Eval | t Clássico (s) | t RVNS (s) | VNS (LS \| +k1/+k2/+k3) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Aalsmeer | 23.37% | 23.37% | 23.57% | 23.57% | 1.296 | 3.221 | LS:509 | +10/26/21 |
| Uithoorn | 48.87% | 48.87% | 48.15% | 48.15% | 2.539 | 4.756 | LS:454 | +0/11/16 |
| Heiloo | 46.79% | 46.79% | 43.99% | 43.99% | 4.702 | 6.460 | LS:453 | +0/9/17 |
| Enkhuizen | 60.57% | 60.56% | 58.07% | 58.11% | 5.137 | 7.212 | LS:427 | +1/3/12 |
| Zaanstad | 53.20% | 53.19% | 49.63% | 49.50% | 6.121 | 6.043 | LS:407 | +2/3/5 |

### Auditoria das Probabilidades Reativas Aprendidas (Zaanstad)

| $\alpha_k$ | Iterações | Média Treino $A_k$ | Probabilidade Final $p_k$ |
| ---: | ---: | ---: | ---: |
| 0.10 | 3 | 52.93% | 2.44% |
| 0.20 | 9 | 53.00% | 10.14% |
| 0.30 | 12 | 53.00% | 9.49% |
| 0.40 | 5 | 52.97% | 7.40% |
| 0.50 | 4 | 52.96% | 6.59% |
| 0.60 | 15 | 53.00% | 10.62% |
| 0.70 | 12 | 53.05% | 17.88% |
| 0.80 | 7 | 52.95% | 4.26% |
| 0.85 | 8 | 52.99% | 9.08% |
| 0.90 | 14 | 53.01% | 13.55% |
| 0.95 | 7 | 52.98% | 8.55% |

## Experimento 3 — AEDs Adicionais (Tabela 6 — Eval %)

| Município | N = 0 | N = 5 (RVNS) | N = 10 (RVNS) | N = 20 (RVNS) | N = 40 (RVNS) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Aalsmeer | 11.28% | 36.14% | 45.98% | 56.80% | 66.39% |
| Uithoorn | 12.89% | 46.97% | 56.97% | 66.82% | 75.12% |
| Zaanstad | 27.22% | 35.90% | 41.54% | 48.10% | 55.67% |
