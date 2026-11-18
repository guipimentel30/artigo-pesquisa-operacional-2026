# Relatório Auditável — Baselines do Artigo Original (Buter et al., 2024)

Métodos avaliados: **BIP (Exato)**, **Greedy (Algoritmo 1)** e **GRASP Clássico (Algoritmo 2)**.

## Experimento 1 — BIP vs. GRASP Clássico vs. Greedy (Tabela 4 / Heiloo)

| Método | Demandas | Treino (%) | Avaliação (%) | Tempo CPU (s) | Gap p/ BIP (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| BIP (Exato) | 250 | 48.93% | 41.68% | 0.8581 | 0.000% |
| GRASP Clássico (Alg. 2) | 250 | 48.93% | 41.68% | 0.4610 | 0.000% |
| Greedy (Alg. 1) | 250 | 47.31% | 40.29% | 0.0003 | 3.296% |
| BIP (Exato) | 500 | 48.07% | 42.57% | 4.5928 | 0.000% |
| GRASP Clássico (Alg. 2) | 500 | 48.07% | 42.57% | 0.8427 | 0.000% |
| Greedy (Alg. 1) | 500 | 46.43% | 41.83% | 0.0004 | 3.413% |
| BIP (Exato) | 1000 | 46.24% | 43.23% | 34.8299 | 0.003% |
| GRASP Clássico (Alg. 2) | 1000 | 46.23% | 43.24% | 1.7623 | 0.028% |
| Greedy (Alg. 1) | 1000 | 44.67% | 42.69% | 0.0013 | 3.413% |

## Experimento 2 — Realocação de AEDs Existentes via GRASP Clássico (Tabela 5)

| Município | AEDs | OHCAs | Candidatos | Atual Eval (%) | Treino (%) | Realocado Eval (%) | Melhoria Rel. | Tempo (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Aalsmeer | 3 | 99 | 1000 | 11.28% | 23.37% | 23.57% | +108.9% | 1.296 |
| Uithoorn | 7 | 105 | 1000 | 12.89% | 48.87% | 48.15% | +273.4% | 2.539 |
| Heiloo | 16 | 97 | 1000 | 30.34% | 46.79% | 43.99% | +45.0% | 4.702 |
| Enkhuizen | 24 | 76 | 1000 | 39.01% | 60.57% | 58.07% | +48.9% | 5.137 |
| Zaanstad | 44 | 616 | 1000 | 27.22% | 53.20% | 49.63% | +82.3% | 6.121 |
| **Overall (Eq. 4)** | 94 | 993 | --- | 25.32% | 49.71% | 46.97% | +85.5% | --- |

## Experimento 3 — Implantação de AEDs Adicionais via GRASP Clássico (Tabela 6)

| Município | N = 0 | N = 5 | N = 10 | N = 20 | N = 40 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Aalsmeer | 11.28% | 36.14% | 45.98% | 56.80% | 66.68% |
| Uithoorn | 12.89% | 46.97% | 56.97% | 66.68% | 75.23% |
| Zaanstad | 27.22% | 35.90% | 41.54% | 48.11% | 55.66% |
| **Overall (Eq. 4)** | 23.46% | 37.34% | 44.05% | 51.54% | 59.50% |

## Experimento 4 — Sensibilidade da Função de Cobertura em Zaanstad (Tabela 7)

| Assumida \ Verdadeira | Piecewise linear | Sigmoid | Binary | Exponential |
| --- | ---: | ---: | ---: | ---: |
| Piecewise linear | 49.63% (100.00%) | 54.24% (99.32%) | 72.84% (96.68%) | 41.32% (100.18%) |
| Sigmoid | 49.70% (100.15%) | 54.61% (100.00%) | 72.28% (95.94%) | 41.43% (100.47%) |
| Binary | 48.85% (98.44%) | 52.79% (96.66%) | 75.34% (100.00%) | 40.64% (98.53%) |
| Exponential | 48.56% (97.84%) | 53.02% (97.09%) | 68.64% (91.11%) | 41.24% (100.00%) |
