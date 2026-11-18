# Relatório Auditável — Escala Ampliada e TTT-Plots (Hollands Kroon)

## Configuração da Instância e Protocolo
- Município (Tabela 3 de Buter et al., 2024): **Hollands Kroon**
- AEDs a realocar (`N`): **80**
- Candidatos selecionados (`|J^c|`): **1200** (Tabela 3 de referência: 30387 pontos)
- Demandas de treino (`|I^train|`): **1500** | Demandas de avaliação (`|I^eval|`): **5000**
- Semente da instância: `42` | Espaçamento entre sementes: `1000003`
- Calibração não censurada: **30 execuções** por algoritmo (`max_iterations = 30`)
- TTT-plots (Slide 94): **100 execuções independentes** por algoritmo e por alvo (`max_iterations = 96`, `time_limit_sec = 20.0`)

## Fase 1 — Calibração Não Censurada (Base para Alvos do Slide 94)

| Algoritmo | Cobertura Treino Média (%) | Desvio Padrão (%) | Mín (%) | Máx (%) | Avaliação Média (%) | Tempo Médio (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GRASP Clássico (Baseline) | 60.4371 | 0.0209 | 60.3936 | 60.4837 | 54.3770 | 6.259 |
| GRASP Reativo + Basic VNS | 60.4456 | 0.0237 | 60.3909 | 60.4882 | 54.4045 | 7.446 |
| GRASP Reativo + Basic VNS + PR | 60.4707 | 0.0208 | 60.4185 | 60.5028 | 54.4495 | 7.940 |

### Alvos Extraídos (Slide 94)
- **Alvo Fácil** (`min` das médias): **60.437055%**
- **Alvo Médio** (`média` das médias): **60.451103%**
- **Alvo Difícil** (`max` das médias): **60.470688%**

## Fase 2 — Distribuição Empírica Time-to-Target (100 Execuções por Alvo)

| Alvo | Valor (%) | Algoritmo | Acertos / 100 | Taxa de Acerto (%) | Mediana Global $t_{50}$ (s) | Mediana Acertos (s) | Q25 Acertos (s) | Q75 Acertos (s) | Média Acertos (s) |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Alvo Fácil | 60.4371 | GRASP Clássico | 92/100 | 92.0% | 4.717 | 4.348 | 1.831 | 8.959 | 5.755 |
| Alvo Fácil | 60.4371 | GRASP Reativo + VNS | 91/100 | 91.0% | 4.939 | 4.402 | 1.552 | 7.622 | 5.156 |
| Alvo Fácil | 60.4371 | GRASP Reativo + VNS + PR | 100/100 | 100.0% | 1.141 | 1.141 | 0.599 | 1.649 | 1.370 |
| Alvo Médio | 60.4511 | GRASP Clássico | 65/100 | 65.0% | 12.113 | 5.139 | 1.951 | 11.014 | 7.015 |
| Alvo Médio | 60.4511 | GRASP Reativo + VNS | 54/100 | 54.0% | 16.885 | 6.710 | 2.831 | 11.036 | 7.357 |
| Alvo Médio | 60.4511 | GRASP Reativo + VNS + PR | 99/100 | 99.0% | 1.184 | 1.157 | 0.645 | 2.288 | 1.775 |
| Alvo Difícil | 60.4707 | GRASP Clássico | 21/100 | 21.0% | >lim. | 2.924 | 1.397 | 3.869 | 3.097 |
| Alvo Difícil | 60.4707 | GRASP Reativo + VNS | 11/100 | 11.0% | >lim. | 5.941 | 3.257 | 8.444 | 5.814 |
| Alvo Difícil | 60.4707 | GRASP Reativo + VNS + PR | 72/100 | 72.0% | 4.531 | 2.611 | 1.456 | 5.808 | 4.002 |
