## Como executar

Pra rodar os 4 primeiros experimentos em sequencia (do baseline ate o metodo completo):

```powershell
.\.venv\Scripts\python.exe simulacao_original.py
.\.venv\Scripts\python.exe simulacao_grasp_reativo_vns.py
.\.venv\Scripts\python.exe simulacao_grasp_reativo_vns_pr.py
```

## Experimento 5: Escala ampliada e TTT-Plots

Esse demora bem mais pq faz a calibracao e depois 100 execucoes por alvo:

```powershell
.\.venv\Scripts\python.exe simulacao_escala_ttt.py
```

A logica principal fica em `src/ttt_plots.py` e as saidas vao pra `resultados/`:
- `escala_ttt_plots.json` e `relatorio_escala_ttt_plots.md` com os numeros consolidados
- os graficos sao salvos em `resultados/figuras/` (`ttt_alvo_facil`, `ttt_alvo_medio`, `ttt_alvo_dificil` e `ttt_consolidado`, em png e pdf)
