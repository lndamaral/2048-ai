# Evolução da IA para 2048 — Narrativa Completa

## Objetivo

Criar uma inteligência artificial capaz de atingir o tile 2048 (e além) no jogo 2048 de Gabriele Cirulli, explorando diferentes abordagens de machine learning e busca em árvore.

---

## Capítulo 1: DQN — A Primeira Tentativa

**Data**: 18/Ago/2026

### Decisão
Começamos com **Deep Q-Learning (DQN)**, a abordagem clássica de deep reinforcement learning. Uma rede neural convolucional aprende a mapear estados do tabuleiro para ações, treinando por self-play.

### Arquitetura
- Estado: grid 4x4 codificado como 16 canais one-hot (log2 dos valores)
- Rede: 2 camadas convolucionais (128 filtros) + 2 fully connected (256 → 4)
- Double DQN com Experience Replay (100k buffer) + Target Network
- Reward shaping: bônus por canto, monotonicidade, células vazias
- Epsilon-greedy: 1.0 → 0.01 ao longo de 200k steps

### Resultados

| Episódios | Tempo | Score médio | 2048 alcançado |
|-----------|-------|-------------|----------------|
| 300 | ~10 min | ~1.500 | 0x |
| 1.000 | ~30 min | ~3.000 | 0x |
| 1.700 | ~3h | ~5.000 | 0x |

**O DQN nunca atingiu 2048 em 1.700 episódios de treino.**

### Análise
O DQN é muito lento para convergir no 2048. O espaço de estados é grande, o epsilon-greedy desperdiça muitas partidas em exploração aleatória, e a rede neural precisa de muitas iterações de backpropagation para aprender padrões espaciais que heurísticas simples capturam imediatamente.

---

## Capítulo 2: Expectimax — Busca Inteligente

**Data**: 18/Ago/2026

### Decisão
Enquanto o DQN treinava (sem resultados), implementamos o **Expectimax** — um algoritmo de busca em árvore que não precisa de treinamento. Usa heurísticas manuais para avaliar posições e simula jogadas futuras considerando a aleatoriedade dos tiles novos.

### Heurísticas implementadas
- **Snake pattern**: valoriza tiles organizados em formato cobra (maior no canto)
- **Monotonicidade**: recompensa linhas/colunas em ordem crescente/decrescente
- **Smoothness**: penaliza diferenças grandes entre tiles vizinhos
- **Células vazias**: mais espaço = mais opções
- **Corner bonus**: bônus por manter o max tile no canto

### Evolução da performance

**Python, depth=3 (~217ms/jogada)**:

| Métrica | Resultado |
|---------|-----------|
| Score | 20.572 |
| Max tile | 2048 |
| Win rate estimada | ~30-50% |

**Primeiro 2048 alcançado!** Score 20.572, 984 jogadas.

### Problema: velocidade
O Python era lento demais para depth > 3. Tentamos otimizar com lookup tables por linha, mas a decomposição 1D perdeu a informação espacial 2D e a qualidade caiu drasticamente (max tile 256).

### Decisão: implementar em C
Reescrevemos o engine em C mantendo as heurísticas 2D originais. Resultado:

| Depth | Python | C | Speedup |
|-------|--------|---|---------|
| 3 | 217ms | 1.4ms | 155x |
| 5 | impossível | 400ms | — |

**Bug do transpose**: a função de transpose via bit manipulation assumia um layout de bits diferente do nosso. Causava movimentos up/down incorretos (AI só movia para baixo). Corrigido com extração direta de colunas.

### C com depth adaptativo + time budget

Implementamos **iterative deepening** com time budget de 100ms:
- Começa em depth=1, vai subindo até acabar o tempo
- Garante ≤100ms por jogada, sempre usa o tempo todo
- Atinge depth=4 consistentemente

| Jogo | Score | Max Tile |
|------|-------|----------|
| 1 | 38.260 | 2048 |
| 2 | 14.508 | 1024 |
| 3 | 46.200 | **4096** |
| 4 | 46.152 | **4096** |
| 5 | 46.216 | **4096** |

**4/5 jogos atingiram 2048+, 3 chegaram a 4096!** Melhor resultado: score **76.516** com 4096+2048+1024 no tabuleiro.

---

## Capítulo 3: N-Tuple Network — O Melhor dos Dois Mundos

**Data**: 18-19/Ago/2026

### Decisão
Implementamos **N-Tuple Networks** — uma abordagem que combina:
- A **aprendizagem** do DQN (aprende sozinho jogando)
- A **avaliação rápida** do Expectimax (lookup tables, não rede neural)
- A **busca em árvore** do Expectimax (olha jogadas à frente)

### Como funciona
- 17 tuplas de 6 posições no tabuleiro (cobertura espacial completa)
- Cada tupla = lookup table com 16^6 = 16.7M entradas
- Avaliação de um board = soma de ~100 lookups (~0.5ms)
- Treinado por **forward TD(0) afterstate learning**
- 8 simetrias (rotações + espelhamentos)

### v1 — Primeira implementação (Python)

7 tuplas (4×6 + 3×4), backward TD, LR=0.0025.

| Episódios | Tempo | Score | 2048 rate | 4096 rate |
|-----------|-------|-------|-----------|-----------|
| 500 | 2 min | 6.000 | 2% | 0% |
| 2.000 | 10 min | 12.000 | 5% | 0% |
| 5.000 | 41 min | 15.000 | 15% | 0% |
| 12.100 | ~13h* | 30.020 | 68% | 7% |

*Treino de 12k incluiu tempo ocioso e múltiplas sessões.

**DQN vs N-Tuple no mesmo tempo de treino**:

| | DQN (2h) | N-Tuple (2 min) |
|---|----------|-----------------|
| 2048 alcançado | 0x | 2x |
| Episódios | 1.100 | 500 |

### Problemas identificados no v1
1. **Backward TD** — atualiza pesos no final do jogo, menos eficiente que forward TD
2. **LR baixo** (0.0025) — aprendizado lento
3. **Poucas tuplas** (7) — capacidade limitada
4. **1-ply no treino** — decisões rasas durante treinamento

### v2 — Otimizações fracassadas

Tentamos LR=0.1 → **overflow** (pesos explodiram para infinito).
Tentamos 8 tuplas com normalização por simetria → aprendeu 2x mais devagar.
Tentamos lookup tables para heurísticas por linha → perdeu qualidade (max 256).

**Lição**: nem toda otimização é melhoria. Testar antes de adotar.

### v3 — Implementação definitiva em C

**Decisões de código**:
- 17 tuplas de 6 posições (cobertura completa)
- Forward TD(0) afterstate learning
- TC-learning (per-weight adaptive LR)
- LR=0.01 com decay para 0.0005
- Gradient clipping (delta clamped a ±1000)
- Move lookup tables (pré-computa merge para 65.536 linhas)
- Grid armazena log2 dos valores (operações mais rápidas)
- Multithreading hogwild (8 threads, sem locks)
- Thread-safe random com seed por thread
- Output sem buffer para monitoramento em tempo real

**Speedup acumulado**:

| Otimização | Speedup |
|------------|---------|
| Python → C | ~50x |
| Move lookup tables | ~2-3x |
| 8 threads | ~3x |
| **Total** | **~300-450x** |

### Treino multi-stage

**Stage 1**: 500k episódios, 1-ply, 8 threads, TC-learning (~14 min)

Aprende padrões básicos e intermediários rapidamente.

| Episódios | Score | 2048 | 4096 | 8192 |
|-----------|-------|------|------|------|
| 50k | ~20.000 | ~30% | ~5% | 0% |
| 127k | ~24.700 | ~38% | ~33% | <1% |
| 500k | ~42.000 | ~40% | ~32% | ~1% |

**Stage 2**: 5M episódios, 3-ply, 8 threads, TC-learning (em andamento)

Refina com busca profunda. Cada decisão no treino simula 3 jogadas à frente.

| Episódios (stage 2) | Score | 2048 | 4096 | 8192 | 2048+ total |
|---------------------|-------|------|------|------|-------------|
| 6.400 | **66.497** | ~22% | **55%** | **12%** | **~87%** |

**Salto brutal**: apenas 6.4k episódios de 3-ply elevaram o score de 42k para 66.5k. A taxa de 4096 quase dobrou (32% → 55%) e o 8192 saltou de 1% para 12%.

---

## Capítulo 4: Comparação Final dos 3 Agentes

### No browser (http://localhost:8080)

| Agente | Tipo | Treino | Velocidade | Score típico | 2048 rate |
|--------|------|--------|------------|-------------|-----------|
| **DQN** | Rede neural pura | 3h (0 resultados) | ~1ms | ~2.000 | ~0% |
| **Expectimax** | Busca + heurísticas | Zero | ~100ms | ~40.000 | ~80% |
| **N-Tuple** | Rede aprendida + busca | ~15 min (stage 1) | ~2.4ms | ~66.000+ | ~87%+ |

### Lições aprendidas

1. **DQN é poderoso mas ineficiente para 2048**. O espaço de estados é pequeno o suficiente para métodos mais diretos.

2. **Heurísticas manuais são surpreendentemente boas**. O Expectimax com snake pattern + monotonicidade atinge 4096 sem nenhum treino.

3. **N-Tuple Networks são o sweet spot**. Combinam a velocidade de lookup tables com a capacidade de aprender padrões que humanos não codificariam.

4. **Forward TD > Backward TD**. Atualizar a cada jogada é mais eficiente que esperar o fim do jogo.

5. **TC-learning estabiliza o treino**. Pesos frequentes recebem LR menor, evitando oscilação.

6. **Multi-stage training é mais eficiente**. 1-ply rápido para padrões básicos, depois 3-ply lento para refinamento.

7. **C é essencial para performance**. O treino em Python levaria semanas; em C com threads, leva horas.

8. **Nem toda otimização funciona**. Lookup tables por linha, LR=0.1, normalização por simetria — todas falharam. Testar é obrigatório.

---

## Capítulo 5: Arquitetura Técnica

### Player (jogo no browser)

```
Browser (JS) → HTTP POST /move → Flask (Python) → N-Tuple C (shared library)
                                                    ↓
                                              5-ply search
                                              17 tuplas × 8 simetrias
                                              ~2.4ms por jogada
                                                    ↓
                                              ← ação (up/right/down/left)
```

### Trainer

```
ntuple_train.c (compilado com -O3 -lpthread)
    ├── 8 threads jogando em paralelo (hogwild)
    ├── Move lookup tables (65.536 entradas)
    ├── Forward TD(0) afterstate learning
    ├── TC-learning (per-weight adaptive LR)
    ├── Checkpoint a cada 1.000 episódios
    └── ~100 ep/s (3-ply) ou ~300 ep/s (1-ply)
```

### Ficheiros

```
ai/
├── game.py              # Motor do jogo em Python
├── dqn_agent.py         # Agente DQN (PyTorch)
├── train.py             # Treino DQN
├── expectimax_agent.py  # Expectimax Python (original)
├── expectimax_c.c       # Expectimax C (otimizado)
├── expectimax_c.so      # Shared library Expectimax
├── expectimax_native.py # Wrapper Python → C
├── ntuple_agent.py      # N-Tuple Python (legado)
├── ntuple_train.c       # Treino N-Tuple em C (definitivo)
├── ntuple_train          # Binário compilado
├── ntuple_c.c           # N-Tuple player C (shared library)
├── ntuple_c.so          # Shared library N-Tuple
├── server.py            # API Flask (3 agentes)
├── play.py              # Jogo no terminal
├── checkpoints/         # Modelos salvos
└── reports/             # Relatórios JSON de cada partida
```

---

## Próximos Passos

- **Completar stage 2** (5M episódios 3-ply) — meta: 95%+ de 2048
- **Avaliar se mais tuplas** (20+) ou tuplas maiores (7-pos) melhoram o teto
- **Análise dos relatórios JSON** — entender em quais situações a IA falha
- **Dashboard de métricas** — visualizar a evolução ao longo do treino
