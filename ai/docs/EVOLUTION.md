# AI Evolution for 2048 — Complete Narrative

## Objective

Create an artificial intelligence capable of reaching the 2048 tile (and beyond) in Gabriele Cirulli's 2048 game, exploring different machine learning and tree search approaches.

---

## Chapter 1: DQN — The First Attempt

**Date**: Aug 18, 2026

### Decision
We started with **Deep Q-Learning (DQN)**, the classic deep reinforcement learning approach. A convolutional neural network learns to map board states to actions, training through self-play.

### Architecture
- State: 4x4 grid encoded as 16 one-hot channels (log2 of values)
- Network: 2 convolutional layers (128 filters) + 2 fully connected (256 -> 4)
- Double DQN with Experience Replay (100k buffer) + Target Network
- Reward shaping: corner bonus, monotonicity, empty cells
- Epsilon-greedy: 1.0 -> 0.01 over 200k steps

### Results

| Episodes | Time | Average Score | 2048 Achieved |
|----------|------|---------------|---------------|
| 300 | ~10 min | ~1,500 | 0x |
| 1,000 | ~30 min | ~3,000 | 0x |
| 1,700 | ~3h | ~5,000 | 0x |

**DQN never reached 2048 in 1,700 training episodes.**

### Analysis
DQN is too slow to converge on 2048. The state space is large, epsilon-greedy wastes many games on random exploration, and the neural network needs many backpropagation iterations to learn spatial patterns that simple heuristics capture immediately.

---

## Chapter 2: Expectimax — Intelligent Search

**Date**: Aug 18, 2026

### Decision
While DQN was training (without results), we implemented **Expectimax** — a tree search algorithm that requires no training. It uses manual heuristics to evaluate positions and simulates future moves considering the randomness of new tiles.

### Implemented heuristics
- **Snake pattern**: values tiles organized in a snake pattern (highest in corner)
- **Monotonicity**: rewards rows/columns in ascending/descending order
- **Smoothness**: penalizes large differences between neighboring tiles
- **Empty cells**: more space = more options
- **Corner bonus**: bonus for keeping the max tile in the corner

### Performance evolution

**Python, depth=3 (~217ms/move)**:

| Metric | Result |
|--------|--------|
| Score | 20,572 |
| Max tile | 2048 |
| Estimated win rate | ~30-50% |

**First 2048 achieved!** Score 20,572, 984 moves.

### Problem: speed
Python was too slow for depth > 3. We tried optimizing with per-row lookup tables, but the 1D decomposition lost the 2D spatial information and quality dropped drastically (max tile 256).

### Decision: implement in C
We rewrote the engine in C keeping the original 2D heuristics. Result:

| Depth | Python | C | Speedup |
|-------|--------|---|---------|
| 3 | 217ms | 1.4ms | 155x |
| 5 | impossible | 400ms | — |

**Transpose bug**: the transpose function via bit manipulation assumed a different bit layout than ours. This caused incorrect up/down moves (AI only moved down). Fixed with direct column extraction.

### C with adaptive depth + time budget

We implemented **iterative deepening** with a 100ms time budget:
- Starts at depth=1, increases until time runs out
- Guarantees <=100ms per move, always uses the full time
- Consistently reaches depth=4

| Game | Score | Max Tile |
|------|-------|----------|
| 1 | 38,260 | 2048 |
| 2 | 14,508 | 1024 |
| 3 | 46,200 | **4096** |
| 4 | 46,152 | **4096** |
| 5 | 46,216 | **4096** |

**4/5 games reached 2048+, 3 reached 4096!** Best result: score **76,516** with 4096+2048+1024 on the board.

---

## Chapter 3: N-Tuple Network — The Best of Both Worlds

**Date**: Aug 18-19, 2026

### Decision
We implemented **N-Tuple Networks** — an approach that combines:
- The **learning** of DQN (learns by itself through playing)
- The **fast evaluation** of Expectimax (lookup tables, not neural network)
- The **tree search** of Expectimax (looks moves ahead)

### How it works
- 17 tuples of 6 board positions (complete spatial coverage)
- Each tuple = lookup table with 16^6 = 16.7M entries
- Evaluating a board = sum of ~100 lookups (~0.5ms)
- Trained by **forward TD(0) afterstate learning**
- 8 symmetries (rotations + reflections)

### v1 — First implementation (Python)

7 tuples (4x6 + 3x4), backward TD, LR=0.0025.

| Episodes | Time | Score | 2048 rate | 4096 rate |
|----------|------|-------|-----------|-----------|
| 500 | 2 min | 6,000 | 2% | 0% |
| 2,000 | 10 min | 12,000 | 5% | 0% |
| 5,000 | 41 min | 15,000 | 15% | 0% |
| 12,100 | ~13h* | 30,020 | 68% | 7% |

*12k training included idle time and multiple sessions.

**DQN vs N-Tuple in the same training time**:

| | DQN (2h) | N-Tuple (2 min) |
|---|----------|-----------------|
| 2048 achieved | 0x | 2x |
| Episodes | 1,100 | 500 |

### Issues identified in v1
1. **Backward TD** — updates weights at end of game, less efficient than forward TD
2. **Low LR** (0.0025) — slow learning
3. **Few tuples** (7) — limited capacity
4. **1-ply in training** — shallow decisions during training

### v2 — Failed optimizations

Tried LR=0.1 -> **overflow** (weights exploded to infinity).
Tried 8 tuples with symmetry normalization -> learned 2x slower.
Tried lookup tables for per-row heuristics -> lost quality (max 256).

**Lesson**: not every optimization is an improvement. Test before adopting.

### v3 — Definitive C implementation

**Code decisions**:
- 17 tuples of 6 positions (complete coverage)
- Forward TD(0) afterstate learning
- TC-learning (per-weight adaptive LR)
- LR=0.01 with decay to 0.0005
- Gradient clipping (delta clamped to +/-1000)
- Move lookup tables (pre-computes merge for 65,536 rows)
- Grid stores log2 of values (faster operations)
- Multithreading hogwild (8 threads, no locks)
- Thread-safe random with per-thread seed
- Unbuffered output for real-time monitoring

**Cumulative speedup**:

| Optimization | Speedup |
|-------------|---------|
| Python -> C | ~50x |
| Move lookup tables | ~2-3x |
| 8 threads | ~3x |
| **Total** | **~300-450x** |

### Multi-stage training

**Stage 1**: 500k episodes, 1-ply, 8 threads, TC-learning (~14 min)

Learns basic and intermediate patterns quickly.

| Episodes | Score | 2048 | 4096 | 8192 |
|----------|-------|------|------|------|
| 50k | ~20,000 | ~30% | ~5% | 0% |
| 127k | ~24,700 | ~38% | ~33% | <1% |
| 500k | ~42,000 | ~40% | ~32% | ~1% |

**Stage 2**: 5M episodes, 3-ply, 8 threads, TC-learning (in progress)

Refines with deep search. Each training decision simulates 3 moves ahead.

| Episodes (stage 2) | Score | 2048 | 4096 | 8192 | 2048+ total |
|---------------------|-------|------|------|------|-------------|
| 6,400 | **66,497** | ~22% | **55%** | **12%** | **~87%** |

**Massive leap**: only 6.4k episodes of 3-ply raised the score from 42k to 66.5k. The 4096 rate nearly doubled (32% -> 55%) and 8192 jumped from 1% to 12%.

---

## Chapter 4: Final Comparison of the 3 Agents

### In the browser (http://localhost:8080)

| Agent | Type | Training | Speed | Typical Score | 2048 rate |
|-------|------|----------|-------|---------------|-----------|
| **DQN** | Pure neural network | 3h (0 results) | ~1ms | ~2,000 | ~0% |
| **Expectimax** | Search + heuristics | Zero | ~100ms | ~40,000 | ~80% |
| **N-Tuple** | Learned network + search | ~15 min (stage 1) | ~2.4ms | ~66,000+ | ~87%+ |

### Lessons learned

1. **DQN is powerful but inefficient for 2048**. The state space is small enough for more direct methods.

2. **Manual heuristics are surprisingly good**. Expectimax with snake pattern + monotonicity reaches 4096 without any training.

3. **N-Tuple Networks are the sweet spot**. They combine the speed of lookup tables with the ability to learn patterns that humans wouldn't code.

4. **Forward TD > Backward TD**. Updating at each move is more efficient than waiting until the end of the game.

5. **TC-learning stabilizes training**. Frequently updated weights receive a lower LR, preventing oscillation.

6. **Multi-stage training is more efficient**. Fast 1-ply for basic patterns, then slow 3-ply for refinement.

7. **C is essential for performance**. Training in Python would take weeks; in C with threads, it takes hours.

8. **Not every optimization works**. Per-row lookup tables, LR=0.1, symmetry normalization — all failed. Testing is mandatory.

---

## Chapter 5: Technical Architecture

### Player (game in the browser)

```
Browser (JS) -> HTTP POST /move -> Flask (Python) -> N-Tuple C (shared library)
                                                    |
                                              5-ply search
                                              17 tuples x 8 symmetries
                                              ~2.4ms per move
                                                    |
                                              <- action (up/right/down/left)
```

### Trainer

```
ntuple_train.c (compiled with -O3 -lpthread)
    |-- 8 threads playing in parallel (hogwild)
    |-- Move lookup tables (65,536 entries)
    |-- Forward TD(0) afterstate learning
    |-- TC-learning (per-weight adaptive LR)
    |-- Checkpoint every 1,000 episodes
    '-- ~100 ep/s (3-ply) or ~300 ep/s (1-ply)
```

### Files

```
ai/
|-- game.py              # Game engine in Python
|-- dqn_agent.py         # DQN agent (PyTorch)
|-- train.py             # DQN training
|-- expectimax_agent.py  # Expectimax Python (original)
|-- expectimax_c.c       # Expectimax C (optimized)
|-- expectimax_c.so      # Expectimax shared library
|-- expectimax_native.py # Python -> C wrapper
|-- ntuple_agent.py      # N-Tuple Python (legacy)
|-- ntuple_train.c       # N-Tuple training in C (definitive)
|-- ntuple_train          # Compiled binary
|-- ntuple_c.c           # N-Tuple player C (shared library)
|-- ntuple_c.so          # N-Tuple shared library
|-- server.py            # Flask API (3 agents)
|-- play.py              # Terminal game
|-- checkpoints/         # Saved models
'-- reports/             # JSON reports for each game
```

---

## Next Steps

- **Complete stage 2** (5M episodes 3-ply) — goal: 95%+ of 2048
- **Evaluate whether more tuples** (20+) or larger tuples (7-pos) improve the ceiling
- **JSON report analysis** — understand in which situations the AI fails
- **Metrics dashboard** — visualize the evolution throughout training
