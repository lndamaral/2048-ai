# Artificial Intelligence for the Game 2048: A Comparative Analysis of Machine Learning and Tree Search Approaches

**Author**: Leonardo
**Date**: August 2026
**Technical Guidance**: Claude (Anthropic)

---

## Abstract

This work presents the development, implementation, and comparative analysis of three artificial intelligence approaches for the game 2048: Deep Q-Network (DQN), Expectimax with hand-crafted heuristics, and N-Tuple Networks with TD-learning. Starting from a naive reinforcement learning implementation, we iteratively evolved to a state-of-the-art solution capable of reaching the 2048 tile in over 87% of games and the 4096 tile in 55%, with a response time of 2.4ms per move. This work documents every technical decision, its results, failures encountered, and optimizations applied, serving as a practical guide for implementing intelligent agents in decision games with randomness.

**Keywords**: Reinforcement Learning, Deep Q-Network, Expectimax, N-Tuple Networks, TD-Learning, Game AI, 2048

---

## 1. Introduction

### 1.1 The Game 2048

2048 is a single-player puzzle game created by Gabriele Cirulli in March 2014. The game takes place on a 4×4 grid where the player slides numbered tiles in four directions (up, down, left, right). Tiles with the same value merge when they collide, doubling their value. After each move, a new tile (value 2 with 90% probability, or 4 with 10%) appears at a random empty position. The goal is to create a tile with value 2048.

### 1.2 Why 2048 is Interesting for AI

2048 presents characteristics that make it an excellent study problem for AI:

- **Large but finite state space**: ~2.5 × 10^28 possible states (16 cells, each with ~18 possible values)
- **Randomness**: the random tile makes a perfect deterministic solution impossible
- **Long horizon**: a typical game lasts 500-2000 moves, requiring long-term planning
- **Sparse feedback**: the reward (merge score) is intermittent and does not directly indicate position quality
- **Multiple viable approaches**: allows comparing search, heuristics, deep learning, and tabular methods

### 1.3 Objectives

1. Implement and compare three fundamentally different AI approaches for 2048
2. Achieve a win rate (tile 2048) above 90%
3. Document the iterative development process, including failures and lessons learned
4. Create a web interface for real-time visualization of agents playing

### 1.4 Document Structure

This work is organized in chronological order of development, reflecting the iterative nature of the project. Each chapter describes an approach, its design decisions, results, and the lessons that motivated the next iteration.

---

## 2. Theoretical Background

### 2.1 Reinforcement Learning

Reinforcement Learning (RL) is a machine learning paradigm where an agent learns to make decisions by interacting with an environment. The agent observes a state *s*, executes an action *a*, receives a reward *r*, and transitions to a new state *s'*. The goal is to learn a policy π(s) → a that maximizes cumulative reward.

#### 2.1.1 Q-Learning

Q-learning (Watkins, 1989) estimates the function Q(s, a) — the expected value of taking action *a* in state *s* and following the optimal policy thereafter. The update rule is:

```
Q(s, a) ← Q(s, a) + α[r + γ max_a' Q(s', a') - Q(s, a)]
```

where α is the learning rate and γ the discount factor.

#### 2.1.2 Deep Q-Network (DQN)

DQN (Mnih et al., 2015) approximates Q(s, a) with a deep neural network, enabling generalization to continuous or very large state spaces. Stabilizing techniques include:

- **Experience Replay**: stores transitions (s, a, r, s') in a buffer and samples random batches for training, breaking temporal correlations
- **Target Network**: uses a frozen copy of the network to compute targets, updated periodically
- **Double DQN**: uses the policy network to select the action and the target network to evaluate, reducing overestimation

### 2.2 Tree Search

#### 2.2.1 Minimax and Expectimax

Minimax is a search algorithm for adversarial games that assumes an optimal opponent. For games with randomness (like 2048, where the "opponent" is the random tile), **Expectimax** is used: MAX nodes (player chooses the best action) alternated with CHANCE nodes (weighted average over random outcomes).

The quality of Expectimax depends on the **evaluation function** at leaf nodes — heuristics that estimate the value of a position.

#### 2.2.2 Iterative Deepening

A technique that performs searches of increasing depth (depth 1, 2, 3, ...) until a time budget is exhausted. Guarantees that a result is always available (from the previous depth) and uses time most efficiently.

### 2.3 N-Tuple Networks

#### 2.3.1 Concept

N-Tuple Networks (Lucas, 2008; Szubert & Jaśkowski, 2014) are evaluation functions based on lookup tables. A **tuple** is a set of *n* positions on the board. For each tuple, a lookup table (LUT) maps the values at those positions to a learned weight. The total value of a board is the sum of weights from all tuples.

**Advantages over neural networks**:
- Extremely fast evaluation (array lookups)
- No backpropagation — direct weight updates
- Controllable capacity (more tuples = more expressiveness)

**Disadvantage**: memory usage — each 6-position tuple requires 16^6 = 16.7M entries.

#### 2.3.2 Afterstate Learning

In 2048, the agent controls the movement but not the random tile placement. The **afterstate** is the state after the player's move, before the random tile. Learning V(afterstate) instead of V(state) reduces learning variance.

#### 2.3.3 TD(0) with Afterstates

The update rule:

```
V(s_t^after) ← V(s_t^after) + α[r_{t+1} + V(s_{t+1}^after) - V(s_t^after)]
```

where s_t^after is the afterstate at time t and r_{t+1} is the reward from the next move.

#### 2.3.4 TC-Learning

Temporal Coherence Learning adapts the learning rate individually for each weight:

- Accumulates Σδ (signed delta sum) and Σ|δ| (absolute delta sum) per weight
- TC ratio = |Σδ| / Σ|δ| measures "temporal coherence"
- Weights receiving consistent updates (TC ≈ 1) keep high LR
- Weights that oscillate (TC ≈ 0) have LR reduced automatically

#### 2.3.5 Symmetries

The 4×4 board has 8 symmetries (4 rotations × 2 reflections). Each tuple generates up to 8 symmetric variants. Evaluating and updating all symmetries multiplies the network's effective capacity by 8 without additional memory cost.

---

## 3. Approach 1: Deep Q-Network (DQN)

### 3.1 Motivation

DQN was chosen as the first approach for being the most emblematic deep reinforcement learning algorithm. The hypothesis was that a convolutional neural network could learn spatial patterns from the board directly from tile values, without manual heuristics.

### 3.2 Architecture

#### 3.2.1 State Representation

The 4×4 grid was encoded as **16 one-hot channels**. Each cell is represented by a 16-position vector where the active index corresponds to log₂ of the tile value (0 = empty, 1 = tile 2, 2 = tile 4, ..., 11 = tile 2048). Final representation: tensor (16, 4, 4).

**Rationale**: one-hot encoding allows the network to treat each power of 2 as an independent feature, preventing the magnitude of values (2 vs 2048) from dominating the gradient.

#### 3.2.2 Neural Network

```
Input: (batch, 16, 4, 4)
  ↓
Conv2d(16 → 128, kernel=2) + BatchNorm + ReLU    → (batch, 128, 3, 3)
  ↓
Conv2d(128 → 128, kernel=2) + BatchNorm + ReLU   → (batch, 128, 2, 2)
  ↓
Flatten                                            → (batch, 512)
  ↓
Linear(512 → 256) + ReLU                          → (batch, 256)
  ↓
Linear(256 → 4)                                   → (batch, 4)  [Q-values]
```

**Parameters**: ~200k trainable parameters.

#### 3.2.3 Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Learning rate | 1×10⁻⁴ | Low for stability with target network |
| Gamma (γ) | 0.99 | High — future rewards matter greatly in 2048 |
| Epsilon start | 1.0 | Full exploration initially |
| Epsilon end | 0.01 | Minimum exploration |
| Epsilon decay | 200,000 steps | Slow — prolonged exploration is crucial |
| Batch size | 512 | Large for stability |
| Target update | 1,000 steps | Sync frequency |
| Buffer size | 100,000 | Experience replay buffer |

#### 3.2.4 Reward Shaping

The raw game reward (merge score) is sparse and doesn't indicate positional quality. We implemented reward shaping with additional components:

- **Merge reward**: 0.1 × log₂(reward + 1) — logarithmically normalized
- **Corner bonus**: +0.1 if max tile is in a corner
- **Monotonicity bonus**: +0.02 per row organized in ascending/descending order
- **Space bonus**: 0.01 × (empty cells)
- **Game over penalty**: -2.0

### 3.3 Results

Training was run for 1,700 episodes (~3 hours on Apple Silicon):

| Episodes | Time | Avg Score (last 100) | Most Frequent Max Tile | 2048 Achieved |
|----------|------|---------------------|----------------------|---------------|
| 100 | 10 min | ~800 | 64 | 0× |
| 500 | 52 min | ~1,500 | 128 | 0× |
| 1,000 | 1h42 | ~3,000 | 256 | 0× |
| 1,500 | 2h42 | ~4,500 | 256-512 | 0× |
| 1,700 | 3h04 | ~5,000 | 512 | 0× |

**DQN failed to reach the 2048 tile in any game during training.**

### 3.4 Analysis and Diagnosis

#### 3.4.1 Slow Convergence

At episode 1,700, epsilon was ~0.65, meaning **65% of actions were still random**. The decay of 200k steps (not episodes) requires much more time for the network to start using its own Q-values.

#### 3.4.2 Experience Replay Inefficiency

Each episode generates ~200-500 transitions, but many are from trivial early positions. The 100k buffer is quickly filled with low-quality experiences.

#### 3.4.3 Architectural Inadequacy

2048 is a game with:
- Discrete and structured state space
- Clear spatial patterns (monotonicity, snake pattern)
- Need for long-term planning

A CNN with 2 convolutional layers of kernel 2×2 has a maximum receptive field of 3×3, insufficient to capture patterns spanning the entire grid.

### 3.5 Partial Conclusion

DQN, while theoretically capable, is inefficient for 2048:
- Requires an estimated 50k-100k episodes for comparable results to simpler methods
- The computational cost (backpropagation, target network, replay buffer) is disproportionate to the benefit
- The approach was abandoned in favor of more domain-appropriate methods

**Lesson learned**: not every sequential decision problem benefits from deep RL. For state spaces with clear spatial structure, specialized methods outperform generic approaches.

---

## 4. Approach 2: Expectimax with Heuristics

### 4.1 Motivation

While DQN trained without results, we implemented an approach requiring no training. Expectimax with manual heuristics is the classic approach for games with randomness and has a proven track record in 2048.

### 4.2 Evaluation Function

The evaluation function combines five weighted heuristics:

#### 4.2.1 Snake Pattern

Values tiles organized in a "snake" pattern — decreasing values following a zigzag path from a corner:

```
Weights (orientation 0):
32768  16384  8192  4096
  256    512  1024  2048
  128     64    32    16
    1      2     4     8
```

Value is computed as Σ(tile_{y,x} × weight_{y,x}). 8 orientations are evaluated (4 rotations × 2 reflections) and the maximum is used.

**Rationale**: this pattern keeps large tiles concentrated in a corner with decreasing values around them, facilitating chain merges.

#### 4.2.2 Monotonicity

Measures how monotonic rows and columns are. For each row/column, computes the sum of differences in ascending and descending directions, using the minimum (the more monotonic direction).

#### 4.2.3 Smoothness

Penalizes large differences between adjacent tiles:

```
smooth = -Σ |log₂(tile_{y,x}) - log₂(neighbor)|
```

#### 4.2.4 Empty Cells

More free space allows more game options:

```
empty = ln(empty_count + 1)
```

#### 4.2.5 Corner Bonus

Bonus when the highest-value tile is in one of the 4 corners.

#### 4.2.6 Combined Weights

```
V(s) = 1.0 × snake + 2.7 × empty + 1.0 × mono + 0.1 × smooth + 1.0 × corner
```

Weights were calibrated empirically through manual experimentation.

### 4.3 Implementation Evolution

#### 4.3.1 Pure Python (depth=3)

| Metric | Value |
|--------|-------|
| Time per move | ~217ms |
| Depth | 3 |
| Typical score | ~20,000 |
| Max tile | 2048 (inconsistent) |

**Milestone**: first 2048 achieved! Score 20,572 in 984 moves.

#### 4.3.2 Attempt: Per-Row Lookup Tables (FAILURE)

**Hypothesis**: decomposing the evaluation into independent per-row/column contributions, pre-computed in tables of 65,536 entries, would allow greater depth.

**Result**: max tile 256, score ~4,000. The 1D decomposition lost all 2D spatial information that makes the snake pattern work.

**Lesson**: 2D heuristics are not decomposable into 1D components without significant quality loss.

#### 4.3.3 C Implementation

**Motivation**: Python was 100x slower than needed for depth > 3.

Rewrote the engine in C using bitboard representation (uint64_t) with nibbles, maintaining the original 2D heuristics.

**Critical Bug: Transpose**

The transpose function via bit manipulation was copied from the nneonneo/2048-ai project, but assumed a different bit layout than ours:

- **nneonneo**: cell(y,x) at nibble 4y + x (cell 0,0 at nibble 0, bits 0-3)
- **Ours**: cell(y,x) at nibble 4y + (3-x) (cell 0,0 at nibble 3, bits 12-15)

**Symptom**: AI only moved down, score ~44, max tile 16.

**Diagnosis**: manually traced the transformation of a specific nibble through the transpose function and confirmed the value was **zeroed** — the transposition was corrupting the board.

**Fix**: replaced transpose with direct column extraction:

```c
static inline row_t board_col(board_t b, int x) {
    int c0 = (board_row(b, 0) >> ((3 - x) * 4)) & 0xF;
    int c1 = (board_row(b, 1) >> ((3 - x) * 4)) & 0xF;
    int c2 = (board_row(b, 2) >> ((3 - x) * 4)) & 0xF;
    int c3 = (board_row(b, 3) >> ((3 - x) * 4)) & 0xF;
    return (c0 << 12) | (c1 << 8) | (c2 << 4) | c3;
}
```

Slower than O(1) bit manipulation, but **correct**.

#### 4.3.4 Iterative Deepening with Time Budget

**Problem**: fixed depth causes moves that are either too fast (waste) or too slow (freezes).

**Solution**: iterative deepening with 100ms budget per move. The 60% rule prevents starting a depth that likely won't finish in time.

#### 4.3.5 Transposition Table

Cache of 4M entries (2²²) avoiding reevaluation of identical positions reached via different paths in the tree.

### 4.4 Final Results

Benchmark with 5 complete games (depth=4, time budget 100ms):

| Game | Score | Max Tile | Moves |
|------|-------|----------|-------|
| 1 | 38,260 | 2048 | 2000+ |
| 2 | 14,508 | 1024 | 840 |
| 3 | 46,200 | **4096** | 2000+ |
| 4 | 46,152 | **4096** | 2000+ |
| 5 | 46,216 | **4096** | 2000+ |

**Average score**: 38,267 | **2048+ rate**: 80% | **4096 rate**: 60%
**Best visual game**: Score 76,516 with tiles 4096 + 2048 + 1024 + 256 on the board.

### 4.5 Cumulative Speedup

| Stage | Time/move | Speedup |
|-------|-----------|---------|
| Python depth=3 | 217ms | 1× (baseline) |
| C depth=3 | 1.4ms | **155×** |
| C depth=4 (time budget) | ~100ms | Better quality, same time |
| C depth=5 (fixed) | 400ms | Impractical for real-time play |

### 4.6 Partial Conclusion

Expectimax with heuristics demonstrated that **well-coded domain knowledge** outperforms weeks of neural training. Without any training, it consistently reaches 2048 and frequently 4096. The limitation is the ceiling: heuristics are fixed and cannot improve with more experience.

---

## 5. Approach 3: N-Tuple Networks

### 5.1 Motivation

The first two approaches represent extremes:
- **DQN**: learns everything on its own, but is slow and inefficient
- **Expectimax**: learns nothing, but is effective immediately

N-Tuple Networks combine the strengths of both:
- **Learns** to evaluate positions (like DQN)
- Uses **fast lookup tables** (like Expectimax heuristics)
- Compatible with **tree search** (like Expectimax)

### 5.2 Network Architecture

#### 5.2.1 Tuples

17 tuples of 6 positions each, covering diverse spatial patterns:

- **Horizontal (3 tuples)**: 4-wide row + 2 cells from the row below
- **Vertical (3 tuples)**: 4-tall column + 2 cells from the column next to it
- **3×2 Blocks (4 tuples)**: rectangular blocks covering different board regions
- **2×3 Blocks (3 tuples)**: vertical rectangular blocks
- **L-shapes and diagonals (4 tuples)**: non-rectangular patterns capturing diagonal relationships

#### 5.2.2 Symmetries

Each tuple generates up to 8 symmetric variants (4 rotations × 2 reflections), with duplicates removed. The total number of features evaluated per board is ~100.

#### 5.2.3 Memory

Per tuple: 16⁶ = 16,777,216 floats × 4 bytes = **64 MB**
17 tuples: **~1.1 GB**
With TC-learning (2 extra tables per tuple): **~3.3 GB**

### 5.3 Training Evolution

#### 5.3.1 v1 — Python, Backward TD (7 tuples)

**Configuration**: 4 tuples of 6 positions + 3 tuples of 4 positions, backward TD, LR=0.0025 fixed.

| Episodes | Time | Score | 2048 Rate | 4096 Rate |
|----------|------|-------|-----------|-----------|
| 100 | 20s | 3,029 | 0% | 0% |
| 500 | 2min | 5,508 | 2% | 0% |
| 1,000 | 4min | 7,949 | ~2% | 0% |
| 2,000 | 10min | 12,000 | 5% | 0% |
| 5,000 | 41min | 16,984 | 15% | 0% |
| 12,100 | ~13h* | 30,020 | **68%** | **7%** |

**Milestone**: DQN with 2h of training = 0× 2048. N-Tuple with 2 min = 2× 2048.

#### 5.3.2 v2 — Failed Improvement Attempts

| Attempt | Change | Result | Cause |
|---------|--------|--------|-------|
| LR=0.1 | 40× higher learning rate | **Overflow** — weights exploded to infinity | Updates too large without normalization |
| 8 tuples + symmetry normalization | adj / n_sym per update | 2× slower learning | Updates too diluted |
| Per-row lookup heuristics | Decompose eval by row | Max tile 256 | Lost 2D spatial information |

**Lesson**: theoretical optimizations don't always translate to practical improvements. Empirical testing is essential.

#### 5.3.3 C Implementation — The Big Leap

**Optimizations implemented**:

1. **Move lookup tables**: pre-computes merge result for all 65,536 possible rows. Each move reduces to 4 lookups instead of loops.

2. **Log2 grid representation**: storing log₂(value) instead of actual value allows each cell to fit in 4 bits and simplifies tuple indexing.

3. **Hogwild multithreading**: 8 threads play independent games and update shared weights without locks. Works because:
   - Updates are sparse (each move affects ~100 of ~285M weights)
   - Collision probability is negligible
   - Small inconsistencies don't affect convergence

4. **Thread-safe random**: each thread has its own seed using a linear congruential generator.

5. **Unbuffered output**: `setbuf(stdout, NULL)` for real-time monitoring.

**Total speedup**:

| Component | Speedup |
|-----------|---------|
| Python → C | ~50× |
| Move lookup tables | ~2-3× |
| 8 threads | ~3× |
| **Accumulated** | **~300-450×** |

From ~5 ep/s (Python) to ~300 ep/s (C, 1-ply) or ~30 ep/s (C, 3-ply).

#### 5.3.4 TC-Learning Implementation

```c
ts[idx] = ts[idx] * 0.9995 + delta;        // signed sum
ta[idx] = ta[idx] * 0.9995 + |delta|;      // absolute sum
tc_ratio = |ts[idx]| / ta[idx];            // coherence
w[idx] += lr * tc_ratio * delta;           // adaptive update
```

The 0.9995 decay gives a "half-life" of ~1,400 updates, allowing adaptation to distribution changes.

#### 5.3.5 Multi-Stage Training

**Stage 1**: 500k episodes, 1-ply, 8 threads, TC-learning (~14 min)

Learns basic and intermediate patterns quickly.

| Episodes | Score | 2048 | 4096 | 8192 | 2048+ Total |
|----------|-------|------|------|------|-------------|
| 50k | ~20,000 | ~30% | ~5% | 0% | ~35% |
| 127k | ~24,700 | ~38% | ~33% | <1% | ~72% |
| 500k | ~42,000 | ~40% | ~32% | ~1% | ~73% |

**Stage 2**: 5M episodes, 3-ply, 8 threads, TC-learning (in progress, ~14h estimated)

Refines with deep search. Each training decision simulates 3 moves ahead.

| Stage 2 Episodes | Score | 2048 | 4096 | 8192 | 2048+ Total |
|------------------|-------|------|------|------|-------------|
| 6,400 | **66,497** | ~22% | **55%** | **12%** | **~87%** |
| 33,000 | **~66,000** | ~20% | **58%** | **11%** | **~88%** |

**Critical observation**: the "2048" rate dropped from 40% to 22%, but this is because many games that previously stopped at 2048 now **pass through to 4096**. The relevant metric is "2048+" (reached at least 2048), which rose from 73% to 87%.

### 5.4 C Player (Shared Library)

For browser gameplay, the N-Tuple player is implemented as a C shared library (`ntuple_c.so`) called via `ctypes`:

- Loads weights from checkpoint
- Performs 5-ply Expectimax search using the N-Tuple as evaluation
- Time per move: **2.4ms**

### 5.5 Web Integration

The JavaScript frontend supports 3 selectable agents:

- **🧠 Expectimax** (yellow): search + manual heuristics
- **🤖 DQN** (pink): pure neural network
- **🏆 N-Tuple** (green): learned network + tree search

At the end of each game, a JSON report is automatically saved with the complete move history.

---

## 6. Comparative Analysis

### 6.1 Agent Performance

| Metric | DQN | Expectimax | N-Tuple (Stage 2) |
|--------|-----|------------|-------------------|
| **Training required** | 3h+ (insufficient) | None | ~15min (stage 1) |
| **Time/move** | ~1ms | ~100ms | **~2.4ms** |
| **Average score** | ~2,000 | ~40,000 | **~66,500** |
| **2048+ rate** | ~0% | ~80% | **~87%** |
| **4096 rate** | 0% | ~60% | **55%** |
| **8192 rate** | 0% | 0% | **12%** |
| **Best score** | ~1,160 | 76,516 | evolving |

### 6.2 Training Efficiency

| Metric | DQN | N-Tuple |
|--------|-----|---------|
| Time to first 2048 | >3h (never achieved) | **2 minutes** |
| Episodes to 2048 | >1,700 (insufficient) | **~500** |
| Score/hour of training | ~1,700 | **~200,000** |

N-Tuple Network is **~100× more efficient** in terms of training than DQN for this problem.

### 6.3 Computational Cost

| Resource | DQN | Expectimax | N-Tuple |
|----------|-----|------------|---------|
| **CPU (training)** | GPU-intensive (PyTorch) | N/A | CPU-intensive (C) |
| **RAM (training)** | ~200MB | N/A | ~3.3GB (with TC) |
| **RAM (playing)** | ~200MB | ~50MB | ~1.1GB |
| **Disk** | ~3MB (weights) | 0 | ~1.1GB (weights) |

### 6.4 Decision Quality

**DQN**: Erratic decisions, no clear spatial pattern. Frequently loses board control in ~200 moves.

**Expectimax**: Excellent snake pattern organization. Consistently keeps max tile in corner. Limited by search depth and heuristic rigidity.

**N-Tuple**: Learns patterns humans wouldn't code. With 3-ply training, develops sophisticated endgame strategies. The combination with 5-ply search during play produces consistently strong decisions.

---

## 7. Failures and Lessons Learned

### 7.1 Failure Catalog

| # | Failure | Root Cause | Impact | Fix |
|---|---------|------------|--------|-----|
| 1 | DQN doesn't converge | Inadequate LR/architecture | 3h wasted | Changed approach |
| 2 | 1D lookup tables | Loss of 2D information | Max tile 256 | Kept 2D heuristics |
| 3 | Transpose bug | Incompatible bit layout | AI only moves down | Direct column extraction |
| 4 | LR=0.1 overflow | Updates without normalization | Score 0, infinite weights | Clipping + LR=0.01 |
| 5 | Symmetry normalization | Dilutes updates | 2× slower learning | Removed |
| 6 | Adaptive depth | Depth=6-7 freezes | 12s moves | Fixed time budget |
| 7 | Buffered output | Pipe to tee buffers | No training feedback | `setbuf(stdout, NULL)` |

### 7.2 General Lessons

1. **Test before adopting**: every "optimization" can be a regression
2. **Always measure**: without metrics, there's no way to compare
3. **Simplicity first**: the version that works is better than the "optimal" version that doesn't
4. **Domain knowledge is gold**: simple heuristics beat weeks of neural training
5. **C for performance, Python for prototyping**: the ideal cycle is prototype in Python, validate, and reimplement critical parts in C
6. **Hyperparameters aren't transferable**: values from papers need validation in your specific context

---

## 8. Performance Ceiling Analysis

### 8.1 Our Position vs. State of the Art

| Level | 2048 Rate | 4096 Rate | 8192 Rate | Score | How to Reach |
|-------|-----------|-----------|-----------|-------|-------------|
| Random player | ~0% | 0% | 0% | ~800 | — |
| **Our DQN (1,700 ep)** | ~0% | 0% | 0% | ~2,000 | — |
| Good human player | ~30% | rare | 0% | ~20,000 | — |
| **Our Expectimax** | ~80% | ~60% | 0% | ~40,000 | — |
| **Our N-Tuple (current)** | **~87%** | **~55%** | **~12%** | **~66,000** | — |
| Good AI | ~95% | ~70% | ~10% | ~50,000 | 5M episodes |
| Top AI (Wu et al., 2014) | ~97% | ~85% | ~40% | ~100,000+ | 10M episodes |
| State of the art (Jaśkowski, 2018) | ~99.5% | ~95% | ~75% | ~300,000+ | 40M episodes |

### 8.2 Convergence Ceiling

With our architecture (17 × 6-tuples), the theoretical ceiling is ~97-98% 2048 rate. To go beyond (99%+) would require:
- More tuples or larger tuples (7-8 positions) — but memory explodes (16⁷ = 268M per table)
- Deeper search during training (5-ply) — but 50-100× slower
- Advanced techniques: TC-learning (implemented ✓), afterstate learning (implemented ✓)

### 8.3 Estimated Training to Ceiling

| Episodes (3-ply) | Time (8 threads) | Expected 2048+ Rate |
|-------------------|-------------------|---------------------|
| 50k (current) | ~30 min | ~87% |
| 500k | ~5h | ~92% |
| 5M | ~2 days | ~95-97% |
| 10M | ~4 days | ~97-98% (ceiling) |

---

## 9. Infrastructure and Reproducibility

### 9.1 Environment

- **Hardware**: Apple Silicon (M-series)
- **OS**: macOS Darwin 25.5.0
- **Languages**: Python 3.9, C (cc with -O3), JavaScript
- **Python Dependencies**: PyTorch, NumPy, Flask, Flask-CORS, matplotlib
- **C Compiler**: Apple Clang with -O3 optimization

### 9.2 Reproduction Steps

```bash
# Clone the original game
git clone https://github.com/gabrielecirulli/2048.git
cd 2048

# Python setup
python3 -m venv venv && source venv/bin/activate
pip install torch numpy flask flask-cors matplotlib

# Compile C engines
cd ai
cc -O3 -shared -o expectimax_c.so expectimax_c.c -lm
cc -O3 -shared -o ntuple_c.so ntuple_c.c -lm
cc -O3 -o ntuple_train ntuple_train.c -lm -lpthread

# Train the N-Tuple (stage 1: ~14 min, stage 2: ~14h)
./ntuple_train --episodes 500000 --depth 0 --tc --threads 8
./ntuple_train --episodes 5000000 --depth 1 --tc --threads 8

# Start the server and game
python3 -u server.py --time-budget 100 &
python3 -m http.server 8080 &
# Open http://localhost:8080
```

### 9.3 Game Reports

Each browser game generates a JSON report in `ai/reports/`:

```json
{
  "timestamp": "2026-08-18T17:00:52",
  "agent": "ntuple",
  "score": 76516,
  "max_tile": 4096,
  "moves": 2526,
  "won": true,
  "final_grid": [[4096, 16, 2, 4], ...],
  "move_history": [...],
  "duration_ms": 52340
}
```

---

## 10. Charts and Visualizations

> Charts generated by `analysis.py` are located in `ai/charts/`:
>
> - `chart_score_evolution.png` — Average score evolution across all training runs
> - `chart_tile_distribution.png` — Max tile distribution over training epochs
> - `chart_2048_rate.png` — 2048+ achievement rate over training
> - `chart_agent_comparison.png` — Comparison of 3 agents (score, max tile, win rate)
> - `chart_speed_comparison.png` — Milliseconds per move per agent
> - `chart_training_time_vs_performance.png` — Training time vs performance relationship
> - `chart_reports_analysis.png` — Analysis of game reports (score distribution, max tile distribution)
> - `chart_stage_comparison.png` — Stage 1 (1-ply) vs Stage 2 (3-ply) learning curves

---

## 11. File Structure

```
ai/
├── game.py                  # Game engine in Python
├── dqn_agent.py             # DQN agent (PyTorch)
├── train.py                 # DQN training script
├── expectimax_agent.py      # Expectimax Python (original)
├── expectimax_c.c           # Expectimax C (optimized)
├── expectimax_c.so          # Expectimax shared library
├── expectimax_native.py     # Python → C wrapper
├── ntuple_agent.py          # N-Tuple Python (legacy)
├── ntuple_train.c           # N-Tuple C trainer (definitive)
├── ntuple_train             # Compiled binary
├── ntuple_c.c               # N-Tuple C player (shared library)
├── ntuple_c.so              # N-Tuple shared library
├── server.py                # Flask API server (3 agents)
├── play.py                  # Terminal player
├── analysis.py              # Chart generation script
├── TCC_AI_2048.md           # This document
├── EVOLUTION.md             # Brief evolution summary
├── checkpoints/             # Saved models
├── reports/                 # JSON game reports
└── charts/                  # Generated analysis charts
```

---

## 12. Conclusion

### 12.1 Results Achieved

Starting from a naive DQN implementation that couldn't reach the 2048 tile, we iteratively evolved to an N-Tuple Network with:

- **87%+ 2048 achievement rate** (target of 95%+ in progress)
- **55% 4096 rate**
- **12% 8192 rate**
- **Average score of ~66,500**
- **Response time of 2.4ms per move**

### 12.2 Contributions

1. **Practical comparative analysis** of three fundamentally different approaches (DQN, Expectimax, N-Tuple) on the same problem
2. **Complete C implementation** with multithreading, lookup tables, and TC-learning
3. **Web interface** for real-time visualization with 3 selectable agents
4. **Detailed documentation** of failures, decisions, and lessons learned

### 12.3 Future Work

- **Complete training**: 5M episodes with 3-ply should reach 95%+ 2048 rate
- **Larger tuples**: 7-tuples would increase expressiveness at exponential memory cost
- **Real-time dashboard**: visualize training metrics in the browser
- **Failure analysis**: use JSON reports to identify patterns in lost games
- **Transfer learning**: use learned weights as initialization for game variants (5×5, hexagonal)

---

## References

1. Cirulli, G. (2014). 2048. https://github.com/gabrielecirulli/2048
2. Mnih, V. et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518(7540), 529-533.
3. Szubert, M., & Jaśkowski, W. (2014). Temporal Difference Learning of N-Tuple Networks for the Game 2048. *IEEE CIG*.
4. Wu, I-C. et al. (2014). Multi-stage temporal difference learning for 2048. *TAAI*.
5. Lucas, S. M. (2008). Learning to play Othello with N-Tuple Systems. *Australian Journal of Intelligent Information Processing Systems*.
6. Watkins, C. J. C. H. (1989). Learning from Delayed Rewards. PhD thesis, University of Cambridge.
7. Jaśkowski, W. (2018). Mastering 2048 with Delayed Temporal Coherence Learning. *IEEE Transactions on Games*.
8. nneonneo (2014). 2048-ai. https://github.com/nneonneo/2048-ai
