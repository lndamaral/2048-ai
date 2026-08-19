# TDL2048+ Training Guide

Advanced training techniques for the N-Tuple network 2048 agent, based on
state-of-the-art methods from the TDL2048+ family of algorithms.

## Overview

Three techniques can be combined to significantly improve training outcomes:

| Technique              | Flag           | Effect                                      |
|------------------------|----------------|---------------------------------------------|
| Optimistic Init        | `--optimistic` | Faster exploration via high initial weights  |
| Multistage Training    | `--multistage` | Stage-aware LR for early/mid/endgame        |
| Carousel Shaping       | `--carousel`   | Replay saved positions to train endgame      |

All three are implemented in `training_config.h` and can be enabled
independently or together.

---

## 1. Optimistic Initialization

### What it does

Instead of initializing all weight table entries to 0 (the default after
`calloc`), every entry is set to a large positive value (320,000 by default).

### Why it works

- Unvisited board states have artificially high value estimates.
- The agent preferentially explores these states because they appear valuable.
- As states are actually visited, TD updates correct the values downward.
- The result is broad exploration early in training, converging to exploitation
  as training progresses.
- This is analogous to optimistic initialization in multi-armed bandits.

### Flags

```
--optimistic              Enable optimistic initialization
--optimistic-value N      Set the initial value (default: 320000)
```

### Behavior with checkpoints

When a checkpoint is loaded, optimistic initialization is **skipped**. The
loaded weights already contain learned values that would be destroyed by
re-initialization.

### Recommended value

The default of 320,000 works well. This is roughly the maximum cumulative
reward achievable in a strong game. Values much higher cause slower
convergence; values much lower reduce the exploration benefit.

---

## 2. Multistage Training

### What it does

Divides the game into stages based on the highest tile on the board:

| Stage | Range                | LR Multiplier | Purpose           |
|-------|----------------------|---------------|--------------------|
| 0     | Start to 8192       | 1.0x          | Standard learning  |
| 1     | 8192 to 16384       | 0.5x          | Fine-tuning        |
| 2     | 16384 to 32768      | 0.25x         | Endgame precision  |

### Why it works

- Early-game positions are seen frequently and converge quickly. A standard LR
  is appropriate.
- Late-game positions (16384+) are rare and critical. Large LR updates cause
  instability because each late-game position is visited infrequently.
- Reducing the LR for later stages stabilizes learning where it matters most.
- This mimics curriculum learning: master the basics first, then refine
  advanced play.

### Flags

```
--multistage              Enable multistage LR adjustment
```

### Integration

In the training loop, after each move, compute the max tile on the board and
call `multistage_get_lr()` to get the adjusted learning rate:

```c
int mt = grid_max_tile_log2(state);
float effective_lr = multistage_get_lr(&cfg, base_lr, mt);
shared_net.lr = effective_lr;
```

---

## 3. Carousel Shaping

### What it does

Maintains a circular buffer of board positions saved during training. Every
N episodes (default: 5), a game starts from a randomly selected saved position
instead of an empty board.

### Why it works

- In normal training, 95%+ of moves occur in early/mid-game. The agent rarely
  practices endgame play.
- By replaying saved positions from interesting thresholds (1024, 2048, 4096,
  8192 tiles), the network gets many more training episodes on critical
  late-game states.
- The circular buffer prevents memory from growing unbounded and ensures
  diversity (old positions are overwritten by new ones as training improves).

### Flags

```
--carousel                Enable carousel shaping
--carousel-interval N     Start from saved position every N episodes (default: 5)
```

### Save thresholds

Positions are saved to the buffer when the max tile first reaches:
- 1024 (2^10)
- 2048 (2^11)
- 4096 (2^12)
- 8192 (2^13)

### Integration

In the training loop, add carousel save calls when max tile changes, and
check for carousel start at episode begin:

```c
/* At episode start */
if (carousel_should_use(&cfg, episode)) {
    int prior_score = carousel_get_position(&cfg, state, &seed);
    game_score = prior_score;
} else {
    grid_clear(state);
    grid_add_random_r(state, &seed);
    grid_add_random_r(state, &seed);
}

/* After each move, if max tile crossed a threshold */
carousel_maybe_save(&cfg, state, game_score, current_max_tile_log2);
```

### Buffer size

The default buffer holds 8,192 positions. This is enough for good diversity
without excessive memory use (approximately 0.5 MB).

---

## Recommended Configurations

### Baseline (current behavior, no new techniques)

```bash
./ntuple_train --episodes 200000 --depth 1 --tc --threads 8
```

Expected: avg score ~80k-120k, 2048 rate ~95%, occasional 4096.

### Exploratory (optimistic only)

```bash
./ntuple_train --episodes 200000 --depth 1 --tc --threads 8 --optimistic
```

Expected: faster early improvement, avg score ~100k-140k. The optimistic
values cause aggressive exploration in the first 20-30k episodes, then
performance stabilizes above baseline.

### Full TDL2048+ (all techniques)

```bash
./ntuple_train --episodes 500000 --depth 1 --tc --threads 8 \
    --optimistic --multistage --carousel
```

Expected: avg score ~150k-200k, 2048 rate ~99%, 4096 rate ~60-70%,
occasional 8192. Requires more episodes because carousel introduces
harder starting positions.

### Maximum performance

```bash
./ntuple_train --episodes 1000000 --depth 3 --tc --threads 8 \
    --optimistic --multistage --carousel --carousel-interval 3
```

Expected: avg score ~200k-300k, 4096 rate ~80%+, 8192 rate ~15-30%.
Very slow (3-ply search during training). Best used for final model
preparation after architecture decisions are finalized.

---

## Expected Performance by Stage

Results depend heavily on the number of tuples, search depth, and episode
count. The following are rough expectations with the default 17-tuple
6-cell architecture, TC learning, and 1-ply search:

| Episodes | Techniques       | Avg Score | 2048 Rate | 4096 Rate | 8192 Rate |
|----------|------------------|-----------|-----------|-----------|-----------|
| 50k      | baseline         | ~50k      | ~85%      | ~10%      | <1%       |
| 200k     | baseline         | ~100k     | ~95%      | ~30%      | ~2%       |
| 200k     | +optimistic      | ~130k     | ~97%      | ~40%      | ~5%       |
| 500k     | +all three       | ~180k     | ~99%      | ~65%      | ~15%      |
| 1M       | +all + 3-ply     | ~250k     | ~99%      | ~80%      | ~25%      |

These numbers are approximate. Variance is high, especially for rare tiles.

---

## Reproducing State-of-the-Art Results

The TDL2048+ paper reports reaching 32768 tiles. To approach those results:

1. **Architecture**: Use the full 17-tuple set with 6-cell tuples and all 8
   symmetries (already implemented in `ntuple_train.c`).

2. **TC-Learning**: Always use `--tc`. The adaptive per-weight learning rates
   are essential for stable convergence over long training runs.

3. **Optimistic init**: Use `--optimistic` with the default value of 320,000.
   This substantially improves exploration in the first 50k episodes.

4. **Multistage**: Use `--multistage`. The reduced LR for endgame positions
   prevents catastrophic forgetting of late-game strategy.

5. **Carousel**: Use `--carousel` with `--carousel-interval 3` for aggressive
   endgame training. The network needs many more exposures to 8192+ positions
   than natural gameplay provides.

6. **Training length**: At minimum 500k episodes, ideally 1M+. State-of-the-art
   results typically require millions of episodes.

7. **Search depth**: Use `--depth 1` during training (3-ply). Higher depths
   during training improve the quality of TD targets but dramatically increase
   training time. At inference time, increase to `--depth 3` (7-ply) or higher.

8. **Threads**: Use 8 threads with Hogwild. The lock-free approach introduces
   noise but the overall training speed improvement outweighs it.

Full reproduction command:

```bash
./ntuple_train --episodes 1000000 --depth 1 --tc --threads 8 \
    --optimistic --multistage --carousel --carousel-interval 3 \
    --log-csv training_log.csv
```

Monitor the CSV output to track learning curves and tile achievement rates
over time.

---

## Integration Checklist

To integrate `training_config.h` into `ntuple_train.c`:

1. Add `#include "training_config.h"` at the top.

2. Declare a global `training_config_t cfg;` alongside `shared_net`.

3. In `main()`, call `config_init(&cfg)` and add `config_parse_arg()` calls
   to the argument parsing loop.

4. After `net_init()` and `net_load()`, call
   `config_apply_optimistic(&cfg, shared_net.weights, N_TUPLES, LUT_SIZE)`.

5. In `train_worker()`:
   - At episode start, check `carousel_should_use()` and optionally load a
     saved position.
   - After each move, call `carousel_maybe_save()` if the max tile changed.
   - Before `net_update()`, adjust the LR with `multistage_get_lr()`.

6. Call `print_config(&cfg)` at startup to display the active configuration.
