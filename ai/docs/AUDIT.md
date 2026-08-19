# Pre-Publication Audit Checklist

Every claim in every document must be verifiable. This checklist ensures
no incorrect statements survive to publication.

**Rule: If we can't prove it with code or data, we can't claim it.**

---

## 1. Numerical Claims to Verify

### 1.1 Architecture Claims

- [ ] "17 tuples of 6 positions" — Verify in ntuple_train.c BASE_TUPLES array. Count them.
- [ ] "285M weights" — Verify: 17 × 16^6 = 17 × 16,777,216 = 285,212,672. Check actual memory.
- [ ] "8 symmetries" — Verify generate_symmetries() produces up to 8 unique variants.
- [ ] "~1.1 GB model size" — Verify: 285M × 4 bytes = 1,140,850,688 bytes. Check actual file size.
- [ ] "TC-learning uses 3× memory" — Verify: weights + tc_sum + tc_abs = 3 tables per tuple.

### 1.2 Performance Claims (CRITICAL — all need re-measurement)

- [ ] **EVERY performance number must come from the benchmark script, not from training logs**
- [ ] Training log metrics are with 3-ply search — document this clearly
- [ ] Browser metrics are with 5-ply search — document this clearly  
- [ ] Szubert comparison must be at 1-ply (their protocol)
- [ ] All claims must specify: episodes trained, search depth at evaluation, number of games, seed

### 1.3 Speed Claims

- [ ] "2.4ms per move (N-Tuple 5-ply)" — Re-benchmark with final weights
- [ ] "100ms per move (Expectimax)" — Re-benchmark
- [ ] "~50ms per move (Attention 3-ply)" — Re-benchmark
- [ ] "~1ms per move (DQN)" — Re-benchmark
- [ ] "29 ep/s training speed" — Verify with current setup
- [ ] "300-450× speedup Python→C" — How was this calculated? Verify.

### 1.4 Szubert Comparison (CRITICAL)

- [ ] Szubert's 97.8% was with 1M episodes, large network (22.8M weights), 1-ply, LR=0.0025
- [ ] Szubert's 90.6% was with 500K episodes, large network, 1-ply
- [ ] Szubert's 51,321 score was with 500K episodes, large network, 1-ply
- [ ] Q-Learning 49.8% was with 500K episodes, small network (860K weights)
- [ ] All these are from Table I and Table II of szubert_cig2014.pdf — cross-check each number

---

## 2. Document Audit

### 2.1 COMPARISON.md
- [ ] Remove or flag any direct comparison between our 3-ply metrics and Szubert's 1-ply metrics
- [ ] "Our score already exceeds Szubert's small network" — This was comparing 3-ply vs 1-ply. INCORRECT. Flag or remove.
- [ ] Projected results table — Mark clearly as PROJECTIONS, not measurements
- [ ] Speed comparison table — Re-verify all numbers

### 2.2 REPORT.md
- [ ] Check all score/rate claims match actual data sources
- [ ] Ensure no training-log metrics are presented as benchmark metrics

### 2.3 THESIS.md  
- [ ] 3,215 lines — comprehensive audit needed. Every number must cite its source.
- [ ] Training progression tables — verify against actual training logs
- [ ] "First 2048 at episode 500" — verify in ntuple_training_5k.log
- [ ] "Score 76,516 with 4096 tile" — verify in game reports JSON

### 2.4 EVOLUTION.md
- [ ] Cross-check all milestone claims against training logs and git history

### 2.5 README.md
- [ ] Performance table — must match benchmark results (not yet run)
- [ ] "87%+ win rate" — source? Training log with 3-ply. Mark as approximate or remove.

---

## 3. Code Correctness

### 3.1 Game Engine Integrity
- [ ] Python game.py produces same results as JavaScript for identical inputs
- [ ] C ntuple_train.c game engine produces same results as Python game.py
- [ ] Tile probabilities: 90% for 2, 10% for 4 — verify in all three implementations
- [ ] Merge logic: verify all three implementations handle edge cases identically

### 3.2 N-Tuple Network
- [ ] encode6() correctly maps grid positions to LUT index — test with known values
- [ ] Symmetries are correctly generated — compare C output with Python output
- [ ] TC-learning update rule matches the formula in the paper
- [ ] Checkpoint save/load preserves exact weight values (save, load, compare)

### 3.3 Benchmark Script
- [ ] Seed reproducibility: run twice with same seed, verify identical results
- [ ] Score computation matches game engine (no off-by-one, no missed rewards)
- [ ] Tile rates are computed correctly (>=2048, not ==2048)
- [ ] Confidence interval formula is correct: 1.96 × std / sqrt(n)

### 3.4 Expectimax
- [ ] Calibrated weights match what's in expectimax_c.c (snake=0.95, empty=2.0)
- [ ] Move simulation in C matches Python (test with identical boards)

---

## 4. Terminology Consistency

- [ ] "1-ply" vs "1-ply search" vs "no search" — pick one, use everywhere
- [ ] "win rate" vs "2048 rate" vs "2048+ rate" — define once, use consistently  
- [ ] "episode" vs "game" vs "training game" — standardize
- [ ] "score" vs "total score" vs "average score" — always specify which one
- [ ] "depth" vs "ply" vs "search depth" — define the relationship (depth=1 means 3-ply? or 1-ply?)

**CRITICAL: In our C code, --depth 0 = 1-ply, --depth 1 = 3-ply, --depth 2 = 5-ply. This mapping must be documented explicitly and used consistently.**

---

## 5. Missing Disclosures

- [ ] Acknowledge that training uses hogwild (lock-free) which introduces non-determinism
- [ ] Acknowledge that chance node sampling (max 5 cells) is an approximation
- [ ] Acknowledge the score vs win rate tradeoff (Szubert Section VI-A)
- [ ] Acknowledge that the Attention N-Tuple is experimental/preliminary

---

## 6. Process

1. **Freeze code** — No more changes after audit is complete
2. **Run benchmark** — All agents, 10k games, multiple seeds
3. **Update all documents** — Use ONLY benchmark numbers
4. **Cross-reference** — Every number in every document traces to a specific benchmark run
5. **Peer review** — Have someone else read and challenge every claim

---

## Status

- [ ] Audit started
- [ ] Audit complete
- [ ] All documents updated
- [ ] All claims verified
- [ ] Ready for submission
