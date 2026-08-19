# Pre-Publication Audit Checklist

Every claim in every document must be verifiable. This checklist ensures
no incorrect statements survive to publication.

**Rule: If we can't prove it with code or data, we can't claim it.**
**Rule: If the measurement conditions differ, it's not a comparison.**
**Rule: Every number needs a source: benchmark run ID, log file + line, or code reference.**

---

## 1. Numerical Claims to Verify

### 1.1 Architecture Claims

- [ ] "17 tuples of 6 positions" — Count BASE_TUPLES in ntuple_train.c AND ntuple_c.c AND ntuple_agent.py. All three must match exactly. If any differ, we have a consistency bug.
- [ ] "285M weights" — Math: 17 × 16^6 = 285,212,672. Verify against actual allocated memory in net_init(). Verify against actual file size of ntuple_best.bin (expected: 285,212,672 × 4 = 1,140,850,688 bytes without TC).
- [ ] "8 symmetries" — Run generate_symmetries() for each of the 17 tuples and log the actual count. Some tuples may produce fewer than 8 due to symmetry (e.g., a symmetric tuple might produce only 4 unique variants). Report the ACTUAL total number of feature evaluations per board, not "up to 8".
- [ ] "~1.1 GB model size" — Measure ntuple_best.bin file size. With TC tables, measure ntuple_latest.bin. Report both.
- [ ] "TC-learning uses 3× memory" — Verify: 3 arrays (weights, tc_sum, tc_abs) × same size. But the checkpoint file may not be 3×: TC tables are only saved when --tc flag is used. Document this.
- [ ] DQN "~200k parameters" — Count actual parameters in DuelingDQN. Old claim was for the original DQN architecture (2 conv + 2 FC). The new Dueling architecture (3 conv + 2 streams) has more. Recount.
- [ ] DQN "18 channels input" — Verify encode_state() produces (18, 4, 4). Old version was 16 channels. Ensure server.py uses the correct encode_state.

### 1.2 Performance Claims (CRITICAL — ALL INVALID UNTIL BENCHMARK)

- [ ] **EVERY performance number currently in documents is PRELIMINARY**
- [ ] Training log metrics use 3-ply search — NOT comparable to Szubert's 1-ply
- [ ] Browser metrics use 5-ply search — never formally measured
- [ ] No confidence intervals exist for any of our numbers
- [ ] No multiple-run statistics exist
- [ ] The "88% → 83%" discrepancy in different conversations — which is correct? Both came from training logs at different episodes. Neither is a formal benchmark.
- [ ] Score of 76,516 and 173,888 from game reports — these are single games, not averages. Cannot be presented as representative without context.

**Action: ALL performance claims in ALL documents must be replaced with benchmark results after training completes. No exceptions.**

### 1.3 Speed Claims

- [ ] "2.4ms per move (N-Tuple 5-ply)" — Was benchmarked with EARLY weights (ntuple_v1, 7 tuples). Re-benchmark with FINAL weights (17 tuples). More tuples = slower evaluation.
- [ ] "100ms per move (Expectimax)" — This is the time BUDGET, not actual measured time. Actual time varies by board state. Report mean + std + worst case.
- [ ] "~50ms per move (Attention 3-ply)" — Was benchmarked? Or estimated? If estimated, mark as such.
- [ ] "~1ms per move (DQN)" — With old 16-channel architecture or new 18-channel Dueling? Re-benchmark.
- [ ] "29 ep/s training speed" — This is for the CURRENT training run. Varies by board complexity (later episodes have longer games). Report range, not single number.
- [ ] "300-450× speedup Python→C" — Calculated how? What Python baseline? What C configuration? This is a compound claim (C speedup × lookup tables × threads). Break it down and verify each component.
- [ ] "22× faster than Python" for C trainer — Measured once with 200 episodes. May not hold at scale. Re-verify.

### 1.4 Szubert Comparison (CRITICAL)

- [ ] Szubert's 97.8% — This is from 1,000,000 test games with the BEST single agent found (not average of 30 runs). Source: Section V-B, last paragraph.
- [ ] Szubert's 90.6% ± 0.51% — This is the AVERAGE across 30 runs of TD-AFTERSTATE with LR=0.0025, 500K training games. Source: Table I.
- [ ] Szubert's 51,321 ± 358 — Same as above (score). Source: Table I.
- [ ] Szubert's small network: 17 × 4-tuples = 17 × 15^4 = 860,625 weights. NOT the same as their large network.
- [ ] Szubert's large network: 2 × 15^4 + 2 × 15^6 = 22,882,500 weights. Uses SYMMETRIC SAMPLING (each tuple evaluated 8 times).
- [ ] **CRITICAL**: Szubert used c=15 (max tile log2 = 14), we use c=16 (max tile log2 = 15). Different encoding! This means our LUT is 16^6 = 16.7M per tuple vs their 15^6 = 11.4M per tuple. Our architecture is NOT directly comparable in weight count.
- [ ] Q-Learning 49.8% — This was with the SMALL network and LR=0.005 (their best for Q-learning). Source: Table I. NOT with the large network.
- [ ] Szubert's Expectimax reference (89%) — NOT from Szubert's paper. From a StackOverflow answer they cited [1]. 100ms per move. Different heuristics than ours.
- [ ] Szubert's "100% Expectimax" — Based on only 100 games (Section VI-B). Szubert himself notes this "may be overestimated."

### 1.5 Wu et al. Comparison

- [ ] "Score 142,727" — Source? Verify from the actual paper, not from secondary citations.
- [ ] "5M episodes" — Verify.
- [ ] "67M weights" — Verify. What tuple configuration?
- [ ] We cite Wu but haven't read the paper. **Do not cite numbers we haven't verified from the primary source.**

### 1.6 Jaśkowski (2018) Comparison

- [ ] "~99.5% win rate" — Source? From which paper exactly? What configuration?
- [ ] "~300,000+ score" — Source?
- [ ] "40M episodes" — Source?
- [ ] Same warning: **do not cite unverified numbers.**

---

## 2. Document Audit

### 2.1 COMPARISON.md
- [ ] Remove ALL direct numerical comparisons until benchmark is run
- [ ] "Our score already exceeds Szubert's small network" — INCORRECT (3-ply vs 1-ply). Remove.
- [ ] Projected results table — These are guesses, not projections. Either remove or mark clearly as SPECULATIVE.
- [ ] Speed comparison table — Mixed sources (some measured, some estimated). Flag each.
- [ ] Section 5.2 "Where we go beyond Szubert" — Claims like "3-ply training" are factual. Claims about performance impact are speculative until benchmarked.
- [ ] The "score vs win rate tradeoff" analysis is sound but our supporting data is from 3-ply, not comparable to Szubert's 1-ply observation.

### 2.2 REPORT.md
- [ ] All performance tables need source annotation
- [ ] "87%+ 2048 achievement rate" — From which measurement? With what search depth?
- [ ] "55% 4096 rate" — Same questions
- [ ] DQN "~0% 2048" — Based on 1,700 episodes of OLD architecture. New Dueling DQN hasn't been trained at all. Cannot compare old DQN with new Expectimax/NTuple.
- [ ] Expectimax "~80%" — Before or after calibration? With what time budget?

### 2.3 THESIS.md
- [ ] 3,215 lines — Every number needs a citation to its source
- [ ] Tables with training progression — Mark which numbers are from training logs (3-ply) vs benchmarks (not yet run)
- [ ] Game report examples — Verify the JSON files still exist and numbers match
- [ ] All "~" prefixed numbers are approximations — either make precise or mark as approximate
- [ ] Check for claims that evolved during development but weren't updated (e.g., early claims about DQN that refer to old architecture)

### 2.4 EVOLUTION.md
- [ ] Chronological accuracy — Cross-reference with git log timestamps
- [ ] Score milestones — Verify each against training logs
- [ ] Decision rationale — Ensure we don't retroactively justify decisions that were actually trial-and-error

### 2.5 README.md
- [ ] Performance table is the first thing people see. Must be 100% accurate.
- [ ] Currently shows preliminary numbers. Must be updated after benchmark.
- [ ] "87%+ win rate" without context is misleading
- [ ] DQN results based on old untrained architecture

### 2.6 TODO.md
- [ ] No factual claims to verify, but ensure completed items are marked accurately

### 2.7 GAME_INTEGRITY.md
- [ ] Verify the diff against original is still accurate (only 2 lines changed)
- [ ] Verify the Python game engine matches JavaScript behavior — need automated test
- [ ] Verify the C game engine matches Python behavior — need automated test

---

## 3. Code Correctness

### 3.1 Game Engine Consistency (CRITICAL)

Three independent implementations must produce identical game behavior:
- JavaScript (browser) — original Cirulli code
- Python (game.py) — our reimplementation for training/benchmark
- C (ntuple_train.c) — our reimplementation for fast training

Tests needed:
- [ ] Given seed=42, play 100 random games in Python and C. Compare final scores. Must match exactly.
- [ ] Given a specific board state, verify all 4 moves produce identical results in Python and C.
- [ ] Edge cases: full board with one merge possible, merge chains (2+2=4, then 4+4=8 in same move), multiple merges in same row.
- [ ] Verify that a tile is added AFTER the move, not during.
- [ ] Verify that each tile can only merge ONCE per move (no chain merging in single move).

### 3.2 N-Tuple Network Consistency

- [ ] Python NTupleNetwork and C ntuple_train.c must produce the SAME evaluation for the same board with the same weights. Test with 100 random boards.
- [ ] ntuple_c.so (player) and ntuple_train.c (trainer) must use the SAME tuple definitions. Diff the BASE_TUPLES arrays.
- [ ] Symmetry generation must be identical across Python/C. For each of 17 tuples, verify same number of symmetries and same positions.
- [ ] encode6() in C and _encode_tuple_fast() in Python must produce the same index for the same board+tuple. Test exhaustively.

### 3.3 Checkpoint Compatibility

- [ ] Weights saved by C trainer can be loaded by C player (ntuple_c.so) — currently works, but verify after any code changes
- [ ] Weights saved by C trainer can be loaded by Python NTupleNetwork — verify
- [ ] TC tables are correctly skipped when loading a non-TC checkpoint
- [ ] File format: verify header (n_tuples count + LUT sizes) matches expectations

### 3.4 Benchmark Script

- [ ] Run benchmark.py twice with --seed 42 --games 100. Results must be IDENTICAL (same scores, same tile distributions, same everything).
- [ ] If NOT identical: find the source of non-determinism (numpy random state, Python hash randomization, thread timing, etc.)
- [ ] Verify Game2048 in Python uses Python's `random` module (which we seed) not numpy.random for tile placement. CHECK THIS — it may use numpy.
- [ ] Verify that the N-Tuple C player (via ctypes) doesn't have internal random state that we can't seed.
- [ ] Verify confidence interval: for n=1000 games, CI = 1.96 × std / sqrt(1000). This assumes normally distributed scores. Is that true? Check with a Q-Q plot or Shapiro-Wilk test. If not normal, use bootstrap CI instead.

### 3.5 Expectimax

- [ ] Calibrated weights in code: snake=0.95, empty=2.0, mono=1.0, smooth=0.1, corner=1.0. Verify against expectimax_calibrate output.
- [ ] The calibration was done with the OLD corner bonus (bugged). After we fixed the corner bonus and added edge monotonicity, the optimal weights may have changed. **The calibration may need to be re-run.**
- [ ] Transposition table: verify no hash collisions corrupt results. Run with and without cache, compare scores over 100 games.
- [ ] Iterative deepening: verify that the 60% budget rule actually prevents starting depths that timeout.
- [ ] Chance node sampling: document that we sample max 5 empty cells. This is an approximation. Quantify the error vs full evaluation.

### 3.6 DQN

- [ ] New Dueling DQN architecture compiles and runs without errors — verified
- [ ] But NEVER TRAINED. All DQN numbers in documents refer to the OLD architecture (2-conv, no dueling, no PER, no n-step). These numbers are INVALID for the new architecture.
- [ ] The old checkpoint (best.pt) is INCOMPATIBLE with the new architecture. The server loads it but falls back to random weights. This means DQN in the browser is currently playing RANDOMLY.
- [ ] **DQN numbers in all documents must be clearly labeled as "old architecture, untrained" or removed entirely.**

### 3.7 Attention N-Tuple

- [ ] Only 400 episodes trained. Results are PRELIMINARY.
- [ ] The 3-ply search in server.py for attention: verify it actually does 3-ply (not 1-ply by mistake). Trace through the code.
- [ ] The attention weights are random at 400 episodes — the agent is essentially playing with random weighting of tuples. This is WORSE than pure N-Tuple sum.
- [ ] Don't present Attention results until properly trained. Any current numbers are meaningless.

---

## 4. Terminology Consistency

- [ ] **ply definition**: In our code, `--depth 0` = player evaluates afterstates directly (1-ply). `--depth 1` = player → chance → player → evaluate (3-ply). `--depth 2` = 5-ply. CREATE A GLOSSARY and reference it everywhere.
- [ ] **"win rate" vs "2048 rate" vs "2048+ rate"**: These might mean different things. "Win rate" = reached 2048. "2048+ rate" = reached AT LEAST 2048 (includes 4096, 8192 games). "2048 rate" = max tile was exactly 2048 (excludes 4096+ games). Our training logs show MAX TILE distribution, so "2048" there means the max tile was exactly 2048, NOT "reached 2048 at some point." VERIFY: does a game with max tile 4096 have a 2048 tile on the board? Yes, necessarily. So 2048+ = 2048 + 4096 + 8192 + ... from the distribution. But we've been computing this inconsistently.
- [ ] **"episode" vs "game"**: In our code these are the same. But in RL literature, an episode can mean different things. Define explicitly.
- [ ] **"score"**: Total game score (sum of merge rewards). Always specify "average score over N games" not just "score."
- [ ] **"training" search depth vs "play" search depth**: Always specify which one. Never say "3-ply" without saying whether it's during training or play.
- [ ] **"N-Tuple" vs "N-Tuple Network" vs "n-tuple network"**: Pick one capitalization style.
- [ ] **Agent names**: "Expectimax" vs "Expectimax agent". "N-Tuple" vs "N-Tuple agent" vs "N-Tuple Network agent". Standardize.

---

## 5. Missing Disclosures

- [ ] **Hogwild non-determinism**: Training with 8 threads and no locks means the same hyperparameters + seed can produce different results. This affects reproducibility. Acknowledge and quantify the variance.
- [ ] **Chance node sampling**: We sample max 5 of up to 15 empty cells. This is an approximation that biases the evaluation. Szubert's TD-STATE evaluates ALL possible transitions. Our TD-AFTERSTATE doesn't need this (afterstate is deterministic), but our SEARCH does sample. Acknowledge.
- [ ] **Score vs win rate tradeoff**: The agent is trained to maximize score, not win rate. This is a design choice with consequences. Acknowledge.
- [ ] **Attention is experimental**: Only 400 episodes trained. Cannot draw conclusions. Acknowledge.
- [ ] **Hardware dependence**: Results may vary on different hardware (CPU speed affects iterative deepening depth reached within time budget). Document exact hardware.
- [ ] **Game version**: We use Cirulli's original 2048. Some variants have different tile probabilities or board sizes. Specify.
- [ ] **No hyperparameter search**: Our hyperparameters were chosen by trial and error, not by systematic search. The "optimal" weights from the calibrator are optimal only within the search space we explored (7 parameters, 4 step sizes). Acknowledge.
- [ ] **Survivorship bias in game reports**: We only have reports from games played in the browser, which we might have selectively played or stopped. The benchmark script eliminates this bias. Acknowledge that game reports are anecdotal, not statistical.
- [ ] **LR decay**: Our LR decays over episodes. But the decay schedule was chosen arbitrarily (0.01 → 0.0005 over N episodes). Different schedules might produce different results. Acknowledge.
- [ ] **C trainer compiles with -O3**: Compiler optimizations could theoretically affect floating-point results. Different compilers or optimization levels might produce slightly different weights. Acknowledge.

---

## 6. Potential Bugs to Investigate

- [ ] **DQN encode_state inconsistency**: The old encode_state produced (16,4,4). The new one produces (18,4,4). If ANY code path still uses the old encoding with the new network, it will silently produce garbage. Search all files for encode_state calls.
- [ ] **Expectimax corner bonus**: We fixed the bug but THEN ran calibration BEFORE the fix. Then we fixed the bug and added edge monotonicity AFTER calibration. The calibrated weights (snake=0.95, empty=2.0) are optimal for the BUGGED version, not the fixed version. **The calibration should be re-run.**
- [ ] **N-Tuple training LR**: Our C trainer uses `lr_start = 0.01f, lr_end = 0.0005f`. But the Python trainer (ntuple_agent.py) uses different values. If someone trains with Python, they get different results. Document which trainer was used.
- [ ] **Attention agent loads ntuple_best.bin**: But during training, the N-Tuple trainer updates ntuple_best.bin. If Attention training runs concurrently, it reads STALE weights. We said "they don't conflict" but the Attention's base weights become outdated as training progresses. Acknowledge this limitation.
- [ ] **Game2048 random module**: game.py uses `import random` and calls `random.choice()` and `random.random()`. But benchmark.py sets `np.random.seed()`. If Game2048 uses Python's `random` module (not numpy), the seed in benchmark.py does NOT control game randomness. **CHECK THIS IMMEDIATELY.**

---

## 7. Process

1. **Fix critical bugs** — encode_state, calibration after bug fix, seed control
2. **Game engine consistency test** — Automated test comparing Python/C engines
3. **Freeze code** — Git tag "v1.0-benchmark", no more changes
4. **Run benchmark** — All agents, 10k games, 5 seeds (42,43,44,45,46), 3 search depths (1,3,5-ply)
5. **Statistical analysis** — Mean, std, 95% CI, tile distributions, normality test
6. **Update ALL documents** — Replace every number with benchmark citation
7. **Cross-reference** — Every number maps to: benchmark_results_seed{N}.json → metric
8. **Internal review** — Re-read every document as a hostile reviewer would
9. **Final commit** — Git tag "v1.0-paper"

---

## Status

- [ ] Critical bugs investigated
- [ ] Game engine consistency verified
- [ ] Code frozen
- [ ] Benchmark run complete
- [ ] All documents updated with benchmark numbers
- [ ] All claims verified against primary sources
- [ ] Terminology standardized
- [ ] Disclosures written
- [ ] Internal review complete
- [ ] Ready for submission
