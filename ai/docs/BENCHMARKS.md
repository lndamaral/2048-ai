# Performance Benchmarks — Reference Targets

## Two targets to track progress against:

### Baseline: Szubert & Jaśkowski (2014)
Source: IEEE CIG 2014, Table I and Section V-B.

| Metric | Szubert (small, 500k ep) | Szubert (large, 1M ep) |
|--------|--------------------------|------------------------|
| Network | 17×4-tuples, 860k weights | 2×4-tuple + 2×6-tuple, 22.8M weights |
| Training | TD-Afterstate, 1-ply, LR=0.0025 | TD-Afterstate, 1-ply, LR=0.0025 |
| Evaluation | 1-ply (no search) | 1-ply (no search) |
| **Avg score** | **51,321 ± 358** | **100,178** |
| **2048 rate** | **90.6% ± 0.5%** | **97.8%** |
| Max single score | — | 261,526 |
| Speed | 330,000 moves/s | 330,000 moves/s |
| Encoding | c=15 | c=15 |

### Ceiling: TDL2048+ (moporgic, 2023)
Source: github.com/moporgic/TDL2048, IEEE ToG.

| Metric | TDL2048+ |
|--------|----------|
| Network | 4×6-tuple (with 8 symmetries = 32 features), 22.8M weights |
| Training | OTD+TC, 1-ply, LR=0.1 distributed (0.003125 per feature) |
| Evaluation | 6-ply tile-downgrading expectimax |
| **Avg score** | **625,377** |
| **2048 rate** | **~99.5%** |
| **32768 rate** | **72%** |
| **65536 rate** | **0.02%** |
| Speed | 6.5M moves/s (single thread), 102.5M moves/s (multi) |

### Our configuration

| Metric | Our setup |
|--------|-----------|
| Network | 17×6-tuple (with 8 symmetries), 285M weights |
| Training | TD-Afterstate + TC + Optimistic(320k) + Multistage + Carousel |
| Training search | Configurable: 1-ply or 3-ply |
| Play search | 5-ply via C shared library, tile-downgrading |
| Encoding | c=16 |

---

## Progress Tracking

**Fill this table as training progresses. All numbers must come from training logs or benchmark.py, with search depth specified.**

### Training metrics (from logs, search depth = training depth)

| Episodes | Depth | Score | 2048+ rate | 4096 rate | 8192 rate | 16384 rate | 32768 rate | vs Szubert | vs TDL2048+ |
|----------|-------|-------|------------|-----------|-----------|------------|------------|------------|-------------|
| 10k | | | | | | | | | |
| 50k | | | | | | | | | |
| 100k | | | | | | | | | |
| 500k | | | | | | | | | |
| 1M | | | | | | | | | |
| 5M | | | | | | | | | |

### Benchmark metrics (from benchmark.py, 10k games, seed=42)

| Checkpoint | Play depth | Score (mean ± CI) | 2048+ | 4096 | 8192 | 16384 | 32768 | vs Szubert | vs TDL2048+ |
|------------|-----------|-------------------|-------|------|------|-------|-------|------------|-------------|
| 100k ep | 1-ply | | | | | | | | |
| 100k ep | 5-ply | | | | | | | | |
| 500k ep | 1-ply | | | | | | | | |
| 500k ep | 5-ply | | | | | | | | |
| 1M ep | 1-ply | | | | | | | | |
| 1M ep | 5-ply | | | | | | | | |
| 5M ep | 1-ply | | | | | | | | |
| 5M ep | 5-ply | | | | | | | | |

### Milestones

- [ ] Match Szubert small (score ≥ 51,321 at 1-ply eval)
- [ ] Match Szubert large (score ≥ 100,178 at 1-ply eval)
- [ ] Exceed Szubert with search (score > 100,178 at 5-ply eval)
- [ ] Reach 300k avg score
- [ ] Match TDL2048+ (score ≥ 625,377)
- [ ] 32768 tile achieved
- [ ] 65536 tile achieved

---

## Important Notes

1. **Szubert uses 1-ply evaluation**: To compare fairly, we MUST benchmark at 1-ply (no search). Our 5-ply numbers are NOT comparable to Szubert's.

2. **TDL2048+ uses 6-ply with tile-downgrading**: To compare fairly with TDL2048+, we benchmark at 5-ply with tile-downgrading (our closest equivalent).

3. **Encoding differs**: Szubert uses c=15, we use c=16. Not directly comparable in weight count, but functionally equivalent for tiles up to 16384.

4. **TDL2048+ uses only 22.8M weights**: They achieve 625k score with FEWER weights than us (285M). Their advantage is techniques (OTD, tile-downgrading, better search), not network size. More weights ≠ better performance.

5. **All numbers in this document must cite their source** (training log line, benchmark JSON, or paper reference).
