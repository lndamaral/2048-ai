# Paper Preparation Checklist

## What a publishable paper needs

### 1. Reproducibility Data

- [ ] **Exact hyperparameters table**: Every parameter for every agent, frozen at publication time
- [ ] **Random seeds**: Record seeds used for every benchmark run (reproducibility)
- [ ] **Hardware specs**: Exact machine, CPU, RAM, OS version
- [ ] **Training time**: Wall-clock time for each agent to reach each milestone
- [ ] **Code version**: Git commit hash for each benchmark run

### 2. Statistical Rigor (CRITICAL — we don't have this yet)

- [ ] **Multiple runs**: Each experiment must be repeated **at least 5 times** (ideally 30, like Szubert did) with different random seeds
- [ ] **Confidence intervals**: Report mean ± 95% CI, not just single numbers
- [ ] **Statistical significance tests**: t-test or Wilcoxon between agents to prove differences are real, not luck
- [ ] **Standard deviation**: Report for all metrics

Current problem: our numbers come from single training runs. A paper reviewer would reject "88% win rate" without confidence intervals. We need: "88.2% ± 1.3% (95% CI, n=30 runs of 1000 games each)".

### 3. Ablation Studies (CRITICAL — proves each component matters)

Must show what happens when you REMOVE each component:

| Experiment | What we test |
|---|---|
| N-Tuple without TC-learning | Does TC-learning actually help? |
| N-Tuple with 1-ply training vs 3-ply training | Does deeper training search help? |
| N-Tuple with 1-ply play vs 3-ply vs 5-ply play | How much does play search depth help? |
| Attention vs pure sum (same N-Tuple base) | Does the attention mechanism add value? |
| 17 tuples vs 4 tuples (same training) | Do more tuples help? |
| With symmetries vs without symmetries | Do 8 symmetries help? |
| LR=0.01 vs 0.0025 vs 0.005 (with 3-ply) | Optimal LR for 3-ply training? |

Each ablation needs multiple runs with confidence intervals.

### 4. Benchmark Protocol

For the final comparison, every agent must be tested with:
- [ ] **Same number of games**: 10,000 minimum (Szubert used 1,000,000 for his best result)
- [ ] **Same conditions**: Fresh game states, no cherry-picking
- [ ] **Multiple metrics**: Score (mean, median, std), 2048/4096/8192/16384 rates, max score achieved
- [ ] **Learning curves**: Performance vs episodes (every 1000 ep), not just final numbers
- [ ] **Score distribution histogram**: Shows if the agent is consistent or volatile

### 5. Figures Needed for Paper

- [ ] **Learning curves** (score vs episodes) for all agents on same plot
- [ ] **Tile distribution evolution** over training (stacked bar)
- [ ] **Attention weight visualization**: Heatmap showing which tuples get more weight in different game phases (early/mid/endgame) — this is the novel contribution, needs strong visual
- [ ] **Score distribution** (histogram/violin plot) for each agent at ceiling
- [ ] **Architecture diagram**: Clean diagram of Attention N-Tuple architecture
- [ ] **Example game sequences**: Show critical decision points where Attention differs from pure N-Tuple
- [ ] **Ablation bar chart**: Performance with/without each component

### 6. Logging Changes Needed

Our current training logs don't capture enough for a paper. We need to add:

- [ ] **Per-episode score logging**: Save raw scores to a file (not just 100-episode averages). Needed for statistical analysis.
- [ ] **Checkpoint at regular intervals**: Save model every 10k episodes for learning curve analysis. Currently only saving best + latest.
- [ ] **Tile distribution per checkpoint**: For each saved checkpoint, run 1000 games and record full tile distribution.
- [ ] **Attention weights logging**: For the Attention agent, log the attention weights periodically to show how they evolve over training.
- [ ] **Move distribution logging**: Track how often each direction (up/right/down/left) is chosen. Reveals learned strategy.

### 7. Writing Structure

Standard ML/Game AI paper format:

1. **Abstract** (200 words)
2. **Introduction** (1-2 pages) — why 2048, what's the gap, what we propose
3. **Related Work** (1 page) — Szubert, Wu, Jaśkowski, Oka
4. **Background** (1 page) — TD-learning, N-Tuple, afterstates
5. **Proposed Method** (2-3 pages) — Attention N-Tuple architecture, training procedure
6. **Experimental Setup** (1 page) — hyperparameters, hardware, protocol
7. **Results** (2-3 pages) — learning curves, ablation, comparison with baselines
8. **Discussion** (1 page) — score vs win rate tradeoff, fragility, broader applications
9. **Conclusion** (0.5 page)
10. **References**

Total: 8-12 pages (typical for IEEE CIG or AAAI workshop)

### 8. Target Venues

| Venue | Type | Deadline | Fit |
|---|---|---|---|
| IEEE CIG (CoG) | Conference | ~March | Perfect — Szubert published here |
| AAAI Workshop on Games | Workshop | ~November | Good fit |
| IEEE Transactions on Games | Journal | Rolling | For extended version |
| arXiv | Preprint | Anytime | Immediate visibility |

### 9. Timeline

| Phase | What | When |
|---|---|---|
| Training complete | N-Tuple 5M ep, DQN 50k ep, Attention 50k ep | +1-2 days |
| Ablation studies | Run all ablation experiments | +2-3 days |
| Benchmark | 10k games per agent, multiple seeds | +1 day |
| Figures | Generate all publication-quality charts | +1 day |
| Writing | Draft paper | +3-5 days |
| Review | Internal review, polish | +2 days |
| Submit | arXiv first, then conference | +1 day |
| **Total** | | **~2 weeks from now** |
