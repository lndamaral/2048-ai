# TODO — Pending Tasks, Observations and Mental Notes

## High Priority

- [ ] **Fragility comparison test**: After N-Tuple training completes, compare how each agent (DQN, Expectimax, N-Tuple) behaves in "fragile" positions — boards where the snake pattern looks organized but a bad random tile placement would disrupt the sequence. This tests whether the N-Tuple learned to avoid greedy snake moves that leave vulnerabilities.

- [ ] **N-Tuple hybrid experiment**: Test adding a small snake pattern bonus to the N-Tuple evaluation during play (not training). Compare pure N-Tuple vs hybrid to see if the learned evaluation already captures the snake strategy or benefits from explicit guidance.

- [ ] **Benchmark script**: Create a unified script that runs all 3 agents (1000+ games each) and produces a single comparison report with statistical analysis (mean, median, std dev, confidence intervals, score distributions, tile distributions).

## Training

- [ ] **N-Tuple Stage 2**: Complete 5M episodes with 3-ply + TC-learning (in progress)
- [ ] **DQN training**: Run 50k+ episodes with the optimized architecture (Dueling DQN + PER + Noisy Networks). Code is ready, training not started. Command: `python3 -u train.py --episodes 50000 2>&1 | tee dqn_training.log`
- [ ] **Reload server after N-Tuple training**: Restart server to pick up the final trained model
- [ ] **Snapshot training logs**: Copy final training logs to ai/logs/ for git

## Analysis

- [ ] **Final 3-agent comparison**: Once all agents are at their ceiling, run 1000+ games each and compare with statistical analysis
- [ ] **Regenerate charts**: Update analysis.py charts with final training data from all 3 agents
- [ ] **Game report analysis**: Analyze JSON reports in ai/reports/ to find patterns in failures — which board configurations lead to game over for each agent?
- [ ] **Screenshots for documentation**: Capture browser screenshots of each agent playing (winning and losing) for the thesis
- [ ] **Move distribution analysis**: Track which directions each agent favors. Does the N-Tuple learn to avoid UP (which disrupts snake patterns)? How does this compare to Expectimax?
- [ ] **Endgame analysis**: At what score/tile does each agent typically die? Is there a common board pattern at game over? This could reveal the "death pattern" each agent is vulnerable to.
- [ ] **Score variance comparison**: Which agent is most consistent? Low variance = reliable strategy. High variance = luck-dependent. Important for understanding robustness.

## Code

- [ ] **DQN checkpoint compatibility**: New Dueling DQN architecture is incompatible with old checkpoints (best.pt). Need fresh training.
- [ ] **Expectimax calibration log**: Commit calibration results to ai/logs/
- [ ] **N-Tuple play depth**: Currently 5-ply during play. Test if 7-ply improves results after training completes (evaluation function will be stronger).

## Documentation

- [ ] **Update THESIS.md with Expectimax ceiling discovery**: Document the structural limitation of fixed heuristics (greedy snake pattern vs positional fragility). This was identified by observing the agent prioritize short-term snake organization over long-term board safety.
- [ ] **Add DQN optimization chapter to THESIS.md**: Document Dueling DQN, Prioritized Experience Replay, Noisy Networks — what they are, why we chose them, results.
- [ ] **Add Expectimax calibration chapter to THESIS.md**: Document the auto-calibration process, hill climbing results, which heuristics helped and which didn't.
- [ ] **Add N-Tuple training progression to THESIS.md**: Document Stage 1 vs Stage 2, the 16384 tile milestone, TC-learning impact.
- [ ] **Final comparison chapter in THESIS.md**: Side-by-side analysis of all 3 agents at their ceiling with charts and statistical analysis.
- [ ] **Final commit with all results**: Push completed training logs, updated charts, final report

## Observations & Lessons Learned (to document)

- [ ] **DQN is structurally inefficient for 2048**: The state space is small enough for tabular methods. DQN's generalization via neural networks adds overhead without proportional benefit. Worth documenting as a "when NOT to use deep RL" case study.

- [ ] **The snake pattern paradox**: The best short-term move (maintain snake) can be the worst long-term move (creates vulnerability). This is a general lesson about greedy heuristics in stochastic games. Expectimax can't resolve this because it evaluates the heuristic at leaf nodes — if the heuristic itself is greedy, deeper search doesn't help.

- [ ] **Forward TD vs Backward TD**: Forward TD(0) converges faster and more stably because it updates weights immediately using fresh information. Backward TD batches all updates to the end of the game, causing large delayed corrections. This was empirically confirmed: v1 (backward) learned slower than the C trainer (forward).

- [ ] **LR sensitivity in N-Tuple networks**: LR=0.1 caused overflow, LR=0.001 was too slow, LR=0.01 was optimal. N-Tuple networks are sensitive because each weight is updated independently — there's no gradient normalization like in neural networks.

- [ ] **The transpose bug**: A reminder that bit manipulation functions are NOT portable across different bit layouts. Always verify with concrete test cases before trusting copy-pasted bit tricks. Cost us hours of debugging.

- [ ] **Lookup tables vs full evaluation tradeoff**: Per-row lookup tables are fast but lose 2D spatial information. The snake pattern fundamentally requires 2D evaluation. This tradeoff is a general lesson for game evaluation functions.

- [ ] **Multi-stage training is more efficient**: 1-ply for basic patterns + 3-ply for refinement. The 1-ply stage builds a reasonable foundation quickly, and 3-ply refines without wasting expensive search on trivial early patterns.

- [ ] **TC-learning prevents weight oscillation**: Weights that receive contradictory updates (sometimes positive, sometimes negative) get their LR automatically reduced. This is especially important in stochastic games where the same board state can lead to different outcomes due to randomness.

- [ ] **C implementation was essential, not optional**: Python was 300-450x slower. The N-Tuple training that takes ~14h in C would take ~6 months in Python. The lesson: prototype in Python, validate the algorithm works, then rewrite performance-critical code in C.

- [ ] **Auto-calibration revealed that merge_potential and trapped_tile heuristics add zero value**: Sometimes the simplest heuristic set is the best. Adding complexity doesn't always help — the calibrator confirmed this empirically.

- [ ] **16384 tile appeared during training**: This is significant — very few AIs reach this tile. It proves our N-Tuple architecture has sufficient capacity to learn deep strategies. Document the exact episode and training configuration when this happened.

## Future Ideas (stretch goals)

- [ ] **Replay viewer**: Build a web-based replay viewer that loads a JSON game report and animates the game move-by-move, showing what the AI was "thinking" (evaluation scores for each direction).

- [ ] **Heatmap of tile placement**: Visualize where each agent tends to place high-value tiles. Expected: Expectimax always uses corners, N-Tuple might discover alternative strategies.

- [ ] **Training curriculum**: Instead of random starting positions, start some training games from difficult mid-game positions to improve endgame play.

- [ ] **Agent ensemble**: Have the 3 agents "vote" on each move. Does consensus lead to better play than any individual agent?

- [ ] **Transfer to 5x5 grid**: Can the learned N-Tuple weights transfer to a larger board? Or does it need retraining from scratch?
