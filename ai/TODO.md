# TODO — Pending Tasks and Future Investigations

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
