# TODO — Pending Tasks and Future Investigations

## High Priority

- [ ] **Fragility comparison test**: After N-Tuple training completes, compare how each agent (DQN, Expectimax, N-Tuple) behaves in "fragile" positions — boards where the snake pattern looks organized but a bad random tile placement would disrupt the sequence. This tests whether the N-Tuple learned to avoid greedy snake moves that leave vulnerabilities.

## Training

- [ ] **N-Tuple Stage 2**: Complete 5M episodes with 3-ply + TC-learning (in progress)
- [ ] **DQN training**: Run 50k+ episodes with the optimized architecture (Dueling DQN + PER + Noisy Networks). Code is ready, training not started.

## Analysis

- [ ] **Final 3-agent comparison**: Once all agents are at their ceiling, run 1000+ games each and compare with statistical analysis (confidence intervals, score distributions)
- [ ] **Regenerate charts**: Update analysis.py charts with final training data
- [ ] **Update THESIS.md**: Add Expectimax calibration chapter, fragility analysis, DQN results, final comparison

## Documentation

- [ ] **Update THESIS.md with Expectimax ceiling discovery**: Document the structural limitation of fixed heuristics (greedy snake pattern vs positional fragility)
- [ ] **Add DQN optimization chapter to THESIS.md**: Document Dueling DQN, PER, Noisy Networks decisions
- [ ] **Final commit with all results**: Push completed training logs, updated charts, final report
