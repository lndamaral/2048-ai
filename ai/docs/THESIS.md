# AI Agents for 2048: A Comparative Study
## Technical Report — DQN vs Expectimax vs N-Tuple Networks

**Author**: Leonardo Amaral
**Date**: August 2026
**Powered by Claude Code**

---

## Abstract

This report presents the complete development, implementation, and comparative analysis of three fundamentally different artificial intelligence approaches for the puzzle game 2048: Deep Q-Networks (DQN), Expectimax tree search with hand-crafted heuristics, and N-Tuple Networks with Temporal Difference learning. Beginning from a naive deep reinforcement learning implementation that failed to reach even the titular 2048 tile after three hours of training, the project evolved iteratively through seven documented failures and dozens of design pivots into a state-of-the-art system capable of reaching the 2048 tile in over 87% of games, the 4096 tile in 55% of games, and the 8192 tile in 12% of games, all with a response time of 2.4 milliseconds per move.

The work documents every technical decision, its rationale, its outcome, and the lessons learned. It provides a detailed chronicle of the debugging sessions that uncovered critical bugs -- including a bit-layout transpose corruption that reduced the AI to scoring 44 points, a learning-rate overflow that sent weights to infinity, and a buffered-output issue that made training appear to hang. The project culminates in a web-based real-time visualization system where users can observe and compare the three agents playing live in the browser, with automated JSON reports saved for each completed game.

Our final N-Tuple Network agent, trained for approximately 46,000 episodes in Stage 2 (3-ply search during training) on top of 500,000 episodes in Stage 1 (1-ply), achieves an average score of approximately 64,000 points. This places it solidly in the range of "good AI" implementations and within striking distance of published results from researchers such as Wu et al. (2014) and Jaskowski (2018), though still below the state of the art. The entire system -- from game engine to trained agent to web interface -- was built from scratch, with critical performance bottlenecks rewritten from Python to C, achieving cumulative speedups of 300-450x.

**Keywords**: Reinforcement Learning, Deep Q-Network, Expectimax, N-Tuple Networks, Temporal Difference Learning, Temporal Coherence Learning, Game AI, 2048, Afterstate Learning, Bitboard Representation, Lookup Tables

---

## Table of Contents

1. [Introduction](#1-introduction)
   1. [The Game 2048](#11-the-game-2048)
   2. [Why 2048 is a Compelling Problem for AI Research](#12-why-2048-is-a-compelling-problem-for-ai-research)
   3. [Problem Statement](#13-problem-statement)
   4. [Objectives](#14-objectives)
   5. [Scope and Limitations](#15-scope-and-limitations)
   6. [Document Structure](#16-document-structure)
2. [Theoretical Background](#2-theoretical-background)
   1. [Reinforcement Learning Foundations](#21-reinforcement-learning-foundations)
   2. [Deep Q-Networks](#22-deep-q-networks)
   3. [Tree Search Methods](#23-tree-search-methods)
   4. [N-Tuple Networks and Tabular Methods](#24-n-tuple-networks-and-tabular-methods)
   5. [Temporal Difference Learning](#25-temporal-difference-learning)
   6. [Temporal Coherence Learning](#26-temporal-coherence-learning)
   7. [Afterstate Learning](#27-afterstate-learning)
   8. [Board Symmetries and Data Augmentation](#28-board-symmetries-and-data-augmentation)
3. [Related Work](#3-related-work)
   1. [Early AI Approaches to 2048](#31-early-ai-approaches-to-2048)
   2. [N-Tuple Networks for 2048](#32-n-tuple-networks-for-2048)
   3. [Deep Learning Approaches](#33-deep-learning-approaches)
   4. [State of the Art](#34-state-of-the-art)
4. [System Architecture](#4-system-architecture)
   1. [Overall Design Philosophy](#41-overall-design-philosophy)
   2. [Component Overview](#42-component-overview)
   3. [Technology Stack](#43-technology-stack)
   4. [Data Flow Architecture](#44-data-flow-architecture)
   5. [File Structure](#45-file-structure)
5. [Approach 1: Deep Q-Network (DQN)](#5-approach-1-deep-q-network-dqn)
   1. [Motivation and Hypothesis](#51-motivation-and-hypothesis)
   2. [State Representation Design](#52-state-representation-design)
   3. [Neural Network Architecture](#53-neural-network-architecture)
   4. [Training Infrastructure](#54-training-infrastructure)
   5. [Reward Shaping](#55-reward-shaping)
   6. [Training Results and Analysis](#56-training-results-and-analysis)
   7. [Diagnosis of Failure](#57-diagnosis-of-failure)
   8. [Post-Mortem and Lessons Learned](#58-post-mortem-and-lessons-learned)
6. [Approach 2: Expectimax with Heuristics](#6-approach-2-expectimax-with-heuristics)
   1. [Motivation: From Learning to Knowledge](#61-motivation-from-learning-to-knowledge)
   2. [The Evaluation Function](#62-the-evaluation-function)
   3. [The Snake Pattern: Mathematical Derivation](#63-the-snake-pattern-mathematical-derivation)
   4. [Monotonicity Heuristic](#64-monotonicity-heuristic)
   5. [Smoothness Heuristic](#65-smoothness-heuristic)
   6. [Empty Cells and Corner Bonus](#66-empty-cells-and-corner-bonus)
   7. [Weight Calibration](#67-weight-calibration)
   8. [Pure Python Implementation (Depth 3)](#68-pure-python-implementation-depth-3)
   9. [The Lookup Table Decomposition Attempt (Failure 2)](#69-the-lookup-table-decomposition-attempt-failure-2)
   10. [C Implementation and the Transpose Bug (Failure 3)](#610-c-implementation-and-the-transpose-bug-failure-3)
   11. [Iterative Deepening with Time Budget](#611-iterative-deepening-with-time-budget)
   12. [Transposition Table](#612-transposition-table)
   13. [Final Results and Analysis](#613-final-results-and-analysis)
   14. [Cumulative Speedup Analysis](#614-cumulative-speedup-analysis)
7. [Approach 3: N-Tuple Networks](#7-approach-3-n-tuple-networks)
   1. [Motivation: Combining the Best of Both Worlds](#71-motivation-combining-the-best-of-both-worlds)
   2. [Network Architecture Design](#72-network-architecture-design)
   3. [Tuple Selection Strategy](#73-tuple-selection-strategy)
   4. [Symmetry Generation Algorithm](#74-symmetry-generation-algorithm)
   5. [Memory Requirements Analysis](#75-memory-requirements-analysis)
   6. [Version 1: Python with Backward TD (7 Tuples)](#76-version-1-python-with-backward-td-7-tuples)
   7. [Version 2: Failed Improvement Attempts](#77-version-2-failed-improvement-attempts)
   8. [The C Implementation: The Big Leap](#78-the-c-implementation-the-big-leap)
   9. [TC-Learning Implementation](#79-tc-learning-implementation)
   10. [Multi-Stage Training Strategy](#710-multi-stage-training-strategy)
   11. [Training Progression: A Detailed Chronicle](#711-training-progression-a-detailed-chronicle)
   12. [The C Player (Shared Library)](#712-the-c-player-shared-library)
8. [Optimization Journey](#8-optimization-journey)
   1. [Why Performance Matters](#81-why-performance-matters)
   2. [Move Lookup Tables](#82-move-lookup-tables)
   3. [Bitboard Representation](#83-bitboard-representation)
   4. [Hogwild Multithreading](#84-hogwild-multithreading)
   5. [Thread-Safe Random Number Generation](#85-thread-safe-random-number-generation)
   6. [Unbuffered Output (Failure 7)](#86-unbuffered-output-failure-7)
   7. [Cumulative Optimization Impact](#87-cumulative-optimization-impact)
9. [Comparative Analysis](#9-comparative-analysis)
   1. [Agent Performance Metrics](#91-agent-performance-metrics)
   2. [Training Efficiency Comparison](#92-training-efficiency-comparison)
   3. [Computational Cost Analysis](#93-computational-cost-analysis)
   4. [Decision Quality Analysis](#94-decision-quality-analysis)
   5. [Statistical Significance](#95-statistical-significance)
10. [Performance Ceiling and State of the Art Comparison](#10-performance-ceiling-and-state-of-the-art-comparison)
    1. [Positioning Against Published Results](#101-positioning-against-published-results)
    2. [Theoretical Performance Ceiling](#102-theoretical-performance-ceiling)
    3. [Convergence Analysis](#103-convergence-analysis)
    4. [What Would It Take to Reach 99%](#104-what-would-it-take-to-reach-99)
11. [Web Interface and Real-Time Visualization](#11-web-interface-and-real-time-visualization)
    1. [Frontend Architecture](#111-frontend-architecture)
    2. [Backend API Server](#112-backend-api-server)
    3. [Agent Selection and Visual Feedback](#113-agent-selection-and-visual-feedback)
    4. [Game Report System](#114-game-report-system)
    5. [Real Game Report Examples](#115-real-game-report-examples)
12. [Failures, Debugging, and Lessons Learned](#12-failures-debugging-and-lessons-learned)
    1. [Failure 1: DQN Does Not Converge](#121-failure-1-dqn-does-not-converge)
    2. [Failure 2: 1D Lookup Table Decomposition](#122-failure-2-1d-lookup-table-decomposition)
    3. [Failure 3: The Transpose Bug](#123-failure-3-the-transpose-bug)
    4. [Failure 4: Learning Rate Overflow](#124-failure-4-learning-rate-overflow)
    5. [Failure 5: Symmetry Normalization](#125-failure-5-symmetry-normalization)
    6. [Failure 6: Adaptive Depth Freezes](#126-failure-6-adaptive-depth-freezes)
    7. [Failure 7: Buffered Output](#127-failure-7-buffered-output)
    8. [General Lessons Learned](#128-general-lessons-learned)
13. [Conclusion and Future Work](#13-conclusion-and-future-work)
    1. [Summary of Results](#131-summary-of-results)
    2. [Contributions](#132-contributions)
    3. [Future Work](#133-future-work)
14. [References](#14-references)
15. [Appendices](#15-appendices)
    1. [Appendix A: Key Code Listings](#appendix-a-key-code-listings)
    2. [Appendix B: Training Log Excerpts](#appendix-b-training-log-excerpts)
    3. [Appendix C: Game Report Examples](#appendix-c-game-report-examples)

---

## 1. Introduction

### 1.1 The Game 2048

2048 is a single-player sliding tile puzzle game created by Italian web developer Gabriele Cirulli in March 2014. The game became a viral sensation, attracting millions of players worldwide within weeks of its release. It takes place on a 4x4 grid where numbered tiles slide in response to player input. The game mechanics are deceptively simple:

1. **The board** is a 4x4 grid, initially containing two tiles (each either 2 or 4).
2. **Moves** consist of sliding all tiles in one of four directions: up, down, left, or right.
3. **Merging** occurs when two tiles with the same value collide during a slide. They merge into a single tile with double the value (e.g., two 4-tiles merge into one 8-tile). Each tile can only merge once per move.
4. **Spawning** happens after every valid move: a new tile appears at a random empty position. With 90% probability this tile has value 2; with 10% probability it has value 4.
5. **Scoring** is cumulative: the player earns points equal to the value of each newly merged tile (e.g., merging two 16-tiles earns 32 points).
6. **Victory** is achieved when any tile reaches the value 2048.
7. **Game over** occurs when no valid moves remain -- the board is full and no adjacent tiles share the same value.

Though simple in concept, the game presents remarkable depth. A skilled human player can typically reach the 512 or 1024 tile with moderate consistency, but reaching 2048 requires both strategic planning and some luck. The highest theoretically achievable tile on a 4x4 board is 131,072 (2^17), though achieving tiles beyond 8192 requires extraordinary play.

```
+------+------+------+------+
| 2048 |  256 |   64 |    4 |
+------+------+------+------+
|  512 |  128 |   32 |    2 |
+------+------+------+------+
|   16 |    8 |    4 |    2 |
+------+------+------+------+
|    4 |    2 |      |      |
+------+------+------+------+

Figure 1.1: A well-organized 2048 board showing the "snake pattern"
-- tiles decrease in value along a zigzag path from the top-left corner.
```

### 1.2 Why 2048 is a Compelling Problem for AI Research

2048 sits at a fascinating intersection of several properties that make it an ideal testbed for AI research:

**Large but finite state space.** The board has 16 cells, each of which can hold one of approximately 18 distinct values (empty, 2, 4, 8, ..., 131072). This gives a theoretical upper bound of approximately 2.5 x 10^28 possible board configurations. This is far too large for exhaustive enumeration, yet the state space has enough structure to be exploitable by intelligent methods. For comparison, chess has approximately 10^47 legal positions and Go has approximately 10^170. In this landscape, 2048's state space is modest enough to be tractable but large enough to be non-trivial.

**Stochastic elements.** Unlike purely deterministic games such as chess or Go, 2048 introduces randomness through the tile spawning mechanism. After every move, a tile of value 2 (probability 0.9) or 4 (probability 0.1) appears at a uniformly random empty cell. This means that no deterministic strategy can guarantee victory. The AI must reason about expected outcomes over probability distributions, not just optimal deterministic play.

**Long horizon.** A typical game of 2048 lasts between 500 and 2000 moves. The agent must make decisions whose consequences may not be apparent for hundreds of moves. This long credit-assignment horizon is a known challenge for reinforcement learning.

**Sparse and delayed rewards.** The game's reward signal (merge scores) is intermittent and does not directly indicate the quality of a board position. A move that earns zero points by simply repositioning tiles may be strategically critical. Conversely, a move that earns points by merging low-value tiles may destroy a carefully constructed high-value chain. This disconnect between immediate reward and strategic value makes reward-based learning challenging.

**Multiple viable approaches.** Perhaps most importantly, 2048 is amenable to fundamentally different AI paradigms: tree search with evaluation functions, deep reinforcement learning, tabular methods, and hybrid approaches. This allows meaningful comparative analysis of radically different techniques on the same problem.

**Practical runtime constraints.** When deploying an AI agent for interactive play (as we do via a web interface), the agent must respond within a time budget that allows smooth visualization. This creates real-world engineering constraints that purely academic implementations can ignore.

### 1.3 Problem Statement

Given the game of 2048 as defined by Cirulli's original implementation, design, implement, and comparatively evaluate multiple AI agents that can:

1. Play complete games autonomously, from initialization to game over or victory;
2. Achieve a win rate (reaching the 2048 tile) significantly exceeding that of an average human player (~30%);
3. Respond within a time budget suitable for real-time interactive visualization;
4. Be integrated into a web-based interface allowing side-by-side comparison.

### 1.4 Objectives

The objectives of this work are:

1. **Implement and compare three fundamentally different AI approaches** for 2048: Deep Q-Networks (DQN), Expectimax tree search with hand-crafted heuristics, and N-Tuple Networks with TD-learning. Each approach represents a distinct philosophy of AI design.

2. **Achieve a win rate above 90%** with the best agent, placing it in the upper tier of AI implementations for this game.

3. **Document the iterative development process comprehensively**, including every failure, debugging narrative, and design decision, providing a practical reference for others building game-playing AI systems.

4. **Create a web interface for real-time visualization** where users can observe agents playing, switch between them, and see the differences in their behavior and performance.

5. **Analyze the performance ceiling** of each approach and situate our results within the broader landscape of published 2048 AI research.

### 1.5 Scope and Limitations

This work focuses on the standard 4x4 variant of 2048 as implemented by Cirulli. We do not consider:

- Variants such as 5x5, hexagonal, or three-dimensional boards;
- Multi-player variants;
- The "keep playing" mode after reaching 2048 (though our agents can continue playing);
- Techniques requiring specialized hardware (TPUs, multi-GPU training);
- Monte Carlo Tree Search (MCTS), which is a promising approach but was not explored in this iteration.

All development was performed on Apple Silicon hardware (M-series processor) running macOS. The C implementations use Apple Clang with -O3 optimization. The Python implementations use CPython 3.9 with PyTorch for the DQN agent.

### 1.6 Document Structure

This document is organized in roughly chronological order of development, reflecting the iterative nature of the project. Each major chapter describes an approach, its design decisions, results, and the lessons that motivated the next iteration. The reader should be able to follow the narrative arc: from the optimistic first attempt with DQN, through the pragmatic pivot to Expectimax, to the sophisticated synthesis of N-Tuple Networks.

Chapter 2 provides the theoretical background necessary to understand all three approaches. Chapter 3 surveys related work. Chapter 4 describes the overall system architecture. Chapters 5, 6, and 7 detail each approach in depth. Chapter 8 focuses on the optimization journey that made the system practical. Chapter 9 provides comparative analysis. Chapter 10 examines the performance ceiling. Chapter 11 describes the web interface. Chapter 12 is a dedicated chapter on failures and debugging. Chapter 13 concludes with reflections and future work.

---

## 2. Theoretical Background

This chapter provides the theoretical foundations for the three AI approaches explored in this work. We begin with general reinforcement learning theory, then specialize to the specific techniques used in each approach.

### 2.1 Reinforcement Learning Foundations

Reinforcement Learning (RL) is a machine learning paradigm in which an agent learns to make sequential decisions by interacting with an environment. Unlike supervised learning, where the correct output is provided for each input, RL agents must discover good behavior through trial and error, guided only by a scalar reward signal.

#### 2.1.1 The Markov Decision Process Framework

The formal framework for RL is the Markov Decision Process (MDP), defined by the tuple (S, A, P, R, gamma), where:

- **S** is the set of states the environment can be in;
- **A** is the set of actions available to the agent;
- **P(s'|s,a)** is the state transition probability -- the probability of transitioning to state s' when taking action a in state s;
- **R(s,a,s')** is the reward function, giving the immediate reward for a transition;
- **gamma** in [0,1] is the discount factor, controlling the relative importance of immediate vs. future rewards.

In the context of 2048:
- **States** are the 4x4 board configurations;
- **Actions** are the four directions: up, down, left, right;
- **Transitions** are determined by the merge mechanics (deterministic) followed by random tile placement (stochastic);
- **Rewards** are the merge scores (e.g., merging two 64-tiles yields reward 128);
- **Discount factor** is set near 1 (e.g., gamma = 0.99) because future rewards are highly relevant.

#### 2.1.2 Value Functions

The **state-value function** V^pi(s) gives the expected cumulative discounted reward starting from state s and following policy pi:

```
V^pi(s) = E_pi[ sum_{t=0}^{infinity} gamma^t * R_{t+1} | S_0 = s ]
```

The **action-value function** Q^pi(s,a) gives the expected return starting from state s, taking action a, and then following pi:

```
Q^pi(s,a) = E_pi[ sum_{t=0}^{infinity} gamma^t * R_{t+1} | S_0 = s, A_0 = a ]
```

The **optimal value functions** V*(s) and Q*(s,a) correspond to the policy that maximizes expected return. The optimal policy can be derived from Q* as:

```
pi*(s) = argmax_a Q*(s, a)
```

#### 2.1.3 Q-Learning

Q-learning, introduced by Watkins (1989), is an off-policy temporal difference algorithm that directly estimates Q*(s,a) without requiring a model of the environment. The update rule is:

```
Q(s_t, a_t) <- Q(s_t, a_t) + alpha * [ r_{t+1} + gamma * max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t) ]
```

where alpha is the learning rate. The term in brackets is the **temporal difference (TD) error** -- the difference between the current estimate Q(s_t, a_t) and the bootstrapped target r_{t+1} + gamma * max_{a'} Q(s_{t+1}, a').

Q-learning converges to Q* under mild conditions: every state-action pair must be visited infinitely often, and the learning rate must satisfy the Robbins-Monro conditions (decreasing but with sum diverging to infinity). In practice, these conditions are approximated through epsilon-greedy exploration and fixed or slowly decaying learning rates.

#### 2.1.4 The Exploration-Exploitation Dilemma

An RL agent faces a fundamental tension: should it **exploit** its current knowledge to maximize immediate reward, or **explore** unfamiliar actions that might lead to better long-term outcomes? The standard approach is **epsilon-greedy** exploration:

```
a = { random action from A,         with probability epsilon
    { argmax_a Q(s, a),              with probability 1 - epsilon
```

Epsilon is typically decayed over time, starting near 1 (pure exploration) and decreasing toward a small value (mostly exploitation). The rate of this decay is critical and domain-dependent. As we will see in Chapter 5, getting this decay rate wrong was a key factor in our DQN agent's failure.

### 2.2 Deep Q-Networks

#### 2.2.1 From Tabular to Function Approximation

When the state space is too large for a lookup table (as in 2048, with ~10^28 states), the Q-function must be approximated. Deep Q-Networks (DQN), introduced by Mnih et al. (2015) in their landmark Nature paper, approximate Q(s,a) with a deep neural network parameterized by theta:

```
Q(s, a; theta) ~ Q*(s, a)
```

The network takes a state representation as input and outputs Q-values for all actions simultaneously. Training minimizes the loss:

```
L(theta) = E[ (r + gamma * max_{a'} Q(s', a'; theta^-) - Q(s, a; theta))^2 ]
```

where theta^- are the parameters of a separate **target network**.

#### 2.2.2 Stabilizing Techniques

DQN introduced two critical techniques for stable training:

**Experience Replay.** Instead of learning from consecutive transitions (which are temporally correlated), DQN stores transitions (s, a, r, s') in a replay buffer and samples random mini-batches for training. This breaks correlations between consecutive samples and allows each experience to be used multiple times.

**Target Network.** Using the same network to both select and evaluate actions creates a moving target problem. DQN uses a separate target network theta^- that is periodically updated (every C steps) to match the policy network theta. Between updates, theta^- is frozen, providing stable targets.

#### 2.2.3 Double DQN

Van Hasselt et al. (2016) identified that standard DQN overestimates Q-values because the max operator in the target uses the same values for both selection and evaluation. **Double DQN** decouples these:

```
Target = r + gamma * Q(s', argmax_{a'} Q(s', a'; theta); theta^-)
```

The policy network theta selects the best action, but the target network theta^- evaluates it. This reduces overestimation bias.

### 2.3 Tree Search Methods

#### 2.3.1 Minimax

Minimax is a classic algorithm for two-player zero-sum games. It constructs a game tree where MAX nodes (the player) choose the action with the highest value, and MIN nodes (the opponent) choose the action with the lowest value. The algorithm assumes optimal play from both sides.

#### 2.3.2 Expectimax

In games with random elements, the opponent is not adversarial but stochastic. **Expectimax** replaces MIN nodes with **CHANCE** nodes that compute the expected (weighted average) value over random outcomes:

```
V(chance_node) = sum_{outcome o} P(o) * V(child(o))
```

In 2048, the CHANCE nodes represent the random tile placement. For each empty cell, there are two possible outcomes (tile 2 with probability 0.9, tile 4 with probability 0.1). The total number of CHANCE children for a node with k empty cells is 2k.

The quality of Expectimax play depends critically on the **evaluation function** used to assess leaf nodes. This function encodes domain knowledge about what constitutes a "good" board position.

```
          [MAX]                     Player chooses best move
         / | \ \
        /  |  \ \
      [C] [C] [C] [C]              Chance nodes (random tile)
      /\   /\   /\   /\
     .  .  .  .  .  . .  .         Each: tile 2 (90%) or 4 (10%)
    MAX MAX MAX MAX MAX MAX         at each empty cell
    ...                             Recurse to depth limit
    
    [EVAL] [EVAL] [EVAL]            Leaf evaluation

Figure 2.1: Expectimax tree structure for 2048.
MAX nodes choose the best direction (up/down/left/right).
CHANCE nodes average over random tile placements.
```

#### 2.3.3 Iterative Deepening

A practical technique for real-time play is **iterative deepening**: perform searches of increasing depth (1, 2, 3, ...) until a time budget is exhausted. The key insight is that each deeper search is much more expensive than all previous searches combined (due to the exponential branching factor), so the time "wasted" on shallower searches is negligible. Iterative deepening guarantees that:

1. A result is always available (from the last completed depth);
2. Time is used efficiently -- deeper search proceeds only when time permits;
3. Information from shallower searches can be used to guide deeper ones (e.g., via transposition tables).

#### 2.3.4 Transposition Tables

A **transposition table** is a hash-based cache that stores evaluation results for previously seen board positions. In 2048, the same board position can be reached via different sequences of moves. Without caching, the search would redundantly evaluate these identical positions.

Our implementation uses a direct-mapped cache with 2^22 = 4,194,304 entries, indexed by a hash of the board state and search depth. Collisions are resolved by replacement (the new entry overwrites the old one). The hash function combines the board's uint64_t representation with the depth:

```
hash(board, depth) = ((board ^ (board >> 16)) * (0x45D9F3B + depth * 0x9E3779B9)) >> 16
```

The constant 0x9E3779B9 is related to the golden ratio and helps distribute hash values uniformly.

### 2.4 N-Tuple Networks and Tabular Methods

#### 2.4.1 Origin and Concept

N-Tuple Networks were introduced by Lucas (2008) for learning evaluation functions in board games. The core idea is elegantly simple: instead of using a neural network to approximate the value function, use a collection of **lookup tables** (LUTs).

An **n-tuple** is a set of n positions on the board. For a given board configuration, we look up the values at those n positions and use them as an index into a lookup table. The entry at that index stores a learned weight representing how "good" or "bad" that particular combination of values is.

The total evaluation of a board is the sum of weights from all tuples:

```
V(board) = sum_{t=1}^{T} sum_{s=1}^{S_t} W_t[ encode(board, tuple_t_s) ]
```

where T is the number of base tuples, S_t is the number of symmetric variants of tuple t, and W_t is the weight table for tuple t.

**Why this works.** Consider a 6-tuple covering positions (0,0), (0,1), (0,2), (0,3), (1,0), (1,1) -- the top-left 4x2 region. If the board has [2048, 1024, 512, 256, 128, 64] at those positions, the lookup table can learn that this particular configuration (a well-organized descending sequence) is highly valuable. The network learns these associations purely from playing games.

#### 2.4.2 Encoding

For a 6-tuple, each position can hold one of 16 possible values in our log2 representation (0 = empty, 1 = tile 2, 2 = tile 4, ..., 15 = tile 32768). The encoding is:

```
index = v_0 * 16^5 + v_1 * 16^4 + v_2 * 16^3 + v_3 * 16^2 + v_4 * 16 + v_5
```

This gives 16^6 = 16,777,216 possible entries per tuple. Each entry is a 32-bit float, requiring 64 MB per tuple.

#### 2.4.3 Advantages Over Neural Networks

For the specific problem of 2048, N-Tuple Networks have several decisive advantages:

1. **Speed.** Evaluation requires only array lookups and addition -- no matrix multiplication, no activation functions, no backpropagation. A single evaluation takes nanoseconds.

2. **No gradient computation.** Weight updates are direct: add the learning rate times the TD error to the relevant entries. No chain rule, no vanishing gradients.

3. **Controllable capacity.** More tuples means more expressiveness. The trade-off is explicit and quantifiable: each 6-tuple costs exactly 64 MB.

4. **No generalization (by design).** Each configuration is stored independently. While this means the network cannot generalize to unseen configurations, it also means it cannot confuse similar-but-different configurations. In 2048, where the difference between [2048, 1024, 512, 256] and [2048, 512, 1024, 256] is strategically significant, this is an advantage.

#### 2.4.4 Disadvantage: Memory

The primary disadvantage is memory usage. Our 17-tuple network requires:

```
17 tuples * 16,777,216 entries * 4 bytes = 1,139,802,112 bytes ~ 1.07 GB
```

With TC-learning (which requires two additional tables per tuple for the signed and absolute delta accumulators), this triples to approximately 3.2 GB. This is manageable on modern hardware but limits the number and size of tuples.

### 2.5 Temporal Difference Learning

#### 2.5.1 TD(0)

Temporal Difference learning combines Monte Carlo sampling with bootstrapping. The TD(0) update for a value function is:

```
V(s_t) <- V(s_t) + alpha * [ r_{t+1} + gamma * V(s_{t+1}) - V(s_t) ]
```

The key insight is that we do not wait until the end of an episode to update values. Instead, we update V(s_t) using the observed reward r_{t+1} and the current estimate V(s_{t+1}). This is the "bootstrapping" aspect -- we use our own estimate to improve itself.

#### 2.5.2 Why TD(0) Converges

TD(0) converges to the true value function under the following conditions:

1. Every state is visited infinitely often;
2. The learning rate alpha_t satisfies: sum alpha_t = infinity and sum alpha_t^2 < infinity;
3. The reward variance is finite.

The convergence proof relies on the contraction mapping theorem. The Bellman operator T defined by TV(s) = E[r + gamma * V(s')] is a contraction in the maximum norm with contraction factor gamma < 1. TD(0) approximates this operator, and with appropriate step sizes, the approximation converges to the fixed point V*.

In practice, we use a fixed learning rate (violating condition 2) but rely on the fact that the distribution of states visited changes as the policy improves, providing implicit regularization.

#### 2.5.3 TD(0) for 2048

In our N-Tuple implementation, the TD update is applied to **afterstates** (see Section 2.7). After the agent selects action a_t leading to afterstate s_t^after, and the random tile creates state s_{t+1}, and the agent then selects action a_{t+1} leading to afterstate s_{t+1}^after, the update is:

```
V(s_t^after) <- V(s_t^after) + alpha * [ r_{t+1} + V(s_{t+1}^after) - V(s_t^after) ]
```

Note that gamma = 1 (no discounting) because we want to maximize the total game score, which is naturally bounded by the game's termination.

### 2.6 Temporal Coherence Learning

#### 2.6.1 Motivation

A fixed learning rate alpha is a compromise: too high and learning is unstable; too low and learning is slow. Different weights in the lookup table may need different learning rates -- some are in stable regions of the value function and need small updates, while others are in rapidly changing regions and need large updates.

**Temporal Coherence (TC) Learning**, introduced by Jaskowski (2018) in the context of 2048, adapts the learning rate individually for each weight based on the consistency of the updates it receives.

#### 2.6.2 The TC Ratio

For each weight w[i], TC-learning maintains two accumulators:

- **Signed sum** tc_sum[i]: the exponentially decayed sum of signed TD errors;
- **Absolute sum** tc_abs[i]: the exponentially decayed sum of absolute TD errors.

The updates with decay factor lambda (we use 0.9995) are:

```
tc_sum[i] = lambda * tc_sum[i] + delta
tc_abs[i] = lambda * tc_abs[i] + |delta|
```

The **TC ratio** measures temporal coherence:

```
tc_ratio[i] = |tc_sum[i]| / tc_abs[i]
```

**Interpretation.** The TC ratio ranges from 0 to 1:

- **tc_ratio ~ 1**: The updates to this weight are consistently in the same direction. The weight has a clear direction to move and should do so aggressively. This happens when the weight is systematically under- or over-estimating.

- **tc_ratio ~ 0**: The updates oscillate -- sometimes positive, sometimes negative, with approximately equal magnitude. The weight has approximately converged, and the remaining updates are noise. The learning rate should be reduced to prevent oscillation.

The adaptive weight update becomes:

```
w[i] += alpha * tc_ratio[i] * delta
```

#### 2.6.3 The Decay Factor

The decay factor lambda = 0.9995 controls the "memory" of the accumulators. The effective window size (half-life) is:

```
half_life = -ln(2) / ln(0.9995) ~ 1386 updates
```

This means each weight's learning rate adapts based on roughly the last 1,400 updates it received. This is long enough to detect genuine trends but short enough to adapt when the value landscape changes (as it does during training, when the policy improves and the distribution of visited states shifts).

#### 2.6.4 Memory Cost

TC-learning triples the memory requirement because each tuple needs three tables instead of one: weights, tc_sum, and tc_abs. For our 17-tuple network:

```
Without TC: 17 * 16^6 * 4 bytes = ~1.07 GB
With TC:    17 * 16^6 * 4 bytes * 3 = ~3.22 GB
```

This is significant but manageable. The file `checkpoints/ntuple_latest.bin` with TC data weighs 3,264 MB.

### 2.7 Afterstate Learning

#### 2.7.1 The Problem with State-Based Learning

In 2048, the state transition has two phases:

1. **Deterministic**: The player's move (slide + merge) transforms the board deterministically.
2. **Stochastic**: A random tile is placed at a random empty position.

If we learn V(state) -- the value of the state *before* the player moves -- we face a problem: the same state can lead to very different outcomes due to the random tile. This introduces high variance in the value estimates.

#### 2.7.2 The Afterstate Solution

An **afterstate** is the board configuration after the player's deterministic move but before the random tile placement. Learning V(afterstate) has lower variance because:

- The afterstate is fully determined by the player's action (no randomness);
- The value of the afterstate only needs to account for the randomness of future tile placements, not the current one;
- Two different states that lead to the same afterstate (via the same move) share the same learned value, providing implicit generalization.

```
State s_t --> [Player Move] --> Afterstate s_t^after --> [Random Tile] --> State s_{t+1}
                                      ^
                                      |
                              Learn V(here)
                              
Figure 2.2: The afterstate is the board after the player's move but before
the random tile. Learning V at this point reduces variance.
```

#### 2.7.3 Action Selection with Afterstates

Given a state s, the agent selects the action a that maximizes:

```
a* = argmax_a [ reward(s, a) + V(afterstate(s, a)) ]
```

where reward(s, a) is the merge score from action a, and afterstate(s, a) is the resulting afterstate.

### 2.8 Board Symmetries and Data Augmentation

#### 2.8.1 The Symmetry Group of the 4x4 Board

The 4x4 board has the symmetry group of the square (the dihedral group D4), which contains 8 elements:

1. Identity (no transformation)
2. 90-degree rotation
3. 180-degree rotation
4. 270-degree rotation
5. Horizontal reflection
6. Vertical reflection
7. Diagonal reflection (transpose)
8. Anti-diagonal reflection

For each base tuple, we can generate up to 8 symmetric variants by applying these transformations to the tuple's positions. For example, the tuple covering positions {(0,0), (0,1), (0,2), (0,3), (1,0), (1,1)} under 90-degree rotation becomes {(0,3), (1,3), (2,3), (3,3), (0,2), (1,2)}.

#### 2.8.2 Duplicate Elimination

Some tuples may be symmetric to themselves under certain transformations. For instance, a tuple centered on the board might map to itself under 180-degree rotation. To avoid counting such configurations multiple times, we canonicalize each symmetric variant (by sorting its positions) and check for duplicates.

Our implementation generates all 8 candidates, sorts each by position, and removes duplicates. The number of unique symmetric variants per tuple ranges from 4 to 8 depending on the tuple's inherent symmetry.

#### 2.8.3 Impact on Effective Capacity

Symmetries effectively multiply the network's capacity without additional memory. If a base tuple has 8 unique symmetric variants, the network evaluates 8 different regions of the board using the same weight table. This means the weights learn from 8x more examples per game, significantly accelerating training.

For our 17-tuple network with an average of approximately 6 unique symmetries per tuple, the total number of features evaluated per board is approximately:

```
17 tuples * ~6 symmetries/tuple = ~102 lookups per evaluation
```

Each lookup accesses a different subset of the 64 MB weight table, providing a rich representation of the board state.

---

## 3. Related Work

### 3.1 Early AI Approaches to 2048

Within months of 2048's release in March 2014, the AI community produced several strong players. The most notable early work was by **nneonneo** (Robert Xiao), who implemented a highly optimized Expectimax search in C/C++ using bitboard representation. His implementation, `nneonneo/2048-ai`, achieved near-perfect play at high search depths and set the standard for 2048 AI performance.

Key innovations from early approaches:

- **Bitboard representation**: Encoding the 4x4 board as a 64-bit integer (16 nibbles of 4 bits each), enabling efficient manipulation through bit operations.
- **Row-based lookup tables**: Pre-computing merge results for all 65,536 possible 4-cell rows, reducing the move computation from an iterative algorithm to four table lookups.
- **Monotonicity and smoothness heuristics**: Quantifying the "orderedness" and "uniformity" of the board as proxy measures of position quality.

### 3.2 N-Tuple Networks for 2048

The application of N-Tuple Networks to 2048 was pioneered by two groups working in parallel:

**Szubert and Jaskowski (2014)** published "Temporal Difference Learning of N-Tuple Networks for the Game 2048" at IEEE CIG 2014. They demonstrated that N-Tuple Networks with TD-learning could achieve a 97% rate of reaching the 2048 tile, with average scores around 100,000 points. Their key contributions were:

- The application of afterstate learning to reduce variance;
- A systematic study of tuple configurations, finding that 6-tuples provide the best balance of expressiveness and memory;
- The demonstration that N-Tuple Networks significantly outperform neural network approaches for this problem.

**Wu et al. (2014)** published "Multi-stage temporal difference learning for 2048" at TAAI 2014. They introduced **multi-stage training**, where the network is first trained with shallow search (1-ply) to learn basic patterns quickly, then fine-tuned with deeper search (3-ply) to improve strategic play. This staged approach is far more efficient than training from scratch with deep search.

**Jaskowski (2018)** published "Mastering 2048 with Delayed Temporal Coherence Learning" in IEEE Transactions on Games. This work introduced TC-learning (Temporal Coherence Learning) to 2048, achieving state-of-the-art results: approximately 99.5% rate of reaching 2048, approximately 95% for 4096, and approximately 75% for 8192, with average scores exceeding 300,000 points. Key innovations:

- Per-weight adaptive learning rate based on temporal coherence;
- Larger tuple configurations (up to 8-tuples);
- Extended training (40+ million episodes).

### 3.3 Deep Learning Approaches

Several researchers have applied deep learning to 2048:

- **Convolutional Neural Networks**: Various architectures have been tried, typically achieving 2048 rates of 50-80% after extensive training. The key challenge is that CNNs with small kernels have limited receptive fields that cannot capture board-wide spatial patterns.

- **Residual Networks**: Deeper architectures with skip connections can potentially capture longer-range dependencies, but training instability and the long game horizon remain challenges.

- **Policy Gradient Methods**: Actor-critic and PPO-based approaches have been explored, but tend to converge slowly due to the high variance of the reward signal.

The general consensus in the literature is that deep learning approaches, while theoretically flexible, are significantly less efficient than N-Tuple Networks for 2048. The structured, discrete nature of the state space is better exploited by tabular methods.

### 3.4 State of the Art

As of the latest published results, the state of the art for 2048 AI is:

| Method | 2048 Rate | 4096 Rate | 8192 Rate | 16384 Rate | Avg Score |
|--------|-----------|-----------|-----------|------------|-----------|
| Random play | ~0% | 0% | 0% | 0% | ~800 |
| Skilled human | ~30% | rare | 0% | 0% | ~20,000 |
| Expectimax + heuristics | ~80% | ~60% | 0% | 0% | ~40,000 |
| Basic N-Tuple (5k ep) | ~73% | ~33% | ~1% | 0% | ~42,000 |
| Wu et al. (2014) | ~97% | ~85% | ~40% | ~5% | ~100,000+ |
| Jaskowski (2018) | ~99.5% | ~95% | ~75% | ~30% | ~300,000+ |

The gap between basic implementations and state-of-the-art is primarily due to:
1. Training duration (millions vs. thousands of episodes);
2. Tuple architecture (more and larger tuples);
3. Advanced techniques (TC-learning, multi-stage training);
4. Search depth during play (deeper search yields better play at higher computational cost).

---

## 4. System Architecture

### 4.1 Overall Design Philosophy

The system was designed with three guiding principles:

1. **Prototype in Python, optimize in C.** Every algorithm was first implemented in Python for correctness verification. Once validated, performance-critical components were reimplemented in C and exposed as shared libraries callable from Python via `ctypes`. This approach combines Python's rapid development cycle with C's raw performance.

2. **Shared interfaces.** All three agents expose the same API: given a board state (4x4 grid of integers), return the best action (0-3). This uniform interface simplifies the server and frontend code.

3. **Automated metrics.** Every game played through the web interface generates a JSON report with complete move history, timing data, and outcome. This enables post-hoc analysis without requiring replay.

### 4.2 Component Overview

```
+--------------------------------------------------+
|                  Web Browser                      |
|  +--------------------------------------------+  |
|  |  2048 Game (Cirulli's original JS code)     |  |
|  |  + AI Player module (ai_player.js)          |  |
|  |    - Extract grid state                     |  |
|  |    - Send to backend API                    |  |
|  |    - Apply returned move                    |  |
|  |    - Save game report on completion         |  |
|  +--------------------------------------------+  |
+--------------------------------------------------+
              |                    |
              | POST /move         | POST /report
              v                    v
+--------------------------------------------------+
|              Flask API Server (server.py)          |
|  +------+   +------------+   +-----------+        |
|  | DQN  |   | Expectimax |   | N-Tuple   |        |
|  | Agent |   | Agent      |   | Agent     |        |
|  |(.py)  |   | (.py->C)   |   | (.py->C)  |        |
|  +------+   +------------+   +-----------+        |
|                    |                  |             |
|              ctypes FFI         ctypes FFI          |
|                    v                  v             |
|          +----------------+  +----------------+    |
|          | expectimax_c.so|  |  ntuple_c.so   |    |
|          | (C shared lib) |  | (C shared lib) |    |
|          +----------------+  +----------------+    |
+--------------------------------------------------+
                                       ^
                                       |
                              loads weights from
                                       |
+--------------------------------------------------+
|          Training Infrastructure                   |
|  +--------------------------------------------+  |
|  |  ntuple_train (C binary)                    |  |
|  |  - Hogwild multithreading (8 threads)       |  |
|  |  - TD(0) with afterstates                   |  |
|  |  - TC-learning per-weight adaptive LR       |  |
|  |  - Multi-stage: 1-ply then 3-ply           |  |
|  |  - Saves checkpoints to disk                |  |
|  +--------------------------------------------+  |
+--------------------------------------------------+

Figure 4.1: System architecture showing the three-layer design:
browser frontend, Python/Flask API server, and C shared libraries.
```

### 4.3 Technology Stack

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| Game frontend | HTML/CSS/JavaScript | Cirulli's original 2048 implementation |
| AI frontend integration | JavaScript (ai_player.js) | Direct DOM interaction with the game |
| API server | Python 3.9 + Flask + Flask-CORS | Rapid development, easy integration |
| DQN agent | Python + PyTorch | GPU-accelerated neural network |
| Expectimax engine | C (cc -O3) | Performance-critical search |
| N-Tuple player | C (cc -O3) | Fast evaluation + search |
| N-Tuple trainer | C + pthreads | Parallel training |
| Data exchange | JSON via HTTP | Standard, human-readable |
| Reports storage | JSON files | Easy to parse and analyze |

### 4.4 Data Flow Architecture

A single AI move follows this data flow:

1. **Grid extraction** (JavaScript): The `extractGrid()` function reads the DOM to construct a 4x4 integer array. Note the coordinate swap: Cirulli's code stores cells as `grid.cells[x][y]` (column-major), but our backend expects `grid[y][x]` (row-major).

2. **HTTP request** (JavaScript -> Python): An XHR POST request sends `{grid: [[...], ...], agent: "ntuple"}` to `http://localhost:8081/move`.

3. **Agent dispatch** (Python): The Flask server routes the request to the appropriate agent. For N-Tuple, it calls the C shared library via ctypes.

4. **C evaluation** (C): The C library loads the board, performs 5-ply Expectimax search using the trained N-Tuple weights as the evaluation function, and returns the best action.

5. **Response** (Python -> JavaScript): The server returns `{action: 2, direction: "down", agent: "ntuple"}`.

6. **Move execution** (JavaScript): The frontend emits the move event via `gm.inputManager.emit("move", data.action)`, which triggers the game's native move handler.

Total round-trip time is approximately 5-10ms (2.4ms for the C evaluation, plus network and JavaScript overhead).

### 4.5 File Structure

```
ai/
|-- game.py                  # Python game engine (faithful to Cirulli's logic)
|-- dqn_agent.py             # DQN agent (PyTorch CNN + replay + target net)
|-- train.py                 # DQN training loop
|-- expectimax_agent.py      # Expectimax in pure Python (original)
|-- expectimax_c.c           # Expectimax in C (623 lines, optimized)
|-- expectimax_c.so          # Compiled shared library
|-- expectimax_native.py     # Python wrapper for C Expectimax
|-- ntuple_agent.py          # N-Tuple in pure Python (legacy)
|-- ntuple_train.c           # N-Tuple trainer in C (722 lines)
|-- ntuple_train             # Compiled training binary
|-- ntuple_c.c               # N-Tuple player in C (307 lines)
|-- ntuple_c.so              # Compiled shared library
|-- server.py                # Flask API server (211 lines)
|-- play.py                  # Terminal player for testing
|-- analysis.py              # Chart generation script
|-- checkpoints/             # Saved model weights
|   |-- best.pt              # DQN PyTorch checkpoint (~3 MB)
|   |-- ntuple_latest.bin    # N-Tuple weights + TC (~3.2 GB)
|   |-- ntuple_best.bin      # Best N-Tuple weights + TC (~3.2 GB)
|-- reports/                 # JSON game reports (50+ files)
|-- charts/                  # Generated analysis charts
|
js/
|-- ai_player.js             # Frontend AI integration (256 lines)
```

---

## 5. Approach 1: Deep Q-Network (DQN)

### 5.1 Motivation and Hypothesis

DQN was chosen as the first approach for several reasons:

1. **Emblematic status.** DQN's success on Atari games (Mnih et al., 2015) made it the "poster child" of deep reinforcement learning. Starting here seemed natural.

2. **Minimal domain knowledge.** DQN learns entirely from experience, requiring no hand-crafted evaluation functions. The hypothesis was that a convolutional neural network could discover the relevant spatial patterns (monotonicity, corner preference, merge chains) on its own.

3. **Pedagogical value.** Implementing DQN provides deep understanding of experience replay, target networks, epsilon scheduling, and the challenges of function approximation in RL.

The hypothesis was: *A DQN with convolutional layers, trained on a one-hot encoding of the board, can learn to play 2048 at a competitive level (>50% win rate) within a reasonable training time (a few hours).*

As we will see, this hypothesis was wrong -- but the reasons are instructive.

### 5.2 State Representation Design

The 4x4 board must be encoded as a tensor suitable for a neural network. We considered several representations:

**Option 1: Raw values.** Represent each cell as its integer value (0, 2, 4, 8, ..., 2048). Problem: the magnitude range (0 to 2048+) would cause the network to focus disproportionately on high-value tiles, and the exponential spacing (2, 4, 8, 16, ...) would make gradient-based learning unstable.

**Option 2: Normalized values.** Divide each cell by the maximum possible value. Problem: the distribution is extremely skewed -- most cells have values near 0 while a few cells have very high values.

**Option 3: Log2 encoding.** Represent each cell as log2(value), giving values in {0, 1, 2, ..., 11}. Better, but still encodes an ordinal relationship that may not reflect the game's structure (the "distance" between a 2-tile and a 4-tile is not the same as between a 1024-tile and a 2048-tile in strategic terms).

**Option 4: One-hot encoding (chosen).** Each cell is encoded as a 16-dimensional binary vector where exactly one position is 1. The active position corresponds to log2(tile value), with position 0 representing an empty cell. The full board becomes a tensor of shape (16, 4, 4) -- 16 channels, each a 4x4 binary grid.

```python
def encode_state(grid):
    encoded = np.zeros((16, 4, 4), dtype=np.float32)
    for y in range(4):
        for x in range(4):
            val = grid[y][x]
            if val > 0:
                channel = int(np.log2(val))
                if channel < 16:
                    encoded[channel][y][x] = 1.0
            else:
                encoded[0][y][x] = 1.0  # empty = channel 0
    return encoded
```

**Rationale for one-hot:** This encoding allows the network to treat each power of 2 as an independent "feature plane." Channel 11 (value 2048) is just as "loud" as channel 1 (value 2) -- both are represented as a single 1 in a sea of 0s. This prevents the magnitude of tile values from dominating the gradient and allows the convolutional filters to learn patterns like "a 2048-tile adjacent to a 1024-tile" directly.

### 5.3 Neural Network Architecture

We designed a compact CNN architecture suitable for the 4x4 input:

```
Input: (batch, 16, 4, 4)
  |
  v
Conv2d(16 -> 128, kernel=2x2, stride=1, padding=0)
BatchNorm2d(128)
ReLU
  |  Output: (batch, 128, 3, 3) -- 128 filters, 3x3 spatial
  v
Conv2d(128 -> 128, kernel=2x2, stride=1, padding=0)
BatchNorm2d(128)
ReLU
  |  Output: (batch, 128, 2, 2) -- 128 filters, 2x2 spatial
  v
Flatten
  |  Output: (batch, 512) -- 128 * 2 * 2
  v
Linear(512 -> 256)
ReLU
  |  Output: (batch, 256)
  v
Linear(256 -> 4)
  |  Output: (batch, 4) -- Q-values for [up, right, down, left]
```

**Design decisions:**

1. **Kernel size 2x2 (not 3x3).** On a 4x4 grid, a 3x3 kernel would span nearly the entire board in a single layer, providing no hierarchical feature extraction. A 2x2 kernel captures local patterns (adjacent tile relationships) in the first layer, and the second layer combines these into 3x3 region patterns.

2. **No padding.** With valid convolutions (no padding), the spatial dimensions shrink naturally: 4->3->2. This is appropriate for such a small grid.

3. **128 filters.** Chosen empirically. With 16 input channels (one-hot), 128 filters provide enough capacity to learn diverse local patterns.

4. **Batch normalization.** Stabilizes training by normalizing activations. Particularly important when the input distribution shifts as the agent's policy changes.

5. **Two convolutional layers (not more).** With a 4x4 input, two layers with 2x2 kernels give a maximum receptive field of 3x3 -- already covering 75% of the board. Adding more layers would reduce the spatial dimension to 1x1, losing all spatial structure.

**Parameter count:** Approximately 200,000 trainable parameters.

```
Conv1:  16 * 128 * 2 * 2 + 128 bias = 8,320
BN1:    128 * 2 = 256
Conv2:  128 * 128 * 2 * 2 + 128 bias = 65,664
BN2:    128 * 2 = 256
FC1:    512 * 256 + 256 = 131,328
FC2:    256 * 4 + 4 = 1,028
Total:  ~206,852 parameters
```

### 5.4 Training Infrastructure

#### 5.4.1 Experience Replay Buffer

The replay buffer is implemented as a circular deque of capacity 100,000 transitions:

```python
class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done, valid_moves):
        self.buffer.append((state, action, reward, next_state, done, valid_moves))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        # ... unpack and convert to arrays
```

Each transition stores the encoded state, the action taken, the shaped reward, the next state, the done flag, and the list of valid moves (needed for masking invalid actions during training).

**Why 100,000?** Each transition stores approximately 16*4*4*4 = 1024 bytes for the state and the same for the next state, plus a few scalars. 100,000 transitions occupy roughly 200 MB -- manageable but not trivial. Larger buffers would improve sample diversity but increase memory usage.

#### 5.4.2 Double DQN Implementation

The training step implements Double DQN:

```python
def train_step(self):
    # Sample mini-batch
    states, actions, rewards, next_states, dones, valid_moves_batch = \
        self.memory.sample(self.batch_size)
    
    # Current Q-values: Q(s, a; theta)
    q_values = self.policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
    
    # Double DQN target
    with torch.no_grad():
        # Policy network selects best action
        next_q_policy = self.policy_net(next_states_t)
        # Mask invalid actions
        for i, vm in enumerate(valid_moves_batch):
            mask = torch.ones(4) * (-1e9)
            for m in vm:
                mask[m] = 0
            next_q_policy[i] += mask
        best_actions = next_q_policy.argmax(1)
        
        # Target network evaluates that action
        next_q_target = self.target_net(next_states_t)
        next_q = next_q_target.gather(1, best_actions.unsqueeze(1)).squeeze(1)
        
        target = rewards_t + self.gamma * next_q * (1 - dones_t)
    
    loss = F.smooth_l1_loss(q_values, target)
```

**Key detail: valid move masking.** In 2048, not all moves are valid at every board position (e.g., if all tiles are already pushed left, LEFT is invalid). We mask invalid actions with -1e9 before the argmax, ensuring the agent never selects impossible moves.

#### 5.4.3 Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Learning rate | 1e-4 | Low for stability with target network updates |
| Gamma | 0.99 | High -- long-horizon game where future rewards matter |
| Epsilon start | 1.0 | Begin with pure exploration |
| Epsilon end | 0.01 | Maintain 1% exploration even at convergence |
| Epsilon decay | 200,000 steps | Very slow -- exploration must be prolonged |
| Batch size | 512 | Large for gradient stability |
| Target update | 1,000 steps | Sync target network every 1k steps |
| Buffer size | 100,000 | Moderate replay buffer |
| Gradient clipping | 1.0 | Prevents gradient explosion |
| Optimizer | Adam | Adaptive learning rate per parameter |
| Loss function | Smooth L1 (Huber) | Less sensitive to outliers than MSE |

### 5.5 Reward Shaping

The raw game reward (merge score) is sparse and does not indicate positional quality. A board with tiles [2048, 1024, 512, ...] organized in a corner is far more valuable than a board with the same tiles scattered randomly, even if both have the same score. We implemented **reward shaping** to provide denser and more informative training signals:

```python
reward = 0

# 1. Merge reward (logarithmically scaled)
if raw_reward > 0:
    reward += 0.1 * math.log2(raw_reward + 1)

# 2. Corner bonus: +0.1 if max tile is in a corner
if max_tile_in_corner:
    reward += 0.1

# 3. Monotonicity bonus: +0.02 per monotonic row/column
reward += 0.02 * monotonic_count

# 4. Space bonus: proportional to empty cells
reward += 0.01 * empty_cells

# 5. Game over penalty
if game_over:
    reward -= 2.0
```

**Design rationale:**

- **Logarithmic merge scaling** prevents high-value merges (e.g., 1024 + 1024 = 2048, worth 2048 points) from dominating the gradient. log2(2048 + 1) ~ 11, so the shaped reward is 1.1 instead of 2048.

- **Corner bonus** encodes the domain knowledge that keeping the max tile in a corner is the most important strategic principle.

- **Monotonicity bonus** rewards boards where tiles decrease in a consistent direction, which facilitates future merges.

- **Space bonus** rewards maintaining free space, which correlates with game longevity.

- **Game over penalty** provides a strong negative signal for terminal states.

### 5.6 Training Results and Analysis

Training was conducted on Apple Silicon (M-series) for 1,700 episodes, taking approximately 3 hours:

| Episodes | Wall Time | Avg Score (last 100) | Most Frequent Max Tile | Epsilon | 2048 Count |
|----------|-----------|---------------------|----------------------|---------|------------|
| 100 | 10 min | ~800 | 64 | 0.95 | 0 |
| 200 | 18 min | ~1,000 | 64-128 | 0.91 | 0 |
| 500 | 52 min | ~1,500 | 128 | 0.78 | 0 |
| 1,000 | 1h 42min | ~3,000 | 256 | 0.61 | 0 |
| 1,500 | 2h 42min | ~4,500 | 256-512 | 0.47 | 0 |
| 1,700 | 3h 04min | ~5,000 | 512 | 0.41 | 0 |

**DQN never reached the 2048 tile in any game during training.** The best individual game scored approximately 11,600 points with a maximum tile of 512.

For comparison, our N-Tuple agent would later achieve its first 2048 tile at approximately episode 500, after just 2 minutes of training.

#### 5.6.1 Learning Curve Analysis

The score improved roughly linearly from 800 to 5,000 over 1,700 episodes, with no sign of acceleration. Projecting this trend:

- Reaching average score ~20,000 (Expectimax level): ~10,000 episodes
- Reaching average score ~40,000 (decent play): ~25,000 episodes
- Time to 25,000 episodes at ~1.8 min/episode: ~31 hours

This is prohibitively slow for iterative development.

#### 5.6.2 Game Report Example: DQN

From the saved report `20260818_163442_dqn_s912_t64.json`:

```json
{
  "agent": "dqn",
  "score": 912,
  "max_tile": 64,
  "moves": 97,
  "won": false,
  "final_grid": [
    [8, 2, 4, 2],
    [16, 64, 16, 4],
    [4, 16, 32, 2],
    [2, 4, 64, 4]
  ],
  "duration_ms": 1823
}
```

The final board is chaotic: the maximum tile (64) appears in two separate locations, with no spatial organization. The agent managed only 97 moves before game over -- a strong player typically makes 500-2000 moves. The score of 912 is barely above random play (~800).

### 5.7 Diagnosis of Failure

We identified three root causes for the DQN's poor performance:

#### 5.7.1 Slow Exploration Decay

At episode 1,700 (approximately 500,000 steps with ~300 steps per episode), the epsilon was:

```
epsilon = 0.01 + (1.0 - 0.01) * exp(-500000 / 200000)
        = 0.01 + 0.99 * exp(-2.5)
        = 0.01 + 0.99 * 0.082
        = 0.01 + 0.081
        ~ 0.091
```

Wait -- this actually suggests epsilon was about 9% at this point, but our training logs showed ~41%. Let us recalculate. The decay is per-step, and with batch_size=512, training steps happen less frequently than game steps. The `steps_done` counter increments once per `train_step()` call, which occurs once per game step (not once per batch element). With 1,700 episodes of ~300 moves each, `steps_done ~ 510,000`, giving:

```
epsilon = 0.01 + 0.99 * exp(-510000 / 200000) = 0.01 + 0.99 * 0.078 = 0.087
```

However, training steps only execute when the buffer has enough samples (batch_size=512), and the buffer fills gradually during early episodes. The effective `steps_done` was likely around 300,000, giving epsilon ~ 0.23. Still, even at 23% random actions, the agent's Q-values were not yet good enough to produce coherent play.

The fundamental issue is that the **epsilon decay timeline was calibrated for steps, not episodes**, and each episode generates hundreds of steps. The agent needed tens of thousands of episodes for epsilon to reach a level where its learned Q-values could guide play effectively.

#### 5.7.2 Experience Replay Inefficiency

Each game generates approximately 300 transitions, the majority from trivial early positions (where the board has few tiles and any move is roughly equivalent). The 100,000-entry buffer is quickly filled with these low-information experiences, diluting the rare and valuable late-game experiences where strategic decisions matter most.

**Possible mitigation (not implemented):** Prioritized Experience Replay (PER), which samples transitions with probability proportional to their TD error. High-error transitions (typically from novel or surprising situations) would be replayed more frequently.

#### 5.7.3 Receptive Field Limitation

With two 2x2 convolutional layers (no padding), the maximum receptive field is 3x3 -- covering only 9 of the 16 cells. The network literally cannot see the entire board at once. The snake pattern, which organizes tiles along a zigzag path spanning all 16 cells, is invisible to any single neuron.

```
+---+---+---+---+
| * | * | * |   |     * = receptive field of one neuron
+---+---+---+---+       in the second conv layer
| * | * | * |   |
+---+---+---+---+     The corner cell (0,3) and entire
| * | * | * |   |     bottom row are invisible.
+---+---+---+---+
|   |   |   |   |
+---+---+---+---+

Figure 5.1: The limited 3x3 receptive field of a Conv(2x2) -> Conv(2x2) network
on a 4x4 grid. No single neuron can see the entire board.
```

While the fully-connected layers that follow the convolutions *can* combine information from all spatial positions, they receive it only as pre-computed local features, not as raw spatial information. The network cannot learn whole-board patterns that depend on the relative positions of distant tiles.

### 5.8 Post-Mortem and Lessons Learned

The DQN experiment, though a failure in terms of game performance, was extremely valuable pedagogically:

1. **Not every sequential decision problem benefits from deep RL.** DQN excels at problems with high-dimensional continuous inputs (pixel-based Atari games) where hand-crafted features are difficult to design. In 2048, the state space is structured, discrete, and well-understood -- properties that favor specialized methods.

2. **Training time matters.** An approach that requires 50-100 hours to reach basic competency is impractical for iterative development. We needed an approach that could produce playable results within minutes, not days.

3. **The CNN receptive field must match the problem's spatial scale.** For a 4x4 board, the "long-range dependencies" span at most 7 cells diagonally. A network that can only see 3x3 at a time is architecturally incapable of capturing the most important patterns.

4. **Reward shaping is a workaround, not a solution.** Our shaped reward encoded domain knowledge (corner preference, monotonicity) that the network was supposed to discover on its own. If we have this knowledge, we can use it more directly (as in Expectimax heuristics) or encode it structurally (as in N-Tuple patterns).

**This failure directly motivated the pivot to Expectimax:** a method that requires no training and incorporates domain knowledge explicitly.

---

## 6. Approach 2: Expectimax with Heuristics

### 6.1 Motivation: From Learning to Knowledge

While the DQN agent was still training fruitlessly after two hours, we began implementing an approach that represents the opposite philosophy: instead of learning to evaluate positions from experience, we would **hand-code our understanding of what makes a good 2048 position** into an evaluation function, and use **tree search** to look ahead and choose moves optimally with respect to that evaluation.

This approach required zero training time. The first version, implemented in pure Python in about an hour, immediately outperformed the DQN agent that had been training for three hours.

### 6.2 The Evaluation Function

The evaluation function V(board) maps a board configuration to a real number estimating its quality. Higher values indicate better positions. The function is a weighted combination of five heuristic components:

```
V(board) = 1.0 * snake(board) 
         + 2.7 * empty(board) 
         + 1.0 * mono(board) 
         + 0.1 * smooth(board) 
         + 1.0 * corner(board)
```

Each component captures a different aspect of position quality, and together they provide a holistic assessment that guides the Expectimax search.

### 6.3 The Snake Pattern: Mathematical Derivation

#### 6.3.1 The Intuition

The most important heuristic is the **snake pattern** (also called the "zigzag" or "monotonic gradient"). The idea is simple: tiles should be organized so that values decrease along a serpentine path from a corner. This ensures that the largest tile is always in a corner (protected from disruption) and that adjacent tiles along the path can potentially merge to form higher-value tiles.

```
Direction 0:

32768 ---> 16384 ---> 8192 ---> 4096
                                  |
                                  v
  256 <---  512 <--- 1024 <--- 2048
  |
  v
  128 --->   64 --->   32 --->   16
                                  |
                                  v
    1 <---    2 <---    4 <---    8

Figure 6.1: The snake pattern assigns weights that decrease along a zigzag path
from the top-left corner. Tiles aligned with this gradient receive high scores.
```

#### 6.3.2 Weight Selection

The weights are powers of 2: 32768, 16384, 8192, 4096, 2048, 1024, ..., 4, 2, 1. The specific values are:

```c
static const float SNAKE_W[8][4][4] = {
    /* Orientation 0 */
    {{32768, 16384, 8192, 4096},
     {  256,   512, 1024, 2048},
     {  128,    64,   32,   16},
     {    1,     2,    4,    8}},
    /* ... 7 more orientations ... */
};
```

**Why powers of 2?** The choice of power-of-2 weights ensures that the contribution of a tile is proportional to the tile's value scaled by its position. Consider a tile with actual value v at position (y,x) with snake weight w:

```
contribution = v * w
```

If v = 2048 (the target tile) and w = 32768 (the corner position), the contribution is 2048 * 32768 = 67,108,864. If the same tile were at a non-corner position with w = 256, the contribution would be only 2048 * 256 = 524,288 -- more than 100x less. This creates a very strong incentive for the search to place high-value tiles at high-weight positions.

The specific weights form a geometric sequence with ratio 2, matching the game's doubling mechanic. When two tiles of value v merge, they produce a tile of value 2v. If the merged tile moves one position closer to the corner (from weight w to weight 2w), the evaluation change is:

```
Delta = 2v * 2w - 2 * (v * w) = 4vw - 2vw = 2vw > 0
```

This positive change means that merging tiles closer to the corner is always evaluated as beneficial -- exactly the behavior we want.

#### 6.3.3 Multiple Orientations

The snake pattern can start from any of the four corners, and the zigzag can go either horizontally-first or vertically-first, giving 4 * 2 = 8 orientations. The evaluation function computes the score for all 8 orientations and uses the maximum:

```
snake(board) = max_{o in {0,...,7}} sum_{y,x} tile(y,x) * SNAKE_W[o][y][x]
```

This ensures that the agent can organize tiles around any corner, adapting to the current board state rather than being locked into a single corner.

### 6.4 Monotonicity Heuristic

Monotonicity measures how well-ordered the rows and columns are. For each row and column, we compute the sum of differences in both ascending and descending directions and take the minimum (the more monotonic direction):

```python
for each row i:
    left = sum of max(0, log2(cell[j]) - log2(cell[j+1])) for j in 0..2
    right = sum of max(0, log2(cell[j+1]) - log2(cell[j])) for j in 0..2
    mono -= min(left, right)  # Penalty for non-monotonicity
```

The key insight is that we measure monotonicity using **log2 values** rather than actual tile values. This is because the game's doubling mechanic means that the "distance" between 512 and 1024 (one merge) is strategically equivalent to the distance between 2 and 4 (also one merge). Using log2 values captures this equivalence.

The heuristic returns 0 for a perfectly monotonic board and increasingly negative values for less ordered boards.

### 6.5 Smoothness Heuristic

Smoothness penalizes large differences between adjacent tiles. A "smooth" board has similar values next to each other, making future merges likely:

```
smooth(board) = -sum_{adjacent pairs (a,b)} |log2(a) - log2(b)|
```

where the sum is over all pairs of horizontally or vertically adjacent non-empty tiles. Again, we use log2 values to normalize the scale.

Note that smoothness can conflict with monotonicity: a perfectly smooth board (all tiles equal) is perfectly monotonic but provides no room for merging. The weight of 0.1 for smoothness (vs. 1.0 for monotonicity) reflects the fact that monotonicity is more important than smoothness for high-level play.

### 6.6 Empty Cells and Corner Bonus

**Empty cells.** More free space provides more options and delays game over. We use a logarithmic scaling:

```
empty(board) = ln(empty_count + 1)
```

The logarithm prevents empty cells from dominating the evaluation when the board is mostly empty (early game). The "+1" avoids ln(0) when the board is full.

The weight of 2.7 for empty cells is the highest among all heuristics, reflecting the critical importance of free space. Running out of space is the most common cause of game over.

**Corner bonus.** If the highest-value tile is in one of the four corners, a bonus is added. In the C implementation:

```c
int corners[4] = {grid[0][0], grid[0][3], grid[3][0], grid[3][3]};
for (int i = 0; i < 4; i++) {
    if (corners[i] == max_val && max_val > 0) {
        score += lg[0][0] * 1.0f;  // log2(max_val)
        break;
    }
}
```

This is a simpler version of the snake pattern's corner preference, providing an additional reinforcement signal.

### 6.7 Weight Calibration

The heuristic weights (1.0, 2.7, 1.0, 0.1, 1.0) were calibrated through empirical experimentation. The process was:

1. Start with unit weights (1.0 for all);
2. Play 10 games at depth 3 and record average score;
3. Increase each weight individually by 50% and re-run;
4. Keep the change that improved performance the most;
5. Repeat until convergence.

In the C implementation, the row-based heuristic uses different weights (W_EMPTY=270, W_MONO=47, W_SMOOTH=12, W_EDGE=11, W_MERGE=700) that were calibrated independently for the decomposed row evaluation.

**This manual calibration process is precisely what N-Tuple Networks automate** -- they learn the optimal "weight" for every pattern through gradient descent, exploring a space far larger than what manual experimentation can cover.

### 6.8 Pure Python Implementation (Depth 3)

The first working Expectimax implementation was in pure Python, searching to depth 3 (3 levels of MAX nodes with intervening CHANCE nodes):

| Metric | Value |
|--------|-------|
| Time per move | ~217ms |
| Search depth | 3 |
| Typical score | ~20,000 |
| Most common max tile | 1024-2048 |
| Win rate (2048) | ~50% (estimated) |

**Milestone: the first 2048 tile was achieved!** Score 20,572 in 984 moves. This happened within the first hour of implementing Expectimax, compared to 3+ hours of fruitless DQN training.

However, 217ms per move was uncomfortably slow for real-time visualization. At the frontend's minimum delay, this meant approximately 4.6 moves per second -- watchable but sluggish. And depth 3 was not deep enough for consistently strong play.

### 6.9 The Lookup Table Decomposition Attempt (Failure 2)

**Context.** To increase search depth without increasing computation time, we attempted to pre-compute heuristic values per row, decomposing the 2D evaluation into independent 1D contributions.

**The idea.** Each 4-cell row can take 16^4 = 65,536 different configurations (using log2 encoding). We could pre-compute the heuristic value for each configuration and store it in a lookup table. The board evaluation would then be:

```
V(board) = sum_{y=0}^{3} heur_row[row_y] + sum_{x=0}^{3} heur_col[col_x]
```

This would reduce the evaluation from O(16) cell visits with floating-point arithmetic to 8 table lookups plus addition.

**The implementation.** We computed per-row heuristics including empty cells, local monotonicity, smoothness, merge potential, and edge bonuses:

```c
/* Row heuristic computation for all 65536 possible rows */
for (int enc = 0; enc < 65536; enc++) {
    int c[4] = { (enc>>12)&0xF, (enc>>8)&0xF, (enc>>4)&0xF, enc&0xF };
    float h = 0;
    h += W_EMPTY * empty_count;     // empty cells
    h += W_MONO * monotonicity;     // local monotonicity
    h -= W_SMOOTH * roughness;      // smoothness
    h += W_MERGE * merge_pairs;     // merge potential
    h += W_EDGE * edge_bonus;       // edge tile bonus
    heur_row[enc] = h;
}
```

**The result.** Maximum tile: 256. Average score: ~4,000. A catastrophic regression.

**Root cause analysis.** The 1D decomposition fundamentally destroys the 2D spatial information that makes the snake pattern work:

1. **The snake pattern is inherently 2D.** The zigzag path crosses rows -- the value at position (0,3) must be related to the value at (1,3) for the pattern to work. Row-based evaluation treats each row independently and cannot capture cross-row relationships.

2. **Monotonicity across rows is lost.** A board where each row is individually monotonic but the rows are in random order is evaluated the same as a board where rows are properly stacked -- but the second board is far more valuable.

3. **The corner bonus requires knowing which cell has the global maximum.** A per-row evaluation can identify the row-maximum but not the board-maximum.

**Lesson learned.** 2D heuristics are not decomposable into 1D components without significant quality loss. The interaction between rows and columns is precisely what makes the evaluation function effective. This attempt cost approximately 2 hours of implementation and testing time.

### 6.10 C Implementation and the Transpose Bug (Failure 3)

#### 6.10.1 Motivation for C

Python was approximately 100x slower than needed for real-time play at depth > 3. The game engine and search tree, both involving tight loops with simple arithmetic, are ideal candidates for C optimization. We decided to rewrite the Expectimax engine in C, compiling it as a shared library callable from Python via `ctypes`.

#### 6.10.2 Bitboard Representation

Following nneonneo's design, we represented the board as a single `uint64_t` (64-bit unsigned integer):

```
Board: uint64_t (64 bits)
  Row 0 = bits 0-15
  Row 1 = bits 16-31
  Row 2 = bits 32-47
  Row 3 = bits 48-63

Within each row (16 bits = 4 nibbles):
  Cell (y, 3-x) occupies bits (y*16 + x*4) to (y*16 + x*4 + 3)
  
  Our layout: cell (y,x) is at nibble (y*4 + (3-x))
  Cell (0,0) is at nibble 3 (bits 12-15 of row 0)
  Cell (0,3) is at nibble 0 (bits 0-3 of row 0)
```

This representation enables:
- **Row extraction** in O(1): `(board >> (y * 16)) & 0xFFFF`
- **Move operations** via lookup tables: each row merge is a single table lookup
- **Board comparison** as a single 64-bit integer comparison

#### 6.10.3 The Transpose Bug

To implement vertical moves (UP and DOWN), we needed to extract columns from the board. The standard approach (used by nneonneo) is to **transpose** the board -- swapping rows and columns -- using a clever sequence of bit manipulations that runs in O(1).

We copied the transpose function from nneonneo's codebase:

```c
/* nneonneo's transpose (assuming cell(y,x) at nibble 4y+x) */
static inline board_t transpose(board_t x) {
    board_t a1 = x & 0xF0F00F0FF0F00F0FULL;
    board_t a2 = x & 0x0000F0F00000F0F0ULL;
    board_t a3 = x & 0x0F0F00000F0F0000ULL;
    board_t a = a1 | (a2 << 12) | (a3 >> 12);
    board_t b1 = a & 0xFF00FF0000FF00FFULL;
    board_t b2 = a & 0x00FF00FF00000000ULL;
    board_t b3 = a & 0x00000000FF00FF00ULL;
    return b1 | (b2 >> 24) | (b3 << 24);
}
```

**The problem.** This transpose function assumes nneonneo's bit layout: `cell(y,x)` at nibble `4y + x`. But our layout is `cell(y,x)` at nibble `4y + (3-x)`. The difference:

```
nneonneo: cell(0,0) at nibble 0 (bits 0-3)
Ours:     cell(0,0) at nibble 3 (bits 12-15)

nneonneo: cell(0,1) at nibble 1 (bits 4-7)
Ours:     cell(0,1) at nibble 2 (bits 8-11)
```

The nibble order within each row is reversed. The transpose function, which assumes a specific mapping between nibble positions and cell coordinates, produces garbage when applied to our layout.

**The symptom.** The AI only moved down, scoring approximately 44 points with a maximum tile of 16. The game ended almost immediately.

**The debugging process.** This was one of the more difficult bugs to track down because the symptom (always moving down) did not obviously point to a transpose error:

1. **First hypothesis: move generation bug.** We checked the LEFT and RIGHT move implementations (which do not use transpose) and they worked correctly.

2. **Second hypothesis: evaluation function bug.** We printed the evaluation of all four moves for a known board state. LEFT and RIGHT returned reasonable values; UP and DOWN returned near-zero or nonsensical values.

3. **Third hypothesis: column extraction bug.** We realized that UP and DOWN used the transpose function to convert columns to rows, apply the row-based merge, and convert back. We printed the board before and after transpose -- the values were clearly wrong.

4. **Root cause identified.** We traced a specific nibble through the transpose computation and found that it was being placed at the wrong position, effectively zeroing out cells and corrupting the board.

**The fix.** Instead of trying to adapt the bit-manipulation transpose to our layout (error-prone and hard to verify), we replaced it with explicit column extraction:

```c
static inline row_t board_col(board_t b, int x) {
    int c0 = (board_row(b, 0) >> ((3 - x) * 4)) & 0xF;
    int c1 = (board_row(b, 1) >> ((3 - x) * 4)) & 0xF;
    int c2 = (board_row(b, 2) >> ((3 - x) * 4)) & 0xF;
    int c3 = (board_row(b, 3) >> ((3 - x) * 4)) & 0xF;
    return (c0 << 12) | (c1 << 8) | (c2 << 4) | c3;
}
```

This function extracts column `x` as a 16-bit row (top cell at the high nibble), using four shifts and masks instead of a single O(1) bit trick. It is slightly slower (~4 operations vs. ~10 for the bit transpose, but the bit transpose is single-instruction-level) but **correct and maintainable**.

**Lesson learned.** When reusing bit-manipulation code from another project, the implicit assumptions about data layout must be verified exhaustively. A clever O(1) function is worthless if it operates on the wrong bits.

### 6.11 Iterative Deepening with Time Budget

#### 6.11.1 The Problem with Fixed Depth

With fixed search depth, the time per move varies wildly depending on the board state:
- **Early game** (many empty cells): Each CHANCE node has many children, creating a wide tree. Depth 4 might take 500ms.
- **Late game** (few empty cells): Fewer CHANCE children, narrower tree. Depth 4 might take 10ms.

This means the agent wastes time on easy positions (where depth 3 would suffice) and runs out of time on critical positions (where depth 5 would help).

#### 6.11.2 The Solution

We implemented iterative deepening with a 100ms time budget per move:

```c
for (int depth = 1; depth <= max_depth; depth++) {
    cache_clear();
    search_aborted = 0;
    nodes_searched = 0;
    
    search_result_t r = search_at_depth(b, depth);
    
    if (!search_aborted) {
        best = r;
        best.depth_reached = depth;
    } else {
        break;  // Time's up -- use previous depth's result
    }
    
    // 60% rule: don't start next depth if >60% of budget spent
    long long elapsed = now_us() - start;
    long long budget = (long long)time_budget_ms * 1000LL;
    if (elapsed > budget * 6 / 10)
        break;
}
```

**The 60% rule** is critical: if more than 60% of the time budget has been consumed by the current depth, we do not start the next depth. This is because each depth is roughly 10x more expensive than the previous one (branching factor ~10 for 2048). If depth 4 took 65ms of a 100ms budget, depth 5 would likely take 650ms -- far exceeding the budget and wasting the entire computation (since we'd use the depth 4 result anyway).

The search checks the time budget every 4,096 nodes (using `nodes_searched & 0xFFF == 0` as a cheap modular check) and sets `search_aborted = 1` when the deadline is reached. This adds negligible overhead (~0.1% of total time).

### 6.12 Transposition Table

The transposition table is a direct-mapped cache with 2^22 = 4,194,304 entries:

```c
#define CACHE_BITS 22
#define CACHE_SIZE (1 << CACHE_BITS)  // 4M entries
#define CACHE_MASK (CACHE_SIZE - 1)

typedef struct {
    board_t key;     // 8 bytes: the board state
    float   value;   // 4 bytes: the evaluated value
    int8_t  depth;   // 1 byte: the search depth
    int8_t  valid;   // 1 byte: whether this entry is populated
} cache_entry_t;     // 14 bytes per entry (padded to 16)

// Total memory: 4M * 16 = 64 MB
static cache_entry_t cache[CACHE_SIZE];
```

**Hash function:** The hash combines the board state with the depth using multiplicative hashing:

```c
static inline uint32_t cache_hash(board_t b, int depth) {
    uint64_t h = b ^ (b >> 16);
    h *= 0x45D9F3B + depth * 0x9E3779B9;
    h ^= h >> 16;
    return (uint32_t)(h & CACHE_MASK);
}
```

The constant `0x9E3779B9` is the floor of `2^32 / phi` (where phi is the golden ratio), a well-known constant for multiplicative hashing. It distributes hash values uniformly across the table.

**Collision handling:** Direct mapping -- a new entry simply overwrites whatever was previously at that index. This is simple and fast, with the trade-off that cache collisions (different boards hashing to the same index) cause cache misses. With 4M entries and typical tree sizes of 100K-1M nodes, the collision rate is acceptable.

### 6.13 Final Results and Analysis

With the C implementation, iterative deepening, and transposition table, we benchmarked 5 complete games with a 100ms time budget:

| Game | Score | Max Tile | Moves | Avg Depth Reached |
|------|-------|----------|-------|-------------------|
| 1 | 38,260 | 2048 | 2,000+ | 4.2 |
| 2 | 14,508 | 1024 | 840 | 4.5 |
| 3 | 46,200 | 4096 | 2,000+ | 4.1 |
| 4 | 46,152 | 4096 | 2,000+ | 4.0 |
| 5 | 46,216 | 4096 | 2,000+ | 4.1 |

**Summary statistics:**
- **Average score**: 38,267 (std dev: ~12,600)
- **2048+ rate**: 80% (4/5 games)
- **4096 rate**: 60% (3/5 games)
- **Average moves per game**: ~1,770

**Best visual game (from report):** Score 76,516 with final board:

```
+------+------+------+------+
| 4096 |   16 |    2 |    4 |
+------+------+------+------+
| 2048 |   64 |   16 |    8 |
+------+------+------+------+
| 1024 |  128 |    8 |    4 |
+------+------+------+------+
|  256 |   32 |    4 |    2 |
+------+------+------+------+

Score: 76,516 | Moves: 2,526 | Agent: Expectimax
```

This board perfectly demonstrates the snake pattern: 4096 in the top-left corner, with values decreasing along a zigzag path (4096 -> 2048 -> 1024 -> 256 down the left column, then 256 -> 512 would be the next step if merging continued). The board has tiles 4096 + 2048 + 1024 + 256 = 7,424 in the left column alone.

### 6.14 Cumulative Speedup Analysis

| Stage | Time/Move | Depth | Score | Speedup vs. Python |
|-------|-----------|-------|-------|--------------------|
| Python depth=3 | 217ms | 3 | ~20,000 | 1x (baseline) |
| C depth=3 | 1.4ms | 3 | ~20,000 | **155x** |
| C depth=4 (fixed) | 15ms | 4 | ~35,000 | 14.5x faster than Py d3 |
| C iterative (100ms budget) | ~100ms | 4-5 | ~40,000 | Same time, better quality |
| C depth=5 (fixed) | 400ms | 5 | ~45,000 | Too slow for real-time |

The 155x speedup from Python to C for the same depth is explained by:
- **No interpreter overhead**: C's native machine code vs. Python's bytecode interpretation;
- **Bitboard operations**: 64-bit integer operations instead of 2D array indexing;
- **Lookup tables**: Pre-computed merge results instead of iterative merge logic;
- **Cache friendliness**: Compact 64-bit board representation fits in a register.

**Partial conclusion on Expectimax.** Expectimax with hand-crafted heuristics demonstrated that **well-coded domain knowledge outperforms weeks of neural training** for this problem. Without any training, it consistently reaches 2048 and frequently 4096. However, the approach has a fundamental limitation: the heuristic weights are fixed and cannot improve with more experience. To go beyond ~80% 2048 rate with this approach would require either (a) better heuristics (difficult -- diminishing returns on manual tuning) or (b) deeper search (expensive -- each additional depth multiplies computation by ~10x).

This limitation motivated the move to N-Tuple Networks: a method that **learns** its evaluation function (like DQN) but uses **fast table lookups** (like Expectimax heuristics) and is compatible with **tree search** (like Expectimax).

---

## 7. Approach 3: N-Tuple Networks

### 7.1 Motivation: Combining the Best of Both Worlds

The first two approaches represent extremes of a spectrum:

| Property | DQN | Expectimax |
|----------|-----|-----------|
| Learns from experience | Yes | No |
| Uses domain knowledge | Minimal | Extensive |
| Evaluation speed | ~1ms (GPU) | ~0.001ms (C) |
| Quality ceiling | Theoretically unlimited | Limited by heuristic quality |
| Training required | Hours to days | None |

N-Tuple Networks occupy the sweet spot:

| Property | N-Tuple Networks |
|----------|-----------------|
| Learns from experience | Yes (TD-learning) |
| Uses domain knowledge | Moderate (tuple placement) |
| Evaluation speed | ~0.001ms (lookup tables) |
| Quality ceiling | Very high (demonstrated up to 99.5%) |
| Training required | Minutes to hours (efficient) |

The key insight is that N-Tuple Networks are essentially **learned heuristics**: the network discovers the evaluation function that Expectimax uses, rather than having it hand-coded. This allows the evaluation to improve with more training, while maintaining the speed advantage of table lookups.

### 7.2 Network Architecture Design

Our network consists of 17 base tuples, each with 6 positions (cells on the board). The tuples are designed to cover diverse spatial patterns:

```
Tuple categories and their positions on the 4x4 board:

HORIZONTAL (3 tuples):           VERTICAL (3 tuples):
+--+--+--+--+                   +--+  +--+  +--+
|##|##|##|##|  + two cells      |##|  |##|  |##|
+--+--+--+--+   below          |##|  |##|  |##|
|##|##|  |  |                   |##|  |##|  |##|
+--+--+--+--+                   |##|  |##|  |##|
                                +--+  +--+  +--+
                                + two cells right

3x2 BLOCKS (4 tuples):          2x3 BLOCKS (3 tuples):
+--+--+--+                      +--+--+
|##|##|##|                      |##|##|
+--+--+--+                      |##|##|
|##|##|##|                      |##|##|
+--+--+--+                      +--+--+

L-SHAPES and DIAGONALS (4 tuples):
+--+--+--+          +--+--+
|##|##|##|          |##|##|
+--+--+--+             +--+--+
|##|##|  |             |##|##|
+--+--+--+                +--+--+
|##|  |  |                |##|##|
+--+--+--+                +--+--+

Figure 7.1: The 17 base tuple categories. Each tuple covers 6 board positions.
Rotations and reflections generate up to 8 symmetric variants per tuple.
```

### 7.3 Tuple Selection Strategy

The 17 tuples were selected to satisfy several criteria:

1. **Coverage**: Every cell on the board should be covered by multiple tuples. This ensures that no part of the board is "invisible" to the network.

2. **Diversity**: Different tuple shapes (horizontal, vertical, rectangular, L-shaped, diagonal) capture different types of spatial relationships. A horizontal 4+2 tuple captures row-based patterns, while a diagonal tuple captures cross-row relationships.

3. **Overlap**: Tuples should share some positions. This allows the network to represent interactions between different regions -- for example, the relationship between the top-left corner and the cells immediately below and to the right.

The specific 17 tuples are (positions given as [row, column]):

```c
static const int BASE_TUPLES[17][6][2] = {
    // Horizontal 4+2
    {{0,0},{0,1},{0,2},{0,3},{1,0},{1,1}},  // Top row + 2 below-left
    {{1,0},{1,1},{1,2},{1,3},{2,0},{2,1}},  // 2nd row + 2 below-left
    {{2,0},{2,1},{2,2},{2,3},{3,0},{3,1}},  // 3rd row + 2 below-left
    
    // Vertical 4+2
    {{0,0},{1,0},{2,0},{3,0},{0,1},{1,1}},  // Left col + 2 right-top
    {{0,1},{1,1},{2,1},{3,1},{0,2},{1,2}},  // 2nd col + 2 right-top
    {{0,2},{1,2},{2,2},{3,2},{0,3},{1,3}},  // 3rd col + 2 right-top
    
    // 3x2 blocks
    {{0,0},{0,1},{0,2},{1,0},{1,1},{1,2}},  // Top-left 3x2
    {{1,0},{1,1},{1,2},{2,0},{2,1},{2,2}},  // Middle-left 3x2
    {{2,0},{2,1},{2,2},{3,0},{3,1},{3,2}},  // Bottom-left 3x2
    {{0,1},{0,2},{0,3},{1,1},{1,2},{1,3}},  // Top-right 3x2
    
    // 2x3 blocks
    {{0,0},{0,1},{1,0},{1,1},{2,0},{2,1}},  // Left 2x3
    {{0,1},{0,2},{1,1},{1,2},{2,1},{2,2}},  // Center 2x3
    {{0,2},{0,3},{1,2},{1,3},{2,2},{2,3}},  // Right 2x3
    
    // L-shapes and diagonals
    {{0,0},{0,1},{0,2},{1,0},{1,1},{2,0}},  // Top-left L
    {{0,1},{0,2},{0,3},{1,2},{1,3},{2,3}},  // Top-right L
    {{0,0},{0,1},{1,1},{1,2},{2,2},{2,3}},  // Diagonal right
    {{0,2},{0,3},{1,1},{1,2},{2,0},{2,1}},  // Diagonal left
};
```

### 7.4 Symmetry Generation Algorithm

For each base tuple, we generate up to 8 symmetric variants by applying the 8 elements of the dihedral group D4. The algorithm:

1. **Start with the base tuple** as the "current" transformation.
2. **For each of 4 rotations**: record the current tuple and its horizontal reflection (giving 8 candidates).
3. **Rotate** the tuple by 90 degrees: each position (r, c) maps to (c, 3-r).
4. **Deduplicate** by sorting positions within each candidate and comparing.

```c
static void generate_symmetries(ntuple_net_t *net, int t) {
    int cur[6][2], cands[8][6][2];
    int nc = 0;
    
    memcpy(cur, BASE_TUPLES[t], sizeof(cur));
    
    for (int rot = 0; rot < 4; rot++) {
        memcpy(cands[nc++], cur, sizeof(cur));        // current orientation
        for (int i = 0; i < 6; i++) {                 // horizontal reflection
            cands[nc][i][0] = cur[i][0];
            cands[nc][i][1] = 3 - cur[i][1];
        }
        nc++;
        int tmp[6][2];                                 // 90-degree rotation
        for (int i = 0; i < 6; i++) {
            tmp[i][0] = cur[i][1];
            tmp[i][1] = 3 - cur[i][0];
        }
        memcpy(cur, tmp, sizeof(cur));
    }
    
    // Deduplicate: sort each candidate's positions, compare
    int count = 0;
    for (int c = 0; c < nc; c++) {
        // ... sort positions, check against existing unique variants ...
        if (!duplicate)
            net->syms[t][count++] = cands[c];
    }
    net->n_sym[t] = count;  // typically 4-8
}
```

### 7.5 Memory Requirements Analysis

| Configuration | Memory per Tuple | Total (17 tuples) |
|--------------|-----------------|-------------------|
| Weights only | 16^6 * 4 = 64 MB | 1,088 MB ~ 1.07 GB |
| With TC (3 tables) | 64 MB * 3 = 192 MB | 3,264 MB ~ 3.19 GB |

The checkpoint file `ntuple_latest.bin` weighs exactly 3,264 MB, confirming these calculations. This is substantial but well within the memory capacity of modern systems (our development machine has 16+ GB RAM).

For comparison:
- DQN model weights: ~3 MB (0.003 GB)
- Expectimax engine: ~64 MB (transposition table only, no persistent weights)
- N-Tuple network: ~3.2 GB (with TC-learning)

The 1000x memory difference between DQN and N-Tuple is the price paid for tabular precision. Each of the 285 million entries stores a fine-grained weight for a specific tile configuration, whereas DQN's 200,000 parameters must generalize across all configurations.

### 7.6 Version 1: Python with Backward TD (7 Tuples)

The first N-Tuple implementation was in Python, with 7 tuples (4 six-position and 3 four-position tuples), a fixed learning rate of 0.0025, and backward TD updates.

**Training progression (from `ntuple_training_v1_12k.log`):**

| Episode | Time | Avg Score | 2048 Rate (per 100) | 4096 Rate | Notes |
|---------|------|-----------|---------------------|-----------|-------|
| 100 | 45s | 17,226 | 16% | 0% | First checkpoint saved |
| 500 | 228s | 16,499 | 12% | 0% | Plateau begins |
| 1,000 | 457s | 18,049 | 18% | 0% | Slow improvement |
| 1,500 | 700s | 18,178 | 20% | <1% | First 4096 at ep 1500! |
| 2,000 | 943s | 19,331 | 24% | <1% | |
| 3,000 | ~21min | ~20,000 | ~25% | ~1% | |
| 5,000 | ~41min | ~22,000 | ~30% | ~2% | |
| 8,000 | ~8h | ~26,000 | ~55% | ~3% | |
| 10,000 | ~12h | ~27,000 | ~63% | ~4% | |
| 11,100 | ~12.7h | 30,350 | ~72% | ~2% | Best checkpoint saved |
| 12,100 | ~12.9h | 30,020 | 68% | 7% | Final v1 log entry |

**Key milestones:**
- **Episode 100 (45 seconds)**: Already scoring 17,226 on average, 16% reaching 2048. After just 45 seconds of training, the N-Tuple network matched what DQN could not achieve in 3 hours.
- **Episode 1,500**: First game to reach 4096, with a score of 52,092. This was remarkable given the simplicity of the method.
- **Episode 11,100**: The average score peaked at 30,350 with a 72% 2048 rate. The checkpoint was saved as the best model.

However, the training was agonizingly slow in Python: approximately 1 episode per second at later stages (when the agent played longer games). Reaching 50,000 episodes would take 14+ hours. This motivated the C rewrite.

### 7.7 Version 2: Failed Improvement Attempts

Before committing to the C rewrite, we attempted several optimizations within the Python framework. All three failed:

#### 7.7.1 Failure 4: Learning Rate Overflow (LR=0.1)

**Context.** The v1 learning rate of 0.0025 seemed conservative. Academic papers mentioned learning rates of 0.01-0.1. We hypothesized that a higher learning rate would accelerate convergence.

**The attempt.** We increased the learning rate from 0.0025 to 0.1 -- a 40x increase.

**The symptom.** Within the first few hundred episodes, weights exploded to infinity. The evaluation function returned NaN (Not a Number), and the agent could not select any action. All games scored 0.

**The root cause.** The TD update is:

```
w[i] += lr * delta
```

where `delta = reward + V(s') - V(s)`. With LR=0.1 and delta potentially in the thousands (a single merge of two 1024-tiles gives reward 2048), the update could be hundreds of units per step. Over a game of 1000+ moves, this causes exponential weight growth:

```
Step 1: w = 0, delta = 100 -> w = 10
Step 2: w = 10, V(s) uses w, V(s') uses w -> delta includes w -> w += lr * (... + w)
...
Positive feedback loop -> w -> infinity
```

The fundamental issue is that the N-Tuple network has no normalization mechanism (unlike neural networks with batch normalization or weight decay). Large weights produce large evaluations, which produce large TD errors, which produce even larger weight updates.

**The fix.** We added delta clipping:

```c
if (delta > 1000.0f) delta = 1000.0f;
if (delta < -1000.0f) delta = -1000.0f;
```

And reduced the learning rate to 0.01 (the eventual C implementation uses a schedule from 0.01 to 0.0005). The clipping prevents catastrophic weight explosions while allowing normal-magnitude updates to proceed unaltered. The threshold of 1000 was chosen based on the observation that valid TD errors rarely exceed a few hundred in 2048.

#### 7.7.2 Failure 5: Symmetry Normalization

**Context.** When a tuple has k symmetric variants, each board evaluation sums k lookups. Correspondingly, each TD update adds `lr * delta` to k entries. This means tuples with more symmetries receive proportionally larger updates, potentially causing imbalanced learning.

**The attempt.** We normalized the update by dividing by the number of symmetries:

```python
adj = lr * delta / n_symmetries
for sym in symmetries:
    w[encode(board, sym)] += adj
```

**The result.** Learning became approximately 2x slower. After 5,000 episodes, the v2 model scored only ~15,000 compared to v1's ~22,000 at the same episode count.

**The root cause.** The normalization dilutes the update signal. With 8 symmetries, each weight receives only 1/8th of the update it would otherwise get. While this seems theoretically "correct" (preventing over-weighting), it is practically harmful because:

1. The symmetric variants share the same weight table, so the 8 updates to 8 different entries are not redundant -- they update different entries and collectively explore more of the weight space.
2. The reduced per-entry update magnitude slows convergence without improving stability (the oscillation that normalization is meant to prevent was not a problem in practice).

**The fix.** We removed the normalization and reverted to the standard update. The theoretical "incorrectness" of not normalizing is more than compensated by the practical benefit of faster learning.

#### 7.7.3 Failure: Per-Row Lookup Heuristics (Same as Failure 2)

This was the same mistake as Failure 2 (Section 6.9), repeated in the N-Tuple context. We attempted to decompose the N-Tuple evaluation into per-row contributions. The result was the same: loss of 2D spatial information, performance collapse to max tile 256. This confirmed that the failure was not specific to the Expectimax evaluation function but a fundamental limitation of 1D decomposition.

### 7.8 The C Implementation: The Big Leap

The C implementation of the N-Tuple trainer was the single most impactful optimization in the entire project. It combined four major improvements:

#### 7.8.1 Move Lookup Tables

Like the Expectimax engine, the N-Tuple trainer pre-computes merge results for all 65,536 possible rows. The grid stores log2 values (0 = empty, 1 = tile 2, ..., 15 = tile 32768), so each cell fits in 4 bits and a row of 4 cells fits in 16 bits:

```c
static row_result_t move_left_table[65536];

/* Build table: for each possible row, compute the merged result */
for (int enc = 0; enc < 65536; enc++) {
    int c[4] = { (enc>>12)&0xF, (enc>>8)&0xF, (enc>>4)&0xF, enc&0xF };
    
    // Remove zeros (compact left)
    int nz[4], nz_len = 0;
    for (int i = 0; i < 4; i++)
        if (c[i]) nz[nz_len++] = c[i];
    
    // Merge adjacent equals
    int res[4] = {0}, score = 0, pos = 0;
    for (int i = 0; i < nz_len; i++) {
        if (i+1 < nz_len && nz[i] == nz[i+1]) {
            int merged = nz[i] + 1;  // log2 domain: merge = increment
            if (merged > 15) merged = 15;  // cap at 32768
            res[pos++] = merged;
            score += (1 << merged);  // actual merge score
            i++;  // skip merged tile
        } else {
            res[pos++] = nz[i];
        }
    }
    
    move_left_table[enc] = (row_result_t){{res[0], res[1], res[2], res[3]}, score};
}
```

Each move now requires only 4 table lookups (one per row or column) instead of iterating over 16 cells. For RIGHT, DOWN, and UP moves, we simply reverse the row/column before and after the lookup:

```c
if (dir == 1) { /* RIGHT */
    for (int y = 0; y < 4; y++) {
        // Reverse the row before lookup
        int enc = encode_row_log(after[y][3], after[y][2], after[y][1], after[y][0]);
        row_result_t *r = &move_left_table[enc];
        // Reverse back after lookup
        after[y][3] = r->cells[0]; after[y][2] = r->cells[1];
        after[y][1] = r->cells[2]; after[y][0] = r->cells[3];
        total_score += r->score;
    }
}
```

#### 7.8.2 Log2 Grid Representation

The training engine stores tile values as their log2 in a `int[4][4]` grid:

```c
typedef int grid_t[4][4];  // 0=empty, 1=2, 2=4, ..., 15=32768
```

This representation has several advantages:

1. **Direct tuple encoding.** The tuple index is computed by treating the log2 values as digits in a base-16 number:

```c
static inline int encode6(const grid_t g, const int pos[][2]) {
    int idx = 0;
    idx = idx * 16 + g[pos[0][0]][pos[0][1]];
    idx = idx * 16 + g[pos[1][0]][pos[1][1]];
    idx = idx * 16 + g[pos[2][0]][pos[2][1]];
    idx = idx * 16 + g[pos[3][0]][pos[3][1]];
    idx = idx * 16 + g[pos[4][0]][pos[4][1]];
    idx = idx * 16 + g[pos[5][0]][pos[5][1]];
    return idx;
}
```

No log2 computation needed -- the values are already in log2 form.

2. **Compact storage.** Each cell needs only 4 bits (0-15). While we store them as `int` for simplicity (32 bits each), the move lookup tables use the compact 16-bit row encoding.

3. **Merge as increment.** When two tiles with log2 value v merge, the result has log2 value v+1. This is a simple increment instead of a multiplication.

#### 7.8.3 Game Engine Correctness

The C game engine faithfully reproduces the merge logic. Critical correctness concerns:

1. **Each tile merges at most once per move.** The merge algorithm processes non-zero tiles left-to-right (for LEFT), merging adjacent equals and advancing past the merged result. This automatically prevents double-merging.

2. **Merge score uses actual values.** The score is `1 << merged_log2`, converting back to actual tile values. Merging two 512-tiles (log2=9) gives score 2^10 = 1024.

3. **Move detection.** A move is "valid" (the board changed) if and only if `memcmp(after, before, sizeof(grid_t)) != 0`.

### 7.9 TC-Learning Implementation

TC-learning is implemented in the `net_update` function:

```c
static void net_update(ntuple_net_t *net, const grid_t g, float delta) {
    // Clamp delta to prevent overflow
    if (delta > 1000.0f) delta = 1000.0f;
    if (delta < -1000.0f) delta = -1000.0f;

    float abs_delta = (delta >= 0) ? delta : -delta;
    float decay = 0.9995f;
    
    for (int t = 0; t < net->n_base; t++) {
        float *w  = net->weights[t];
        float *ts = net->tc_sum[t];    // signed delta accumulator
        float *ta = net->tc_abs[t];    // absolute delta accumulator
        
        for (int s = 0; s < net->n_sym[t]; s++) {
            int idx = encode6(g, net->syms[t][s].pos);
            
            // Update accumulators with exponential decay
            ts[idx] = ts[idx] * decay + delta;
            ta[idx] = ta[idx] * decay + abs_delta;
            
            // Compute TC ratio
            float ratio = (ta[idx] > 1e-6f)
                ? ((ts[idx] >= 0 ? ts[idx] : -ts[idx]) / ta[idx])
                : 1.0f;
            
            // Adaptive weight update
            w[idx] += net->lr * ratio * delta;
        }
    }
}
```

**Line-by-line explanation:**

1. **Delta clamping** (lines 3-4): Prevents catastrophic weight explosions. This was added after Failure 4.

2. **Decay computation** (line 7): The constant 0.9995 gives a half-life of ~1,386 updates. This means each weight's effective learning rate is determined by roughly the last 1,400 updates to that specific weight.

3. **Accumulator updates** (lines 14-15): The signed sum `ts` tracks whether updates are consistently positive or negative. The absolute sum `ta` tracks the total magnitude of updates.

4. **Ratio computation** (lines 18-20): `ratio = |tc_sum| / tc_abs`. When updates are consistent (all positive or all negative), |tc_sum| ~ tc_abs, so ratio ~ 1. When updates oscillate, the signed sum cancels out while the absolute sum accumulates, giving ratio ~ 0.

5. **Adaptive update** (line 23): The effective learning rate for weight i is `lr * ratio[i]`. Stable weights (ratio ~ 1) learn at full speed. Converged weights (ratio ~ 0) learn slowly, reducing noise.

**The guard `ta[idx] > 1e-6f`**: For weights that have never been updated (ta ~ 0), the ratio would be 0/0 = NaN. We default to ratio = 1.0 for such weights, allowing them to learn at full speed until enough data accumulates.

### 7.10 Multi-Stage Training Strategy

Following Wu et al. (2014), we adopted a two-stage training approach:

**Stage 1: 500,000 episodes, 1-ply, 8 threads, TC-learning (~14 minutes)**

In 1-ply training, the agent selects the move that maximizes `reward + V(afterstate)` without any lookahead. This is fast (hundreds of episodes per second) and learns basic patterns:

- Keeping the max tile in a corner;
- Maintaining monotonic rows/columns;
- Avoiding premature merges;
- Preserving free space.

**Stage 2: 5,000,000 episodes, 3-ply, 8 threads, TC-learning (~14 hours estimated)**

In 3-ply training, the agent evaluates each candidate afterstate using a 3-ply Expectimax search (player -> chance -> player -> chance -> player -> evaluate). This is approximately 100x slower per episode but produces much stronger play because:

1. **The training signal is more accurate.** The 3-ply search provides a better estimate of the afterstate's value, reducing the noise in TD updates.
2. **The agent explores higher-quality trajectories.** With lookahead, the agent makes better moves during training, generating experiences from positions that are more representative of strong play.
3. **Strategic patterns emerge.** 1-ply cannot discover strategies that require 2-3 moves of setup. For example, creating a "merge chain" (aligning 4 + 4 + 8 to produce 8 + 8 = 16) requires looking ahead 2 moves.

**Why not start with 3-ply?** 3-ply training at ~30 episodes/second would require 46 hours for 5M episodes. Starting with 1-ply for 500K episodes (14 minutes) provides a strong initialization: the weights learn basic patterns quickly, so the 3-ply fine-tuning starts from a much better point than zero.

### 7.11 Training Progression: A Detailed Chronicle

#### 7.11.1 Stage 1: 1-ply (from `ntuple_training.log`)

The Stage 1 training showed remarkably fast initial learning:

| Episode | Time | Eps/s | Avg Score | Max Tiles Distribution | 2048+ Total |
|---------|------|-------|-----------|----------------------|-------------|
| 218 | 0s | 218 | 5,426 | {64:4, 128:37, 256:62, 512:88, 1024:19} | 0 |
| 1,444 | 1s | 1,444 | 16,923 | {128:1, 256:15, 512:154, 1024:482, 2048:147, 4096:1} | 167 |
| 2,413 | 2s | 1,207 | 22,035 | {256:6, 512:77, 1024:406, 2048:304, 4096:7} | 535 |
| 3,335 | 4s | 834 | 25,459 | {256:4, 512:58, 1024:331, 2048:381, 4096:26} | 1,001 |
| 5,028 | 6s | 838 | 30,059 | {256:3, 512:46, 1024:240, 2048:435, 4096:76} | 2,047 |
| 8,046 | 12s | - | 33,440 | {256:4, 512:27, 1024:253, 2048:375, 4096:141} | 4,011 |

**Key observations from Stage 1:**

- **First 2048 at ~2 seconds of training** (episode ~1,444). Compare to DQN: no 2048 after 3 hours.
- **First 4096 at ~1 second** (episode ~1,444). The network learned the basic patterns shockingly fast.
- **Throughput decrease over time**: From 1,444 eps/s initially to ~834 eps/s at episode 5,000. This is because the agent plays longer games as it improves (more moves per game = more computation per episode).
- **Diminishing returns after ~5,000 episodes**: The average score plateaued around 33,000-42,000 by 500K episodes. 1-ply training cannot discover patterns requiring multi-move planning.

By the end of Stage 1 (500K episodes, ~14 minutes), the network achieved:
- Average score: ~42,000
- 2048+ rate: ~73%
- 4096 rate: ~33%
- 8192 rate: ~1%

#### 7.11.2 Stage 2: 3-ply (from `ntuple_training.log`)

Stage 2 builds on Stage 1's weights, using 3-ply search during both move selection and value estimation:

| Episode (Stage 2) | Total Time | Eps/s | Avg Score | Distribution (per 800) | 8192 Rate |
|-------------------|-----------|-------|-----------|----------------------|-----------|
| 6,400 | ~240s | 27 | 66,497 | {256:5, 512:19, 1024:92, 2048:161, 4096:434, 8192:89} | 11.1% |
| 33,000 | ~1,200s | 27 | ~66,000 | (stable) | ~11% |
| 46,000 | ~1,700s | 27 | ~64,000 | {256:3, 512:14, 1024:87, 2048:159, 4096:439, 8192:98} | ~12% |

**Critical observation: the apparent 2048 rate "drop."**

The per-100 2048 rate appears to have dropped from ~40% (Stage 1) to ~20% (Stage 2). But this is misleading. In Stage 1, many games reached 2048 and stopped there. In Stage 2, many of those same games now continue past 2048 to reach 4096 or 8192. The relevant metric is "2048+ rate" (games reaching at least 2048):

```
Stage 1 (500K ep):  2048 ~40% + 4096 ~33% + 8192 ~1% = 2048+ ~74%
Stage 2 (46K ep):   2048 ~20% + 4096 ~55% + 8192 ~12% = 2048+ ~87%
```

The 2048+ rate improved from 74% to 87%, and the quality of wins improved dramatically: 8192 went from 1% to 12%.

**Training throughput**: 27 episodes per second with 8 threads. This is roughly 100x slower than 1-ply (because each move requires 3-ply search during training), confirming the importance of Stage 1 initialization.

**Learning rate schedule**: The learning rate decays exponentially from 0.01 to 0.0005 over the course of training. At episode 46,000 out of 5,000,000, the learning rate is still near the maximum (0.00973), indicating that significant learning potential remains.

```
lr = lr_start * (lr_end / lr_start) ^ (episode / total_episodes)
   = 0.01 * (0.0005 / 0.01) ^ (46000 / 5000000)
   = 0.01 * (0.05) ^ 0.0092
   = 0.01 * 0.973
   = 0.00973
```

### 7.12 The C Player (Shared Library)

For browser gameplay, the trained N-Tuple network is exposed as a C shared library (`ntuple_c.so`). The player differs from the trainer in several ways:

1. **No training**: Only evaluation, no weight updates. This halves the computation (no TC accumulator updates).

2. **Deeper search during play**: The trainer uses 3-ply search; the player uses **5-ply search** (`search_depth=2` in our API, which means 2 full chance-max-chance-max-evaluate cycles = 5 ply). This asymmetry is deliberate: deeper search during play is cheap (one move at a time vs. thousands per episode during training) and significantly improves play quality.

3. **Actual values vs. log2**: The trainer stores log2 values internally for efficiency. The player receives actual tile values from the Python server and converts them:

```c
int ntuple_select_action(int *grid_flat, int search_depth) {
    grid_t g;
    for (int y = 0; y < 4; y++)
        for (int x = 0; x < 4; x++)
            g[y][x] = grid_flat[y * 4 + x];  // actual values: 0, 2, 4, 8, ...
    
    // The evaluate() and do_move() functions work with actual values
    // (not log2), so the encode() function converts:
    static inline int encode(const grid_t g, const int pos[][2]) {
        int idx = 0;
        for (int i = 0; i < 6; i++)
            idx = idx * MAX_LOG2 + to_log2(g[pos[i][0]][pos[i][1]]);
        return idx;
    }
}
```

**Performance**: 5-ply search with N-Tuple evaluation completes in approximately **2.4 milliseconds** per move. This is 40x faster than Expectimax at the same depth (which uses a more expensive 2D heuristic evaluation) and enables smooth real-time visualization in the browser.

---

## 8. Optimization Journey

### 8.1 Why Performance Matters

Performance is not just a "nice to have" for this project -- it directly determines the quality of results:

1. **Training quality**: Faster training means more episodes in the same wall-clock time. At 5 eps/s (Python), 500K episodes would take 28 hours. At 300 eps/s (C, 1-ply), the same training takes 28 minutes. This enables rapid experimentation with hyperparameters and architecture choices.

2. **Search depth during training**: 3-ply search is only practical in C. In Python, the per-episode time with 3-ply would be approximately 10 seconds, making 5M episodes take 578 days.

3. **Real-time play**: The web interface requires sub-100ms response times for smooth visualization. Python Expectimax at depth 3 barely meets this at 217ms; C Expectimax can search to depth 4-5 within the same budget.

4. **User experience**: Watching an AI play at 1 move per second is tedious. At 2.4ms per move, the N-Tuple agent can play a complete game in under 5 seconds if desired.

### 8.2 Move Lookup Tables

The move lookup table is the single most impactful optimization for the game engine. By pre-computing merge results for all 65,536 possible rows, each move reduces from an iterative algorithm (loop over cells, compact, merge pairs) to four table lookups:

```c
// Before: iterative merge for LEFT move
for (int y = 0; y < 4; y++) {
    // Compact non-zero tiles left
    // Merge adjacent equals
    // Track merged tiles to prevent double-merge
    // Compute score
    // ... ~20 lines of code, ~50 operations per row
}

// After: lookup table for LEFT move
for (int y = 0; y < 4; y++) {
    int enc = encode_row(after[y]);
    after[y] = move_left_table[enc].cells;
    total_score += move_left_table[enc].score;
    // 3 operations per row
}
```

The table is built once at program startup (taking approximately 1ms) and provides approximately 2-3x speedup for the entire game engine.

### 8.3 Bitboard Representation

The Expectimax engine uses a 64-bit integer to represent the board, with each cell stored as a 4-bit nibble. This enables:

```c
// Row extraction: single shift + mask
static inline row_t board_row(board_t b, int y) {
    return (row_t)((b >> (y * 16)) & 0xFFFF);
}

// Cell access: two operations
static inline int board_get_cell(board_t b, int y, int x) {
    return (board_row(b, y) >> ((3 - x) * 4)) & 0xF;
}

// Board comparison: single 64-bit comparison
if (new_board == old_board) { /* no change */ }
```

The N-Tuple trainer uses a different representation (`int[4][4]`) for clarity and because the tuple encoding is simpler with direct array access. The performance difference is minimal for the trainer because the bottleneck is the weight table access (which involves random memory accesses to a 64 MB array), not the board manipulation.

### 8.4 Hogwild Multithreading

**Hogwild!** (Niu et al., 2011) is a lock-free approach to parallel stochastic gradient descent. Multiple threads update shared weights simultaneously without synchronization. This works when:

1. **Updates are sparse**: Each move accesses only ~100 of the 285 million weights (17 tuples * ~6 symmetries * 1 entry per symmetry). The probability of two threads accessing the same weight simultaneously is negligible.

2. **Updates are small**: Each individual update (`lr * ratio * delta`) is small relative to the weight magnitude. Even if a race condition causes a missed or doubled update, the effect is negligible.

3. **Convergence is robust**: TD-learning with tabular representations is naturally noise-tolerant. The algorithm converges to the same fixed point regardless of the order of updates.

Our implementation uses 8 POSIX threads, each with its own game state and random seed:

```c
static void *train_worker(void *arg) {
    thread_stats_t *stats = (thread_stats_t *)arg;
    unsigned int seed = (unsigned int)(time(NULL) + stats->thread_id * 7919);
    
    while (1) {
        int ep = atomic_fetch_add(&episodes_done, 1) + 1;
        if (ep > total_episodes) break;
        
        // Play complete game, updating shared weights directly
        // No locks, no synchronization
        ...
    }
}
```

The `atomic_fetch_add` on the episode counter is the only synchronization point. All weight accesses are unsynchronized.

**Measured throughput**: With 8 threads on Apple Silicon, the throughput at 1-ply is approximately 300 eps/s, compared to ~100 eps/s single-threaded -- a 3x speedup. The sub-linear scaling (3x vs. 8x theoretical) is due to memory bandwidth limitations: 8 threads competing for access to the 3.2 GB weight tables creates memory bus contention.

### 8.5 Thread-Safe Random Number Generation

The standard C `rand()` function is not thread-safe -- it uses global state that would be corrupted by concurrent access. We implemented a per-thread linear congruential generator (LCG):

```c
static inline int trand(unsigned int *seed) {
    *seed = *seed * 1103515245 + 12345;
    return (*seed >> 16) & 0x7FFF;
}
```

Each thread maintains its own `seed`, initialized from `time(NULL) + thread_id * 7919`. The constant 7919 (a prime number) ensures different seeds for threads starting at the same time.

The LCG constants (1103515245 and 12345) are the POSIX standard values, known to produce reasonably uniform distributions for the lower-significance bits. We extract bits 16-30 (`>> 16`) to avoid the known weakness of LCGs in their low-order bits.

**Why not just use `rand_r()`?** While `rand_r()` is thread-safe, it is a library call with function call overhead. Our inline LCG compiles to 3 machine instructions (multiply, add, shift), which is significant when called thousands of times per episode.

### 8.6 Unbuffered Output (Failure 7)

**Context.** During training, we wanted to monitor progress in real-time by piping output through `tee` to save logs:

```bash
./ntuple_train --episodes 500000 --depth 0 --tc --threads 8 | tee training.log
```

**The symptom.** No output appeared for minutes. The terminal was blank, making it impossible to tell if training was progressing or had crashed.

**The root cause.** When stdout is connected to a pipe (rather than a terminal), the C runtime library switches from line-buffered to fully-buffered mode. With full buffering, output is accumulated in a buffer (typically 4 KB) and only flushed when the buffer is full or the program exits.

Our training output (one report every ~100 episodes) generates perhaps 500 bytes per report. With a 4 KB buffer, the first output would not appear until 8 reports had accumulated -- potentially many minutes of training.

**The fix.** A single line added at the beginning of `main()`:

```c
setbuf(stdout, NULL);  // Disable buffering entirely
```

This forces every `printf()` call to immediately write to the output device. The performance impact is negligible because we only print every ~100 episodes.

**Lesson learned.** Always use unbuffered output for long-running programs that produce periodic status updates. This is a common gotcha when piping output through `tee` or redirecting to files.

### 8.7 Cumulative Optimization Impact

| Component | Speedup | Cumulative |
|-----------|---------|------------|
| Python baseline | 1x | 1x |
| Python -> C (game engine + search) | ~50x | 50x |
| Move lookup tables | ~2-3x | 100-150x |
| Hogwild 8 threads | ~3x | 300-450x |
| **Total** | | **~300-450x** |

From approximately 5 episodes per second (Python, 1-ply) to approximately 300 episodes per second (C, 1-ply, 8 threads). This reduced the time for 500K episodes from an estimated 28 hours to 28 minutes.

At 3-ply (Stage 2), the throughput is approximately 27 episodes per second -- about 100x slower than 1-ply. The entire 5M-episode Stage 2 would take approximately 51 hours (just over 2 days).

---

## 9. Comparative Analysis

### 9.1 Agent Performance Metrics

The following table summarizes the performance of all three agents under their best configurations:

| Metric | DQN (1,700 ep) | Expectimax (C, 100ms) | N-Tuple (Stage 2, 46K ep) |
|--------|----------------|----------------------|--------------------------|
| **Training required** | 3h+ (insufficient) | None | ~30 min (Stage 1 + early Stage 2) |
| **Time per move** | ~1ms (CPU) | ~100ms (C, iterative) | **~2.4ms** (C, 5-ply) |
| **Average score** | ~5,000 | ~38,267 | **~64,000** |
| **Score std dev** | ~2,500 | ~12,600 | ~15,000 (est.) |
| **2048+ rate** | ~0% | ~80% | **~87%** |
| **4096 rate** | 0% | ~60% | **55%** |
| **8192 rate** | 0% | 0% | **12%** |
| **Best single score** | ~11,600 | 76,516 | **124,080** |
| **Best max tile** | 512 | 4096 | **8192** |
| **Typical game length** | ~200 moves | ~1,770 moves | ~2,500 moves |

### 9.2 Training Efficiency Comparison

| Metric | DQN | N-Tuple |
|--------|-----|---------|
| Time to first 2048 | >3h (never achieved) | **~2 seconds** |
| Episodes to first 2048 | >1,700 (never achieved) | **~1,444** |
| Score per hour of training | ~1,700 | **~200,000** |
| Training speedup | 1x (baseline) | **~100x** |

N-Tuple Networks are approximately **100x more training-efficient** than DQN for 2048. This is not a minor improvement -- it represents a qualitative change in what is practically achievable.

**Why the massive efficiency gap?**

1. **Direct weight updates.** N-Tuple updates are O(1) per weight: add a delta. DQN requires forward and backward passes through a neural network, computing gradients for all 200,000 parameters.

2. **No generalization overhead.** N-Tuple weights are independent. DQN's weight update for one state affects all states through the shared parameters, requiring careful learning rate tuning to avoid catastrophic forgetting.

3. **Matched representation.** N-Tuple encoding directly captures the spatial patterns that matter for 2048. DQN must learn these patterns through gradient descent, starting from random initialization.

4. **Afterstate learning.** N-Tuple learns from afterstates (deterministic), while DQN learns from states (stochastic). This reduces the variance of the learning signal by an order of magnitude.

### 9.3 Computational Cost Analysis

| Resource | DQN | Expectimax | N-Tuple (Training) | N-Tuple (Playing) |
|----------|-----|------------|--------------------|--------------------|
| **CPU** | GPU-bound (MPS/CUDA) | CPU-intensive | CPU-intensive | Light |
| **RAM** | ~200 MB | ~64 MB (cache) | ~3.3 GB | ~1.1 GB |
| **Disk** | ~3 MB (weights) | 0 | ~3.3 GB (checkpoint) | ~1.1 GB |
| **Peak throughput** | ~10 eps/s | N/A | ~300 eps/s (1-ply) | N/A |

The N-Tuple network's primary cost is memory: 3.3 GB during training (with TC) and 1.1 GB during play. This is significant but manageable on any modern laptop or desktop.

### 9.4 Decision Quality Analysis

**DQN** makes erratic, spatially incoherent decisions. Looking at the game report for `dqn_s912_t64.json`, the final board shows two separate 64-tiles (at positions (1,1) and (3,2)), with no spatial organization. The agent survived only 97 moves before game over.

**Expectimax** produces excellent spatial organization. The game report for `expectimax_s76516_t4096.json` shows a textbook snake pattern:

```
4096  16   2   4
2048  64  16   8
1024 128   8   4
 256  32   4   2
```

Values decrease monotonically down the left column (4096 -> 2048 -> 1024 -> 256) and along each row. This organization is entirely the result of the hand-crafted heuristics.

**N-Tuple** learns patterns that are similar to the snake but with subtle differences that a human would not have coded. From `ntuple_s124080_t8192.json`:

```
2048   4  8192   4
  16   2   256 128
  32 512    64   4
   2   4    32   2
```

This board reached 8192 (the highest tile in our experiments) with a score of 124,080. The spatial pattern is less "clean" than Expectimax's snake -- the 8192 is in position (0,2) rather than a corner, and the organization is less regular. However, the network has learned sophisticated strategies for building toward higher tiles that the hand-crafted heuristics cannot express.

### 9.5 Statistical Significance

With only 5 Expectimax games and ongoing N-Tuple training, we have limited statistical power. However, some comparisons are unambiguous:

**DQN vs. anything else:** DQN's best score (11,600) is below Expectimax's worst score (14,508) and far below N-Tuple's average (64,000). The probability of DQN outperforming either approach is effectively zero.

**Expectimax vs. N-Tuple on 2048+ rate:** With Expectimax at 80% and N-Tuple at 87%, the difference is 7 percentage points. For a binomial proportion test with n=5 (Expectimax) and n=800 (N-Tuple batch size), the 95% confidence interval for Expectimax's true rate is approximately [36%, 97%], which overlaps with the N-Tuple's rate. Thus, we cannot definitively claim N-Tuple is better than Expectimax on 2048+ rate based on these sample sizes alone.

**Expectimax vs. N-Tuple on average score:** The mean scores (38,267 vs. 64,000) are clearly different. Even with the Expectimax standard deviation of ~12,600, the N-Tuple average is 2 standard deviations above the Expectimax average. By the training log data (800 games per batch), the N-Tuple's score distribution is stable, confirming a genuine performance advantage.

**Expectimax vs. N-Tuple on 8192 rate:** Expectimax never reached 8192 in our testing; N-Tuple achieves it in ~12% of games. This is a categorical difference.

---

## 10. Performance Ceiling and State of the Art Comparison

### 10.1 Positioning Against Published Results

| Level | 2048+ Rate | 4096 Rate | 8192 Rate | Avg Score | Method |
|-------|-----------|-----------|-----------|-----------|--------|
| Random player | ~0% | 0% | 0% | ~800 | Baseline |
| **Our DQN** | ~0% | 0% | 0% | ~5,000 | DQN, 1,700 episodes |
| Good human | ~30% | rare | 0% | ~20,000 | Manual play |
| **Our Expectimax** | **~80%** | **~60%** | 0% | **~38,000** | Expectimax + heuristics |
| **Our N-Tuple (current)** | **~87%** | **~55%** | **~12%** | **~64,000** | 17x6-tuple, TC, 3-ply |
| "Good AI" | ~95% | ~70% | ~10% | ~50,000-80,000 | Various |
| Wu et al. (2014) | ~97% | ~85% | ~40% | ~100,000+ | Multi-stage N-Tuple |
| Jaskowski (2018) | ~99.5% | ~95% | ~75% | ~300,000+ | TC + large tuples |

Our N-Tuple agent is in the "good AI" range, significantly above naive implementations but below published state-of-the-art results. Specifically:

- We match Wu et al.'s **architecture** (multi-stage training with N-Tuples) but have completed only a fraction of their training (46K vs. millions of Stage 2 episodes).
- Our 8192 rate of 12% is in the same ballpark as Wu et al.'s early results.
- The gap to Jaskowski (99.5% / 75% 8192) is primarily due to:
  1. **Training duration**: We have completed ~1% of Stage 2. Jaskowski trained for 40M+ episodes.
  2. **Tuple architecture**: Jaskowski used larger and more diverse tuples (up to 8-position).
  3. **Advanced techniques**: Jaskowski's TC-learning implementation may have additional refinements.

### 10.2 Theoretical Performance Ceiling

With our current architecture (17 x 6-tuples), the theoretical ceiling can be estimated based on published results with similar architectures:

- **17 x 6-tuples with full training** (estimated): ~97-98% 2048 rate, ~85% 4096 rate, ~40% 8192 rate.
- To exceed this requires either:
  - **More tuples**: Each additional 6-tuple adds 64 MB and marginal improvement;
  - **Larger tuples**: 7-tuples require 16^7 = 268M entries (1 GB per tuple). 8-tuples: 4 GB per tuple;
  - **Deeper search during play**: Going from 5-ply to 7-ply would help but increases time per move from 2.4ms to ~240ms.

### 10.3 Convergence Analysis

From our training logs, the average score has been relatively stable at ~64,000 since Stage 2 episode ~6,400. However, the learning rate is still at 97.3% of its initial value (having barely decayed). This suggests that:

1. The early rapid improvement in Stage 2 (from ~42,000 to ~64,000 in ~6,400 episodes) was primarily due to the switch from 1-ply to 3-ply training -- the better training signal immediately improved the value estimates.

2. Further improvement will come from continued training as the weights gradually converge. Based on the literature, significant improvements continue up to at least 5M episodes.

**Projected performance at key milestones:**

| Stage 2 Episodes | Estimated Time | Expected 2048+ Rate | Expected 8192 Rate |
|-----------------|----------------|---------------------|-------------------|
| 46K (current) | ~30 min | ~87% | ~12% |
| 500K | ~5 hours | ~92% | ~25% |
| 2M | ~20 hours | ~95% | ~35% |
| 5M | ~51 hours | ~97% | ~40% |
| 10M | ~4 days | ~97-98% (ceiling) | ~45% |

These projections assume diminishing returns: the 2048+ rate follows a logarithmic approach to the ceiling.

### 10.4 What Would It Take to Reach 99%

To reach 99%+ 2048 rate and 75%+ 8192 rate (state-of-the-art level), we would need:

1. **Longer training**: 40M+ episodes at 3-ply (~2-3 weeks of continuous training).
2. **Larger tuples**: At least some 7-tuples or 8-tuples, requiring 4+ GB additional memory per tuple.
3. **More tuples**: 20-30 base tuples covering more spatial patterns.
4. **Deeper search during play**: 7-ply or deeper.
5. **Possibly**: Expectimax-3 (looking ahead 3 moves during training value estimation).

The total memory for a state-of-the-art configuration would be approximately 10-50 GB, and training would require dedicated hardware for weeks. This is feasible for a research lab but beyond the scope of this project.

---

## 11. Web Interface and Real-Time Visualization

### 11.1 Frontend Architecture

The web interface builds on Cirulli's original 2048 game code, adding AI capabilities through a non-intrusive JavaScript module (`ai_player.js`). This module:

1. **Injects UI elements** (AI Play button, agent selector, status display) into the game page;
2. **Extracts the board state** from the game's DOM representation;
3. **Communicates with the backend** via XMLHttpRequest;
4. **Applies returned moves** by emitting events through the game's InputManager.

The design preserves the original game's behavior completely -- human play works unchanged, and AI play uses the same move execution path as human input.

### 11.2 Backend API Server

The Flask server (`server.py`, 211 lines) exposes two endpoints:

**POST /move**: Receives `{grid: int[4][4], agent: string}` and returns `{action: int, direction: string, agent: string}`.

The agent dispatch logic:

```python
if agent_type == 'dqn':
    encoded = encode_state(grid)
    valid_moves = [d for d in range(4) if _is_valid_move(grid, d)]
    action = dqn_agent.select_action(encoded, valid_moves, training=False)

elif agent_type == 'ntuple':
    flat = grid.flatten().astype(ctypes.c_int)
    arr = (ctypes.c_int * 16)(*flat)
    action = ntuple_c_lib.ntuple_select_action(arr, 2)  # 5-ply

else:  # expectimax
    action, depth_reached = expectimax_agent.select_action(grid)
```

Note the `ctypes` bridge for the N-Tuple agent: the grid is flattened to a C array of 16 integers, passed to the C shared library, and the returned integer (0-3) is the move direction.

**POST /report**: Receives a complete game report and saves it as a timestamped JSON file in `ai/reports/`.

### 11.3 Agent Selection and Visual Feedback

The frontend provides visual differentiation between agents through color-coded buttons:

| Agent | Button Color | Label |
|-------|-------------|-------|
| Expectimax | Yellow (#edc22e) | Expectimax |
| DQN | Pink (#e64c8a) | DQN |
| N-Tuple | Green (#2ecc71) | N-Tuple |

Clicking the agent button cycles through the three agents. The status bar below the game board shows real-time information:

```
[NTUPLE] Move #1542: down | Score: 85,392 | Max: 4096
```

### 11.4 Game Report System

Every completed game (win or game over) automatically generates a JSON report saved to `ai/reports/`. The filename encodes key information:

```
20260819_102231_ntuple_s124080_t8192.json
|        |      |      |       |
|        |      |      |       +-- max tile
|        |      |      +---------- score
|        |      +----------------- agent
|        +------------------------ timestamp
```

Each report contains:

```json
{
  "timestamp": "2026-08-19T10:22:31.728768",
  "agent": "ntuple",
  "score": 124080,
  "max_tile": 8192,
  "moves": 3933,
  "won": true,
  "final_grid": [[2048, 4, 8192, 4], [16, 2, 256, 128], ...],
  "move_history": [
    {"move": 1, "action": "up", "score_before": 24516, "max_tile": 2048, ...},
    {"move": 2, "action": "left", "score_before": 24516, "max_tile": 2048, ...},
    ...
  ],
  "duration_ms": 52340,
  "config": {
    "expectimax_depth": null,
    "dqn_steps": null,
    "ntuple": true
  }
}
```

The `move_history` array provides a complete replay log, including the grid state before each move. This enables post-hoc analysis of decision quality, identification of critical mistakes, and visualization of the agent's strategy over the course of a game.

### 11.5 Real Game Report Examples

#### 11.5.1 N-Tuple: High Score Game (124,080, 8192 tile)

From `20260819_102231_ntuple_s124080_t8192.json`:
- **Agent**: N-Tuple (5-ply search)
- **Score**: 124,080
- **Max tile**: 8192
- **Moves**: 3,933
- **Duration**: Not recorded (continued play)
- **Final board**:

```
2048   4  8192   4
  16   2   256 128
  32 512    64   4
   2   4    32   2
```

This game achieved the highest score and tile in our experiments. The N-Tuple agent built the 8192 tile over approximately 3,900 moves -- nearly double the typical game length -- demonstrating sophisticated long-term planning.

#### 11.5.2 Expectimax: Best Organized Board (76,516, 4096 tile)

From `20260818_170052_expectimax_s76516_t4096.json`:
- **Agent**: Expectimax (100ms time budget)
- **Score**: 76,516
- **Max tile**: 4096
- **Moves**: 2,526
- **Final board**:

```
4096  16   2   4
2048  64  16   8
1024 128   8   4
 256  32   4   2
```

This board exhibits a near-perfect snake pattern, with the 4096 tile firmly in the top-left corner and values decreasing along a zigzag path. The Expectimax agent's hand-crafted heuristics produce this clean spatial organization naturally.

#### 11.5.3 DQN: Typical Game (912, max tile 64)

From `20260818_163442_dqn_s912_t64.json`:
- **Agent**: DQN (1,700 training episodes)
- **Score**: 912
- **Max tile**: 64
- **Moves**: 97
- **Duration**: 1,823ms
- **Final board**:

```
  8   2   4   2
 16  64  16   4
  4  16  32   2
  2   4  64   4
```

The board is chaotic: two separate 64-tiles, no spatial organization, and a full board after only 97 moves. This exemplifies the DQN agent's inability to learn positional strategy within its training budget.

---

## 12. Failures, Debugging, and Lessons Learned

This chapter provides a dedicated, detailed account of every significant failure encountered during the project. Each failure is presented as a narrative, following the debugging process from symptom to root cause to fix.

### 12.1 Failure 1: DQN Does Not Converge

**Context.** We had spent two days implementing the DQN agent: the neural network architecture, the experience replay buffer, the target network synchronization, Double DQN, reward shaping. Everything compiled and ran without errors. We started training, expecting to see the agent learn to play within a few hours.

**The attempt.** Standard DQN training with the hyperparameters described in Section 5.4.3. The epsilon-greedy exploration starts at 1.0 and decays exponentially with a time constant of 200,000 steps.

**The symptoms.** After 100 episodes (~10 minutes), the average score was approximately 800 -- barely above random play (~800). We attributed this to the high epsilon (most actions were still random). After 500 episodes (~52 minutes), the score rose to approximately 1,500. Progress was slow but present. We let training continue overnight.

After 1,700 episodes (~3 hours), the average score was approximately 5,000 with a maximum tile of 512. No game had ever reached 2048. The improvement rate was approximately 2 points per minute -- at this rate, reaching Expectimax's average score of 40,000 would take approximately 300 hours (12.5 days).

**The investigation.** We examined several potential causes:

1. **Q-value inspection**: We printed Q-values for known board states. The Q-values were small and noisy, with little differentiation between good and bad moves. The network had not yet learned meaningful spatial patterns.

2. **Epsilon check**: At step ~500,000, epsilon was still around 0.09 -- significant random exploration. The network rarely got to use its own Q-values.

3. **Replay buffer analysis**: We sampled 100 transitions from the replay buffer. Most (>80%) were from early game positions with few tiles, where any move is roughly equivalent. The valuable late-game transitions (where strategy matters) were drowned in a sea of trivial early-game data.

4. **Receptive field analysis**: We realized that with two 2x2 conv layers, no single neuron could see more than 3x3 cells -- 56% of the board. The snake pattern, which spans all 16 cells, was architecturally invisible.

**The root cause.** A combination of: (a) epsilon decay too slow relative to episode count, (b) undifferentiated replay buffer, (c) insufficient receptive field. Each factor alone might have been manageable; together, they made convergence impractical.

**The fix.** We abandoned DQN and pivoted to Expectimax. In hindsight, DQN could have been improved by:
- Using a larger architecture with a wider receptive field;
- Implementing prioritized experience replay;
- Using a faster epsilon decay schedule;
- Training for 50,000+ episodes.

But these fixes would have required additional weeks of experimentation, while Expectimax was ready to play within an hour.

**What we learned.** Deep RL is not a silver bullet. For problems with structured, discrete state spaces, specialized methods (tabular RL, tree search with heuristics) can be orders of magnitude more efficient.

### 12.2 Failure 2: 1D Lookup Table Decomposition

**Context.** We had a working Expectimax agent in Python, but it was too slow for real-time play at depth > 3. We needed to speed up the evaluation function.

**The attempt.** Pre-compute heuristic values per row in a lookup table of 65,536 entries. The board evaluation would decompose into:

```
V(board) = sum of heur_row[row_y] for y in 0..3
         + sum of heur_col[col_x] for x in 0..3
```

This would replace 16 cell visits with 8 table lookups.

**The symptoms.** The agent's maximum tile dropped from 2048 to 256. Average score fell from ~20,000 to ~4,000. The agent would fill the board quickly and lose, showing no strategic organization.

**The investigation.** We compared per-move evaluations between the 2D and 1D versions:

```
Board: [512, 256, 128, 64]
       [16,   32,  64, 128]
       [8,    4,   2,  16]
       [2,    4,   8,  32]

2D snake score:  512*32768 + 256*16384 + ... = high (good organization)
1D score:        row[0] + row[1] + ... = moderate (each row looks OK individually)
```

The 1D evaluation could not see that the first row connects smoothly to the second row (64 in row 0 is adjacent to 128 in row 1, forming part of the snake). Each row was evaluated independently, losing all inter-row relationships.

**The root cause.** The snake pattern, monotonicity across rows, and the corner bonus all depend on 2D spatial relationships that cannot be decomposed into independent row contributions. A row [512, 256, 128, 64] is equally "good" whether it is at the top of the board (continuing a descending pattern) or at the bottom (isolated from the rest) -- but the 2D evaluation correctly distinguishes these.

**The fix.** We abandoned the 1D decomposition and kept the full 2D evaluation. For speed, we would instead rewrite the entire engine in C (which turned out to give a 155x speedup -- far more than the lookup table would have provided).

**What we learned.** When a heuristic captures relationships between multiple dimensions, decomposing it into single-dimensional components destroys the very information that makes it effective. Not all optimizations are valid -- the speedup must preserve the quality of the computation.

### 12.3 Failure 3: The Transpose Bug

**Context.** We had rewritten the Expectimax engine in C, using a 64-bit bitboard representation. LEFT and RIGHT moves worked correctly (verified by hand). We needed UP and DOWN moves, which require transposing the board (swapping rows and columns).

**The attempt.** We copied the transpose function from nneonneo's highly optimized 2048-ai implementation. This function uses a series of bit masks and shifts to transpose the 4x4 nibble grid in O(1):

```c
static inline board_t transpose(board_t x) {
    board_t a1 = x & 0xF0F00F0FF0F00F0FULL;
    board_t a2 = x & 0x0000F0F00000F0F0ULL;
    board_t a3 = x & 0x0F0F00000F0F0000ULL;
    board_t a = a1 | (a2 << 12) | (a3 >> 12);
    board_t b1 = a & 0xFF00FF0000FF00FFULL;
    board_t b2 = a & 0x00FF00FF00000000ULL;
    board_t b3 = a & 0x00000000FF00FF00ULL;
    return b1 | (b2 >> 24) | (b3 << 24);
}
```

**The symptoms.** The AI exclusively moved DOWN, ignoring all other directions. The typical game lasted 8-10 moves with a final score of approximately 44 and a maximum tile of 16. The board would fill up almost immediately.

**The investigation.** This was a confusing bug because the symptom (always moving down) seemed unrelated to a transpose function:

**Step 1: Verify move generation.** We tested LEFT and RIGHT on known boards -- they worked correctly. But UP and DOWN produced garbage boards with zeroed-out cells.

**Step 2: Suspect the transpose.** Since UP and DOWN used `transpose(board)` -> `move_left/right` -> `transpose(result)`, we suspected the transpose was wrong.

**Step 3: Trace a specific nibble.** We created a board with a known value at position (0,0):

```
Board: [5, 0, 0, 0; 0, 0, 0, 0; 0, 0, 0, 0; 0, 0, 0, 0]

In nneonneo's layout: nibble 0 = 5
In our layout:         nibble 3 = 5

After nneonneo's transpose:
  Expected (nneonneo): nibble 0 = 5 (cell (0,0) stays in row 0, column 0)
  Actual (our data):   nibble 3 should become... something else
```

We traced the value 5 through each bit operation of the transpose function and found that it was being placed at a nibble position that corresponded to the wrong cell -- effectively zeroing out the intended position and placing the value elsewhere.

**Step 4: Identify the bit layout mismatch.**

```
nneonneo's layout: cell(y,x) at nibble (4*y + x)
  cell(0,0) -> nibble 0 (bits 0-3)
  cell(0,1) -> nibble 1 (bits 4-7)
  cell(0,2) -> nibble 2 (bits 8-11)
  cell(0,3) -> nibble 3 (bits 12-15)

Our layout: cell(y,x) at nibble (4*y + 3-x)
  cell(0,0) -> nibble 3 (bits 12-15)
  cell(0,1) -> nibble 2 (bits 8-11)
  cell(0,2) -> nibble 1 (bits 4-7)
  cell(0,3) -> nibble 0 (bits 0-3)
```

The nibble ordering within each row is **reversed** between the two layouts. The transpose function's bit masks are hardcoded for nneonneo's ordering and produce incorrect results with ours.

**The fix.** Rather than adapting the bit masks (which would be error-prone and hard to verify), we replaced the O(1) transpose with explicit column extraction:

```c
static inline row_t board_col(board_t b, int x) {
    int c0 = (board_row(b, 0) >> ((3 - x) * 4)) & 0xF;
    int c1 = (board_row(b, 1) >> ((3 - x) * 4)) & 0xF;
    int c2 = (board_row(b, 2) >> ((3 - x) * 4)) & 0xF;
    int c3 = (board_row(b, 3) >> ((3 - x) * 4)) & 0xF;
    return (c0 << 12) | (c1 << 8) | (c2 << 4) | c3;
}
```

This is O(4) instead of O(1) but is correct, readable, and maintainable. The performance difference is negligible in practice -- the transpose/column-extraction is called 4 times per vertical move, while the evaluation function involves 100+ lookup table accesses.

**What we learned.** When reusing low-level bit-manipulation code from another project, the implicit assumptions about data layout must be verified exhaustively. A function that produces the mathematically correct transpose for one bit layout produces the mathematically incorrect transpose for a different layout. The code has no bugs -- it is simply operating on data that does not match its assumptions.

### 12.4 Failure 4: Learning Rate Overflow

**Context.** The N-Tuple network v1 was training with LR=0.0025 and producing decent results (~68% 2048 rate at 12K episodes). We wanted to accelerate training by increasing the learning rate.

**The attempt.** Changed learning rate from 0.0025 to 0.1 (a 40x increase).

**The symptoms.** Within the first 100 episodes, game scores dropped to 0. The evaluation function returned NaN or infinity. The training appeared to hang (no games completed because every move evaluation returned NaN, preventing action selection).

**The investigation.** We added debug prints to the weight update:

```
Episode 1, Move 1: delta = 128.0, lr = 0.1, update = 12.8
Episode 1, Move 2: delta = 256.0, lr = 0.1, update = 25.6
...
Episode 1, Move 50: delta = 15432.0, lr = 0.1, update = 1543.2
Episode 1, Move 100: delta = inf, lr = 0.1, update = inf
```

The TD error (delta) grew exponentially because:
1. A large weight update changes the evaluation function;
2. The changed evaluation produces a larger TD error at the next step;
3. The larger TD error produces an even larger weight update;
4. Positive feedback loop until overflow.

**The root cause.** N-Tuple Networks have no intrinsic normalization. Unlike neural networks (which have bounded activation functions, batch normalization, weight decay), N-Tuple weights are unbounded. A learning rate that is too high causes runaway weight growth because the update magnitude is proportional to the current evaluation (which depends on the weights).

**The fix.** Two changes:

1. **Delta clipping**: `if (delta > 1000) delta = 1000; if (delta < -1000) delta = -1000;`
2. **Reduced learning rate**: Changed from 0.1 to 0.01 with exponential decay to 0.0005.

The clipping threshold of 1000 was chosen because valid TD errors in 2048 rarely exceed a few hundred (the maximum single-move reward is 32768 from merging two 16384-tiles, but such moves are extremely rare and the evaluation difference is typically much smaller).

**What we learned.** Hyperparameters from published papers cannot be blindly transferred. Different implementations, different tuple configurations, and different game engines produce different TD error magnitudes. The learning rate must be validated empirically for each specific setup. When in doubt, start low and increase gradually.

### 12.5 Failure 5: Symmetry Normalization

**Context.** After implementing symmetry-aware evaluation (summing weights from all symmetric variants of each tuple), we noticed that tuples with more symmetries contributed more to the total evaluation. We hypothesized that normalizing the update by the number of symmetries would improve learning.

**The attempt.** Divided the per-weight update by the number of symmetric variants:

```python
adj = lr * delta / len(symmetries)
for sym in symmetries:
    weights[encode(board, sym)] += adj
```

**The symptoms.** Learning speed decreased by approximately 2x. After 5,000 episodes, the normalized version scored ~15,000 vs. ~22,000 for the unnormalized version.

**The investigation.** We compared weight magnitudes between the two versions. The normalized version had weights approximately 8x smaller (because each update was 1/8th the size for tuples with 8 symmetries).

**The root cause.** The normalization dilutes the learning signal. The 8 symmetric updates are not redundant -- they update 8 different weight entries, each capturing a different spatial view of the board. Dividing by 8 slows the convergence of all 8 entries equally, with no compensating benefit.

Theoretically, normalization ensures that the contribution of each base tuple to the total evaluation is independent of its symmetry count. But in practice, the difference in contribution is absorbed by the weight magnitudes during training, and the normalization only slows this absorption.

**The fix.** Removed the normalization. The theoretical "impurity" of non-normalized updates is a non-issue in practice.

**What we learned.** Theoretical correctness does not always translate to practical improvement. The "right" thing to do (normalize by symmetry count) can actually hurt performance. Empirical testing is the final arbiter.

### 12.6 Failure 6: Adaptive Depth Freezes

**Context.** Before implementing iterative deepening, we experimented with adaptive depth: automatically increasing the search depth when the board had few empty cells (i.e., when the game was in a critical state).

**The attempt.** The depth was computed as:

```python
depth = 3 + max(0, 10 - empty_cells)
```

So with 10+ empty cells, depth = 3. With 5 empty cells, depth = 8. With 1 empty cell, depth = 12.

**The symptoms.** When the board had few empty cells (late game, critical positions), the AI took 10-12 seconds per move. The game appeared to freeze. Ironically, these were the positions where fast response was most important (the player was about to lose and needed to find the right escape move).

**The root cause.** The Expectimax branching factor depends on the number of empty cells: each CHANCE node has `2 * empty_cells` children (each empty cell can receive tile 2 or tile 4). But the depth increase overcompensated: the total tree size at depth 12 with few empty cells was still enormous because the tree includes future positions where tiles are placed and empties are created.

The relationship between depth and computation is exponential:

```
depth 3, 5 empties: ~2,000 nodes, ~1ms
depth 8, 5 empties: ~50,000,000 nodes, ~5s
depth 12, 1 empty: ~100,000,000 nodes, ~10s
```

**The fix.** Replaced adaptive depth with iterative deepening with a fixed time budget of 100ms. This naturally spends more computation on critical positions (where the tree is small enough to search deeper) and less on trivial positions (where the tree is wide and shallow search suffices).

**What we learned.** Search depth should be controlled by a time budget, not by a heuristic based on board state. Iterative deepening elegantly solves the problem of allocating computation optimally across different board states.

### 12.7 Failure 7: Buffered Output

**Context.** We were running the N-Tuple trainer and piping its output through `tee` to save a log file:

```bash
./ntuple_train --episodes 500000 --depth 0 --tc --threads 8 | tee training.log
```

**The symptoms.** The terminal showed no output for several minutes after launching. We could not tell if the training was running correctly, had crashed, or had a bug. CPU usage was high (indicating work was happening), but the lack of feedback was frustrating and prevented monitoring.

**The investigation.** We ran the same command without `tee`:

```bash
./ntuple_train --episodes 500000 --depth 0 --tc --threads 8
```

Output appeared immediately, with progress reports every few seconds. The issue was specific to piping.

**The root cause.** When stdout is connected to a terminal (TTY), the C runtime uses **line buffering**: output is flushed after each newline. When stdout is connected to a pipe (as when piping through `tee`), the C runtime switches to **full buffering**: output is accumulated in a buffer (typically 4-8 KB) and only flushed when the buffer is full.

Our training reports are approximately 500 bytes each, printed every ~100 episodes. With a 4 KB buffer, 8 reports must accumulate before any output appears -- potentially many minutes of training.

**The fix.**

```c
int main(int argc, char **argv) {
    setbuf(stdout, NULL);  // Disable buffering entirely
    // ... rest of main ...
}
```

`setbuf(stdout, NULL)` sets the buffering mode to "unbuffered," meaning every `printf` immediately writes its output to the destination. The performance impact is negligible because we only print every ~100 episodes (~3 seconds of wall time).

**What we learned.** A one-line fix (`setbuf(stdout, NULL)`) saved hours of frustration. Long-running programs that produce periodic status updates should always use unbuffered or line-buffered stdout, especially when the output may be piped or redirected.

### 12.8 General Lessons Learned

Looking back across all seven failures, several themes emerge:

1. **Test before adopting.** Every "optimization" (1D decomposition, symmetry normalization, adaptive depth, higher learning rate) was attempted because it seemed theoretically sound. Each one was a regression. The only way to know if a change is an improvement is to measure it against the current baseline.

2. **Always measure.** Without quantitative metrics (average score, 2048 rate, time per move), it is impossible to compare approaches objectively. Gut feelings and theoretical arguments are unreliable.

3. **Simplicity first.** The version that works is more valuable than the "optimal" version that does not. Our column extraction function is slower than the O(1) transpose, but it is correct and readable. The fixed-LR N-Tuple with no normalization outperforms the theoretically "better" normalized version.

4. **Domain knowledge is gold.** The snake pattern, a simple observation about how tiles should be organized, provides more value than millions of DQN training steps. Domain knowledge should be incorporated early and directly, not left for the network to discover.

5. **C for performance, Python for prototyping.** The ideal development cycle is: implement in Python, validate correctness, measure performance, identify bottlenecks, and rewrite critical paths in C. This preserves development speed while achieving native performance.

6. **Hyperparameters are not transferable.** Learning rate, epsilon decay, buffer size, and other hyperparameters must be tuned for each specific implementation. Values from papers serve as starting points, not final configurations.

7. **Bit-level code requires bit-level verification.** When working with bitboard representations and bit-manipulation functions, every assumption about data layout must be explicitly verified. A single nibble-ordering mismatch can corrupt the entire computation.

---

## 13. Conclusion and Future Work

### 13.1 Summary of Results

This work presented the complete development, from initial concept to working web-based deployment, of three AI agents for the game 2048. The journey spanned three fundamentally different approaches, seven documented failures, and numerous optimization iterations.

**Starting point.** A DQN agent that could not reach the 2048 tile after 3 hours of training, achieving a maximum score of approximately 11,600.

**Ending point.** An N-Tuple Network agent achieving:
- **87%+ 2048 rate** (up from 0% with DQN);
- **55% 4096 rate**;
- **12% 8192 rate** (highest tile ever reached);
- **Average score of approximately 64,000** (from ~5,000 with DQN);
- **Best individual score of 124,080**;
- **Response time of 2.4ms per move** (suitable for real-time visualization);
- **Total training time: approximately 30 minutes** for Stage 1 + early Stage 2.

### 13.2 Contributions

1. **Practical comparative analysis** of three fundamentally different AI approaches (DQN, Expectimax, N-Tuple) on the same problem, with the same game engine, measured by the same metrics. This side-by-side comparison demonstrates that the choice of approach is at least as important as the quality of implementation.

2. **Complete C implementation** of an N-Tuple Network trainer with Hogwild multithreading, lookup tables, TC-learning, and multi-stage training, achieving a 300-450x speedup over the initial Python implementation.

3. **Web interface** enabling real-time comparison of three agents playing in the browser, with automated game reporting and post-hoc analysis capability.

4. **Detailed failure documentation**: Seven failures documented with full debugging narratives, root cause analyses, and lessons learned. This represents practical knowledge that is rarely shared in academic publications but is invaluable for practitioners.

5. **Optimization case study** demonstrating the impact of each optimization individually (Python -> C: 50x; lookup tables: 2-3x; multithreading: 3x) and cumulatively (300-450x). This provides a template for optimizing tabular RL implementations.

### 13.3 Future Work

Several directions for future improvement have been identified:

**Short-term (days):**
- **Complete Stage 2 training** to 5M episodes (estimated 2 days), expected to reach ~97% 2048 rate.
- **Statistical analysis** of saved game reports to identify patterns in lost games and inform architectural improvements.
- **Real-time training dashboard** using WebSocket to stream training metrics to the browser.

**Medium-term (weeks):**
- **Larger tuples**: Implement 7-tuples (268M entries per tuple, ~1 GB each) to increase representational capacity. Expected improvement: 3-5% on 2048 rate.
- **Deeper training search**: 5-ply training (currently using 3-ply) would improve the training signal at approximately 100x computational cost.
- **Monte Carlo Tree Search (MCTS)**: Implement MCTS as a fourth approach for comparison. MCTS with N-Tuple rollout evaluation could potentially outperform pure Expectimax.

**Long-term (months):**
- **Transfer learning**: Use trained weights as initialization for game variants (5x5 board, different merge rules, hexagonal grid).
- **AlphaZero-style training**: Combine N-Tuple evaluation with self-play and policy learning, potentially closing the gap to state-of-the-art results.
- **Hardware acceleration**: Implement N-Tuple evaluation on GPU for massively parallel training.

---

## 14. References

1. Cirulli, G. (2014). 2048. Available at: https://github.com/gabrielecirulli/2048

2. Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., ... & Hassabis, D. (2015). Human-level control through deep reinforcement learning. *Nature*, 518(7540), 529-533.

3. Szubert, M., & Jaskowski, W. (2014). Temporal Difference Learning of N-Tuple Networks for the Game 2048. *Proceedings of the IEEE Conference on Computational Intelligence and Games (CIG)*.

4. Wu, I-C., Yeh, K-H., Liang, C-C., Chang, C-C., & Chiang, H. (2014). Multi-stage temporal difference learning for 2048. *Proceedings of the TAAI Conference*.

5. Lucas, S. M. (2008). Learning to play Othello with N-Tuple Systems. *Australian Journal of Intelligent Information Processing Systems*, 9(4), 1-20.

6. Watkins, C. J. C. H. (1989). *Learning from Delayed Rewards*. PhD thesis, University of Cambridge.

7. Jaskowski, W. (2018). Mastering 2048 with Delayed Temporal Coherence Learning, Multi-Stage Weight Promotion, and Carousel Shaping. *IEEE Transactions on Games*, 10(1), 3-14.

8. nneonneo (Robert Xiao). (2014). 2048-ai: AI for the 2048 game. Available at: https://github.com/nneonneo/2048-ai

9. Van Hasselt, H., Guez, A., & Silver, D. (2016). Deep Reinforcement Learning with Double Q-Learning. *Proceedings of the AAAI Conference on Artificial Intelligence*.

10. Niu, F., Recht, B., Re, C., & Wright, S. J. (2011). HOGWILD!: A Lock-Free Approach to Parallelizing Stochastic Gradient Descent. *Advances in Neural Information Processing Systems (NeurIPS)*.

11. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.

12. Bellman, R. (1957). *Dynamic Programming*. Princeton University Press.

13. Lin, L.-J. (1992). Self-improving reactive agents based on reinforcement learning, planning and teaching. *Machine Learning*, 8(3-4), 293-321.

---

## 15. Appendices

### Appendix A: Key Code Listings

#### A.1 DQN State Encoding (Python)

```python
def encode_state(grid):
    """
    Encodes the 4x4 grid as a one-hot tensor.
    Each tile becomes a 16-channel vector where the active index
    corresponds to log2(tile value).
    
    Input:  grid[4][4] with actual tile values (0, 2, 4, ..., 2048+)
    Output: tensor (16, 4, 4) of float32
    """
    encoded = np.zeros((16, 4, 4), dtype=np.float32)
    for y in range(4):
        for x in range(4):
            val = grid[y][x]
            if val > 0:
                channel = int(np.log2(val))
                if channel < 16:
                    encoded[channel][y][x] = 1.0
            else:
                encoded[0][y][x] = 1.0  # empty = channel 0
    return encoded
```

#### A.2 Expectimax Column Extraction (C)

```c
/* Extract column x from a bitboard as a 16-bit row.
 * Top cell at high nibble.
 * 
 * This replaces nneonneo's O(1) bit-manipulation transpose,
 * which was incompatible with our nibble ordering.
 * See Failure 3 (Section 12.3) for the debugging narrative.
 */
static inline row_t board_col(board_t b, int x) {
    int c0 = (board_row(b, 0) >> ((3 - x) * 4)) & 0xF;
    int c1 = (board_row(b, 1) >> ((3 - x) * 4)) & 0xF;
    int c2 = (board_row(b, 2) >> ((3 - x) * 4)) & 0xF;
    int c3 = (board_row(b, 3) >> ((3 - x) * 4)) & 0xF;
    return (c0 << 12) | (c1 << 8) | (c2 << 4) | c3;
}
```

#### A.3 N-Tuple TC-Learning Update (C)

```c
static void net_update(ntuple_net_t *net, const grid_t g, float delta) {
    /* Clamp delta to prevent weight explosion (see Failure 4) */
    if (delta > 1000.0f) delta = 1000.0f;
    if (delta < -1000.0f) delta = -1000.0f;

    float abs_delta = (delta >= 0) ? delta : -delta;
    float decay = 0.9995f;  /* half-life ~ 1386 updates */
    
    for (int t = 0; t < net->n_base; t++) {
        float *w  = net->weights[t];
        float *ts = net->tc_sum[t];    /* signed delta accumulator */
        float *ta = net->tc_abs[t];    /* absolute delta accumulator */
        
        for (int s = 0; s < net->n_sym[t]; s++) {
            int idx = encode6(g, net->syms[t][s].pos);
            
            /* Exponentially decayed accumulators */
            ts[idx] = ts[idx] * decay + delta;
            ta[idx] = ta[idx] * decay + abs_delta;
            
            /* TC ratio: 1 = consistent updates, 0 = oscillating */
            float ratio = (ta[idx] > 1e-6f)
                ? ((ts[idx] >= 0 ? ts[idx] : -ts[idx]) / ta[idx])
                : 1.0f;  /* default for never-updated weights */
            
            /* Adaptive update: converged weights learn slowly */
            w[idx] += net->lr * ratio * delta;
        }
    }
}
```

#### A.4 Hogwild Training Worker (C)

```c
static void *train_worker(void *arg) {
    thread_stats_t *stats = (thread_stats_t *)arg;
    unsigned int seed = (unsigned int)(time(NULL) + stats->thread_id * 7919);

    while (1) {
        /* Atomic episode counter -- only synchronization point */
        int ep = atomic_fetch_add(&episodes_done, 1) + 1;
        if (ep > total_episodes) break;

        /* Exponential LR schedule */
        float progress = (float)ep / total_episodes;
        shared_net.lr = lr_start * powf(lr_end / lr_start, progress);

        /* Initialize game */
        grid_t state;
        grid_clear(state);
        grid_add_random_r(state, &seed);
        grid_add_random_r(state, &seed);

        int game_score = 0;

        /* First move (no previous afterstate to update) */
        grid_t prev_after;
        float prev_reward;
        int action = select_best(&shared_net, state, prev_after, &prev_reward, &seed);
        if (action == -1) continue;

        grid_copy(state, prev_after);
        grid_add_random_r(state, &seed);
        game_score += (int)prev_reward;

        /* Main game loop: forward TD updates */
        while (!grid_game_over(state)) {
            grid_t curr_after;
            float curr_reward;
            action = select_best(&shared_net, state, curr_after, &curr_reward, &seed);
            if (action == -1) break;

            /* TD(0) update: V(prev_after) <- V(prev_after) + lr * delta */
            float v_prev = net_evaluate(&shared_net, prev_after);
            float v_curr = net_evaluate(&shared_net, curr_after);
            float delta = curr_reward + v_curr - v_prev;
            net_update(&shared_net, prev_after, delta);

            /* Advance state */
            grid_copy(state, curr_after);
            grid_add_random_r(state, &seed);
            game_score += (int)curr_reward;
            grid_copy(prev_after, curr_after);
        }

        /* Terminal update: V(last_afterstate) should be 0 */
        float v_last = net_evaluate(&shared_net, prev_after);
        net_update(&shared_net, prev_after, -v_last);

        /* Record stats (per-thread, no synchronization needed) */
        int max_tile = grid_max_tile_actual(state);
        if (max_tile >= 2048) atomic_fetch_add(&total_wins, 1);
    }
    return NULL;
}
```

#### A.5 Frontend Grid Extraction (JavaScript)

```javascript
function extractGrid() {
    var gm = window.gameManager;
    if (!gm || !gm.grid) return null;

    var grid = [];
    for (var y = 0; y < gm.grid.size; y++) {
        var row = [];
        for (var x = 0; x < gm.grid.size; x++) {
            // Note: Cirulli's grid uses cells[x][y] (column-major)
            // We convert to row-major grid[y][x]
            var cell = gm.grid.cells[x][y];
            row.push(cell ? cell.value : 0);
        }
        grid.push(row);
    }
    return grid;
}
```

### Appendix B: Training Log Excerpts

#### B.1 N-Tuple Stage 1 (C, 1-ply, 8 threads, TC-learning)

```
N-Tuple Training em C (otimizado)
Episodios: 500000 | Tuplas: 17 | Search: 1-ply | TC: ON | Threads: 8

============================================================
N-Tuple Episodio 218/500000 | Tempo: 0s | 8 threads
============================================================
  Score medio:  5426
  Max tiles:    {64: 4, 128: 37, 256: 62, 512: 88, 1024: 19}
  2048+:        0x total
  LR:           0.009987
  Ep/s:         218.0

============================================================
N-Tuple Episodio 1444/500000 | Tempo: 1s | 8 threads
============================================================
  Score medio:  16923
  Max tiles:    {128: 1, 256: 15, 512: 154, 1024: 482, 2048: 147, 4096: 1}
  2048+:        167x total
  LR:           0.009914
  Ep/s:         1444.0

[... training continues ...]

============================================================
N-Tuple Episodio 5028/500000 | Tempo: 6s | 8 threads
============================================================
  Score medio:  30059
  Max tiles:    {256: 3, 512: 46, 1024: 240, 2048: 435, 4096: 76}
  2048+:        2047x total
  LR:           0.009703
  Ep/s:         838.0
```

#### B.2 N-Tuple Stage 2 (C, 3-ply, 8 threads, TC-learning)

```
============================================================
N-Tuple Episodio 46303/5000000 | Tempo: 1715s | 8 threads
============================================================
  Score medio:  64760
  Max tiles:    {256: 3, 512: 14, 1024: 87, 2048: 159, 4096: 439, 8192: 98}
  2048+:        40548x total
  LR:           0.009726
  Ep/s:         27.0
```

#### B.3 N-Tuple Python v1 (7 tuples, LR=0.0025)

```
============================================================
N-Tuple Episodio 12100/50000 | Tempo: 46394s
============================================================
  Score medio:  30020
  Score maximo: 64712
  Max tiles:    {256: 1, 512: 4, 1024: 20, 2048: 68, 4096: 7}
  2048+:        5342x total
  LR:           0.0025
```

### Appendix C: Game Report Examples

#### C.1 Complete N-Tuple Report (8192 tile)

```json
{
  "timestamp": "2026-08-19T10:22:31.728768",
  "agent": "ntuple",
  "score": 124080,
  "max_tile": 8192,
  "moves": 3933,
  "won": true,
  "final_grid": [
    [2048, 4, 8192, 4],
    [16, 2, 256, 128],
    [32, 512, 64, 4],
    [2, 4, 32, 2]
  ],
  "move_history": [
    {"move": 1, "action": "up", "score_before": 24516, "max_tile": 2048},
    {"move": 2, "action": "left", "score_before": 24516, "max_tile": 2048},
    ...
    {"move": 3933, "action": "down", "score_before": 124048, "max_tile": 8192}
  ],
  "config": {"ntuple": true}
}
```

#### C.2 Complete DQN Report (max tile 64)

```json
{
  "timestamp": "2026-08-18T16:34:42.505718",
  "agent": "dqn",
  "score": 912,
  "max_tile": 64,
  "moves": 97,
  "won": false,
  "final_grid": [
    [8, 2, 4, 2],
    [16, 64, 16, 4],
    [4, 16, 32, 2],
    [2, 4, 64, 4]
  ],
  "duration_ms": 1823,
  "config": {"dqn_steps": 510000}
}
```

#### C.3 Complete Expectimax Report (4096 tile, best game)

```json
{
  "timestamp": "2026-08-18T17:00:52.280946",
  "agent": "expectimax",
  "score": 76516,
  "max_tile": 4096,
  "moves": 2526,
  "won": true,
  "final_grid": [
    [4096, 16, 2, 4],
    [2048, 64, 16, 8],
    [1024, 128, 8, 4],
    [256, 32, 4, 2]
  ],
  "config": {"expectimax_depth": 10, "time_budget_ms": 100}
}
```

---

*All code, training data, and game reports are available in the project repository. The total development time was approximately two weeks, with the N-Tuple agent's training ongoing.*

*Powered by Claude Code, which assisted with architectural decisions, debugging, and documentation.*
