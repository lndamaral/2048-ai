# 2048 AI — Three Approaches to Game Intelligence

A comparative study of three AI approaches for the game 2048: **Deep Q-Network (DQN)**, **Expectimax with hand-crafted heuristics**, and **N-Tuple Networks with TD-learning**.

Built on top of the [original 2048 game](https://github.com/gabrielecirulli/2048) by Gabriele Cirulli.

## Results

| Agent | 2048+ Rate | 4096 Rate | 8192 Rate | Avg Score | ms/move |
|-------|-----------|-----------|-----------|-----------|---------|
| **DQN** | ~0%* | 0% | 0% | ~2,000 | ~1ms |
| **Expectimax** | ~80% | ~60% | 0% | ~40,000 | ~100ms |
| **N-Tuple** | **~87%** | **55%** | **12%** | **~66,000** | **2.4ms** |

*DQN requires significantly more training to converge. Results shown are from early training.

## Architecture

```
Browser (localhost:8080)
  │
  ├── Original 2048 game (HTML/CSS/JS)
  ├── AI Player UI (js/ai_player.js)
  │     └── Agent selector: DQN | Expectimax | N-Tuple
  │
  └── HTTP API (localhost:8081)
        └── Flask server (ai/server.py)
              ├── DQN Agent      → PyTorch neural network
              ├── Expectimax     → C engine with iterative deepening
              └── N-Tuple Agent  → C engine with learned lookup tables
```

## Quick Start

```bash
# Setup
python3 -m venv venv && source venv/bin/activate
pip install torch numpy flask flask-cors matplotlib

# Compile C engines
cd ai && make

# Train N-Tuple (recommended — ~15 min for basic, ~14h for best)
./ntuple_train --episodes 500000 --depth 0 --tc --threads 8    # Stage 1
./ntuple_train --episodes 5000000 --depth 1 --tc --threads 8   # Stage 2

# Start servers
python3 -u server.py --time-budget 100 &
cd .. && python3 -m http.server 8080 &

# Open http://localhost:8080 and click "AI Play"
```

## The Three Agents

### 1. DQN (Deep Q-Network)

Pure deep reinforcement learning. A convolutional neural network learns to play by trial and error.

- **Architecture**: Dueling DQN + Prioritized Experience Replay + Noisy Networks
- **Training**: `python3 ai/train.py --episodes 50000`
- **Strengths**: Learns entirely from scratch, no domain knowledge needed
- **Weaknesses**: Very slow to converge for 2048

### 2. Expectimax

Tree search with hand-crafted heuristics. No training required.

- **Engine**: C with bitboard, iterative deepening, transposition table (4M entries)
- **Heuristics**: Snake pattern, monotonicity, smoothness, empty cells, corner bonus
- **Calibration**: `./ai/expectimax_calibrate --games 200 --threads 8`
- **Strengths**: Works immediately, no training needed
- **Weaknesses**: Limited by heuristic quality (fixed ceiling)

### 3. N-Tuple Network (State of the Art)

Learned evaluation function + tree search. Combines the best of both worlds.

- **Architecture**: 17 tuples of 6 positions, 8 symmetries, ~285M learned weights
- **Training**: Forward TD(0) afterstate learning with TC-learning
- **Search**: 5-ply Expectimax during play, 3-ply during training
- **Training**: `./ai/ntuple_train --episodes 5000000 --depth 1 --tc --threads 8`
- **Strengths**: Fast (2.4ms/move), learns from experience, highest performance

## File Structure

```
├── index.html                    # Game UI
├── js/
│   ├── ai_player.js              # AI integration (3 agents)
│   ├── game_manager.js           # Game logic (original)
│   └── ...
├── ai/
│   ├── Makefile                  # Build all C components
│   ├── server.py                 # Flask API server
│   ├── game.py                   # Python game engine
│   │
│   ├── dqn_agent.py              # DQN: Dueling DQN + PER + Noisy Nets
│   ├── train.py                  # DQN: Training script
│   │
│   ├── expectimax_c.c            # Expectimax: C engine
│   ├── expectimax_native.py      # Expectimax: Python wrapper
│   ├── expectimax_agent.py       # Expectimax: Python engine (legacy)
│   ├── expectimax_calibrate.c    # Expectimax: Weight auto-calibrator
│   │
│   ├── ntuple_train.c            # N-Tuple: C trainer (multithreaded)
│   ├── ntuple_c.c                # N-Tuple: C player (shared library)
│   ├── ntuple_agent.py           # N-Tuple: Python trainer (legacy)
│   │
│   ├── analysis.py               # Chart generation
│   ├── charts/                   # Generated analysis charts
│   ├── checkpoints/              # Model weights (not in git)
│   ├── reports/                  # Game reports (JSON)
│   └── docs/
│       ├── TCC_AI_2048.md        # Summary document
│       └── EVOLUTION.md          # Evolution narrative
└── style/                        # Game CSS (original)
```

## Training Commands

### N-Tuple (recommended)

```bash
cd ai

# Stage 1: Fast pattern learning (~15 min)
caffeinate -dims ./ntuple_train --episodes 500000 --depth 0 --tc --threads 8

# Stage 2: Deep refinement (~14h, run overnight)
caffeinate -dims ./ntuple_train --episodes 5000000 --depth 1 --tc --threads 8
```

### DQN

```bash
cd ai
python3 -u train.py --episodes 50000 2>&1 | tee dqn_training.log
```

### Expectimax Calibration

```bash
cd ai
./expectimax_calibrate --games 200 --depth 1 --threads 8
```

## Game Reports

Every game played in the browser automatically saves a JSON report to `ai/reports/`:

```json
{
  "agent": "ntuple",
  "score": 76516,
  "max_tile": 4096,
  "moves": 2526,
  "won": true,
  "move_history": [...]
}
```

## References

1. Mnih et al. (2015). Human-level control through deep reinforcement learning. *Nature*.
2. Szubert & Jaśkowski (2014). TD Learning of N-Tuple Networks for 2048. *IEEE CIG*.
3. Wu et al. (2014). Multi-stage TD learning for 2048. *TAAI*.

## License

The original 2048 game is by [Gabriele Cirulli](https://github.com/gabrielecirulli/2048) under MIT License.
AI components added by Leonardo.
