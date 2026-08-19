"""
Unified benchmark script for all agents.

Runs N games per agent under identical conditions and produces
statistical analysis for paper publication.

Usage:
    python benchmark.py --games 1000 --agents expectimax ntuple attention dqn

Output: benchmark_results.json + console summary with confidence intervals.
"""

import argparse
import json
import os
import time
import ctypes
import numpy as np
from datetime import datetime

from game import Game2048
from expectimax_agent import fast_move


def run_expectimax(n_games, time_budget_ms=100):
    """Benchmark Expectimax agent."""
    from expectimax_native import ExpectimaxAgent
    agent = ExpectimaxAgent(depth=10, time_budget_ms=time_budget_ms)
    return _run_agent(agent, n_games, "expectimax",
                      lambda g: agent.select_action(g)[0])


def run_ntuple(n_games, search_depth=2):
    """Benchmark N-Tuple agent via C."""
    lib_path = os.path.join(os.path.dirname(__file__), 'ntuple_c.so')
    lib = ctypes.CDLL(lib_path)
    lib.ntuple_load.argtypes = [ctypes.c_char_p]
    lib.ntuple_load.restype = ctypes.c_int
    lib.ntuple_select_action.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    lib.ntuple_select_action.restype = ctypes.c_int

    for name in ['ntuple_best.bin', 'ntuple_latest.bin']:
        path = os.path.join(os.path.dirname(__file__), 'checkpoints', name)
        if os.path.exists(path):
            lib.ntuple_load(path.encode())
            break

    def select(grid):
        flat = grid.flatten().astype(ctypes.c_int)
        arr = (ctypes.c_int * 16)(*flat)
        return lib.ntuple_select_action(arr, search_depth)

    return _run_agent(None, n_games, "ntuple", select)


def run_attention(n_games):
    """Benchmark Attention N-Tuple agent."""
    from attention_ntuple import AttentionNTupleAgent

    ntuple_path = None
    for name in ['ntuple_best.bin', 'ntuple_latest.bin']:
        p = os.path.join(os.path.dirname(__file__), 'checkpoints', name)
        if os.path.exists(p):
            ntuple_path = p
            break

    attn_path = None
    for name in ['attention_best.pt', 'attention_latest.pt']:
        p = os.path.join(os.path.dirname(__file__), 'checkpoints', name)
        if os.path.exists(p):
            attn_path = p
            break

    if not attn_path:
        print("  Attention model not found, skipping")
        return None

    agent = AttentionNTupleAgent(ntuple_path=ntuple_path)
    agent.load(attn_path)

    # 3-ply search
    def select(grid):
        best_action = 0
        best_value = -1e18
        for d in range(4):
            after, reward, moved = fast_move(grid, d)
            if not moved:
                continue
            empties = list(zip(*np.where(after == 0)))
            if not empties:
                value = reward + agent.evaluate(after)
            else:
                sample = empties if len(empties) <= 4 else \
                    [empties[i] for i in np.random.choice(len(empties), 4, replace=False)]
                total = 0.0
                for y, x in sample:
                    for tv, prob in [(2, 0.9), (4, 0.1)]:
                        g2 = after.copy()
                        g2[y, x] = tv
                        best_resp = -1e18
                        for d2 in range(4):
                            a3, r2, m2 = fast_move(g2, d2)
                            if not m2:
                                continue
                            v = r2 + agent.evaluate(a3)
                            if v > best_resp:
                                best_resp = v
                        if best_resp == -1e18:
                            best_resp = agent.evaluate(g2)
                        total += prob * best_resp
                value = reward + total / len(sample)
            if value > best_value:
                best_value = value
                best_action = d
        return best_action

    return _run_agent(None, n_games, "attention", select)


def run_dqn(n_games):
    """Benchmark DQN agent."""
    from dqn_agent import DQNAgent, encode_state

    agent = DQNAgent()
    for name in ['dqn_best.pt', 'dqn_latest.pt', 'best.pt']:
        path = os.path.join(os.path.dirname(__file__), 'checkpoints', name)
        if os.path.exists(path):
            agent.load(path)
            break

    def select(grid):
        encoded = encode_state(grid)
        game = Game2048()
        game.grid = grid.copy()
        valid = game.get_valid_moves()
        if not valid:
            return 0
        return agent.select_action(encoded, valid, training=False)

    return _run_agent(None, n_games, "dqn", select)


def _run_agent(agent, n_games, name, select_fn):
    """Run N games and collect statistics."""
    scores = []
    max_tiles = []
    move_counts = []
    move_dist = [0, 0, 0, 0]  # UP, RIGHT, DOWN, LEFT

    start = time.time()

    for i in range(n_games):
        game = Game2048()
        game.reset()
        moves = 0

        while not game.over:
            action = select_fn(game.grid)
            _, moved = game.move(action)
            if not moved:
                for d in range(4):
                    _, m = game.move(d)
                    if m:
                        break
            move_dist[action] += 1
            moves += 1

        scores.append(game.score)
        max_tiles.append(game.max_tile())
        move_counts.append(moves)

        if (i + 1) % 100 == 0:
            elapsed = time.time() - start
            print(f"  [{name}] {i+1}/{n_games} | "
                  f"avg={np.mean(scores):.0f} | "
                  f"2048+={sum(1 for t in max_tiles if t >= 2048)}/{i+1} | "
                  f"{elapsed:.0f}s")

    elapsed = time.time() - start
    scores = np.array(scores)
    max_tiles = np.array(max_tiles)

    # Tile rate computation
    tile_rates = {}
    for tile in [512, 1024, 2048, 4096, 8192, 16384, 32768]:
        count = int(np.sum(max_tiles >= tile))
        tile_rates[str(tile)] = {
            "count": count,
            "rate": float(count / n_games),
            "rate_pct": f"{count / n_games * 100:.1f}%"
        }

    # Move distribution
    total_moves = sum(move_dist)
    move_pct = {
        "UP": f"{move_dist[0]/total_moves*100:.1f}%",
        "RIGHT": f"{move_dist[1]/total_moves*100:.1f}%",
        "DOWN": f"{move_dist[2]/total_moves*100:.1f}%",
        "LEFT": f"{move_dist[3]/total_moves*100:.1f}%",
    }

    # 95% confidence interval
    ci_95 = 1.96 * np.std(scores) / np.sqrt(n_games)

    result = {
        "agent": name,
        "n_games": n_games,
        "elapsed_seconds": round(elapsed, 1),
        "score": {
            "mean": round(float(np.mean(scores)), 1),
            "median": round(float(np.median(scores)), 1),
            "std": round(float(np.std(scores)), 1),
            "ci_95": round(float(ci_95), 1),
            "min": int(np.min(scores)),
            "max": int(np.max(scores)),
            "formatted": f"{np.mean(scores):.0f} ± {ci_95:.0f} (95% CI)"
        },
        "tile_rates": tile_rates,
        "move_distribution": move_pct,
        "avg_moves_per_game": round(float(np.mean(move_counts)), 0),
        "timestamp": datetime.now().isoformat(),
    }

    # Print summary
    print(f"\n{'='*60}")
    print(f"  {name.upper()} — {n_games} games in {elapsed:.0f}s")
    print(f"{'='*60}")
    print(f"  Score: {result['score']['formatted']}")
    print(f"  Median: {result['score']['median']:.0f} | Max: {result['score']['max']}")
    for tile, data in tile_rates.items():
        if data['count'] > 0:
            print(f"  {tile}+ rate: {data['rate_pct']} ({data['count']}/{n_games})")
    print(f"  Moves: {move_pct}")
    print(f"  Avg moves/game: {result['avg_moves_per_game']:.0f}")

    return result


def main():
    parser = argparse.ArgumentParser(description='Benchmark all agents')
    parser.add_argument('--games', type=int, default=1000)
    parser.add_argument('--agents', nargs='+',
                        default=['expectimax', 'ntuple', 'attention', 'dqn'],
                        choices=['expectimax', 'ntuple', 'attention', 'dqn'])
    parser.add_argument('--output', type=str, default='benchmark_results.json')
    args = parser.parse_args()

    print(f"Benchmark: {args.games} games per agent")
    print(f"Agents: {', '.join(args.agents)}\n")

    runners = {
        'expectimax': run_expectimax,
        'ntuple': run_ntuple,
        'attention': run_attention,
        'dqn': run_dqn,
    }

    results = {}
    for agent_name in args.agents:
        print(f"\n>>> Running {agent_name}...")
        result = runners[agent_name](args.games)
        if result:
            results[agent_name] = result

    # Save results
    output_path = os.path.join(os.path.dirname(__file__), args.output)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
