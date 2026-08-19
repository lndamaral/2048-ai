"""
API server for the AI to play 2048 in the browser.
Supports three agents: DQN, Expectimax, and N-Tuple.

Usage:
    python ai/server.py [--port 8081] [--time-budget 100]
"""

import argparse
import json
import os
from datetime import datetime
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS

from dqn_agent import DQNAgent, encode_state
from expectimax_native import ExpectimaxAgent
from ntuple_agent import NTupleNetwork
from expectimax_agent import fast_move
from attention_ntuple import AttentionNTupleAgent
import ctypes

app = Flask(__name__)
CORS(app)

dqn_agent = None
expectimax_agent = ExpectimaxAgent(depth=3)
ntuple_net = None
ntuple_c_lib = None
attention_agent = None


def _is_valid_move(grid, direction):
    from game import Game2048
    g = Game2048()
    g.grid = grid.copy()
    return g.is_valid_move(direction)


def _attention_chance_node(grid, depth):
    """CHANCE node for attention agent 3-ply search."""
    empties = list(zip(*np.where(grid == 0)))
    if not empties:
        return attention_agent.evaluate(grid)

    sample = empties if len(empties) <= 4 else [empties[i] for i in np.random.choice(len(empties), 4, replace=False)]
    total = 0.0
    for y, x in sample:
        for tile_val, prob in [(2, 0.9), (4, 0.1)]:
            g2 = grid.copy()
            g2[y, x] = tile_val
            if depth <= 0:
                total += prob * attention_agent.evaluate(g2)
            else:
                total += prob * _attention_max_node(g2, depth)
    return total / len(sample)


def _attention_max_node(grid, depth):
    """MAX node for attention agent 3-ply search."""
    best = -1e18
    any_moved = False
    for d in range(4):
        after, reward, moved = fast_move(grid, d)
        if not moved:
            continue
        any_moved = True
        v = reward + _attention_chance_node(after, depth - 1)
        if v > best:
            best = v
    return attention_agent.evaluate(grid) if not any_moved else best


def _attention_select_action(grid):
    """Attention agent with 3-ply search."""
    best_action = 0
    best_value = -1e18
    for d in range(4):
        after, reward, moved = fast_move(grid, d)
        if not moved:
            continue
        value = reward + _attention_chance_node(after, 0)  # depth=0 → 3-ply total
        if value > best_value:
            best_value = value
            best_action = d
    return best_action


def _ntuple_select_action(grid):
    """N-Tuple search via C — instant 3-ply."""
    if ntuple_c_lib:
        flat = grid.flatten().astype(ctypes.c_int)
        arr = (ctypes.c_int * 16)(*flat)
        return ntuple_c_lib.ntuple_select_action(arr, 2)  # search_depth=2 → 5-ply
    # Fallback Python 1-ply
    best_action = 0
    best_value = -1e18
    for d in range(4):
        after, reward, moved = fast_move(grid, d)
        if not moved:
            continue
        value = reward + ntuple_net.evaluate(after)
        if value > best_value:
            best_value = value
            best_action = d
    return best_action


@app.route('/move', methods=['POST'])
def get_move():
    """Receives the grid and the desired agent, returns the best action."""
    data = request.json
    grid = np.array(data['grid'], dtype=np.int32)
    agent_type = data.get('agent', 'expectimax')

    direction_names = ['up', 'right', 'down', 'left']

    if agent_type == 'dqn':
        if dqn_agent is None:
            return jsonify({'action': -1, 'message': 'DQN model not loaded'})

        encoded = encode_state(grid)
        valid_moves = [d for d in range(4) if _is_valid_move(grid, d)]
        if not valid_moves:
            return jsonify({'action': -1, 'message': 'No valid moves'})

        action = dqn_agent.select_action(encoded, valid_moves, training=False)
        return jsonify({
            'action': action,
            'direction': direction_names[action],
            'agent': 'dqn',
        })

    elif agent_type == 'ntuple':
        if ntuple_net is None:
            return jsonify({'action': -1, 'message': 'N-Tuple model not loaded'})

        action = _ntuple_select_action(grid)
        return jsonify({
            'action': action,
            'direction': direction_names[action],
            'agent': 'ntuple',
        })

    elif agent_type == 'attention':
        if attention_agent is None:
            return jsonify({'action': -1, 'message': 'Attention model not loaded'})

        action = _attention_select_action(grid)
        return jsonify({
            'action': action,
            'direction': direction_names[action],
            'agent': 'attention',
        })

    else:  # expectimax
        action, depth_reached = expectimax_agent.select_action(grid)
        return jsonify({
            'action': action,
            'direction': direction_names[action],
            'agent': 'expectimax',
            'depth_reached': int(depth_reached),
        })


@app.route('/report', methods=['POST'])
def save_report():
    """Saves a JSON report at the end of each game."""
    data = request.json
    reports_dir = os.path.join(os.path.dirname(__file__), 'reports')
    os.makedirs(reports_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    agent = data.get('agent', 'unknown')
    max_tile = data.get('max_tile', 0)
    score = data.get('score', 0)

    report = {
        'timestamp': datetime.now().isoformat(),
        'agent': agent,
        'score': score,
        'max_tile': max_tile,
        'moves': data.get('moves', 0),
        'won': data.get('won', False),
        'final_grid': data.get('final_grid', []),
        'move_history': data.get('move_history', []),
        'duration_ms': data.get('duration_ms', 0),
        'config': {
            'expectimax_depth': expectimax_agent.depth if agent == 'expectimax' else None,
            'dqn_steps': dqn_agent.steps_done if dqn_agent and agent == 'dqn' else None,
            'ntuple': agent == 'ntuple',
        },
    }

    filename = f'{timestamp}_{agent}_s{score}_t{max_tile}.json'
    filepath = os.path.join(reports_dir, filename)
    with open(filepath, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"📊 Report saved: {filename} | Score: {score} | Max: {max_tile} | {'WIN' if data.get('won') else 'GAME OVER'}")
    return jsonify({'saved': filename})


@app.route('/status', methods=['GET'])
def status():
    return jsonify({
        'dqn_loaded': dqn_agent is not None,
        'dqn_steps': dqn_agent.steps_done if dqn_agent else 0,
        'expectimax_depth': expectimax_agent.depth,
        'ntuple_loaded': ntuple_net is not None,
        'agents': ['expectimax'] +
                  (['dqn'] if dqn_agent else []) +
                  (['ntuple'] if ntuple_net else []),
    })


def main():
    global dqn_agent, ntuple_net, ntuple_c_lib, attention_agent

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='checkpoints/best.pt')
    parser.add_argument('--port', type=int, default=8081)
    parser.add_argument('--depth', type=int, default=10, help='Expectimax max search depth')
    parser.add_argument('--time-budget', type=int, default=100, help='Time budget per move in ms')
    args = parser.parse_args()

    expectimax_agent.depth = args.depth
    expectimax_agent.time_budget_ms = args.time_budget

    # Load DQN
    model_path = os.path.join(os.path.dirname(__file__), args.model)
    if os.path.exists(model_path):
        dqn_agent = DQNAgent()
        dqn_agent.load(model_path)
    else:
        print(f"DQN not found ({model_path})")

    # Load N-Tuple via C (fast)
    ntuple_path = os.path.join(os.path.dirname(__file__), 'checkpoints', 'ntuple_best.bin')
    if not os.path.exists(ntuple_path):
        ntuple_path = os.path.join(os.path.dirname(__file__), 'checkpoints', 'ntuple_latest.bin')
    if os.path.exists(ntuple_path):
        try:
            ntuple_c_so = os.path.join(os.path.dirname(__file__), 'ntuple_c.so')
            ntuple_c_lib = ctypes.CDLL(ntuple_c_so)
            ntuple_c_lib.ntuple_load.argtypes = [ctypes.c_char_p]
            ntuple_c_lib.ntuple_load.restype = ctypes.c_int
            ntuple_c_lib.ntuple_select_action.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
            ntuple_c_lib.ntuple_select_action.restype = ctypes.c_int
            ntuple_c_lib.ntuple_load(ntuple_path.encode())
            ntuple_net = True  # flag para status endpoint
        except Exception as e:
            print(f"N-Tuple C failed ({e}), trying Python...")
            ntuple_net = NTupleNetwork()
            ntuple_net.load(ntuple_path)
            ntuple_c_lib = None
    else:
        print(f"N-Tuple not found — train with: ./ntuple_train --episodes 50000")

    # Load Attention N-Tuple
    attn_path = os.path.join(os.path.dirname(__file__), 'checkpoints', 'attention_best.pt')
    if not os.path.exists(attn_path):
        attn_path = os.path.join(os.path.dirname(__file__), 'checkpoints', 'attention_latest.pt')
    if os.path.exists(attn_path) and os.path.exists(ntuple_path):
        attention_agent = AttentionNTupleAgent(ntuple_path=ntuple_path)
        attention_agent.load(attn_path)
    else:
        print("Attention N-Tuple not found — train with: python attention_ntuple.py")

    agents = ['Expectimax']
    if dqn_agent: agents.append(f'DQN ({dqn_agent.steps_done} steps)')
    if ntuple_net: agents.append('N-Tuple')
    if attention_agent: agents.append('Attention N-Tuple')

    print(f"\nAI server running at http://localhost:{args.port}")
    print(f"Agents: {' + '.join(agents)}")
    print(f"Open http://localhost:8080 and click 'AI Play'\n")
    app.run(host='0.0.0.0', port=args.port, debug=False)


if __name__ == '__main__':
    main()
