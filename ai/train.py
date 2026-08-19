"""
DQN Training Script for 2048 — Rainbow-style.

Usage:
    python ai/train.py [--episodes 50000] [--resume] [--no-distributional]

Features:
- Dueling DQN with Prioritized Replay
- Noisy Networks for exploration
- Quantile Regression DQN (distributional RL, enabled by default)
- N-step returns
- Improved reward shaping
- Detailed logging every 100 episodes
"""

import argparse
import time
import os
import numpy as np

from game import Game2048
from dqn_agent import DQNAgent, encode_state


def shape_reward(reward, game, moved):
    """
    Reward shaping to accelerate learning.

    Components:
    - Direct merge reward (log-normalized)
    - Corner bonus for keeping max tile in corner
    - Monotonicity bonus for organized rows
    - Empty space bonus
    - Game over penalty
    - Invalid move penalty
    """
    if not moved:
        return -1.0

    shaped = np.log2(reward + 1) * 0.1 if reward > 0 else 0.0

    # Corner bonus
    max_val = game.max_tile()
    corners = [game.grid[0][0], game.grid[0][3], game.grid[3][0], game.grid[3][3]]
    if max_val in corners:
        shaped += 0.1

    # Monotonicity bonus
    grid = game.grid
    mono_bonus = 0
    for row in grid:
        vals = [v for v in row if v > 0]
        if vals == sorted(vals) or vals == sorted(vals, reverse=True):
            mono_bonus += 0.02
    shaped += mono_bonus

    # Empty cells bonus
    empty = len(game._empty_cells())
    shaped += empty * 0.01

    # Game over penalty
    if game.over:
        shaped -= 2.0

    return shaped


def train(episodes=50000, resume=False, distributional=True):
    save_dir = os.path.join(os.path.dirname(__file__), 'checkpoints')
    os.makedirs(save_dir, exist_ok=True)

    agent = DQNAgent(distributional=distributional)

    # Try to resume from checkpoint
    checkpoint_path = os.path.join(save_dir, 'dqn_latest.pt')
    if resume and os.path.exists(checkpoint_path):
        agent.load(checkpoint_path)
    elif resume and os.path.exists(os.path.join(save_dir, 'best.pt')):
        agent.load(os.path.join(save_dir, 'best.pt'))

    game = Game2048()

    # Metrics
    scores = []
    max_tiles = []
    best_score = 0
    reached_2048 = 0

    start_time = time.time()

    for episode in range(1, episodes + 1):
        state = game.reset()
        encoded = encode_state(state)
        total_reward = 0
        moves = 0

        while not game.over:
            valid_moves = game.get_valid_moves()
            if not valid_moves:
                break

            action = agent.select_action(encoded, valid_moves, training=True)
            reward, moved = game.move(action)

            next_state = game.get_state()
            next_encoded = encode_state(next_state)
            next_valid = game.get_valid_moves()

            shaped = shape_reward(reward, game, moved)
            agent.store_transition(
                encoded, action, shaped, next_encoded,
                game.over, next_valid if next_valid else [0]
            )

            loss = agent.train_step()

            encoded = next_encoded
            total_reward += shaped
            moves += 1

        scores.append(game.score)
        max_tiles.append(game.max_tile())
        if game.won:
            reached_2048 += 1

        if game.score > best_score:
            best_score = game.score
            agent.save(os.path.join(save_dir, 'dqn_best.pt'))

        # Log every 100 episodes
        if episode % 100 == 0:
            recent_scores = scores[-100:]
            recent_tiles = max_tiles[-100:]
            elapsed = time.time() - start_time
            eps = agent.get_epsilon()

            tile_dist = {}
            for t in recent_tiles:
                tile_dist[t] = tile_dist.get(t, 0) + 1

            print(f"\n{'='*60}")
            print(f"Episode {episode}/{episodes} | Time: {elapsed:.0f}s")
            print(f"{'='*60}")
            print(f"  Avg score:    {np.mean(recent_scores):.0f}")
            print(f"  Max score:    {max(recent_scores)}")
            print(f"  Best score:   {best_score}")
            print(f"  Epsilon:      {eps:.4f}")
            print(f"  Max tiles:    {dict(sorted(tile_dist.items()))}")
            print(f"  2048 reached: {reached_2048}x total")
            if agent.training_losses:
                print(f"  Avg loss:     {np.mean(agent.training_losses[-1000:]):.6f}")

            agent.save(checkpoint_path)

        # Save every 1000
        if episode % 1000 == 0:
            agent.save(os.path.join(save_dir, f'dqn_ep{episode}.pt'))

    # Save final metrics
    np.savez(
        os.path.join(save_dir, 'dqn_metrics.npz'),
        scores=np.array(scores),
        max_tiles=np.array(max_tiles),
    )

    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Total episodes: {episodes}")
    print(f"Best score: {best_score}")
    print(f"2048 reached: {reached_2048}x")
    print(f"Avg score (last 100): {np.mean(scores[-100:]):.0f}")

    return agent


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train DQN agent for 2048')
    parser.add_argument('--episodes', type=int, default=50000, help='Number of episodes')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--no-distributional', action='store_true',
                        help='Disable QR-DQN (use standard Dueling DQN)')
    args = parser.parse_args()

    train(
        episodes=args.episodes,
        resume=args.resume,
        distributional=not args.no_distributional,
    )
