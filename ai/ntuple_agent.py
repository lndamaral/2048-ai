"""
N-Tuple Network for 2048 — state of the art version.

Based on Wu et al. (2014) and Szubert & Jaskowski (2014).

Keys for 90%+:
- 17 tuples of 6 positions (complete spatial coverage)
- High LR (0.1) with decay — n-tuple updates are sparse
- Forward TD(0) afterstate learning
- 8 symmetries in evaluation and update
"""

import numpy as np
import os
import struct
from game import Game2048
from expectimax_agent import fast_move


class NTupleNetwork:
    """
    N-Tuple Network with 17 tuples of 6 positions.
    Patterns based on Wu et al. (2014) — complete 4x4 coverage.
    """

    TUPLES = [
        # Horizontais 4+2
        [(0,0),(0,1),(0,2),(0,3),(1,0),(1,1)],
        [(1,0),(1,1),(1,2),(1,3),(2,0),(2,1)],
        [(2,0),(2,1),(2,2),(2,3),(3,0),(3,1)],
        # Verticais 4+2
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1)],
        [(0,1),(1,1),(2,1),(3,1),(0,2),(1,2)],
        [(0,2),(1,2),(2,2),(3,2),(0,3),(1,3)],
        # Blocos 3x2
        [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2)],
        [(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)],
        [(2,0),(2,1),(2,2),(3,0),(3,1),(3,2)],
        [(0,1),(0,2),(0,3),(1,1),(1,2),(1,3)],
        # Blocos 2x3
        [(0,0),(0,1),(1,0),(1,1),(2,0),(2,1)],
        [(0,1),(0,2),(1,1),(1,2),(2,1),(2,2)],
        [(0,2),(0,3),(1,2),(1,3),(2,2),(2,3)],
        # L-shapes e diagonais
        [(0,0),(0,1),(0,2),(1,0),(1,1),(2,0)],
        [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)],
        [(0,0),(0,1),(1,1),(1,2),(2,2),(2,3)],
        [(0,2),(0,3),(1,1),(1,2),(2,0),(2,1)],
    ]

    MAX_TILE_LOG2 = 16

    def __init__(self):
        self.all_tuples = []
        self.tuple_sizes = []

        for t in self.TUPLES:
            symmetries = self._generate_symmetries(t)
            self.all_tuples.append(symmetries)
            self.tuple_sizes.append(len(t))

        self.weights = []
        for size in self.tuple_sizes:
            lut_size = self.MAX_TILE_LOG2 ** size
            self.weights.append(np.zeros(lut_size, dtype=np.float32))

        self.learning_rate = 0.01

    def _generate_symmetries(self, positions):
        """Generates the 8 symmetries (4 rotations x 2 reflections), without duplicates."""
        symmetries = []
        pos = list(positions)

        for _ in range(4):
            symmetries.append(list(pos))
            symmetries.append([(y, 3-x) for y, x in pos])
            pos = [(x, 3-y) for y, x in pos]

        unique = []
        seen = set()
        for s in symmetries:
            key = tuple(sorted(s))
            if key not in seen:
                seen.add(key)
                unique.append(s)

        return unique

    def _grid_to_log(self, grid):
        """Converts grid to log2 values (cached)."""
        log_grid = np.zeros((4, 4), dtype=np.int32)
        for y in range(4):
            for x in range(4):
                v = int(grid[y, x])
                if v > 0:
                    l = 0
                    while v > 1:
                        v >>= 1
                        l += 1
                    log_grid[y, x] = l
        return log_grid

    def _encode_tuple(self, grid, positions):
        """Encodes grid values at positions as an index into the LUT."""
        index = 0
        for y, x in positions:
            val = grid[y, x]
            log_val = 0
            if val > 0:
                v = int(val)
                while v > 1:
                    v >>= 1
                    log_val += 1
            index = index * self.MAX_TILE_LOG2 + log_val
        return index

    def _encode_tuple_fast(self, log_grid, positions):
        """Fast version using pre-computed log_grid."""
        index = 0
        for y, x in positions:
            index = index * self.MAX_TILE_LOG2 + log_grid[y, x]
        return index

    def evaluate(self, grid):
        """Evaluates a board state."""
        lg = self._grid_to_log(grid)
        total = 0.0
        for i, symmetries in enumerate(self.all_tuples):
            w = self.weights[i]
            for positions in symmetries:
                idx = self._encode_tuple_fast(lg, positions)
                total += w[idx]
        return total

    def update(self, grid, delta):
        """Updates weights for the given board state."""
        delta = max(-1000.0, min(1000.0, delta))
        adj = self.learning_rate * delta
        lg = self._grid_to_log(grid)
        for i, symmetries in enumerate(self.all_tuples):
            w = self.weights[i]
            for positions in symmetries:
                idx = self._encode_tuple_fast(lg, positions)
                w[idx] += adj

    def save(self, path):
        with open(path, 'wb') as f:
            f.write(struct.pack('I', len(self.weights)))
            for w in self.weights:
                f.write(struct.pack('I', len(w)))
                f.write(w.tobytes())
        size_mb = os.path.getsize(path) / 1024 / 1024
        print(f"N-Tuple saved to {path} ({size_mb:.1f} MB)")

    def load(self, path):
        with open(path, 'rb') as f:
            n_tables = struct.unpack('I', f.read(4))[0]
            self.weights = []
            for _ in range(n_tables):
                size = struct.unpack('I', f.read(4))[0]
                data = np.frombuffer(f.read(size * 4), dtype=np.float32).copy()
                self.weights.append(data)
        print(f"N-Tuple loaded from {path} ({n_tables} tables)")


def select_best_afterstate(net, grid):
    """Chooses the best action by evaluating afterstates."""
    best_action = -1
    best_value = -1e18
    best_after = None
    best_reward = 0

    for d in range(4):
        new_grid, reward, moved = fast_move(grid, d)
        if not moved:
            continue
        value = reward + net.evaluate(new_grid)
        if value > best_value:
            best_value = value
            best_action = d
            best_after = new_grid.copy()
            best_reward = reward

    return best_action, best_after, best_reward


def train_ntuple(episodes=50000, save_every=1000):
    """
    Forward TD(0) afterstate learning.

    At each move:
      V(s_prev) += lr * (r + V(s_curr) - V(s_prev))

    Where s_prev and s_curr are afterstates (after moving, before the random tile).
    """
    save_dir = os.path.join(os.path.dirname(__file__), 'checkpoints')
    os.makedirs(save_dir, exist_ok=True)

    net = NTupleNetwork()

    model_path = os.path.join(save_dir, 'ntuple_latest.bin')
    if os.path.exists(model_path):
        try:
            net.load(model_path)
        except Exception as e:
            print(f"Incompatible checkpoint ({e}), starting from scratch")

    scores = []
    max_tiles = []
    wins = 0
    best_avg = 0

    # LR decay: 0.01 → 0.0005
    lr_start = 0.01
    lr_end = 0.0005

    import time
    start = time.time()

    for episode in range(1, episodes + 1):
        progress = episode / episodes
        net.learning_rate = lr_start * (lr_end / lr_start) ** progress

        game = Game2048()
        game.reset()

        # First afterstate
        action, prev_after, prev_reward = select_best_afterstate(net, game.grid)
        if action == -1:
            continue
        game.move(action)

        while not game.over:
            action, curr_after, curr_reward = select_best_afterstate(net, game.grid)
            if action == -1:
                break

            # Forward TD(0): V(prev) += lr * (r + V(curr) - V(prev))
            if prev_after is not None:
                v_prev = net.evaluate(prev_after)
                v_curr = net.evaluate(curr_after)
                delta = curr_reward + v_curr - v_prev
                net.update(prev_after, delta)

            game.move(action)
            prev_after = curr_after
            prev_reward = curr_reward

        # Terminal: V(last) += lr * (0 - V(last))
        if prev_after is not None:
            net.update(prev_after, -net.evaluate(prev_after))

        scores.append(game.score)
        max_tiles.append(game.max_tile())
        if game.max_tile() >= 2048:
            wins += 1

        if episode % 100 == 0:
            recent = scores[-100:]
            recent_tiles = max_tiles[-100:]
            elapsed = time.time() - start

            tile_dist = {}
            for t in recent_tiles:
                tile_dist[t] = tile_dist.get(t, 0) + 1

            avg = np.mean(recent)
            print(f"\n{'='*60}")
            print(f"N-Tuple Episode {episode}/{episodes} | Time: {elapsed:.0f}s")
            print(f"{'='*60}")
            print(f"  Average score:  {avg:.0f}")
            print(f"  Max score: {max(recent)}")
            print(f"  Max tiles:    {dict(sorted(tile_dist.items()))}")
            print(f"  2048+:        {wins}x total")
            print(f"  LR:           {net.learning_rate:.6f}")

            if avg > best_avg:
                best_avg = avg
                net.save(os.path.join(save_dir, 'ntuple_best.bin'))

        if episode % save_every == 0:
            net.save(model_path)

    net.save(model_path)
    print(f"\nTraining complete! Best avg score: {best_avg:.0f}")
    return net


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=50000)
    args = parser.parse_args()
    train_ntuple(episodes=args.episodes)
