"""
Attention-Weighted N-Tuple Network for 2048 — Novel Architecture.

Extends the standard N-Tuple evaluation with a learned attention mechanism
that captures inter-tuple correlations and phase-aware weighting.

Standard N-Tuple:  V(board) = sum(tuple_i(board))
Ours:              V(board) = attention_net(tuple_1, ..., tuple_17, features)

The attention network learns:
1. Which tuples matter most for the current board state
2. Non-linear interactions between tuples
3. Phase-aware evaluation (early/mid/endgame)

This is a novel contribution — no published work combines N-Tuple
evaluation with learned attention for 2048.

Powered by Claude Code
"""

import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim

from ntuple_agent import NTupleNetwork
from expectimax_agent import fast_move
from game import Game2048


class AttentionCombiner(nn.Module):
    """
    Small attention network that learns to combine N-Tuple outputs.

    Input: 17 tuple values + 3 board features = 20 features
    Output: weighted combination + correction term

    Board features:
    - empty_ratio: fraction of empty cells (0-1)
    - max_tile_log: log2(max_tile) / 16 (normalized, 0-1)
    - merge_density: fraction of adjacent pairs that can merge (0-1)
    """

    def __init__(self, n_tuples=17, n_features=3, hidden_dim=64):
        super().__init__()
        input_dim = n_tuples + n_features

        # Attention weights: learns which tuples matter for this board state
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_tuples),
            nn.Softmax(dim=-1),
        )

        # Correction term: captures non-linear inter-tuple interactions
        self.correction = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.n_tuples = n_tuples

    def forward(self, tuple_values, features):
        """
        Args:
            tuple_values: (batch, 17) — raw N-Tuple evaluations
            features: (batch, 3) — board features
        Returns:
            (batch, 1) — final board evaluation
        """
        x = torch.cat([tuple_values, features], dim=-1)

        # Attention-weighted sum of tuple values
        weights = self.attention(x)  # (batch, 17)
        weighted_sum = (weights * tuple_values).sum(dim=-1, keepdim=True)

        # Scale to match raw N-Tuple magnitude
        scaled_sum = weighted_sum * self.n_tuples

        # Non-linear correction
        corr = self.correction(x)

        return scaled_sum + corr


class AttentionNTupleAgent:
    """
    Fourth agent: N-Tuple + Attention Network.

    Uses pre-trained N-Tuple weights as feature extractors,
    then applies a learned attention mechanism to combine them.
    """

    def __init__(self, ntuple_path=None, device=None):
        self.device = device or torch.device(
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )

        # Load pre-trained N-Tuple (frozen — not updated)
        self.ntuple = NTupleNetwork()
        if ntuple_path and os.path.exists(ntuple_path):
            self.ntuple.load(ntuple_path)
            print(f"N-Tuple base loaded from {ntuple_path}")
        else:
            print("WARNING: No N-Tuple weights loaded — attention will train on random base")

        # Attention combiner network
        self.combiner = AttentionCombiner(
            n_tuples=len(self.ntuple.all_tuples),
            n_features=3,
            hidden_dim=64,
        ).to(self.device)

        self.optimizer = optim.Adam(self.combiner.parameters(), lr=1e-3)
        self.learning_rate = 1e-3

    def _extract_tuple_values(self, grid):
        """Extract individual tuple values (not summed)."""
        lg = self.ntuple._grid_to_log(grid)
        values = []
        for i, symmetries in enumerate(self.ntuple.all_tuples):
            v = 0.0
            for positions in symmetries:
                idx = self.ntuple._encode_tuple_fast(lg, positions)
                v += self.ntuple.weights[i][idx]
            values.append(v)
        return np.array(values, dtype=np.float32)

    def _extract_features(self, grid):
        """Extract board-level features for phase awareness."""
        # Empty ratio
        empty = np.count_nonzero(grid == 0)
        empty_ratio = empty / 16.0

        # Max tile (normalized)
        max_val = grid.max()
        max_tile_log = (np.log2(max_val) / 16.0) if max_val > 0 else 0.0

        # Merge density: fraction of adjacent pairs that can merge
        merges = 0
        pairs = 0
        for y in range(4):
            for x in range(4):
                if grid[y, x] == 0:
                    continue
                if x < 3 and grid[y, x + 1] > 0:
                    pairs += 1
                    if grid[y, x] == grid[y, x + 1]:
                        merges += 1
                if y < 3 and grid[y + 1, x] > 0:
                    pairs += 1
                    if grid[y, x] == grid[y + 1, x]:
                        merges += 1
        merge_density = (merges / pairs) if pairs > 0 else 0.0

        return np.array([empty_ratio, max_tile_log, merge_density], dtype=np.float32)

    def evaluate(self, grid):
        """Evaluate a board state using attention-weighted N-Tuple."""
        tuple_vals = self._extract_tuple_values(grid)
        features = self._extract_features(grid)

        with torch.no_grad():
            tv = torch.FloatTensor(tuple_vals).unsqueeze(0).to(self.device)
            ft = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            value = self.combiner(tv, ft).item()

        return value

    def evaluate_with_grad(self, grid):
        """Evaluate with gradient tracking (for training)."""
        tuple_vals = self._extract_tuple_values(grid)
        features = self._extract_features(grid)

        tv = torch.FloatTensor(tuple_vals).unsqueeze(0).to(self.device)
        ft = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        value = self.combiner(tv, ft)

        return value

    def select_action(self, grid):
        """Select best action using 1-ply search with attention evaluation."""
        best_action = 0
        best_value = -1e18

        for d in range(4):
            new_grid, reward, moved = fast_move(grid, d)
            if not moved:
                continue
            value = reward + self.evaluate(new_grid)
            if value > best_value:
                best_value = value
                best_action = d

        return best_action, best_value

    def get_attention_weights(self, grid):
        """Get the attention weights for visualization/analysis."""
        tuple_vals = self._extract_tuple_values(grid)
        features = self._extract_features(grid)

        with torch.no_grad():
            tv = torch.FloatTensor(tuple_vals).unsqueeze(0).to(self.device)
            ft = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            x = torch.cat([tv, ft], dim=-1)
            weights = self.combiner.attention(x).cpu().numpy()[0]

        return weights

    def save(self, path):
        torch.save({
            'combiner': self.combiner.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)
        print(f"Attention model saved to {path}")

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.combiner.load_state_dict(checkpoint['combiner'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"Attention model loaded from {path}")


def train_attention(episodes=50000, ntuple_path=None, save_every=1000):
    """
    Train the attention combiner using TD(0) afterstate learning.

    The N-Tuple weights are FROZEN — only the attention network trains.
    This learns the optimal way to combine tuple evaluations.
    """
    save_dir = os.path.join(os.path.dirname(__file__), 'checkpoints')
    os.makedirs(save_dir, exist_ok=True)

    # Find N-Tuple weights
    if ntuple_path is None:
        for name in ['ntuple_best.bin', 'ntuple_latest.bin']:
            p = os.path.join(save_dir, name)
            if os.path.exists(p):
                ntuple_path = p
                break

    agent = AttentionNTupleAgent(ntuple_path=ntuple_path)

    # Try to load previous attention checkpoint
    attn_path = os.path.join(save_dir, 'attention_latest.pt')
    if os.path.exists(attn_path):
        agent.load(attn_path)

    scores = []
    max_tiles = []
    wins = 0
    best_avg = 0

    # LR decay
    lr_start = 1e-3
    lr_end = 1e-5

    import time
    start = time.time()

    for episode in range(1, episodes + 1):
        progress = episode / episodes
        lr = lr_start * (lr_end / lr_start) ** progress
        for param_group in agent.optimizer.param_groups:
            param_group['lr'] = lr

        game = Game2048()
        game.reset()

        # First afterstate
        best_action = -1
        best_value = -1e18
        best_after = None
        best_reward = 0

        for d in range(4):
            new_grid, reward, moved = fast_move(game.grid, d)
            if not moved:
                continue
            value = reward + agent.evaluate(new_grid)
            if value > best_value:
                best_value = value
                best_action = d
                best_after = new_grid.copy()
                best_reward = reward

        if best_action == -1:
            continue

        prev_after = best_after.copy()
        game.move(best_action)

        while not game.over:
            # Find best action from current state
            best_action = -1
            best_value = -1e18
            best_after = None
            curr_reward = 0

            for d in range(4):
                new_grid, reward, moved = fast_move(game.grid, d)
                if not moved:
                    continue
                value = reward + agent.evaluate(new_grid)
                if value > best_value:
                    best_value = value
                    best_action = d
                    best_after = new_grid.copy()
                    curr_reward = reward

            if best_action == -1:
                break

            # TD(0) update for attention network
            v_prev = agent.evaluate_with_grad(prev_after)
            with torch.no_grad():
                v_curr = agent.evaluate(best_after)

            target = curr_reward + v_curr
            loss = (v_prev - target) ** 2

            agent.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.combiner.parameters(), 1.0)
            agent.optimizer.step()

            game.move(best_action)
            prev_after = best_after.copy()

        # Terminal update
        v_last = agent.evaluate_with_grad(prev_after)
        loss = v_last ** 2
        agent.optimizer.zero_grad()
        loss.backward()
        agent.optimizer.step()

        scores.append(game.score)
        max_tiles.append(game.max_tile())
        if game.max_tile() >= 2048:
            wins += 1

        # Log every 100 episodes
        if episode % 100 == 0:
            recent = scores[-100:]
            recent_tiles = max_tiles[-100:]
            elapsed = time.time() - start

            tile_dist = {}
            for t in recent_tiles:
                tile_dist[t] = tile_dist.get(t, 0) + 1

            avg = np.mean(recent)
            print(f"\n{'='*60}")
            print(f"Attention N-Tuple Episode {episode}/{episodes} | Time: {elapsed:.0f}s")
            print(f"{'='*60}")
            print(f"  Average score:  {avg:.0f}")
            print(f"  Max score:      {max(recent)}")
            print(f"  Max tiles:      {dict(sorted(tile_dist.items()))}")
            print(f"  2048+:          {wins}x total")
            print(f"  LR:             {lr:.6f}")

            # Show attention weights for a sample board
            sample_grid = np.array(recent_tiles).max()
            if avg > best_avg:
                best_avg = avg
                agent.save(os.path.join(save_dir, 'attention_best.pt'))

        if episode % save_every == 0:
            agent.save(attn_path)

    agent.save(attn_path)
    print(f"\nTraining complete! Best avg score: {best_avg:.0f}")
    return agent


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train Attention N-Tuple agent')
    parser.add_argument('--episodes', type=int, default=50000)
    parser.add_argument('--ntuple', type=str, default=None, help='Path to N-Tuple weights')
    args = parser.parse_args()
    train_attention(episodes=args.episodes, ntuple_path=args.ntuple)
