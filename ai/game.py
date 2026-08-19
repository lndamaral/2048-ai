"""
2048 game engine in Python.
Faithfully reproduces the logic of Gabriele Cirulli's original game.
"""

import numpy as np
import random


class Game2048:
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

    VECTORS = {
        0: (0, -1),   # up
        1: (1, 0),    # right
        2: (0, 1),    # down
        3: (-1, 0),   # left
    }

    def __init__(self, size=4):
        self.size = size
        self.reset()

    def reset(self):
        self.grid = np.zeros((self.size, self.size), dtype=np.int32)
        self.score = 0
        self.won = False
        self.over = False
        self._add_random_tile()
        self._add_random_tile()
        return self.get_state()

    def _empty_cells(self):
        return list(zip(*np.where(self.grid == 0)))

    def _add_random_tile(self):
        empty = self._empty_cells()
        if empty:
            y, x = random.choice(empty)
            self.grid[y][x] = 4 if random.random() < 0.1 else 2

    def get_state(self):
        return self.grid.copy()

    def _build_traversals(self, direction):
        """Determines the traversal order based on the move direction."""
        dx, dy = self.VECTORS[direction]
        xs = list(range(self.size))
        ys = list(range(self.size))
        if dx == 1:
            xs = list(reversed(xs))
        if dy == 1:
            ys = list(reversed(ys))
        return xs, ys

    def _find_farthest(self, x, y, dx, dy):
        """Finds the farthest position a tile can reach."""
        while True:
            prev_x, prev_y = x, y
            x += dx
            y += dy
            if not (0 <= x < self.size and 0 <= y < self.size) or self.grid[y][x] != 0:
                return prev_x, prev_y, x, y

    def move(self, direction):
        """
        Executes a move. Returns (reward, changed).
        reward = points earned in this move.
        changed = True if the board changed.
        """
        if self.over:
            return 0, False

        dx, dy = self.VECTORS[direction]
        xs, ys = self._build_traversals(direction)

        moved = False
        reward = 0
        merged = np.zeros((self.size, self.size), dtype=bool)

        for y in ys:
            for x in xs:
                if self.grid[y][x] == 0:
                    continue

                value = self.grid[y][x]
                far_x, far_y, next_x, next_y = self._find_farthest(x, y, dx, dy)

                # Check merge
                if (0 <= next_x < self.size and 0 <= next_y < self.size
                        and self.grid[next_y][next_x] == value
                        and not merged[next_y][next_x]):
                    # Merge (unchanged)
                    new_value = value * 2
                    self.grid[y][x] = 0
                    self.grid[next_y][next_x] = new_value
                    merged[next_y][next_x] = True
                    reward += new_value
                    self.score += new_value
                    if new_value == 2048:
                        self.won = True
                    moved = True
                elif far_x != x or far_y != y:
                    # Move without merge
                    self.grid[y][x] = 0
                    self.grid[far_y][far_x] = value
                    moved = True

        if moved:
            self._add_random_tile()
            if not self._moves_available():
                self.over = True

        return reward, moved

    def _moves_available(self):
        """Checks if there are still possible moves."""
        if self._empty_cells():
            return True
        # Check possible merges
        for y in range(self.size):
            for x in range(self.size):
                val = self.grid[y][x]
                if x < self.size - 1 and self.grid[y][x + 1] == val:
                    return True
                if y < self.size - 1 and self.grid[y + 1][x] == val:
                    return True
        return False

    def is_valid_move(self, direction):
        """Checks if a move is valid without executing it."""
        backup = self.grid.copy()
        score_backup = self.score
        won_backup = self.won
        over_backup = self.over

        # Simulate without adding a random tile
        dx, dy = self.VECTORS[direction]
        xs, ys = self._build_traversals(direction)
        merged = np.zeros((self.size, self.size), dtype=bool)
        moved = False

        for y_pos in ys:
            for x_pos in xs:
                if self.grid[y_pos][x_pos] == 0:
                    continue
                value = self.grid[y_pos][x_pos]
                far_x, far_y, next_x, next_y = self._find_farthest(x_pos, y_pos, dx, dy)

                if (0 <= next_x < self.size and 0 <= next_y < self.size
                        and self.grid[next_y][next_x] == value
                        and not merged[next_y][next_x]):
                    moved = True
                    break
                elif far_x != x_pos or far_y != y_pos:
                    moved = True
                    break
            if moved:
                break

        self.grid = backup
        self.score = score_backup
        self.won = won_backup
        self.over = over_backup
        return moved

    def get_valid_moves(self):
        """Returns list of valid moves."""
        return [d for d in range(4) if self.is_valid_move(d)]

    def max_tile(self):
        return int(self.grid.max())

    def __str__(self):
        lines = []
        for row in self.grid:
            lines.append(' '.join(f'{v:6d}' if v else '     .' for v in row))
        return '\n'.join(lines) + f'\nScore: {self.score}'
