"""
Agente Expectimax otimizado para o jogo 2048.

Usa simulação de movimentos inline (sem criar objetos Game2048)
para máxima velocidade na busca em árvore.
"""

import numpy as np


def _merge_line(line):
    """Merge uma linha para a esquerda. Retorna (nova_linha, score, changed)."""
    # Compacta: remove zeros
    non_zero = line[line != 0]
    result = np.zeros(4, dtype=np.int32)
    score = 0
    pos = 0
    skip = False

    for i in range(len(non_zero)):
        if skip:
            skip = False
            continue
        if i + 1 < len(non_zero) and non_zero[i] == non_zero[i + 1]:
            merged = non_zero[i] * 2
            result[pos] = merged
            score += merged
            skip = True
        else:
            result[pos] = non_zero[i]
        pos += 1

    return result, score


def fast_move(grid, direction):
    """
    Executa um movimento no grid (numpy array 4x4).
    Retorna (new_grid, score, moved) sem efeitos colaterais.
    """
    new_grid = np.zeros((4, 4), dtype=np.int32)
    total_score = 0

    for i in range(4):
        if direction == 0:    # UP
            line = grid[:, i].copy()
        elif direction == 1:  # RIGHT
            line = grid[i, ::-1].copy()
        elif direction == 2:  # DOWN
            line = grid[::-1, i].copy()
        else:                 # LEFT
            line = grid[i, :].copy()

        merged, score = _merge_line(line)
        total_score += score

        if direction == 0:
            new_grid[:, i] = merged
        elif direction == 1:
            new_grid[i, :] = merged[::-1]
        elif direction == 2:
            new_grid[:, i] = merged[::-1]
        else:
            new_grid[i, :] = merged

    moved = not np.array_equal(grid, new_grid)
    return new_grid, total_score, moved


class ExpectimaxAgent:
    """
    Agente Expectimax otimizado.
    Depth adaptativo: usa profundidade maior quando há poucas células vazias.
    """

    def __init__(self, depth=3):
        self.base_depth = depth
        # Pesos das heurísticas
        self.snake_weights = [
            np.array([
                [2**15, 2**14, 2**13, 2**12],
                [2**8,  2**9,  2**10, 2**11],
                [2**7,  2**6,  2**5,  2**4],
                [2**0,  2**1,  2**2,  2**3],
            ], dtype=np.float64),
        ]
        # Pré-computa as 8 variações (4 rotações x 2 espelhamentos)
        base = self.snake_weights[0]
        self.snake_weights = []
        for _ in range(4):
            self.snake_weights.append(base.copy())
            self.snake_weights.append(base[:, ::-1].copy())
            base = np.rot90(base)

    def evaluate(self, grid):
        """Avalia um estado do tabuleiro com heurísticas combinadas."""
        empty = np.count_nonzero(grid == 0)

        # Snake pattern — melhor das 8 orientações
        g = grid.astype(np.float64)
        snake_score = max(np.sum(g * w) for w in self.snake_weights)

        # Monotonicidade
        mono = self._monotonicity(grid)

        # Smoothness
        smooth = self._smoothness(grid)

        # Penalidade por game over iminente
        if empty == 0:
            has_merge = False
            for y in range(4):
                for x in range(4):
                    v = grid[y, x]
                    if x < 3 and grid[y, x + 1] == v:
                        has_merge = True
                        break
                    if y < 3 and grid[y + 1, x] == v:
                        has_merge = True
                        break
                if has_merge:
                    break
            if not has_merge:
                return -100000.0

        return (
            snake_score * 1.0 +
            empty * 500.0 +
            mono * 100.0 +
            smooth * 10.0
        )

    def _monotonicity(self, grid):
        """Mede quão monotônicas são as linhas e colunas."""
        score = 0.0
        log_grid = np.zeros((4, 4), dtype=np.float64)
        mask = grid > 0
        log_grid[mask] = np.log2(grid[mask].astype(np.float64))

        for i in range(4):
            # Linhas
            left = right = 0.0
            for j in range(3):
                diff = log_grid[i, j] - log_grid[i, j + 1]
                if diff > 0:
                    left += diff
                else:
                    right -= diff
            score -= min(left, right)

            # Colunas
            up = down = 0.0
            for j in range(3):
                diff = log_grid[j, i] - log_grid[j + 1, i]
                if diff > 0:
                    up += diff
                else:
                    down -= diff
            score -= min(up, down)

        return score

    def _smoothness(self, grid):
        """Penaliza diferenças entre tiles adjacentes."""
        score = 0.0
        log_grid = np.zeros((4, 4), dtype=np.float64)
        mask = grid > 0
        log_grid[mask] = np.log2(grid[mask].astype(np.float64))

        for y in range(4):
            for x in range(4):
                if grid[y, x] == 0:
                    continue
                v = log_grid[y, x]
                if x < 3 and grid[y, x + 1] > 0:
                    score -= abs(v - log_grid[y, x + 1])
                if y < 3 and grid[y + 1, x] > 0:
                    score -= abs(v - log_grid[y + 1, x])
        return score

    def select_action(self, grid):
        """Escolhe a melhor ação usando busca Expectimax."""
        depth = self.base_depth
        best_action = 0
        best_score = -np.inf

        for direction in range(4):
            new_grid, move_score, moved = fast_move(grid, direction)
            if not moved:
                continue

            score = self._chance_node(new_grid, depth - 1)

            if score > best_score:
                best_score = score
                best_action = direction

        return best_action, best_score

    def _chance_node(self, grid, depth):
        """Nó CHANCE: média ponderada sobre tiles aleatórios."""
        empty_cells = list(zip(*np.where(grid == 0)))
        if not empty_cells:
            return self.evaluate(grid)

        # Amostra se muitas células vazias (otimização crucial)
        if len(empty_cells) > 3:
            indices = np.random.choice(len(empty_cells), 3, replace=False)
            cells = [empty_cells[i] for i in indices]
        else:
            cells = empty_cells

        total = 0.0
        for y, x in cells:
            for value, prob in [(2, 0.9), (4, 0.1)]:
                new_grid = grid.copy()
                new_grid[y, x] = value

                if depth <= 0:
                    total += prob * self.evaluate(new_grid)
                else:
                    total += prob * self._max_node(new_grid, depth)

        return total / len(cells)

    def _max_node(self, grid, depth):
        """Nó MAX: jogador escolhe o melhor movimento."""
        best_score = -np.inf
        any_moved = False

        for direction in range(4):
            new_grid, _, moved = fast_move(grid, direction)
            if not moved:
                continue

            any_moved = True
            score = self._chance_node(new_grid, depth - 1)
            best_score = max(best_score, score)

        return self.evaluate(grid) if not any_moved else best_score
