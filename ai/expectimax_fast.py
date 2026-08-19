"""
Agente Expectimax otimizado com Lookup Tables para 2048.

Representação: cada célula = 4 bits (0=vazio, 1=2, 2=4, ..., 15=32768).
Uma linha = 16 bits = 65536 estados possíveis.
Tabelas pré-computadas para merge e heurísticas.
"""

import numpy as np

# ─── Build Lookup Tables ────────────────────────────────────────

def _build_all_tables():
    """Constrói todas as lookup tables de uma vez."""
    merge_left = np.zeros(65536, dtype=np.int32)
    merge_score = np.zeros(65536, dtype=np.int32)
    merge_right = np.zeros(65536, dtype=np.int32)
    merge_right_score = np.zeros(65536, dtype=np.int32)
    heur_score = np.zeros(65536, dtype=np.float64)

    for encoded in range(65536):
        c0 = (encoded >> 12) & 0xF
        c1 = (encoded >> 8) & 0xF
        c2 = (encoded >> 4) & 0xF
        c3 = encoded & 0xF
        cells = [c0, c1, c2, c3]

        # ── Merge left ──
        non_zero = [c for c in cells if c != 0]
        result = []
        score = 0
        i = 0
        while i < len(non_zero):
            if i + 1 < len(non_zero) and non_zero[i] == non_zero[i + 1]:
                m = min(non_zero[i] + 1, 15)
                result.append(m)
                score += (1 << m)
                i += 2
            else:
                result.append(non_zero[i])
                i += 1
        while len(result) < 4:
            result.append(0)

        left_enc = (result[0] << 12) | (result[1] << 8) | (result[2] << 4) | result[3]
        merge_left[encoded] = left_enc
        merge_score[encoded] = score

        # ── Merge right (reverse, merge left, reverse) ──
        rev = (c3 << 12) | (c2 << 8) | (c1 << 4) | c0
        rev_cells = [c3, c2, c1, c0]
        non_zero_r = [c for c in rev_cells if c != 0]
        result_r = []
        score_r = 0
        i = 0
        while i < len(non_zero_r):
            if i + 1 < len(non_zero_r) and non_zero_r[i] == non_zero_r[i + 1]:
                m = min(non_zero_r[i] + 1, 15)
                result_r.append(m)
                score_r += (1 << m)
                i += 2
            else:
                result_r.append(non_zero_r[i])
                i += 1
        while len(result_r) < 4:
            result_r.append(0)
        # Reverse back
        right_enc = (result_r[3] << 12) | (result_r[2] << 8) | (result_r[1] << 4) | result_r[0]
        merge_right[encoded] = right_enc
        merge_right_score[encoded] = score_r

        # ── Heurística da linha ──
        h = 0.0
        empty = sum(1 for c in cells if c == 0)
        h += empty * 270.0

        # Monotonicidade
        mono_inc = mono_dec = 0.0
        for j in range(3):
            if cells[j] > cells[j + 1]:
                mono_dec += (cells[j] - cells[j + 1]) * cells[j]
            elif cells[j] < cells[j + 1]:
                mono_inc += (cells[j + 1] - cells[j]) * cells[j + 1]
        h += 47.0 * max(mono_inc, mono_dec)

        # Smoothness + merge potential
        for j in range(3):
            if cells[j] != 0 and cells[j + 1] != 0:
                if cells[j] == cells[j + 1]:
                    h += 700.0
                else:
                    h -= 12.0 * abs(cells[j] - cells[j + 1])

        # Edge bonus
        if cells[0] != 0:
            h += 11.0 * cells[0] * cells[0]
        if cells[3] != 0:
            h += 11.0 * cells[3] * cells[3]

        heur_score[encoded] = h

    return merge_left, merge_score, merge_right, merge_right_score, heur_score


print("Construindo lookup tables...", end=" ", flush=True)
MERGE_LEFT, MERGE_SCORE, MERGE_RIGHT, MERGE_RIGHT_SCORE, HEUR_SCORE = _build_all_tables()
print("OK")


# ─── Fast board operations ──────────────────────────────────────
# Board = tuple of 4 ints (each row encoded as 16-bit)

def grid_to_board(grid):
    """Numpy grid 4x4 → board tuple."""
    rows = []
    for y in range(4):
        enc = 0
        for x in range(4):
            val = int(grid[y, x])
            log_val = 0
            if val > 0:
                v = val
                while v > 1:
                    v >>= 1
                    log_val += 1
            enc = (enc << 4) | log_val
        rows.append(enc)
    return tuple(rows)


def board_move(board, direction):
    """Executa movimento. Retorna (new_board, score, moved)."""
    r0, r1, r2, r3 = board
    total_score = 0

    if direction == 3:  # LEFT
        n0 = MERGE_LEFT[r0]; total_score += MERGE_SCORE[r0]
        n1 = MERGE_LEFT[r1]; total_score += MERGE_SCORE[r1]
        n2 = MERGE_LEFT[r2]; total_score += MERGE_SCORE[r2]
        n3 = MERGE_LEFT[r3]; total_score += MERGE_SCORE[r3]
        new_board = (int(n0), int(n1), int(n2), int(n3))

    elif direction == 1:  # RIGHT
        n0 = MERGE_RIGHT[r0]; total_score += MERGE_RIGHT_SCORE[r0]
        n1 = MERGE_RIGHT[r1]; total_score += MERGE_RIGHT_SCORE[r1]
        n2 = MERGE_RIGHT[r2]; total_score += MERGE_RIGHT_SCORE[r2]
        n3 = MERGE_RIGHT[r3]; total_score += MERGE_RIGHT_SCORE[r3]
        new_board = (int(n0), int(n1), int(n2), int(n3))

    elif direction == 0:  # UP — transpõe, merge left, transpõe
        # Extrai colunas
        cols = []
        for x in range(4):
            c = (((r0 >> (12 - x * 4)) & 0xF) << 12 |
                 ((r1 >> (12 - x * 4)) & 0xF) << 8 |
                 ((r2 >> (12 - x * 4)) & 0xF) << 4 |
                 ((r3 >> (12 - x * 4)) & 0xF))
            merged = int(MERGE_LEFT[c])
            total_score += int(MERGE_SCORE[c])
            cols.append(merged)
        # Reconstrói linhas a partir das colunas mergidas
        n0 = (((cols[0] >> 12) & 0xF) << 12 | ((cols[1] >> 12) & 0xF) << 8 |
               ((cols[2] >> 12) & 0xF) << 4 | ((cols[3] >> 12) & 0xF))
        n1 = (((cols[0] >> 8) & 0xF) << 12 | ((cols[1] >> 8) & 0xF) << 8 |
               ((cols[2] >> 8) & 0xF) << 4 | ((cols[3] >> 8) & 0xF))
        n2 = (((cols[0] >> 4) & 0xF) << 12 | ((cols[1] >> 4) & 0xF) << 8 |
               ((cols[2] >> 4) & 0xF) << 4 | ((cols[3] >> 4) & 0xF))
        n3 = ((cols[0] & 0xF) << 12 | (cols[1] & 0xF) << 8 |
               (cols[2] & 0xF) << 4 | (cols[3] & 0xF))
        new_board = (n0, n1, n2, n3)

    else:  # DOWN — transpõe, merge right, transpõe
        cols = []
        for x in range(4):
            c = (((r0 >> (12 - x * 4)) & 0xF) << 12 |
                 ((r1 >> (12 - x * 4)) & 0xF) << 8 |
                 ((r2 >> (12 - x * 4)) & 0xF) << 4 |
                 ((r3 >> (12 - x * 4)) & 0xF))
            merged = int(MERGE_RIGHT[c])
            total_score += int(MERGE_RIGHT_SCORE[c])
            cols.append(merged)
        n0 = (((cols[0] >> 12) & 0xF) << 12 | ((cols[1] >> 12) & 0xF) << 8 |
               ((cols[2] >> 12) & 0xF) << 4 | ((cols[3] >> 12) & 0xF))
        n1 = (((cols[0] >> 8) & 0xF) << 12 | ((cols[1] >> 8) & 0xF) << 8 |
               ((cols[2] >> 8) & 0xF) << 4 | ((cols[3] >> 8) & 0xF))
        n2 = (((cols[0] >> 4) & 0xF) << 12 | ((cols[1] >> 4) & 0xF) << 8 |
               ((cols[2] >> 4) & 0xF) << 4 | ((cols[3] >> 4) & 0xF))
        n3 = ((cols[0] & 0xF) << 12 | (cols[1] & 0xF) << 8 |
               (cols[2] & 0xF) << 4 | (cols[3] & 0xF))
        new_board = (n0, n1, n2, n3)

    moved = new_board != board
    return new_board, int(total_score), moved


def board_evaluate(board):
    """Avalia o board usando lookup tables de heurísticas."""
    r0, r1, r2, r3 = board
    score = HEUR_SCORE[r0] + HEUR_SCORE[r1] + HEUR_SCORE[r2] + HEUR_SCORE[r3]

    # Colunas também
    for x in range(4):
        c = (((r0 >> (12 - x * 4)) & 0xF) << 12 |
             ((r1 >> (12 - x * 4)) & 0xF) << 8 |
             ((r2 >> (12 - x * 4)) & 0xF) << 4 |
             ((r3 >> (12 - x * 4)) & 0xF))
        score += HEUR_SCORE[c]

    # Corner bonus: max tile no canto
    corners = [
        (r0 >> 12) & 0xF, r0 & 0xF,
        (r3 >> 12) & 0xF, r3 & 0xF,
    ]
    max_corner = max(corners)
    # Find global max
    all_max = 0
    for r in board:
        for shift in [12, 8, 4, 0]:
            v = (r >> shift) & 0xF
            if v > all_max:
                all_max = v
    if max_corner == all_max and all_max > 0:
        score += all_max * all_max * 40.0

    return score


def board_empty_cells(board):
    """Retorna lista de (row_idx, shift) para células vazias."""
    empties = []
    for y, row in enumerate(board):
        for shift in [12, 8, 4, 0]:
            if (row >> shift) & 0xF == 0:
                empties.append((y, shift))
    return empties


def board_set_cell(board, y, shift, val_log2):
    """Define uma célula. Retorna novo board."""
    lst = list(board)
    lst[y] = lst[y] | (val_log2 << shift)
    return tuple(lst)


# ─── Expectimax Agent ────────────────────────────────────────────

class ExpectimaxAgent:
    def __init__(self, depth=4):
        self.depth = depth
        self.base_depth = depth

    def select_action(self, grid):
        """Recebe numpy grid 4x4, retorna (ação, score)."""
        self._cache = {}  # transposition table — limpa a cada jogada
        board = grid_to_board(grid)
        best_action = 0
        best_score = -1e18

        for direction in range(4):
            new_board, move_score, moved = board_move(board, direction)
            if not moved:
                continue

            score = move_score + self._chance_node(new_board, self.base_depth - 1)
            if score > best_score:
                best_score = score
                best_action = direction

        self._cache = None
        return best_action, best_score

    def _chance_node(self, board, depth):
        empties = board_empty_cells(board)
        if not empties:
            return board_evaluate(board)

        # Menos vazias → avalia todas. Muitas vazias → amostra.
        if len(empties) > 4:
            import random
            cells = random.sample(empties, 4)
        else:
            cells = empties

        total = 0.0
        for y, shift in cells:
            for val_log2, prob in [(1, 0.9), (2, 0.1)]:
                new_board = board_set_cell(board, y, shift, val_log2)
                if depth <= 0:
                    total += prob * board_evaluate(new_board)
                else:
                    total += prob * self._max_node(new_board, depth)

        return total / len(cells)

    def _max_node(self, board, depth):
        # Transposition table lookup
        key = (board, depth)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        best_score = -1e18
        any_moved = False

        for direction in range(4):
            new_board, move_score, moved = board_move(board, direction)
            if not moved:
                continue

            any_moved = True
            score = move_score + self._chance_node(new_board, depth - 1)
            if score > best_score:
                best_score = score

        result = board_evaluate(board) if not any_moved else best_score
        self._cache[key] = result
        return result
