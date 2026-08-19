/*
 * engine_bitboard.h — Bitboard game engine for 2048.
 *
 * Self-contained header with all functions defined static inline.
 * No separate .c file needed.
 *
 * Board representation:
 *   uint64_t with 16 nibbles (4 bits each), storing log2 of tile values.
 *   0 = empty, 1 = tile 2, 2 = tile 4, ..., 15 = tile 32768.
 *
 * Bit layout:
 *   Row 0 = bits  0-15   (top row)
 *   Row 1 = bits 16-31
 *   Row 2 = bits 32-47
 *   Row 3 = bits 48-63   (bottom row)
 *
 * Within each 16-bit row:
 *   Cell 0 (leftmost)  = bits 12-15
 *   Cell 1              = bits  8-11
 *   Cell 2              = bits  4-7
 *   Cell 3 (rightmost) = bits  0-3
 *
 * Compile with: -O3 for best performance (enables inlining).
 */

#ifndef ENGINE_BITBOARD_H
#define ENGINE_BITBOARD_H

#include <stdint.h>
#include <string.h>

typedef uint64_t board_t;

/* ─── Thread-safe PRNG ─────────────────────────────────────────── */

static inline int trand(unsigned int *seed) {
    *seed = *seed * 1103515245 + 12345;
    return (*seed >> 16) & 0x7FFF;
}

/* ─── Move lookup tables ───────────────────────────────────────── */

/*
 * Pre-computed tables for row merges. Indexed by a 16-bit row value
 * (4 nibbles packed into a uint16_t). There are 65536 possible rows.
 *
 * move_left_row[row]  = merged row after sliding left
 * move_right_row[row] = merged row after sliding right
 * move_left_score[row]  = score earned by the left merge
 * move_right_score[row] = score earned by the right merge
 */
static uint16_t move_left_row[65536];
static uint16_t move_right_row[65536];
static int move_left_score[65536];
static int move_right_score[65536];

/*
 * Build all move lookup tables. Must be called once before any
 * do_move_* function. Safe to call multiple times (idempotent).
 */
static void build_move_tables(void) {
    static int built = 0;
    if (built) return;

    for (int enc = 0; enc < 65536; enc++) {
        /* Extract the four nibbles (cells) from the encoded row. */
        int c[4] = {
            (enc >> 12) & 0xF,
            (enc >>  8) & 0xF,
            (enc >>  4) & 0xF,
             enc        & 0xF
        };

        /* --- Compute left merge --- */

        /* Step 1: Compact non-zero cells to the left. */
        int nz[4], nz_len = 0;
        for (int i = 0; i < 4; i++)
            if (c[i]) nz[nz_len++] = c[i];

        /* Step 2: Merge adjacent equal tiles, accumulate score. */
        int res[4] = {0, 0, 0, 0};
        int score = 0, pos = 0;
        for (int i = 0; i < nz_len; i++) {
            if (i + 1 < nz_len && nz[i] == nz[i + 1]) {
                int merged = nz[i] + 1;
                if (merged > 15) merged = 15; /* clamp to max nibble */
                res[pos++] = merged;
                score += (1 << merged);
                i++; /* skip the second tile of the pair */
            } else {
                res[pos++] = nz[i];
            }
        }

        /* Pack left-merged result into a 16-bit row. */
        uint16_t left_row = (uint16_t)(
            (res[0] << 12) | (res[1] << 8) | (res[2] << 4) | res[3]);
        move_left_row[enc] = left_row;
        move_left_score[enc] = score;

        /*
         * Right merge: reverse the input, merge left, reverse the output.
         * The reversed input index maps c[0]<->c[3] and c[1]<->c[2].
         */
        /* Recompute: compact right = compact reversed left. */
        int nz2[4], nz2_len = 0;
        for (int i = 3; i >= 0; i--)
            if (c[i]) nz2[nz2_len++] = c[i];

        int res2[4] = {0, 0, 0, 0};
        int score2 = 0, pos2 = 0;
        for (int i = 0; i < nz2_len; i++) {
            if (i + 1 < nz2_len && nz2[i] == nz2[i + 1]) {
                int merged = nz2[i] + 1;
                if (merged > 15) merged = 15;
                res2[pos2++] = merged;
                score2 += (1 << merged);
                i++;
            } else {
                res2[pos2++] = nz2[i];
            }
        }

        /* Pack right-merged result: result fills from the right side. */
        int rr[4] = {0, 0, 0, 0};
        for (int i = 0; i < pos2; i++)
            rr[3 - i] = res2[pos2 - 1 - i];
        uint16_t right_row = (uint16_t)(
            (rr[0] << 12) | (rr[1] << 8) | (rr[2] << 4) | rr[3]);
        move_right_row[enc] = right_row;
        move_right_score[enc] = score2;
    }

    built = 1;
}

/* ─── Board accessors ──────────────────────────────────────────── */

/*
 * Extract the 16-bit row at position y (0 = top, 3 = bottom).
 */
static inline uint16_t board_row(board_t b, int y) {
    return (uint16_t)((b >> (y * 16)) & 0xFFFF);
}

/*
 * Replace row y in the board with a new 16-bit value.
 */
static inline board_t board_set_row(board_t b, int y, uint16_t row) {
    int shift = y * 16;
    b &= ~((uint64_t)0xFFFF << shift);
    b |= (uint64_t)row << shift;
    return b;
}

/*
 * Get the nibble (log2 tile value) at cell (y, x).
 * y = row (0-3), x = column (0-3, 0 = leftmost).
 */
static inline int board_get(board_t b, int y, int x) {
    int shift = y * 16 + (3 - x) * 4;
    return (int)((b >> shift) & 0xF);
}

/*
 * Set the nibble at cell (y, x) to val (0-15).
 * Returns the updated board.
 */
static inline board_t board_set(board_t b, int y, int x, int val) {
    int shift = y * 16 + (3 - x) * 4;
    b &= ~((uint64_t)0xF << shift);
    b |= (uint64_t)(val & 0xF) << shift;
    return b;
}

/*
 * Extract column x as a 16-bit value in row format.
 * The nibble from row 0 goes to bits 12-15 (leftmost cell position),
 * and the nibble from row 3 goes to bits 0-3 (rightmost cell position).
 * This lets us reuse the row merge lookup tables for column moves.
 */
static inline uint16_t board_col(board_t b, int x) {
    int bit_offset = (3 - x) * 4; /* nibble position within each row */
    uint16_t col = 0;
    col |= (uint16_t)(((b >> ( 0 + bit_offset)) & 0xF) << 12); /* row 0 -> bits 12-15 */
    col |= (uint16_t)(((b >> (16 + bit_offset)) & 0xF) <<  8); /* row 1 -> bits  8-11 */
    col |= (uint16_t)(((b >> (32 + bit_offset)) & 0xF) <<  4); /* row 2 -> bits  4-7  */
    col |= (uint16_t)(((b >> (48 + bit_offset)) & 0xF)      ); /* row 3 -> bits  0-3  */
    return col;
}

/*
 * Write a 16-bit column value back into the board at column x.
 * The high nibble (bits 12-15) goes to row 0, the low nibble (bits 0-3)
 * goes to row 3.
 */
static inline board_t board_set_col(board_t b, int x, uint16_t col) {
    int bit_offset = (3 - x) * 4;
    uint64_t mask = (uint64_t)0xF << bit_offset;

    /* Clear all four nibbles in this column. */
    b &= ~(mask | (mask << 16) | (mask << 32) | (mask << 48));

    /* Set each row's nibble from the column value. */
    b |= (uint64_t)((col >> 12) & 0xF) << ( 0 + bit_offset); /* row 0 */
    b |= (uint64_t)((col >>  8) & 0xF) << (16 + bit_offset); /* row 1 */
    b |= (uint64_t)((col >>  4) & 0xF) << (32 + bit_offset); /* row 2 */
    b |= (uint64_t)((col      ) & 0xF) << (48 + bit_offset); /* row 3 */
    return b;
}

/* ─── Conversion to/from int[4][4] grid ────────────────────────── */

/*
 * Convert a legacy int[4][4] grid (log2 values) to a bitboard.
 */
static inline board_t board_from_grid(int grid[4][4]) {
    board_t b = 0;
    for (int y = 0; y < 4; y++)
        for (int x = 0; x < 4; x++)
            b = board_set(b, y, x, grid[y][x]);
    return b;
}

/*
 * Convert a bitboard back to a legacy int[4][4] grid (log2 values).
 */
static inline void board_to_grid(board_t b, int grid[4][4]) {
    for (int y = 0; y < 4; y++)
        for (int x = 0; x < 4; x++)
            grid[y][x] = board_get(b, y, x);
}

/* ─── Board queries ────────────────────────────────────────────── */

/*
 * Count the number of empty cells (zero nibbles) on the board.
 * Uses a bit-parallel technique: a nibble is zero iff none of its
 * four bits are set.
 */
static inline int board_empty_count(board_t b) {
    /*
     * For each nibble, determine if it is zero by ORing its bits together,
     * then count the zero nibbles.
     *
     * Step 1: Collapse each nibble to a single bit indicating "non-empty".
     * Step 2: Count those bits. Empty count = 16 - non_empty_count.
     */
    uint64_t v = b;
    v |= (v >> 1);
    v |= (v >> 2);
    /* Now bit 0 of each nibble is 1 iff the nibble was non-zero. */
    v &= 0x1111111111111111ULL;

    /* Sum the 16 flag bits. We split into two 32-bit halves so each
     * half has at most 8 flag bits, safely fitting in a nibble sum. */
    uint32_t lo = (uint32_t)(v & 0xFFFFFFFF);
    uint32_t hi = (uint32_t)(v >> 32);
    lo *= 0x11111111U;
    hi *= 0x11111111U;
    int non_empty = (int)((lo >> 28) + (hi >> 28));
    return 16 - non_empty;
}

/*
 * Find the maximum nibble value on the board and return the
 * corresponding actual tile value (1 << max_nibble), or 0 if
 * the board is empty.
 */
static inline int board_max_tile(board_t b) {
    int max_val = 0;
    uint64_t tmp = b;
    for (int i = 0; i < 16; i++) {
        int nibble = (int)(tmp & 0xF);
        if (nibble > max_val) max_val = nibble;
        tmp >>= 4;
    }
    return (max_val > 0) ? (1 << max_val) : 0;
}

/* ─── Fast moves using lookup tables ───────────────────────────── */

/*
 * Slide the entire board left. Each row is independently looked up
 * in the pre-computed merge table.
 * If score is not NULL, the merge score is accumulated there.
 */
static inline board_t do_move_left(board_t b, int *score) {
    board_t result = 0;
    int total = 0;
    for (int y = 0; y < 4; y++) {
        uint16_t row = board_row(b, y);
        result |= (uint64_t)move_left_row[row] << (y * 16);
        total += move_left_score[row];
    }
    if (score) *score = total;
    return result;
}

/*
 * Slide the entire board right.
 */
static inline board_t do_move_right(board_t b, int *score) {
    board_t result = 0;
    int total = 0;
    for (int y = 0; y < 4; y++) {
        uint16_t row = board_row(b, y);
        result |= (uint64_t)move_right_row[row] << (y * 16);
        total += move_right_score[row];
    }
    if (score) *score = total;
    return result;
}

/*
 * Slide the entire board up. Columns are extracted as 16-bit row-format
 * values, merged using the left table (since "up" = "left" when the column
 * is read top-to-bottom), then distributed back.
 */
static inline board_t do_move_up(board_t b, int *score) {
    board_t result = b;
    int total = 0;
    for (int x = 0; x < 4; x++) {
        uint16_t col = board_col(b, x);
        uint16_t merged = move_left_row[col];
        total += move_left_score[col];
        result = board_set_col(result, x, merged);
    }
    if (score) *score = total;
    return result;
}

/*
 * Slide the entire board down. Columns are extracted and merged using
 * the right table (since "down" = "right" when the column is read
 * top-to-bottom).
 */
static inline board_t do_move_down(board_t b, int *score) {
    board_t result = b;
    int total = 0;
    for (int x = 0; x < 4; x++) {
        uint16_t col = board_col(b, x);
        uint16_t merged = move_right_row[col];
        total += move_right_score[col];
        result = board_set_col(result, x, merged);
    }
    if (score) *score = total;
    return result;
}

/*
 * Perform a move in the given direction.
 *   dir: 0 = UP, 1 = RIGHT, 2 = DOWN, 3 = LEFT
 *   score: receives the merge score (may be NULL)
 *   moved: set to 1 if the board changed, 0 otherwise (may be NULL)
 *
 * Returns the new board state.
 */
static inline board_t do_move(board_t b, int dir, int *score, int *moved) {
    board_t result;
    switch (dir) {
        case 0: result = do_move_up(b, score);    break;
        case 1: result = do_move_right(b, score);  break;
        case 2: result = do_move_down(b, score);   break;
        case 3: result = do_move_left(b, score);   break;
        default:
            if (score) *score = 0;
            if (moved) *moved = 0;
            return b;
    }
    if (moved) *moved = (result != b);
    return result;
}

/* ─── Random tile placement ────────────────────────────────────── */

/*
 * Add a random tile to an empty cell on the board.
 * 90% chance of tile 2 (nibble = 1), 10% chance of tile 4 (nibble = 2).
 * Uses the thread-safe trand() PRNG with the given seed.
 * Returns the updated board, or the original if no empty cells exist.
 */
static inline board_t board_add_random(board_t b, unsigned int *seed) {
    /* Collect positions of empty cells. */
    int empty_positions[16];
    int n = 0;
    for (int i = 0; i < 16; i++) {
        if (((b >> (i * 4)) & 0xF) == 0)
            empty_positions[n++] = i;
    }
    if (n == 0) return b;

    int idx = trand(seed) % n;
    int pos = empty_positions[idx];
    int tile = (trand(seed) % 10 < 9) ? 1 : 2; /* 1 = tile 2, 2 = tile 4 */

    b |= (uint64_t)tile << (pos * 4);
    return b;
}

/* ─── Game over detection ──────────────────────────────────────── */

/*
 * Check if no valid moves remain on the board.
 * Returns 1 if the game is over, 0 if at least one move is possible.
 */
static inline int board_game_over(board_t b) {
    /* If there are any empty cells, a move is possible. */
    for (int i = 0; i < 16; i++) {
        if (((b >> (i * 4)) & 0xF) == 0)
            return 0;
    }

    /* No empty cells. Check if any adjacent pair can merge. */
    for (int y = 0; y < 4; y++) {
        for (int x = 0; x < 3; x++) {
            /* Horizontal neighbor */
            if (board_get(b, y, x) == board_get(b, y, x + 1))
                return 0;
        }
    }
    for (int y = 0; y < 3; y++) {
        for (int x = 0; x < 4; x++) {
            /* Vertical neighbor */
            if (board_get(b, y, x) == board_get(b, y + 1, x))
                return 0;
        }
    }

    return 1; /* no moves possible */
}

#endif /* ENGINE_BITBOARD_H */
