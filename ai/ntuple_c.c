/*
 * N-Tuple Network player in C — shared library for the Python server.
 * Uses the bitboard engine for fast move computation.
 * Loads trained weights and performs expectimax search with
 * tile-downgrading support for boards exceeding the training range.
 *
 * Compile: cc -O3 -shared -o ntuple_c.so ntuple_c.c -lm
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "engine_bitboard.h"

/* ─── N-Tuple Network ──────────────────────────────────────── */

#define MAX_TUPLES 20
#define TUPLE_SIZE 6
#define LUT_SIZE   16777216  /* 16^6 */

typedef struct {
    int pos[TUPLE_SIZE][2];
} sym_t;

static int    n_tuples = 0;
static int    n_sym[MAX_TUPLES];
static sym_t  syms[MAX_TUPLES][8];
static float *weights[MAX_TUPLES];
static int    loaded = 0;

/* Base tuples: 17 tuples of 6 positions each */
static const int BASE_TUPLES[17][TUPLE_SIZE][2] = {
    {{0,0},{0,1},{0,2},{0,3},{1,0},{1,1}},
    {{1,0},{1,1},{1,2},{1,3},{2,0},{2,1}},
    {{2,0},{2,1},{2,2},{2,3},{3,0},{3,1}},
    {{0,0},{1,0},{2,0},{3,0},{0,1},{1,1}},
    {{0,1},{1,1},{2,1},{3,1},{0,2},{1,2}},
    {{0,2},{1,2},{2,2},{3,2},{0,3},{1,3}},
    {{0,0},{0,1},{0,2},{1,0},{1,1},{1,2}},
    {{1,0},{1,1},{1,2},{2,0},{2,1},{2,2}},
    {{2,0},{2,1},{2,2},{3,0},{3,1},{3,2}},
    {{0,1},{0,2},{0,3},{1,1},{1,2},{1,3}},
    {{0,0},{0,1},{1,0},{1,1},{2,0},{2,1}},
    {{0,1},{0,2},{1,1},{1,2},{2,1},{2,2}},
    {{0,2},{0,3},{1,2},{1,3},{2,2},{2,3}},
    {{0,0},{0,1},{0,2},{1,0},{1,1},{2,0}},
    {{0,1},{0,2},{0,3},{1,2},{1,3},{2,3}},
    {{0,0},{0,1},{1,1},{1,2},{2,2},{2,3}},
    {{0,2},{0,3},{1,1},{1,2},{2,0},{2,1}},
};

/* Generate unique symmetry variants (rotations + reflections) for tuple t */
static void gen_symmetries(int t) {
    int cur[TUPLE_SIZE][2];
    int cands[8][TUPLE_SIZE][2];
    int nc = 0;

    memcpy(cur, BASE_TUPLES[t], sizeof(cur));

    for (int rot = 0; rot < 4; rot++) {
        /* Identity (or current rotation) */
        memcpy(cands[nc++], cur, sizeof(cur));
        /* Horizontal reflection */
        for (int i = 0; i < TUPLE_SIZE; i++) {
            cands[nc][i][0] = cur[i][0];
            cands[nc][i][1] = 3 - cur[i][1];
        }
        nc++;
        /* Rotate 90 degrees clockwise for next iteration */
        int tmp[TUPLE_SIZE][2];
        for (int i = 0; i < TUPLE_SIZE; i++) {
            tmp[i][0] = cur[i][1];
            tmp[i][1] = 3 - cur[i][0];
        }
        memcpy(cur, tmp, sizeof(cur));
    }

    /* Deduplicate by comparing sorted position sets */
    int count = 0;
    for (int c = 0; c < nc; c++) {
        int sorted[TUPLE_SIZE][2];
        memcpy(sorted, cands[c], sizeof(sorted));
        for (int i = 0; i < TUPLE_SIZE - 1; i++)
            for (int j = i + 1; j < TUPLE_SIZE; j++)
                if (sorted[i][0] > sorted[j][0] ||
                    (sorted[i][0] == sorted[j][0] && sorted[i][1] > sorted[j][1])) {
                    int ty = sorted[i][0], tx = sorted[i][1];
                    sorted[i][0] = sorted[j][0]; sorted[i][1] = sorted[j][1];
                    sorted[j][0] = ty; sorted[j][1] = tx;
                }

        int dup = 0;
        for (int k = 0; k < count; k++) {
            int s2[TUPLE_SIZE][2];
            memcpy(s2, syms[t][k].pos, sizeof(s2));
            for (int i = 0; i < TUPLE_SIZE - 1; i++)
                for (int j = i + 1; j < TUPLE_SIZE; j++)
                    if (s2[i][0] > s2[j][0] ||
                        (s2[i][0] == s2[j][0] && s2[i][1] > s2[j][1])) {
                        int ty = s2[i][0], tx = s2[i][1];
                        s2[i][0] = s2[j][0]; s2[i][1] = s2[j][1];
                        s2[j][0] = ty; s2[j][1] = tx;
                    }
            if (memcmp(sorted, s2, sizeof(sorted)) == 0) { dup = 1; break; }
        }
        if (!dup) {
            memcpy(syms[t][count].pos, cands[c], sizeof(cands[c]));
            count++;
        }
    }
    n_sym[t] = count;
}

/* ─── Bitboard Encoding ───────────────────────────────────── */

/*
 * Encode 6 board cells into a single LUT index.
 * Each cell contributes 4 bits (one nibble), packed big-endian.
 * pos[i] = {row, col} in grid coordinates.
 */
static inline int encode6_board(board_t b, const int pos[][2]) {
    int idx = 0;
    for (int i = 0; i < 6; i++) {
        int y = pos[i][0], x = pos[i][1];
        int shift = y * 16 + (3 - x) * 4;
        int val = (int)((b >> shift) & 0xF);
        idx = idx * 16 + val;
    }
    return idx;
}

/* ─── Evaluation with Tile Downgrading ────────────────────── */

/*
 * Evaluate a board position using the N-Tuple network.
 * If the maximum nibble exceeds the safe training threshold (14 = tile 16384),
 * all non-zero nibbles are decremented until the max fits.  This lets the
 * network handle boards with tiles larger than it was trained on.
 */
static float evaluate(board_t b) {
    /* Find maximum nibble value */
    int max_nib = 0;
    board_t tmp = b;
    for (int i = 0; i < 16; i++) {
        int v = (int)(tmp & 0xF);
        if (v > max_nib) max_nib = v;
        tmp >>= 4;
    }

    /* Downgrade: subtract 1 from all non-zero nibbles until max <= 14 */
    while (max_nib > 14) {
        board_t ones = 0;
        for (int i = 0; i < 16; i++) {
            if ((b >> (i * 4)) & 0xF)
                ones |= (1ULL << (i * 4));
        }
        b -= ones;
        max_nib--;
    }

    /* Normal N-Tuple evaluation */
    float total = 0;
    for (int t = 0; t < n_tuples; t++) {
        const float *w = weights[t];
        for (int s = 0; s < n_sym[t]; s++)
            total += w[encode6_board(b, syms[t][s].pos)];
    }
    return total;
}

/* ─── Transposition Table ─────────────────────────────────── */

#define TT_SIZE (1 << 22)   /* 4M entries */
#define TT_MASK (TT_SIZE - 1)

typedef struct {
    uint64_t key;
    float    value;
    int      depth;
} tt_entry_t;

static tt_entry_t *tt = NULL;

static void tt_init(void) {
    if (!tt) {
        tt = (tt_entry_t *)calloc(TT_SIZE, sizeof(tt_entry_t));
    }
}

static void tt_clear(void) {
    if (tt)
        memset(tt, 0, TT_SIZE * sizeof(tt_entry_t));
}

static inline uint32_t tt_hash(board_t b) {
    /* Mix the board bits into a 32-bit hash */
    uint64_t h = b;
    h ^= h >> 16;
    h *= 0x45d9f3b;
    h ^= h >> 16;
    return (uint32_t)(h & TT_MASK);
}

static inline int tt_lookup(board_t b, int depth, float *value) {
    uint32_t idx = tt_hash(b);
    tt_entry_t *e = &tt[idx];
    if (e->key == b && e->depth == depth) {
        *value = e->value;
        return 1;
    }
    return 0;
}

static inline void tt_store(board_t b, int depth, float value) {
    uint32_t idx = tt_hash(b);
    tt_entry_t *e = &tt[idx];
    e->key   = b;
    e->value = value;
    e->depth = depth;
}

/* ─── Expectimax Search ───────────────────────────────────── */

/*
 * Search depths:
 *   0 = 1-ply  (move + evaluate)
 *   1 = 3-ply  (move + chance + move + evaluate)
 *   2 = 5-ply  (move + chance + move + chance + move + evaluate)
 *   3 = 7-ply  (move + chance + move + chance + move + chance + move + evaluate)
 */

static float chance_node(board_t b, int depth);

static float max_node(board_t b, int depth) {
    float best = -1e18f;
    int any = 0;
    for (int d = 0; d < 4; d++) {
        int score;
        board_t after = do_move(b, d, &score, NULL);
        if (after == b) continue;  /* move had no effect */
        any = 1;
        float v;
        if (depth <= 0)
            v = (float)score + evaluate(after);
        else
            v = (float)score + chance_node(after, depth - 1);
        if (v > best) best = v;
    }
    return any ? best : evaluate(b);
}

static float chance_node(board_t b, int depth) {
    /* Check transposition table */
    float cached;
    if (tt_lookup(b, depth, &cached))
        return cached;

    /* Find all empty cells */
    int empty_pos[16];
    int ne = 0;
    for (int i = 0; i < 16; i++) {
        if (((b >> (i * 4)) & 0xF) == 0)
            empty_pos[ne++] = i;
    }

    if (ne == 0) {
        float val = evaluate(b);
        tt_store(b, depth, val);
        return val;
    }

    /* Evaluate ALL empty cells deterministically */
    float total = 0;
    for (int i = 0; i < ne; i++) {
        int pos = empty_pos[i];
        int shift = pos * 4;

        /* Place tile 2 (nibble = 1) */
        board_t b2 = b | ((uint64_t)1 << shift);
        total += 0.9f * max_node(b2, depth);

        /* Place tile 4 (nibble = 2) */
        board_t b4 = b | ((uint64_t)2 << shift);
        total += 0.1f * max_node(b4, depth);
    }
    float result = total / ne;
    tt_store(b, depth, result);
    return result;
}

/* ─── Grid-to-Bitboard Conversion ─────────────────────────── */

/*
 * Convert a flat int[16] grid of actual tile values (0, 2, 4, 8, ...)
 * to a bitboard with log2-encoded nibbles.
 */
static inline board_t grid_flat_to_board(const int *grid_flat) {
    board_t b = 0;
    for (int y = 0; y < 4; y++)
        for (int x = 0; x < 4; x++) {
            int val = grid_flat[y * 4 + x];
            int log_val = 0;
            if (val > 0) {
                int v = val;
                while (v > 1) { v >>= 1; log_val++; }
            }
            int shift = y * 16 + (3 - x) * 4;
            b |= ((uint64_t)log_val << shift);
        }
    return b;
}

/* ─── Public API ──────────────────────────────────────────── */

/*
 * Load N-Tuple weights from a binary file.
 * Format: int n_tuples, then for each tuple: int lut_size, float[lut_size].
 * Returns 1 on success, 0 on failure.
 */
int ntuple_load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return 0;

    /* Build the bitboard move lookup tables */
    build_move_tables();

    /* Initialize transposition table */
    tt_init();

    int n;
    if (fread(&n, sizeof(int), 1, f) != 1) { fclose(f); return 0; }
    n_tuples = n;

    for (int t = 0; t < n; t++) {
        int size;
        if (fread(&size, sizeof(int), 1, f) != 1) { fclose(f); return 0; }
        weights[t] = (float *)malloc(size * sizeof(float));
        if (!weights[t]) { fclose(f); return 0; }
        if ((int)fread(weights[t], sizeof(float), size, f) != size) {
            fclose(f);
            return 0;
        }
        gen_symmetries(t);
    }
    fclose(f);
    loaded = 1;
    printf("N-Tuple C: loaded %s (%d tuples)\n", path, n);
    return 1;
}

/*
 * Select the best action for the given board state.
 *
 * grid_flat: int[16] row-major, actual tile values (0, 2, 4, 8, ...)
 * search_depth: 0 = 1-ply, 1 = 3-ply, 2 = 5-ply, 3 = 7-ply
 * Returns: best direction (0 = UP, 1 = RIGHT, 2 = DOWN, 3 = LEFT)
 */
int ntuple_select_action(int *grid_flat, int search_depth) {
    if (!loaded) return 0;

    board_t b = grid_flat_to_board(grid_flat);

    /* Clear the transposition table for each new decision */
    if (search_depth > 0)
        tt_clear();

    int best_action = 0;
    float best_value = -1e18f;

    for (int d = 0; d < 4; d++) {
        int score;
        board_t after = do_move(b, d, &score, NULL);
        if (after == b) continue;  /* invalid move */

        float value;
        if (search_depth <= 0)
            value = (float)score + evaluate(after);
        else
            value = (float)score + chance_node(after, search_depth - 1);

        if (value > best_value) {
            best_value = value;
            best_action = d;
        }
    }
    return best_action;
}
