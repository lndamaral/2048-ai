/*
 * Optimized Expectimax engine for 2048 (bitboard version).
 *
 * Uses engine_bitboard.h for all board operations and move lookup tables.
 * Board: uint64_t (board_t), 16 nibbles of 4 bits each.
 *
 * Optimizations:
 *   - Lookup tables for merge via engine_bitboard.h (65536 entries)
 *   - Transposition table with Zobrist hashing (16M entries)
 *   - Iterative deepening with time budget
 *
 * Heuristic weights below were calibrated with the previous engine.
 * NOTE: They may need re-calibration after switching to the bitboard engine.
 *
 * Compile: cc -O3 -shared -o expectimax_c.so expectimax_c.c -lm
 */

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>

#include "engine_bitboard.h"

/* ─── Timing ────────────────────────────────────────────────── */

static inline long long now_us(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (long long)tv.tv_sec * 1000000LL + tv.tv_usec;
}

static long long deadline_us = 0;  /* 0 = no deadline */

/* ─── Snake pattern weights (8 orientations: 4 rotations x 2 reflections) ── */

static const float SNAKE_W[8][4][4] = {
    /* 0 */ {{32768,16384,8192,4096},{256,512,1024,2048},{128,64,32,16},{1,2,4,8}},
    /* 1 */ {{4096,8192,16384,32768},{2048,1024,512,256},{16,32,64,128},{8,4,2,1}},
    /* 2 */ {{1,128,256,32768},{2,64,512,16384},{4,32,1024,8192},{8,16,2048,4096}},
    /* 3 */ {{32768,256,128,1},{16384,512,64,2},{8192,1024,32,4},{4096,2048,16,8}},
    /* 4 */ {{8,4,2,1},{16,32,64,128},{2048,1024,512,256},{32768,16384,8192,4096}},
    /* 5 */ {{1,2,4,8},{128,64,32,16},{256,512,1024,2048},{4096,8192,16384,32768}},
    /* 6 */ {{8,16,2048,4096},{4,32,1024,8192},{2,64,512,16384},{1,128,256,32768}},
    /* 7 */ {{4096,2048,16,8},{8192,1024,32,4},{16384,512,64,2},{32768,256,128,1}},
};

/* ─── 2D Heuristic evaluation ──────────────────────────────── */
/*
 * Calibrated weights (from previous engine). NOTE: need re-calibration
 * after the bitboard migration since move/scoring behavior is identical
 * but sampling may differ slightly.
 *
 * snake=0.95  empty=2.0  mono=1.0  smooth=0.1  corner=1.0
 * edge_mono=0.5  corner_penalty=0.5
 */

static float evaluate(board_t b) {
    /* Extract nibble values to compute heuristics */
    int grid[4][4];
    float lg[4][4]; /* log2 values (same as nibble) */
    int empty = 0;
    int max_val = 0;

    for (int y = 0; y < 4; y++)
        for (int x = 0; x < 4; x++) {
            int v = board_get(b, y, x);
            grid[y][x] = v;
            lg[y][x] = (float)v;
            if (v == 0) empty++;
            if (v > max_val) max_val = v;
        }

    float score = 0;

    /* 1. Snake pattern -- best of 8 orientations, using actual tile values */
    float best_snake = -1e18f;
    for (int o = 0; o < 8; o++) {
        float s = 0;
        for (int y = 0; y < 4; y++)
            for (int x = 0; x < 4; x++) {
                float tile_val = (grid[y][x] > 0) ? (float)(1 << grid[y][x]) : 0;
                s += tile_val * SNAKE_W[o][y][x];
            }
        if (s > best_snake) best_snake = s;
    }
    score += best_snake * 0.95f;

    /* 2. Empty cells */
    score += (empty > 0) ? logf((float)empty + 1.0f) * 2.0f : 0;

    /* 3. Monotonicity (using log2 values) */
    float mono = 0;
    for (int i = 0; i < 4; i++) {
        float left = 0, right = 0;
        for (int j = 0; j < 3; j++) {
            float diff = lg[i][j] - lg[i][j+1];
            if (diff > 0) left += diff;
            else right -= diff;
        }
        mono -= (left < right) ? left : right;

        float up = 0, down = 0;
        for (int j = 0; j < 3; j++) {
            float diff = lg[j][i] - lg[j+1][i];
            if (diff > 0) up += diff;
            else down -= diff;
        }
        mono -= (up < down) ? up : down;
    }
    score += mono * 1.0f;

    /* 4. Smoothness */
    float smooth = 0;
    for (int y = 0; y < 4; y++)
        for (int x = 0; x < 4; x++) {
            if (grid[y][x] == 0) continue;
            float v = lg[y][x];
            if (x < 3 && grid[y][x+1] > 0)
                smooth -= fabsf(v - lg[y][x+1]);
            if (y < 3 && grid[y+1][x] > 0)
                smooth -= fabsf(v - lg[y+1][x]);
        }
    score += smooth * 0.1f;

    /* 5. Max tile in corner -- strong bonus/penalty */
    int corners[4] = {grid[0][0], grid[0][3], grid[3][0], grid[3][3]};
    int max_in_corner = 0;
    for (int i = 0; i < 4; i++) {
        if (corners[i] == max_val) { max_in_corner = 1; break; }
    }
    if (max_val > 0) {
        float tile_val = (float)(1 << max_val);
        if (max_in_corner)
            score += tile_val * 1.0f;     /* bonus: reward corner placement */
        else
            score -= tile_val * 0.5f;     /* penalty: max tile not in corner */
    }

    /* 6. Edge monotonicity -- tiles along edges should decrease from corner */
    float edge_mono = 0;
    /* Top edge */
    for (int x = 0; x < 3; x++) {
        if (grid[0][x] > 0 && grid[0][x+1] > 0 && grid[0][x] >= grid[0][x+1])
            edge_mono += (float)grid[0][x];
    }
    /* Left edge */
    for (int y = 0; y < 3; y++) {
        if (grid[y][0] > 0 && grid[y+1][0] > 0 && grid[y][0] >= grid[y+1][0])
            edge_mono += (float)grid[y][0];
    }
    score += edge_mono * 0.5f;

    return score;
}

/* ─── Transposition table (16M entries) ────────────────────── */

#define CACHE_BITS 24
#define CACHE_SIZE (1 << CACHE_BITS)  /* 16M entries */
#define CACHE_MASK (CACHE_SIZE - 1)

typedef struct {
    board_t key;
    float   value;
    int8_t  depth;
    int8_t  valid;
} cache_entry_t;

static cache_entry_t cache[CACHE_SIZE];

static void cache_clear(void) {
    memset(cache, 0, sizeof(cache));
}

static inline uint32_t cache_hash(board_t b, int depth) {
    uint64_t h = b ^ (b >> 16);
    h *= 0x45D9F3B + depth * 0x9E3779B9;
    h ^= h >> 16;
    return (uint32_t)(h & CACHE_MASK);
}

/* ─── Expectimax search ─────────────────────────────────────── */

static int search_aborted = 0;
static long long nodes_searched = 0;

static float chance_node(board_t b, int depth);

static float max_node(board_t b, int depth) {
    /* Check time budget periodically */
    if (deadline_us && (nodes_searched & 0xFFF) == 0) {
        if (now_us() >= deadline_us) {
            search_aborted = 1;
            return 0;
        }
    }
    nodes_searched++;

    /* Cache lookup */
    uint32_t idx = cache_hash(b, depth);
    cache_entry_t *e = &cache[idx];
    if (e->valid && e->key == b && e->depth == depth)
        return e->value;

    float best = -1e18f;
    int any = 0;

    for (int d = 0; d < 4; d++) {
        int mscore;
        board_t nb = do_move(b, d, &mscore, NULL);
        if (nb == b) continue;
        any = 1;
        float v = (float)mscore + chance_node(nb, depth - 1);
        if (search_aborted) return 0;
        if (v > best) best = v;
    }

    float result = any ? best : evaluate(b);

    /* Cache store */
    e->key = b;
    e->depth = (int8_t)depth;
    e->value = result;
    e->valid = 1;

    return result;
}

static float chance_node(board_t b, int depth) {
    if (search_aborted) return 0;

    /* Find empty cells */
    int positions[16];
    int n_empty = 0;
    for (int i = 0; i < 16; i++) {
        if (((b >> (i * 4)) & 0xF) == 0)
            positions[n_empty++] = i;
    }

    if (n_empty == 0)
        return evaluate(b);

    /* Sample if many empties */
    int sample_n = n_empty;
    int indices[16];
    for (int i = 0; i < 16; i++) indices[i] = i;

    if (n_empty > 5) {
        sample_n = 5;
        for (int i = 0; i < sample_n; i++) {
            int j = i + rand() % (n_empty - i);
            int tmp = indices[i]; indices[i] = indices[j]; indices[j] = tmp;
        }
    }

    float total = 0;
    for (int i = 0; i < sample_n; i++) {
        int pos = positions[indices[i]];
        int shift = pos * 4;

        /* tile=2 (log2=1), prob=0.9 */
        board_t b2 = b | ((uint64_t)1 << shift);
        /* tile=4 (log2=2), prob=0.1 */
        board_t b4 = b | ((uint64_t)2 << shift);

        if (depth <= 0) {
            total += 0.9f * evaluate(b2) + 0.1f * evaluate(b4);
        } else {
            total += 0.9f * max_node(b2, depth) + 0.1f * max_node(b4, depth);
        }
        if (search_aborted) return 0;
    }

    return total / sample_n;
}

/* ─── Iterative Deepening ───────────────────────────────────── */

typedef struct {
    int action;
    float score;
    int depth_reached;
} search_result_t;

static search_result_t search_at_depth(board_t b, int depth) {
    search_result_t result = {0, -1e18f, depth};

    for (int d = 0; d < 4; d++) {
        int mscore;
        board_t nb = do_move(b, d, &mscore, NULL);
        if (nb == b) continue;

        float v = (float)mscore + chance_node(nb, depth - 1);
        if (search_aborted) break;
        if (v > result.score) {
            result.score = v;
            result.action = d;
        }
    }

    return result;
}

/* ─── Board conversion from int[16] ────────────────────────── */

static inline board_t board_from_flat(const int *grid) {
    board_t b = 0;
    for (int y = 0; y < 4; y++) {
        for (int x = 0; x < 4; x++) {
            int val = grid[y * 4 + x];
            int log_val = 0;
            if (val > 0) {
                int v = val;
                while (v > 1) { v >>= 1; log_val++; }
            }
            b = board_set(b, y, x, log_val);
        }
    }
    return b;
}

/* ─── Public API ────────────────────────────────────────────── */

void init(void) {
    build_move_tables();
    srand((unsigned)time(NULL));
}

/*
 * Iterative deepening with time budget.
 * grid: int[16] row-major, actual values (0, 2, 4, ...)
 * time_budget_ms: max milliseconds per move (0 = use max_depth directly)
 * max_depth: absolute maximum depth
 * Returns: best direction (0=up, 1=right, 2=down, 3=left)
 */
int select_action(int *grid, int time_budget_ms, int max_depth) {
    build_move_tables();

    board_t b = board_from_flat(grid);

    if (time_budget_ms <= 0) {
        /* Fixed depth mode */
        cache_clear();
        search_aborted = 0;
        nodes_searched = 0;
        deadline_us = 0;
        search_result_t r = search_at_depth(b, max_depth);
        return r.action;
    }

    /* Iterative deepening with time budget */
    long long start = now_us();
    deadline_us = start + (long long)time_budget_ms * 1000LL;

    search_result_t best = {0, -1e18f, 0};

    for (int depth = 1; depth <= max_depth; depth++) {
        cache_clear();
        search_aborted = 0;
        nodes_searched = 0;

        search_result_t r = search_at_depth(b, depth);

        if (!search_aborted) {
            best = r;
            best.depth_reached = depth;
        } else {
            break;  /* Time's up -- use result from previous depth */
        }

        /* If we've used >60% of budget, don't start next depth */
        long long elapsed = now_us() - start;
        long long budget = (long long)time_budget_ms * 1000LL;
        if (elapsed > budget * 6 / 10)
            break;
    }

    deadline_us = 0;
    return best.action;
}

/*
 * Like select_action but also returns depth reached.
 * result[0] = action, result[1] = depth reached
 */
void select_action_info(int *grid, int time_budget_ms, int max_depth, int *result) {
    build_move_tables();

    board_t b = board_from_flat(grid);

    long long start = now_us();
    if (time_budget_ms > 0)
        deadline_us = start + (long long)time_budget_ms * 1000LL;
    else
        deadline_us = 0;

    search_result_t best = {0, -1e18f, 0};

    for (int depth = 1; depth <= max_depth; depth++) {
        cache_clear();
        search_aborted = 0;
        nodes_searched = 0;

        search_result_t r = search_at_depth(b, depth);

        if (!search_aborted) {
            best = r;
            best.depth_reached = depth;
        } else {
            break;
        }

        if (time_budget_ms <= 0) break; /* fixed depth: single iteration */

        long long elapsed = now_us() - start;
        long long budget = (long long)time_budget_ms * 1000LL;
        if (elapsed > budget * 6 / 10)
            break;
    }

    deadline_us = 0;
    result[0] = best.action;
    result[1] = best.depth_reached;
}
