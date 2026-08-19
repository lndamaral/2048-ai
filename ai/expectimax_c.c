/*
 * Optimized Expectimax engine for 2048.
 *
 * Board: uint64_t, 4 rows of 16 bits.
 *   Row 0 = bits 0-15, Row 1 = bits 16-31, Row 2 = bits 32-47, Row 3 = bits 48-63
 *   Each cell = 4 bits (nibble): 0=empty, 1=2, 2=4, ..., 15=32768
 *
 * Optimizations:
 *   - Lookup tables for merge (65536 entries)
 *   - Lookup tables for per-row heuristics (65536 entries x 4 positions)
 *   - Transpose via bit manipulation O(1)
 *   - Transposition table with Zobrist hashing (4M entries)
 *   - Iterative deepening with time budget
 *
 * Compile: cc -O3 -shared -o expectimax_c.so expectimax_c.c -lm
 */

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>

/* ─── Types ─────────────────────────────────────────────────── */

typedef uint64_t board_t;
typedef uint16_t row_t;

/* ─── Timing ────────────────────────────────────────────────── */

static inline long long now_us(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (long long)tv.tv_sec * 1000000LL + tv.tv_usec;
}

static long long deadline_us = 0;  /* 0 = no deadline */

/* ─── Lookup tables ─────────────────────────────────────────── */

static row_t  move_left[65536];
static row_t  move_right[65536];
static int    move_score[65536];     /* score for left move */
static int    move_score_r[65536];   /* score for right move */

/* Per-row heuristic: mono + smooth + empty + edge */
static float  heur_row[65536];

/* Snake pattern: 8 orientations x 4 row positions */
static float  snake_row[8][4][65536];

static int    tables_built = 0;

/* Snake weights for the 8 orientations (4 rotations x 2 reflections) */
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

/* Heuristic weights */
#define W_EMPTY     270.0f
#define W_MONO       47.0f
#define W_SMOOTH     12.0f
#define W_EDGE       11.0f
#define W_MERGE     700.0f
#define W_SNAKE       1.0f
#define W_CORNER     40.0f

static void build_tables(void) {
    if (tables_built) return;

    for (int enc = 0; enc < 65536; enc++) {
        int c[4] = {
            (enc >> 12) & 0xF,
            (enc >> 8)  & 0xF,
            (enc >> 4)  & 0xF,
            enc & 0xF
        };

        /* ── Merge left ── */
        int nz[4], nz_len = 0;
        for (int i = 0; i < 4; i++)
            if (c[i]) nz[nz_len++] = c[i];

        int res[4] = {0, 0, 0, 0};
        int score = 0, pos = 0;
        for (int i = 0; i < nz_len; i++) {
            if (i + 1 < nz_len && nz[i] == nz[i + 1]) {
                int m = nz[i] + 1;
                if (m > 15) m = 15;
                res[pos++] = m;
                score += (1 << m);
                i++;
            } else {
                res[pos++] = nz[i];
            }
        }
        row_t left = (res[0] << 12) | (res[1] << 8) | (res[2] << 4) | res[3];
        move_left[enc] = left;
        move_score[enc] = score;

        /* ── Merge right = reverse(merge_left(reverse)) ── */
        int rev = (c[3] << 12) | (c[2] << 8) | (c[1] << 4) | c[0];
        /* Will fill after all left are done */

        /* ── Row heuristic ── */
        float h = 0;

        /* Empty cells */
        int empty = 0;
        for (int i = 0; i < 4; i++) if (c[i] == 0) empty++;
        h += W_EMPTY * empty;

        /* Monotonicity (best of left-to-right / right-to-left) */
        float mono_inc = 0, mono_dec = 0;
        for (int i = 0; i < 3; i++) {
            if (c[i] > c[i+1])
                mono_dec += (float)(c[i] - c[i+1]) * c[i];
            else if (c[i] < c[i+1])
                mono_inc += (float)(c[i+1] - c[i]) * c[i+1];
        }
        h += W_MONO * (mono_inc > mono_dec ? mono_inc : mono_dec);

        /* Smoothness + merge potential */
        for (int i = 0; i < 3; i++) {
            if (c[i] != 0 && c[i+1] != 0) {
                if (c[i] == c[i+1])
                    h += W_MERGE;
                else
                    h -= W_SMOOTH * abs(c[i] - c[i+1]);
            }
        }

        /* Edge bonus */
        if (c[0]) h += W_EDGE * c[0] * c[0];
        if (c[3]) h += W_EDGE * c[3] * c[3];

        heur_row[enc] = h;

        /* ── Snake tables ── */
        for (int o = 0; o < 8; o++) {
            for (int y = 0; y < 4; y++) {
                float s = 0;
                for (int x = 0; x < 4; x++)
                    s += (float)c[x] * SNAKE_W[o][y][x];
                snake_row[o][y][enc] = s;
            }
        }
    }

    /* Second pass: merge_right */
    for (int enc = 0; enc < 65536; enc++) {
        int c[4] = {
            (enc >> 12) & 0xF,
            (enc >> 8)  & 0xF,
            (enc >> 4)  & 0xF,
            enc & 0xF
        };
        int rev = (c[3] << 12) | (c[2] << 8) | (c[1] << 4) | c[0];
        row_t lr = move_left[rev];
        int r0 = (lr >> 12) & 0xF, r1 = (lr >> 8) & 0xF;
        int r2 = (lr >> 4) & 0xF, r3 = lr & 0xF;
        move_right[enc] = (r3 << 12) | (r2 << 8) | (r1 << 4) | r0;
        move_score_r[enc] = move_score[rev];
    }

    tables_built = 1;
}

/* ─── Board operations ──────────────────────────────────────── */

static inline row_t board_row(board_t b, int y) {
    return (row_t)((b >> (y * 16)) & 0xFFFF);
}

static inline board_t board_set_row(board_t b, int y, row_t r) {
    int shift = y * 16;
    b &= ~((uint64_t)0xFFFF << shift);
    b |= ((uint64_t)r << shift);
    return b;
}

/* Extract column x as a row_t (top cell at high nibble) */
static inline row_t board_col(board_t b, int x) {
    int c0 = (board_row(b, 0) >> ((3 - x) * 4)) & 0xF;
    int c1 = (board_row(b, 1) >> ((3 - x) * 4)) & 0xF;
    int c2 = (board_row(b, 2) >> ((3 - x) * 4)) & 0xF;
    int c3 = (board_row(b, 3) >> ((3 - x) * 4)) & 0xF;
    return (c0 << 12) | (c1 << 8) | (c2 << 4) | c3;
}

/* Set column x from a row_t */
static inline board_t board_set_col(board_t b, int x, row_t col) {
    int shift = (3 - x) * 4;
    uint64_t col_mask = 0;
    for (int y = 0; y < 4; y++)
        col_mask |= ((uint64_t)0xF << (y * 16 + shift));

    b &= ~col_mask;
    b |= ((uint64_t)((col >> 12) & 0xF)) << (0 * 16 + shift);
    b |= ((uint64_t)((col >> 8)  & 0xF)) << (1 * 16 + shift);
    b |= ((uint64_t)((col >> 4)  & 0xF)) << (2 * 16 + shift);
    b |= ((uint64_t)( col        & 0xF)) << (3 * 16 + shift);
    return b;
}

/* Move operations using lookup tables */
static board_t do_move(board_t b, int dir, int *score) {
    *score = 0;
    board_t nb;

    if (dir == 3) { /* LEFT */
        nb = 0;
        for (int y = 0; y < 4; y++) {
            row_t r = board_row(b, y);
            *score += move_score[r];
            nb |= ((uint64_t)move_left[r]) << (y * 16);
        }
    } else if (dir == 1) { /* RIGHT */
        nb = 0;
        for (int y = 0; y < 4; y++) {
            row_t r = board_row(b, y);
            *score += move_score_r[r];
            nb |= ((uint64_t)move_right[r]) << (y * 16);
        }
    } else if (dir == 0) { /* UP — merge columns upward */
        nb = b;
        for (int x = 0; x < 4; x++) {
            row_t col = board_col(b, x);
            *score += move_score[col];
            nb = board_set_col(nb, x, move_left[col]);
        }
    } else { /* DOWN — merge columns downward */
        nb = b;
        for (int x = 0; x < 4; x++) {
            row_t col = board_col(b, x);
            *score += move_score_r[col];
            nb = board_set_col(nb, x, move_right[col]);
        }
    }

    return nb;
}

/* ─── 2D heuristic evaluation (matches working Python version) ── */

static inline int board_get_cell(board_t b, int y, int x) {
    return (board_row(b, y) >> ((3 - x) * 4)) & 0xF;
}

static float evaluate(board_t b) {
    /* Extract grid as nibbles and compute actual values */
    int grid[4][4];     /* nibble (log2) values */
    float lg[4][4];     /* same as nibble, used as log2 */
    int empty = 0;
    int max_val = 0;

    for (int y = 0; y < 4; y++) {
        for (int x = 0; x < 4; x++) {
            int v = board_get_cell(b, y, x);
            grid[y][x] = v;
            lg[y][x] = (float)v;
            if (v == 0) empty++;
            if (v > max_val) max_val = v;
        }
    }

    float score = 0;

    /* 1. Snake pattern — best of 8 orientations, using ACTUAL tile values */
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

    /* 5. Max tile in corner — strong bonus/penalty */
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

    /* 6. Edge monotonicity — tiles along edges should decrease from corner */
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

/* ─── Transposition table ───────────────────────────────────── */

#define CACHE_BITS 22
#define CACHE_SIZE (1 << CACHE_BITS)  /* 4M entries */
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
        int score;
        board_t nb = do_move(b, d, &score);
        if (nb == b) continue;
        any = 1;
        float v = (float)score + chance_node(nb, depth - 1);
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
    int ey[16], ex[16];
    int n_empty = 0;
    for (int y = 0; y < 4; y++) {
        row_t r = board_row(b, y);
        for (int x = 0; x < 4; x++) {
            if (((r >> ((3 - x) * 4)) & 0xF) == 0) {
                ey[n_empty] = y;
                ex[n_empty] = x;
                n_empty++;
            }
        }
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
        int idx = indices[i];
        int y = ey[idx], x = ex[idx];
        int shift = y * 16 + (3 - x) * 4;

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
        int score;
        board_t nb = do_move(b, d, &score);
        if (nb == b) continue;

        float v = (float)score + chance_node(nb, depth - 1);
        if (search_aborted) break;
        if (v > result.score) {
            result.score = v;
            result.action = d;
        }
    }

    return result;
}

/* ─── Public API ────────────────────────────────────────────── */

void init(void) {
    build_tables();
    srand((unsigned)time(NULL));
}

/*
 * Iterative deepening com time budget.
 * grid: int[16] row-major, actual values (0, 2, 4, ...)
 * time_budget_ms: max milliseconds per move (0 = use max_depth directly)
 * max_depth: absolute maximum depth
 * Returns: best direction (0=up, 1=right, 2=down, 3=left)
 */
int select_action(int *grid, int time_budget_ms, int max_depth) {
    build_tables();

    /* Convert grid to board */
    board_t b = 0;
    for (int y = 0; y < 4; y++) {
        for (int x = 0; x < 4; x++) {
            int val = grid[y * 4 + x];
            int log_val = 0;
            if (val > 0) {
                int v = val;
                while (v > 1) { v >>= 1; log_val++; }
            }
            int shift = y * 16 + (3 - x) * 4;
            b |= ((uint64_t)log_val << shift);
        }
    }

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
            break;  /* Time's up — use result from previous depth */
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
    build_tables();

    board_t b = 0;
    for (int y = 0; y < 4; y++) {
        for (int x = 0; x < 4; x++) {
            int val = grid[y * 4 + x];
            int log_val = 0;
            if (val > 0) {
                int v = val;
                while (v > 1) { v >>= 1; log_val++; }
            }
            int shift = y * 16 + (3 - x) * 4;
            b |= ((uint64_t)log_val << shift);
        }
    }

    long long start = now_us();
    if (time_budget_ms > 0)
        deadline_us = start + (long long)time_budget_ms * 1000LL;
    else
        deadline_us = 0;

    search_result_t best = {0, -1e18f, 0};

    int max_d = (time_budget_ms > 0) ? max_depth : max_depth;
    for (int depth = 1; depth <= max_d; depth++) {
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
