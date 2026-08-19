/*
 * N-Tuple Network Training for 2048 — optimized version.
 *
 * Optimizations:
 *   - Move lookup tables (pre-computes merge for all 65536 rows)
 *   - Pre-computed log2
 *   - Multithreading hogwild (N games in parallel, no locks)
 *   - TC-learning (per-weight adaptive LR)
 *   - 3-ply search during training
 *
 * Compile: cc -O3 -o ntuple_train ntuple_train.c -lm -lpthread
 * Usage:   ./ntuple_train --episodes 200000 --depth 1 --tc --threads 8
 */

#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdatomic.h>

/* ─── Pre-computed tables ───────────────────────────────────── */

/* Move lookup: row (4 nibbles as actual values) -> merged row + score */
typedef struct {
    int cells[4];
    int score;
} row_result_t;

/* We encode a row of actual values as an index into these tables.
 * Since actual values can be 0,2,4,...,32768 (log2: 0-15), we use
 * the log2 encoding: row_index = c0*16^3 + c1*16^2 + c2*16 + c3
 * where ci = log2(value) or 0 if empty.
 */
static row_result_t move_left_table[65536];
static int tables_built = 0;

/* Log2 lookup for tile values */
static int log2_table[65537]; /* index by actual value, max 65536 */

static void build_log2_table(void) {
    memset(log2_table, 0, sizeof(log2_table));
    for (int i = 1; i <= 16; i++)
        log2_table[1 << i] = i;
}

static inline int fast_log2(int val) {
    return (val > 0 && val <= 65536) ? log2_table[val] : 0;
}

static int encode_row_log(int c0, int c1, int c2, int c3) {
    return (c0 << 12) | (c1 << 8) | (c2 << 4) | c3;
}

static void build_move_tables(void) {
    if (tables_built) return;
    build_log2_table();

    for (int enc = 0; enc < 65536; enc++) {
        int c[4] = {
            (enc >> 12) & 0xF,
            (enc >> 8)  & 0xF,
            (enc >> 4)  & 0xF,
            enc & 0xF
        };

        /* Merge left (using log2 values) */
        int nz[4], nz_len = 0;
        for (int i = 0; i < 4; i++)
            if (c[i]) nz[nz_len++] = c[i];

        int res[4] = {0, 0, 0, 0};
        int score = 0, pos = 0;
        for (int i = 0; i < nz_len; i++) {
            if (i + 1 < nz_len && nz[i] == nz[i+1]) {
                int m = nz[i] + 1;
                if (m > 15) m = 15;
                res[pos++] = m;
                score += (1 << m);
                i++;
            } else {
                res[pos++] = nz[i];
            }
        }

        move_left_table[enc].cells[0] = res[0];
        move_left_table[enc].cells[1] = res[1];
        move_left_table[enc].cells[2] = res[2];
        move_left_table[enc].cells[3] = res[3];
        move_left_table[enc].score = score;
    }
    tables_built = 1;
}

/* ─── Fast Game Engine ──────────────────────────────────────── */

typedef int grid_t[4][4];  /* stores log2 values: 0=empty, 1=2, 2=4, ..., 15=32768 */

static void grid_clear(grid_t g) { memset(g, 0, sizeof(grid_t)); }
static void grid_copy(grid_t dst, const grid_t src) { memcpy(dst, src, sizeof(grid_t)); }

static int grid_empty_count(const grid_t g) {
    int c = 0;
    for (int y = 0; y < 4; y++)
        for (int x = 0; x < 4; x++)
            if (g[y][x] == 0) c++;
    return c;
}

/* Thread-safe random using per-thread seed */
static inline int trand(unsigned int *seed) {
    *seed = *seed * 1103515245 + 12345;
    return (*seed >> 16) & 0x7FFF;
}

static void grid_add_random_r(grid_t g, unsigned int *seed) {
    int ey[16], ex[16], n = 0;
    for (int y = 0; y < 4; y++)
        for (int x = 0; x < 4; x++)
            if (g[y][x] == 0) { ey[n] = y; ex[n] = x; n++; }
    if (n == 0) return;
    int idx = trand(seed) % n;
    g[ey[idx]][ex[idx]] = (trand(seed) % 10 < 9) ? 1 : 2; /* log2: 1=tile2, 2=tile4 */
}

static int grid_max_tile_actual(const grid_t g) {
    int m = 0;
    for (int y = 0; y < 4; y++)
        for (int x = 0; x < 4; x++)
            if (g[y][x] > m) m = g[y][x];
    return (m > 0) ? (1 << m) : 0;
}

/* Move using lookup tables. Grid stores log2 values. */
static int grid_move(grid_t after, const grid_t g, int dir, int *moved) {
    grid_copy(after, g);
    int total_score = 0;

    if (dir == 3) { /* LEFT */
        for (int y = 0; y < 4; y++) {
            int enc = encode_row_log(after[y][0], after[y][1], after[y][2], after[y][3]);
            row_result_t *r = &move_left_table[enc];
            after[y][0] = r->cells[0]; after[y][1] = r->cells[1];
            after[y][2] = r->cells[2]; after[y][3] = r->cells[3];
            total_score += r->score;
        }
    } else if (dir == 1) { /* RIGHT */
        for (int y = 0; y < 4; y++) {
            int enc = encode_row_log(after[y][3], after[y][2], after[y][1], after[y][0]);
            row_result_t *r = &move_left_table[enc];
            after[y][3] = r->cells[0]; after[y][2] = r->cells[1];
            after[y][1] = r->cells[2]; after[y][0] = r->cells[3];
            total_score += r->score;
        }
    } else if (dir == 0) { /* UP */
        for (int x = 0; x < 4; x++) {
            int enc = encode_row_log(after[0][x], after[1][x], after[2][x], after[3][x]);
            row_result_t *r = &move_left_table[enc];
            after[0][x] = r->cells[0]; after[1][x] = r->cells[1];
            after[2][x] = r->cells[2]; after[3][x] = r->cells[3];
            total_score += r->score;
        }
    } else { /* DOWN */
        for (int x = 0; x < 4; x++) {
            int enc = encode_row_log(after[3][x], after[2][x], after[1][x], after[0][x]);
            row_result_t *r = &move_left_table[enc];
            after[3][x] = r->cells[0]; after[2][x] = r->cells[1];
            after[1][x] = r->cells[2]; after[0][x] = r->cells[3];
            total_score += r->score;
        }
    }

    *moved = (memcmp(after, g, sizeof(grid_t)) != 0);
    return total_score;
}

static int grid_game_over(const grid_t g) {
    for (int y = 0; y < 4; y++)
        for (int x = 0; x < 4; x++) {
            if (g[y][x] == 0) return 0;
            if (x < 3 && g[y][x] == g[y][x+1]) return 0;
            if (y < 3 && g[y][x] == g[y+1][x]) return 0;
        }
    return 1;
}

/* ─── N-Tuple Network ──────────────────────────────────────── */

#define MAX_LOG2 16
#define N_TUPLES 17
#define TUPLE_SIZE 6
#define LUT_SIZE 16777216  /* 16^6 */

typedef struct { int pos[TUPLE_SIZE][2]; } sym_t;

typedef struct {
    int n_base;
    int n_sym[N_TUPLES];
    sym_t syms[N_TUPLES][8];
    float *weights[N_TUPLES];
    float *tc_abs[N_TUPLES];
    float *tc_sum[N_TUPLES];
    float lr;
    int use_tc;
} ntuple_net_t;

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

/* Grid already stores log2 values, so encode is direct */
static inline int encode_tuple(const grid_t g, const int pos[][2]) {
    return g[pos[0][0]][pos[0][1]] * (MAX_LOG2*MAX_LOG2*MAX_LOG2*MAX_LOG2) +
           g[pos[1][0]][pos[1][1]] * (MAX_LOG2*MAX_LOG2*MAX_LOG2) +
           g[pos[2][0]][pos[2][1]] * (MAX_LOG2*MAX_LOG2) +
           g[pos[3][0]][pos[3][1]] * MAX_LOG2 +
           g[pos[4][0]][pos[4][1]];
    /* This only handles 5 positions. We need 6. Let me fix: */
}

/* Proper 6-position encode */
static inline int encode6(const grid_t g, const int pos[][2]) {
    int idx = 0;
    idx = idx * MAX_LOG2 + g[pos[0][0]][pos[0][1]];
    idx = idx * MAX_LOG2 + g[pos[1][0]][pos[1][1]];
    idx = idx * MAX_LOG2 + g[pos[2][0]][pos[2][1]];
    idx = idx * MAX_LOG2 + g[pos[3][0]][pos[3][1]];
    idx = idx * MAX_LOG2 + g[pos[4][0]][pos[4][1]];
    idx = idx * MAX_LOG2 + g[pos[5][0]][pos[5][1]];
    return idx;
}

static void generate_symmetries(ntuple_net_t *net, int t) {
    int cur[TUPLE_SIZE][2];
    int cands[8][TUPLE_SIZE][2];
    int nc = 0;

    memcpy(cur, BASE_TUPLES[t], sizeof(cur));

    for (int rot = 0; rot < 4; rot++) {
        memcpy(cands[nc++], cur, sizeof(cur));
        for (int i = 0; i < TUPLE_SIZE; i++) {
            cands[nc][i][0] = cur[i][0];
            cands[nc][i][1] = 3 - cur[i][1];
        }
        nc++;
        int tmp[TUPLE_SIZE][2];
        for (int i = 0; i < TUPLE_SIZE; i++) {
            tmp[i][0] = cur[i][1];
            tmp[i][1] = 3 - cur[i][0];
        }
        memcpy(cur, tmp, sizeof(cur));
    }

    int count = 0;
    for (int c = 0; c < nc; c++) {
        int sorted[TUPLE_SIZE][2];
        memcpy(sorted, cands[c], sizeof(sorted));
        for (int i = 0; i < TUPLE_SIZE-1; i++)
            for (int j = i+1; j < TUPLE_SIZE; j++)
                if (sorted[i][0] > sorted[j][0] ||
                    (sorted[i][0] == sorted[j][0] && sorted[i][1] > sorted[j][1])) {
                    int ty=sorted[i][0], tx=sorted[i][1];
                    sorted[i][0]=sorted[j][0]; sorted[i][1]=sorted[j][1];
                    sorted[j][0]=ty; sorted[j][1]=tx;
                }
        int dup = 0;
        for (int k = 0; k < count; k++) {
            int s2[TUPLE_SIZE][2];
            memcpy(s2, net->syms[t][k].pos, sizeof(s2));
            for (int i = 0; i < TUPLE_SIZE-1; i++)
                for (int j = i+1; j < TUPLE_SIZE; j++)
                    if (s2[i][0] > s2[j][0] ||
                        (s2[i][0] == s2[j][0] && s2[i][1] > s2[j][1])) {
                        int ty=s2[i][0], tx=s2[i][1];
                        s2[i][0]=s2[j][0]; s2[i][1]=s2[j][1];
                        s2[j][0]=ty; s2[j][1]=tx;
                    }
            if (memcmp(sorted, s2, sizeof(sorted)) == 0) { dup = 1; break; }
        }
        if (!dup) {
            memcpy(net->syms[t][count].pos, cands[c], sizeof(cands[c]));
            count++;
        }
    }
    net->n_sym[t] = count;
}

static void net_init(ntuple_net_t *net, int use_tc) {
    net->n_base = N_TUPLES;
    net->lr = 0.01f;
    net->use_tc = use_tc;
    for (int t = 0; t < N_TUPLES; t++) {
        net->weights[t] = (float *)calloc(LUT_SIZE, sizeof(float));
        if (use_tc) {
            net->tc_abs[t] = (float *)calloc(LUT_SIZE, sizeof(float));
            net->tc_sum[t] = (float *)calloc(LUT_SIZE, sizeof(float));
        } else {
            net->tc_abs[t] = NULL;
            net->tc_sum[t] = NULL;
        }
        if (!net->weights[t]) { fprintf(stderr, "OOM\n"); exit(1); }
        generate_symmetries(net, t);
    }
}

static void net_free(ntuple_net_t *net) {
    for (int t = 0; t < net->n_base; t++) {
        free(net->weights[t]);
        if (net->tc_abs[t]) free(net->tc_abs[t]);
        if (net->tc_sum[t]) free(net->tc_sum[t]);
    }
}

static float net_evaluate(const ntuple_net_t *net, const grid_t g) {
    float total = 0;
    for (int t = 0; t < net->n_base; t++) {
        const float *w = net->weights[t];
        for (int s = 0; s < net->n_sym[t]; s++)
            total += w[encode6(g, net->syms[t][s].pos)];
    }
    return total;
}

static void net_update(ntuple_net_t *net, const grid_t g, float delta) {
    if (delta > 1000.0f) delta = 1000.0f;
    if (delta < -1000.0f) delta = -1000.0f;

    if (!net->use_tc) {
        float adj = net->lr * delta;
        for (int t = 0; t < net->n_base; t++) {
            float *w = net->weights[t];
            for (int s = 0; s < net->n_sym[t]; s++)
                w[encode6(g, net->syms[t][s].pos)] += adj;
        }
    } else {
        float abs_delta = (delta >= 0) ? delta : -delta;
        float decay = 0.9995f;
        for (int t = 0; t < net->n_base; t++) {
            float *w = net->weights[t];
            float *ts = net->tc_sum[t];
            float *ta = net->tc_abs[t];
            for (int s = 0; s < net->n_sym[t]; s++) {
                int idx = encode6(g, net->syms[t][s].pos);
                ts[idx] = ts[idx] * decay + delta;
                ta[idx] = ta[idx] * decay + abs_delta;
                float ratio = (ta[idx] > 1e-6f)
                    ? ((ts[idx] >= 0 ? ts[idx] : -ts[idx]) / ta[idx])
                    : 1.0f;
                w[idx] += net->lr * ratio * delta;
            }
        }
    }
}

/* ─── Save / Load (backward-compatible) ─────────────────────── */

static void net_save(const ntuple_net_t *net, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) { perror("save"); return; }
    int n = net->n_base;
    fwrite(&n, sizeof(int), 1, f);
    for (int t = 0; t < n; t++) {
        int size = LUT_SIZE;
        fwrite(&size, sizeof(int), 1, f);
        fwrite(net->weights[t], sizeof(float), size, f);
    }
    if (net->use_tc) {
        int marker = 0x5443;
        fwrite(&marker, sizeof(int), 1, f);
        for (int t = 0; t < n; t++) {
            fwrite(net->tc_sum[t], sizeof(float), LUT_SIZE, f);
            fwrite(net->tc_abs[t], sizeof(float), LUT_SIZE, f);
        }
    }
    fclose(f);
    FILE *fs = fopen(path, "rb");
    fseek(fs, 0, SEEK_END);
    long sz = ftell(fs);
    fclose(fs);
    printf("N-Tuple saved to %s (%.1f MB)\n", path, sz / 1024.0 / 1024.0);
}

static int net_load(ntuple_net_t *net, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    int n;
    if (fread(&n, sizeof(int), 1, f) != 1) { fclose(f); return 0; }
    if (n != net->n_base) {
        fprintf(stderr, "Incompatible checkpoint: %d vs %d tuples\n", n, net->n_base);
        fclose(f);
        return 0;
    }
    for (int t = 0; t < n; t++) {
        int size;
        fread(&size, sizeof(int), 1, f);
        if (size != LUT_SIZE) { fclose(f); return 0; }
        fread(net->weights[t], sizeof(float), size, f);
    }
    int marker = 0;
    if (fread(&marker, sizeof(int), 1, f) == 1 && marker == 0x5443 && net->use_tc) {
        for (int t = 0; t < n; t++) {
            fread(net->tc_sum[t], sizeof(float), LUT_SIZE, f);
            fread(net->tc_abs[t], sizeof(float), LUT_SIZE, f);
        }
        printf("N-Tuple + TC loaded from %s\n", path);
    } else {
        printf("N-Tuple loaded from %s (TC initialized)\n", path);
    }
    fclose(f);
    return 1;
}

/* ─── Search for training ───────────────────────────────────── */

static int train_search_depth = 0;

static float train_chance(const ntuple_net_t *net, const grid_t g, int depth, unsigned int *seed);

static float train_max(const ntuple_net_t *net, const grid_t g, int depth, unsigned int *seed) {
    float best = -1e18f;
    int any = 0;
    for (int d = 0; d < 4; d++) {
        grid_t after;
        int moved;
        int reward = grid_move(after, g, d, &moved);
        if (!moved) continue;
        any = 1;
        float v = (depth <= 0)
            ? (float)reward + net_evaluate(net, after)
            : (float)reward + train_chance(net, after, depth - 1, seed);
        if (v > best) best = v;
    }
    return any ? best : net_evaluate(net, g);
}

static float train_chance(const ntuple_net_t *net, const grid_t g, int depth, unsigned int *seed) {
    int ey[16], ex[16], ne = 0;
    for (int y = 0; y < 4; y++)
        for (int x = 0; x < 4; x++)
            if (g[y][x] == 0) { ey[ne] = y; ex[ne] = x; ne++; }
    if (ne == 0) return net_evaluate(net, g);

    int sn = ne;
    int idx[16];
    for (int i = 0; i < 16; i++) idx[i] = i;
    if (ne > 4) {
        sn = 4;
        for (int i = 0; i < sn; i++) {
            int j = i + trand(seed) % (ne - i);
            int tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp;
        }
    }

    float total = 0;
    for (int i = 0; i < sn; i++) {
        int ii = idx[i];
        grid_t g2;
        grid_copy(g2, g);
        g2[ey[ii]][ex[ii]] = 1; /* tile 2 = log2(2) = 1 */
        total += 0.9f * train_max(net, g2, depth, seed);
        grid_copy(g2, g);
        g2[ey[ii]][ex[ii]] = 2; /* tile 4 = log2(4) = 2 */
        total += 0.1f * train_max(net, g2, depth, seed);
    }
    return total / sn;
}

static int select_best(const ntuple_net_t *net, const grid_t state,
                       grid_t best_after, float *best_reward_out, unsigned int *seed) {
    int best_action = -1;
    float best_value = -1e18f;
    *best_reward_out = 0;

    for (int d = 0; d < 4; d++) {
        grid_t after;
        int moved;
        int reward = grid_move(after, state, d, &moved);
        if (!moved) continue;

        float value = (train_search_depth <= 0)
            ? (float)reward + net_evaluate(net, after)
            : (float)reward + train_chance(net, after, train_search_depth - 1, seed);

        if (value > best_value) {
            best_value = value;
            best_action = d;
            grid_copy(best_after, after);
            *best_reward_out = (float)reward;
        }
    }
    return best_action;
}

/* ─── Multithreaded Training ────────────────────────────────── */

static ntuple_net_t shared_net;
static atomic_int episodes_done = 0;
static atomic_int total_wins = 0;
static int total_episodes = 0;
static float lr_start = 0.01f, lr_end = 0.0005f;

/* Per-thread stats */
typedef struct {
    int thread_id;
    int local_episodes;
    int local_scores[100];
    int local_tiles[100];
    int local_buf_idx;
} thread_stats_t;

static void *train_worker(void *arg) {
    thread_stats_t *stats = (thread_stats_t *)arg;
    unsigned int seed = (unsigned int)(time(NULL) + stats->thread_id * 7919);

    while (1) {
        int ep = atomic_fetch_add(&episodes_done, 1) + 1;
        if (ep > total_episodes) break;

        float progress = (float)ep / total_episodes;
        shared_net.lr = lr_start * powf(lr_end / lr_start, progress);

        grid_t state;
        grid_clear(state);
        grid_add_random_r(state, &seed);
        grid_add_random_r(state, &seed);

        int game_score = 0;

        grid_t prev_after;
        float prev_reward;
        int action = select_best(&shared_net, state, prev_after, &prev_reward, &seed);
        if (action == -1) continue;

        grid_copy(state, prev_after);
        grid_add_random_r(state, &seed);
        game_score += (int)prev_reward;

        while (!grid_game_over(state)) {
            grid_t curr_after;
            float curr_reward;
            action = select_best(&shared_net, state, curr_after, &curr_reward, &seed);
            if (action == -1) break;

            float v_prev = net_evaluate(&shared_net, prev_after);
            float v_curr = net_evaluate(&shared_net, curr_after);
            float delta = curr_reward + v_curr - v_prev;
            net_update(&shared_net, prev_after, delta);

            grid_copy(state, curr_after);
            grid_add_random_r(state, &seed);
            game_score += (int)curr_reward;

            grid_copy(prev_after, curr_after);
        }

        float v_last = net_evaluate(&shared_net, prev_after);
        net_update(&shared_net, prev_after, -v_last);

        int max_tile = grid_max_tile_actual(state);
        if (max_tile >= 2048) atomic_fetch_add(&total_wins, 1);

        int bi = stats->local_buf_idx % 100;
        stats->local_scores[bi] = game_score;
        stats->local_tiles[bi] = max_tile;
        stats->local_buf_idx++;
    }
    return NULL;
}

static void train(int episodes, const char *save_dir, int use_tc, int n_threads) {
    net_init(&shared_net, use_tc);

    char path_latest[512], path_best[512];
    snprintf(path_latest, sizeof(path_latest), "%s/ntuple_latest.bin", save_dir);
    snprintf(path_best, sizeof(path_best), "%s/ntuple_best.bin", save_dir);

    net_load(&shared_net, path_latest);

    total_episodes = episodes;
    episodes_done = 0;
    total_wins = 0;
    float best_avg = 0;

    thread_stats_t *stats = calloc(n_threads, sizeof(thread_stats_t));
    pthread_t *threads = malloc(n_threads * sizeof(pthread_t));

    time_t start_time = time(NULL);

    /* Launch threads */
    for (int i = 0; i < n_threads; i++) {
        stats[i].thread_id = i;
        pthread_create(&threads[i], NULL, train_worker, &stats[i]);
    }

    /* Monitor progress from main thread */
    int last_reported = 0;
    while (1) {
        int done = atomic_load(&episodes_done);
        if (done >= episodes) break;

        /* Report every 100 episodes */
        int current_100 = (done / 100) * 100;
        if (current_100 > last_reported && current_100 > 0) {
            last_reported = current_100;

            /* Aggregate stats from all threads */
            float avg = 0;
            int tile_counts[20] = {0};
            int n_samples = 0;
            for (int t = 0; t < n_threads; t++) {
                int n = (stats[t].local_buf_idx < 100) ? stats[t].local_buf_idx : 100;
                for (int i = 0; i < n; i++) {
                    avg += stats[t].local_scores[i];
                    int lt = 0, tv = stats[t].local_tiles[i];
                    while (tv > 1) { tv >>= 1; lt++; }
                    if (lt < 20) tile_counts[lt]++;
                    n_samples++;
                }
            }
            if (n_samples > 0) avg /= n_samples;

            time_t elapsed = time(NULL) - start_time;
            int wins = atomic_load(&total_wins);

            printf("\n============================================================\n");
            printf("N-Tuple Episode %d/%d | Time: %lds | %d threads\n",
                   done, episodes, elapsed, n_threads);
            printf("============================================================\n");
            printf("  Average score:  %.0f\n", avg);
            printf("  Max tiles:    {");
            int first = 1;
            for (int i = 0; i < 20; i++) {
                if (tile_counts[i] > 0) {
                    if (!first) printf(", ");
                    printf("%d: %d", 1 << i, tile_counts[i]);
                    first = 0;
                }
            }
            printf("}\n");
            printf("  2048+:        %dx total\n", wins);
            printf("  LR:           %f\n", shared_net.lr);
            printf("  Ep/s:         %.1f\n", (float)done / (elapsed > 0 ? elapsed : 1));

            if (avg > best_avg) {
                best_avg = avg;
                net_save(&shared_net, path_best);
            }

            if (current_100 % 1000 == 0) {
                net_save(&shared_net, path_latest);
            }
        }

        /* Sleep briefly to avoid busy-waiting */
        struct timespec ts = {0, 100000000}; /* 100ms */
        nanosleep(&ts, NULL);
    }

    /* Join threads */
    for (int i = 0; i < n_threads; i++)
        pthread_join(threads[i], NULL);

    net_save(&shared_net, path_latest);
    printf("\nTraining complete! Best avg score: %.0f\n", best_avg);

    free(stats);
    free(threads);
    net_free(&shared_net);
}

/* ─── Main ──────────────────────────────────────────────────── */

int main(int argc, char **argv) {
    int episodes = 50000;
    const char *save_dir = "checkpoints";
    int use_tc = 0;
    int n_threads = 1;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--episodes") == 0 && i + 1 < argc)
            episodes = atoi(argv[++i]);
        else if (strcmp(argv[i], "--save-dir") == 0 && i + 1 < argc)
            save_dir = argv[++i];
        else if (strcmp(argv[i], "--depth") == 0 && i + 1 < argc)
            train_search_depth = atoi(argv[++i]);
        else if (strcmp(argv[i], "--tc") == 0)
            use_tc = 1;
        else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc)
            n_threads = atoi(argv[++i]);
    }

    srand((unsigned)time(NULL));
    setbuf(stdout, NULL);
    printf("N-Tuple Training in C (optimized)\n");
    printf("Episodes: %d | Tuples: %d | Search: %d-ply | TC: %s | Threads: %d\n",
           episodes, N_TUPLES, 1 + train_search_depth * 2,
           use_tc ? "ON" : "OFF", n_threads);

    build_move_tables();
    train(episodes, save_dir, use_tc, n_threads);
    return 0;
}
