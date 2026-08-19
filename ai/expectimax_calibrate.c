/*
 * Expectimax Heuristic Weight Auto-Calibrator.
 *
 * Runs hundreds of games with different weight combinations
 * and finds the optimal weights via hill climbing.
 *
 * Compile: cc -O3 -o expectimax_calibrate expectimax_calibrate.c -lm -lpthread
 * Run:     ./expectimax_calibrate --games 200 --threads 8
 */

#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdatomic.h>
#include <sys/time.h>

/* ─── Game Engine ───────────────────────────────────────────── */

typedef int grid_t[4][4];

static void grid_clear(grid_t g) { memset(g, 0, sizeof(grid_t)); }
static void grid_copy(grid_t dst, const grid_t src) { memcpy(dst, src, sizeof(grid_t)); }

static inline int trand(unsigned int *seed) {
    *seed = *seed * 1103515245 + 12345;
    return (*seed >> 16) & 0x7FFF;
}

static void grid_add_random(grid_t g, unsigned int *seed) {
    int ey[16], ex[16], n = 0;
    for (int y = 0; y < 4; y++)
        for (int x = 0; x < 4; x++)
            if (g[y][x] == 0) { ey[n] = y; ex[n] = x; n++; }
    if (n == 0) return;
    int idx = trand(seed) % n;
    g[ey[idx]][ex[idx]] = (trand(seed) % 10 < 9) ? 2 : 4;
}

static int merge_line(int *line) {
    int tmp[4] = {0}, pos = 0, score = 0;
    for (int i = 0; i < 4; i++)
        if (line[i]) tmp[pos++] = line[i];
    int result[4] = {0};
    pos = 0;
    for (int i = 0; i < 4 && tmp[i]; i++) {
        if (i + 1 < 4 && tmp[i] == tmp[i+1]) {
            result[pos++] = tmp[i] * 2;
            score += tmp[i] * 2;
            i++;
        } else {
            result[pos++] = tmp[i];
        }
    }
    memcpy(line, result, 4 * sizeof(int));
    return score;
}

static int grid_move(grid_t after, const grid_t g, int dir, int *moved) {
    grid_copy(after, g);
    int score = 0, line[4];
    if (dir == 3) {
        for (int y = 0; y < 4; y++) {
            for (int x = 0; x < 4; x++) line[x] = after[y][x];
            score += merge_line(line);
            for (int x = 0; x < 4; x++) after[y][x] = line[x];
        }
    } else if (dir == 1) {
        for (int y = 0; y < 4; y++) {
            for (int x = 0; x < 4; x++) line[x] = after[y][3-x];
            score += merge_line(line);
            for (int x = 0; x < 4; x++) after[y][3-x] = line[x];
        }
    } else if (dir == 0) {
        for (int x = 0; x < 4; x++) {
            for (int y = 0; y < 4; y++) line[y] = after[y][x];
            score += merge_line(line);
            for (int y = 0; y < 4; y++) after[y][x] = line[y];
        }
    } else {
        for (int x = 0; x < 4; x++) {
            for (int y = 0; y < 4; y++) line[y] = after[3-y][x];
            score += merge_line(line);
            for (int y = 0; y < 4; y++) after[3-y][x] = line[y];
        }
    }
    *moved = (memcmp(after, g, sizeof(grid_t)) != 0);
    return score;
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

static int grid_max_tile(const grid_t g) {
    int m = 0;
    for (int y = 0; y < 4; y++)
        for (int x = 0; x < 4; x++)
            if (g[y][x] > m) m = g[y][x];
    return m;
}

static int to_log2(int val) {
    if (val <= 0) return 0;
    int l = 0;
    while (val > 1) { val >>= 1; l++; }
    return l;
}

/* ─── Heuristic Evaluation with Tunable Weights ──────────── */

/* Snake weights */
static const float SNAKE_W[8][4][4] = {
    {{32768,16384,8192,4096},{256,512,1024,2048},{128,64,32,16},{1,2,4,8}},
    {{4096,8192,16384,32768},{2048,1024,512,256},{16,32,64,128},{8,4,2,1}},
    {{1,128,256,32768},{2,64,512,16384},{4,32,1024,8192},{8,16,2048,4096}},
    {{32768,256,128,1},{16384,512,64,2},{8192,1024,32,4},{4096,2048,16,8}},
    {{8,4,2,1},{16,32,64,128},{2048,1024,512,256},{32768,16384,8192,4096}},
    {{1,2,4,8},{128,64,32,16},{256,512,1024,2048},{4096,8192,16384,32768}},
    {{8,16,2048,4096},{4,32,1024,8192},{2,64,512,16384},{1,128,256,32768}},
    {{4096,2048,16,8},{8192,1024,32,4},{16384,512,64,2},{32768,256,128,1}},
};

typedef struct {
    float w_snake;
    float w_empty;
    float w_mono;
    float w_smooth;
    float w_corner;
    float w_merge_potential;
    float w_trapped;
} weights_t;

static float evaluate(const grid_t g, const weights_t *w) {
    int grid[4][4];
    float lg[4][4];
    int empty = 0;
    int max_val = 0;

    for (int y = 0; y < 4; y++)
        for (int x = 0; x < 4; x++) {
            grid[y][x] = to_log2(g[y][x]);
            lg[y][x] = (float)grid[y][x];
            if (g[y][x] == 0) empty++;
            if (grid[y][x] > max_val) max_val = grid[y][x];
        }

    float score = 0;

    /* 1. Snake pattern */
    float best_snake = -1e18f;
    for (int o = 0; o < 8; o++) {
        float s = 0;
        for (int y = 0; y < 4; y++)
            for (int x = 0; x < 4; x++)
                s += (float)g[y][x] * SNAKE_W[o][y][x];
        if (s > best_snake) best_snake = s;
    }
    score += best_snake * w->w_snake;

    /* 2. Empty cells */
    score += (empty > 0) ? logf((float)empty + 1.0f) * w->w_empty : 0;

    /* 3. Monotonicity */
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
    score += mono * w->w_mono;

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
    score += smooth * w->w_smooth;

    /* 5. Corner bonus */
    int corners[4] = {grid[0][0], grid[0][3], grid[3][0], grid[3][3]};
    for (int i = 0; i < 4; i++) {
        if (corners[i] == max_val && max_val > 0) {
            score += (float)(max_val * max_val) * w->w_corner;
            break;
        }
    }

    /* 6. Merge potential (NEW) — bonus for adjacent equal tiles */
    float merge_pot = 0;
    for (int y = 0; y < 4; y++)
        for (int x = 0; x < 4; x++) {
            if (grid[y][x] == 0) continue;
            if (x < 3 && grid[y][x] == grid[y][x+1])
                merge_pot += (float)grid[y][x];
            if (y < 3 && grid[y][x] == grid[y+1][x])
                merge_pot += (float)grid[y][x];
        }
    score += merge_pot * w->w_merge_potential;

    /* 7. Trapped tile penalty (NEW) — penalize large tiles surrounded by much smaller ones */
    float trapped = 0;
    for (int y = 0; y < 4; y++)
        for (int x = 0; x < 4; x++) {
            if (grid[y][x] < 5) continue; /* only care about tiles >= 32 */
            int v = grid[y][x];
            int blocked = 0;
            int neighbors = 0;
            int dx[] = {0, 0, 1, -1};
            int dy[] = {1, -1, 0, 0};
            for (int d = 0; d < 4; d++) {
                int ny = y + dy[d], nx = x + dx[d];
                if (ny < 0 || ny >= 4 || nx < 0 || nx >= 4) continue;
                neighbors++;
                if (grid[ny][nx] > 0 && abs(grid[ny][nx] - v) > 2)
                    blocked++;
            }
            if (neighbors > 0 && blocked == neighbors)
                trapped -= (float)(v * v);
        }
    score += trapped * w->w_trapped;

    return score;
}

/* ─── Expectimax Search ─────────────────────────────────────── */

static float chance_node(const grid_t g, int depth, const weights_t *w, unsigned int *seed);

static float max_node(const grid_t g, int depth, const weights_t *w, unsigned int *seed) {
    float best = -1e18f;
    int any = 0;
    for (int d = 0; d < 4; d++) {
        grid_t after;
        int moved;
        int reward = grid_move(after, g, d, &moved);
        if (!moved) continue;
        any = 1;
        float v = (depth <= 0)
            ? (float)reward + evaluate(after, w)
            : (float)reward + chance_node(after, depth - 1, w, seed);
        if (v > best) best = v;
    }
    return any ? best : evaluate(g, w);
}

static float chance_node(const grid_t g, int depth, const weights_t *w, unsigned int *seed) {
    int ey[16], ex[16], ne = 0;
    for (int y = 0; y < 4; y++)
        for (int x = 0; x < 4; x++)
            if (g[y][x] == 0) { ey[ne] = y; ex[ne] = x; ne++; }
    if (ne == 0) return evaluate(g, w);

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
        g2[ey[ii]][ex[ii]] = 2;
        total += 0.9f * max_node(g2, depth, w, seed);
        grid_copy(g2, g);
        g2[ey[ii]][ex[ii]] = 4;
        total += 0.1f * max_node(g2, depth, w, seed);
    }
    return total / sn;
}

/* ─── Game Runner ───────────────────────────────────────────── */

typedef struct {
    int score;
    int max_tile;
    int moves;
} game_result_t;

static game_result_t play_game(const weights_t *w, int search_depth, unsigned int *seed) {
    grid_t state;
    grid_clear(state);
    grid_add_random(state, seed);
    grid_add_random(state, seed);

    int total_score = 0, moves = 0;

    while (!grid_game_over(state)) {
        int best_dir = -1;
        float best_val = -1e18f;

        for (int d = 0; d < 4; d++) {
            grid_t after;
            int moved;
            int reward = grid_move(after, state, d, &moved);
            if (!moved) continue;
            float v = (search_depth <= 0)
                ? (float)reward + evaluate(after, w)
                : (float)reward + chance_node(after, search_depth - 1, w, seed);
            if (v > best_val) {
                best_val = v;
                best_dir = d;
            }
        }

        if (best_dir == -1) break;

        grid_t after;
        int moved;
        total_score += grid_move(after, state, best_dir, &moved);
        grid_copy(state, after);
        grid_add_random(state, seed);
        moves++;
    }

    game_result_t r;
    r.score = total_score;
    r.max_tile = grid_max_tile(state);
    r.moves = moves;
    return r;
}

/* ─── Parallel Evaluation ───────────────────────────────────── */

typedef struct {
    const weights_t *w;
    int search_depth;
    int n_games;
    float avg_score;
    float win_rate;
    int thread_id;
} eval_task_t;

static void *eval_worker(void *arg) {
    eval_task_t *task = (eval_task_t *)arg;
    unsigned int seed = (unsigned int)(time(NULL) + task->thread_id * 9973);

    long total_score = 0;
    int wins = 0;

    for (int i = 0; i < task->n_games; i++) {
        game_result_t r = play_game(task->w, task->search_depth, &seed);
        total_score += r.score;
        if (r.max_tile >= 2048) wins++;
    }

    task->avg_score = (float)total_score / task->n_games;
    task->win_rate = (float)wins / task->n_games * 100.0f;
    return NULL;
}

static void evaluate_weights(const weights_t *w, int n_games, int search_depth, int n_threads,
                             float *avg_score, float *win_rate) {
    int games_per_thread = n_games / n_threads;
    eval_task_t *tasks = calloc(n_threads, sizeof(eval_task_t));
    pthread_t *threads = malloc(n_threads * sizeof(pthread_t));

    for (int i = 0; i < n_threads; i++) {
        tasks[i].w = w;
        tasks[i].search_depth = search_depth;
        tasks[i].n_games = games_per_thread;
        tasks[i].thread_id = i;
        pthread_create(&threads[i], NULL, eval_worker, &tasks[i]);
    }

    for (int i = 0; i < n_threads; i++)
        pthread_join(threads[i], NULL);

    float total_score = 0, total_win = 0;
    for (int i = 0; i < n_threads; i++) {
        total_score += tasks[i].avg_score;
        total_win += tasks[i].win_rate;
    }
    *avg_score = total_score / n_threads;
    *win_rate = total_win / n_threads;

    free(tasks);
    free(threads);
}

/* ─── Hill Climbing Calibrator ──────────────────────────────── */

static void print_weights(const char *label, const weights_t *w, float score, float win) {
    printf("%s: snake=%.3f empty=%.3f mono=%.3f smooth=%.3f corner=%.3f merge=%.3f trapped=%.3f | "
           "score=%.0f win=%.1f%%\n",
           label, w->w_snake, w->w_empty, w->w_mono, w->w_smooth,
           w->w_corner, w->w_merge_potential, w->w_trapped, score, win);
}

static void calibrate(int n_games, int search_depth, int n_threads, int iterations) {
    /* Start with current weights */
    weights_t best = {
        .w_snake = 0.5f,
        .w_empty = 2.7f,
        .w_mono = 1.0f,
        .w_smooth = 0.1f,
        .w_corner = 1.0f,
        .w_merge_potential = 0.0f,
        .w_trapped = 0.0f,
    };

    float best_score, best_win;
    evaluate_weights(&best, n_games, search_depth, n_threads, &best_score, &best_win);
    print_weights("INITIAL", &best, best_score, best_win);

    float step_sizes[] = {0.5f, 0.2f, 0.1f, 0.05f};
    int n_steps = sizeof(step_sizes) / sizeof(step_sizes[0]);

    for (int phase = 0; phase < n_steps; phase++) {
        float step = step_sizes[phase];
        printf("\n--- Phase %d: step=%.2f ---\n", phase + 1, step);
        int improved = 1;

        while (improved) {
            improved = 0;
            float *params[] = {
                &best.w_snake, &best.w_empty, &best.w_mono, &best.w_smooth,
                &best.w_corner, &best.w_merge_potential, &best.w_trapped
            };
            const char *names[] = {
                "snake", "empty", "mono", "smooth", "corner", "merge", "trapped"
            };
            int n_params = 7;

            for (int p = 0; p < n_params; p++) {
                float original = *params[p];

                /* Try increase */
                *params[p] = original + step;
                float score_up, win_up;
                evaluate_weights(&best, n_games, search_depth, n_threads, &score_up, &win_up);

                /* Try decrease */
                *params[p] = original - step;
                if (*params[p] < 0) *params[p] = 0;
                float score_dn, win_dn;
                evaluate_weights(&best, n_games, search_depth, n_threads, &score_dn, &win_dn);

                /* Keep best */
                if (score_up > best_score && score_up >= score_dn) {
                    *params[p] = original + step;
                    best_score = score_up;
                    best_win = win_up;
                    improved = 1;
                    printf("  %s += %.2f → ", names[p], step);
                    print_weights("BETTER", &best, best_score, best_win);
                } else if (score_dn > best_score) {
                    *params[p] = (original - step > 0) ? original - step : 0;
                    best_score = score_dn;
                    best_win = win_dn;
                    improved = 1;
                    printf("  %s -= %.2f → ", names[p], step);
                    print_weights("BETTER", &best, best_score, best_win);
                } else {
                    *params[p] = original;
                }
            }
        }
    }

    printf("\n============================================================\n");
    printf("FINAL OPTIMAL WEIGHTS:\n");
    printf("============================================================\n");
    print_weights("OPTIMAL", &best, best_score, best_win);
    printf("\nUpdate expectimax_c.c evaluate() with these weights.\n");
}

/* ─── Main ──────────────────────────────────────────────────── */

int main(int argc, char **argv) {
    int n_games = 200;
    int search_depth = 1; /* 3-ply */
    int n_threads = 8;
    int iterations = 50;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--games") == 0 && i + 1 < argc)
            n_games = atoi(argv[++i]);
        else if (strcmp(argv[i], "--depth") == 0 && i + 1 < argc)
            search_depth = atoi(argv[++i]);
        else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc)
            n_threads = atoi(argv[++i]);
    }

    srand((unsigned)time(NULL));
    setbuf(stdout, NULL);

    printf("Expectimax Heuristic Auto-Calibrator\n");
    printf("Games per eval: %d | Search: %d-ply | Threads: %d\n\n",
           n_games, 1 + search_depth * 2, n_threads);

    calibrate(n_games, search_depth, n_threads, iterations);
    return 0;
}
