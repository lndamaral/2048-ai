/*
 * N-Tuple Network player in C — shared library for the Python server.
 * Loads trained weights and performs instant 1-ply or 3-ply search.
 *
 * Compile: cc -O3 -shared -o ntuple_c.so ntuple_c.c -lm
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ─── Game Engine (inline) ──────────────────────────────────── */

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

typedef int grid_t[4][4];

static int do_move(grid_t after, const grid_t g, int dir, int *moved) {
    memcpy(after, g, sizeof(grid_t));
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

/* ─── N-Tuple Network ──────────────────────────────────────── */

#define MAX_LOG2 16
#define MAX_TUPLES 20
#define TUPLE_SIZE 6
#define LUT_SIZE 16777216

typedef struct {
    int pos[TUPLE_SIZE][2];
} sym_t;

static int n_tuples = 0;
static int n_sym[MAX_TUPLES];
static sym_t syms[MAX_TUPLES][8];
static float *weights[MAX_TUPLES];
static int loaded = 0;

/* Base tuples */
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

static inline int to_log2(int val) {
    if (val <= 0) return 0;
    int l = 0;
    while (val > 1) { val >>= 1; l++; }
    return l;
}

static inline int encode(const grid_t g, const int pos[][2]) {
    int idx = 0;
    for (int i = 0; i < TUPLE_SIZE; i++)
        idx = idx * MAX_LOG2 + to_log2(g[pos[i][0]][pos[i][1]]);
    return idx;
}

static void gen_symmetries(int t) {
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
            memcpy(s2, syms[t][k].pos, sizeof(s2));
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
            memcpy(syms[t][count].pos, cands[c], sizeof(cands[c]));
            count++;
        }
    }
    n_sym[t] = count;
}

static float evaluate(const grid_t g) {
    float total = 0;
    for (int t = 0; t < n_tuples; t++) {
        const float *w = weights[t];
        for (int s = 0; s < n_sym[t]; s++)
            total += w[encode(g, syms[t][s].pos)];
    }
    return total;
}

/* ─── Search ────────────────────────────────────────────────── */

/* 3-ply: player → chance → player → chance → player → evaluate */
static float chance_node(const grid_t g, int depth);

static float max_node(const grid_t g, int depth) {
    float best = -1e18f;
    int any = 0;
    for (int d = 0; d < 4; d++) {
        grid_t after;
        int moved;
        int reward = do_move(after, g, d, &moved);
        if (!moved) continue;
        any = 1;
        float v;
        if (depth <= 0)
            v = (float)reward + evaluate(after);
        else
            v = (float)reward + chance_node(after, depth - 1);
        if (v > best) best = v;
    }
    return any ? best : evaluate(g);
}

static float chance_node(const grid_t g, int depth) {
    int ey[16], ex[16], ne = 0;
    for (int y = 0; y < 4; y++)
        for (int x = 0; x < 4; x++)
            if (g[y][x] == 0) { ey[ne] = y; ex[ne] = x; ne++; }

    if (ne == 0) return evaluate(g);

    /* Evaluate ALL empty cells (no sampling — deterministic for reproducibility) */
    int sample_n = ne;

    float total = 0;
    for (int i = 0; i < sample_n; i++) {
        grid_t g2;

        memcpy(g2, g, sizeof(grid_t));
        g2[ey[i]][ex[i]] = 2;
        total += 0.9f * max_node(g2, depth);

        memcpy(g2, g, sizeof(grid_t));
        g2[ey[i]][ex[i]] = 4;
        total += 0.1f * max_node(g2, depth);
    }
    return total / sample_n;
}

/* ─── Public API ────────────────────────────────────────────── */

int ntuple_load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return 0;

    int n;
    fread(&n, sizeof(int), 1, f);
    n_tuples = n;

    for (int t = 0; t < n; t++) {
        int size;
        fread(&size, sizeof(int), 1, f);
        weights[t] = (float *)malloc(size * sizeof(float));
        fread(weights[t], sizeof(float), size, f);
        gen_symmetries(t);
    }
    fclose(f);
    loaded = 1;
    printf("N-Tuple C: loaded %s (%d tuples)\n", path, n);
    return 1;
}

/*
 * grid: int[16] row-major, actual tile values
 * search_depth: 0 = 1-ply (instant), 1 = 3-ply, 2 = 5-ply
 * Returns: best direction (0-3)
 */
int ntuple_select_action(int *grid_flat, int search_depth) {
    if (!loaded) return 0;

    grid_t g;
    for (int y = 0; y < 4; y++)
        for (int x = 0; x < 4; x++)
            g[y][x] = grid_flat[y * 4 + x];

    int best_action = 0;
    float best_value = -1e18f;

    for (int d = 0; d < 4; d++) {
        grid_t after;
        int moved;
        int reward = do_move(after, g, d, &moved);
        if (!moved) continue;

        float value;
        if (search_depth <= 0)
            value = (float)reward + evaluate(after);
        else
            value = (float)reward + chance_node(after, search_depth - 1);

        if (value > best_value) {
            best_value = value;
            best_action = d;
        }
    }
    return best_action;
}
