/*
 * N-Tuple Network Training for 2048 — State-of-the-Art Version.
 *
 * Architecture:
 *   - Bitboard engine (engine_bitboard.h): board_t = uint64_t with nibbles
 *   - Training config (training_config.h): optimistic init, multistage, carousel
 *   - 17 base tuples x 8 symmetries, 6-position tuples, LUT_SIZE = 16^6
 *
 * Techniques:
 *   - Forward TD(0) afterstate learning
 *   - TC-learning (temporal coherence per-weight adaptive LR)
 *   - Optimistic initialization (weights start at 320,000)
 *   - Multistage training (LR adjustment by game phase)
 *   - Carousel shaping (replay from saved positions)
 *   - Weight promotion (copy weights when new high tile appears)
 *   - Redundant encoding (5-tuple sub-features for generalization)
 *   - Tile downgrading (reuse weights for max_nibble > 15)
 *   - Configurable search depth (1/3/5-ply)
 *   - LR decay with gradient clipping
 *   - Distributed LR (base_lr / active_features)
 *   - Hogwild multithreading (no locks on weights)
 *   - Per-episode CSV logging, periodic checkpoints, move distribution
 *
 * Compile: cc -O3 -o ntuple_train ntuple_train.c -lm -lpthread
 * Usage:   ./ntuple_train --episodes 5000000 --threads 8 --depth 1 --tc \
 *              --optimistic --multistage --carousel --log-csv logs/scores.csv
 */

#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdatomic.h>
#include <sys/stat.h>

#include "engine_bitboard.h"
#include "training_config.h"

/* ─── N-Tuple Network Constants ───────────────────────────────── */

#define MAX_TUPLES    20
#define TUPLE_SIZE    6
#define N_SYMMETRIES  8
#define LUT_SIZE      16777216  /* 16^6 = 2^24 */

/* Redundant encoding: 5-tuple sub-features */
#define SUB_TUPLE_SIZE    5
#define SUB_LUT_SIZE      1048576  /* 16^5 = 2^20 */
#define SUBS_PER_TUPLE    6        /* C(6,5) = 6 sub-tuples per base tuple */
#define MAX_SUB_TUPLES    (MAX_TUPLES * SUBS_PER_TUPLE)

/* Transposition table for search (configurable, default 2^24 = 16M) */
#ifndef TT_SIZE_LOG2
#define TT_SIZE_LOG2  24
#endif
#define TT_SIZE       (1 << TT_SIZE_LOG2)
#define TT_MASK       (TT_SIZE - 1)

/* ─── Tuple Configurations ──────────────────────────────────────── */

/*
 * TDL2048+ configuration: 4 tuples × 8 symmetries = 32 features.
 * Exact patterns from moporgic/TDL2048 4x6patt:
 *   012345, 456789, 012456, 45689a (hex board positions)
 * This is the proven state-of-the-art configuration.
 * Use --tdl2048 flag to select this.
 */
static const int TDL2048_TUPLES[4][TUPLE_SIZE][2] = {
    {{0,0},{0,1},{0,2},{0,3},{1,0},{1,1}},  /* 012345 */
    {{1,0},{1,1},{1,2},{1,3},{2,0},{2,1}},  /* 456789 */
    {{0,0},{0,1},{0,2},{1,0},{1,1},{1,2}},  /* 012456 */
    {{1,0},{1,1},{1,2},{2,0},{2,1},{2,2}},  /* 45689a */
};

/*
 * Extended configuration: 17 tuples × 8 symmetries = ~136 features.
 * More spatial coverage but requires different LR tuning.
 * This is the default configuration.
 */
static const int EXTENDED_TUPLES[17][TUPLE_SIZE][2] = {
    /* Row-based 2x3 blocks */
    {{0,0},{0,1},{0,2},{0,3},{1,0},{1,1}},
    {{1,0},{1,1},{1,2},{1,3},{2,0},{2,1}},
    {{2,0},{2,1},{2,2},{2,3},{3,0},{3,1}},
    /* Column-based 3x2 blocks */
    {{0,0},{1,0},{2,0},{3,0},{0,1},{1,1}},
    {{0,1},{1,1},{2,1},{3,1},{0,2},{1,2}},
    {{0,2},{1,2},{2,2},{3,2},{0,3},{1,3}},
    /* 2x3 rectangles (top-aligned) */
    {{0,0},{0,1},{0,2},{1,0},{1,1},{1,2}},
    {{1,0},{1,1},{1,2},{2,0},{2,1},{2,2}},
    {{2,0},{2,1},{2,2},{3,0},{3,1},{3,2}},
    {{0,1},{0,2},{0,3},{1,1},{1,2},{1,3}},
    /* 3x2 rectangles */
    {{0,0},{0,1},{1,0},{1,1},{2,0},{2,1}},
    {{0,1},{0,2},{1,1},{1,2},{2,1},{2,2}},
    {{0,2},{0,3},{1,2},{1,3},{2,2},{2,3}},
    /* L-shaped / diagonal patterns */
    {{0,0},{0,1},{0,2},{1,0},{1,1},{2,0}},
    {{0,1},{0,2},{0,3},{1,2},{1,3},{2,3}},
    {{0,0},{0,1},{1,1},{1,2},{2,2},{2,3}},
    {{0,2},{0,3},{1,1},{1,2},{2,0},{2,1}},
};

/* ─── Active Tuple Configuration ───────────────────────────────── */

static int n_tuples = 17;
static const int (*active_tuples)[TUPLE_SIZE][2] = EXTENDED_TUPLES;

/* ─── Symmetry Definitions ────────────────────────────────────── */

typedef struct { int pos[TUPLE_SIZE][2]; } sym_t;
typedef struct { int pos[SUB_TUPLE_SIZE][2]; } sym5_t;

typedef struct {
    int n_base;
    int n_sym[MAX_TUPLES];
    sym_t syms[MAX_TUPLES][N_SYMMETRIES];
    float *weights[MAX_TUPLES];
    float *tc_abs[MAX_TUPLES];
    float *tc_sum[MAX_TUPLES];
    int use_tc;

    /* Redundant encoding: 5-position sub-tuples derived from each 6-tuple */
    int use_redundant;
    int n_sub_tuples;
    int sub_tuple_size;   /* always SUB_TUPLE_SIZE (5) */
    int n_sub_sym[MAX_SUB_TUPLES];
    sym5_t sub_syms[MAX_SUB_TUPLES][N_SYMMETRIES];
    float *sub_weights[MAX_SUB_TUPLES];
    float *sub_tc_abs[MAX_SUB_TUPLES];
    float *sub_tc_sum[MAX_SUB_TUPLES];
} ntuple_net_t;

/* Generate all unique symmetries (rotations + reflections) for tuple t */
static void generate_symmetries(ntuple_net_t *net, int t) {
    int cur[TUPLE_SIZE][2];
    int cands[8][TUPLE_SIZE][2];
    int nc = 0;

    memcpy(cur, active_tuples[t], sizeof(cur));

    for (int rot = 0; rot < 4; rot++) {
        /* Current orientation */
        memcpy(cands[nc++], cur, sizeof(cur));
        /* Mirror (horizontal flip) */
        for (int i = 0; i < TUPLE_SIZE; i++) {
            cands[nc][i][0] = cur[i][0];
            cands[nc][i][1] = 3 - cur[i][1];
        }
        nc++;
        /* Rotate 90 degrees clockwise: (y, x) -> (x, 3-y) */
        int tmp[TUPLE_SIZE][2];
        for (int i = 0; i < TUPLE_SIZE; i++) {
            tmp[i][0] = cur[i][1];
            tmp[i][1] = 3 - cur[i][0];
        }
        memcpy(cur, tmp, sizeof(cur));
    }

    /* Deduplicate symmetries */
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
            memcpy(s2, net->syms[t][k].pos, sizeof(s2));
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
            memcpy(net->syms[t][count].pos, cands[c], sizeof(cands[c]));
            count++;
        }
    }
    net->n_sym[t] = count;
}

/* Generate all unique symmetries for a 5-position sub-tuple */
static void generate_sub_symmetries(ntuple_net_t *net, int st,
                                    const int base[SUB_TUPLE_SIZE][2]) {
    int cur[SUB_TUPLE_SIZE][2];
    int cands[8][SUB_TUPLE_SIZE][2];
    int nc = 0;

    memcpy(cur, base, sizeof(int) * SUB_TUPLE_SIZE * 2);

    for (int rot = 0; rot < 4; rot++) {
        /* Current orientation */
        memcpy(cands[nc++], cur, sizeof(int) * SUB_TUPLE_SIZE * 2);
        /* Mirror (horizontal flip) */
        for (int i = 0; i < SUB_TUPLE_SIZE; i++) {
            cands[nc][i][0] = cur[i][0];
            cands[nc][i][1] = 3 - cur[i][1];
        }
        nc++;
        /* Rotate 90 degrees clockwise: (y, x) -> (x, 3-y) */
        int tmp[SUB_TUPLE_SIZE][2];
        for (int i = 0; i < SUB_TUPLE_SIZE; i++) {
            tmp[i][0] = cur[i][1];
            tmp[i][1] = 3 - cur[i][0];
        }
        memcpy(cur, tmp, sizeof(int) * SUB_TUPLE_SIZE * 2);
    }

    /* Deduplicate symmetries */
    int count = 0;
    for (int c = 0; c < nc; c++) {
        int sorted[SUB_TUPLE_SIZE][2];
        memcpy(sorted, cands[c], sizeof(sorted));
        for (int i = 0; i < SUB_TUPLE_SIZE - 1; i++)
            for (int j = i + 1; j < SUB_TUPLE_SIZE; j++)
                if (sorted[i][0] > sorted[j][0] ||
                    (sorted[i][0] == sorted[j][0] && sorted[i][1] > sorted[j][1])) {
                    int ty = sorted[i][0], tx = sorted[i][1];
                    sorted[i][0] = sorted[j][0]; sorted[i][1] = sorted[j][1];
                    sorted[j][0] = ty; sorted[j][1] = tx;
                }
        int dup = 0;
        for (int k = 0; k < count; k++) {
            int s2[SUB_TUPLE_SIZE][2];
            memcpy(s2, net->sub_syms[st][k].pos, sizeof(s2));
            for (int i = 0; i < SUB_TUPLE_SIZE - 1; i++)
                for (int j = i + 1; j < SUB_TUPLE_SIZE; j++)
                    if (s2[i][0] > s2[j][0] ||
                        (s2[i][0] == s2[j][0] && s2[i][1] > s2[j][1])) {
                        int ty = s2[i][0], tx = s2[i][1];
                        s2[i][0] = s2[j][0]; s2[i][1] = s2[j][1];
                        s2[j][0] = ty; s2[j][1] = tx;
                    }
            if (memcmp(sorted, s2, sizeof(sorted)) == 0) { dup = 1; break; }
        }
        if (!dup) {
            memcpy(net->sub_syms[st][count].pos, cands[c],
                   sizeof(int) * SUB_TUPLE_SIZE * 2);
            count++;
        }
    }
    net->n_sub_sym[st] = count;
}

/* ─── Tuple Encoding (bitboard-native) ────────────────────────── */

/*
 * Encode a 6-position tuple directly from the bitboard representation.
 * Each position yields a 4-bit nibble; the 6 nibbles form a 24-bit index.
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

/*
 * Encode a 5-position sub-tuple directly from the bitboard representation.
 * Each position yields a 4-bit nibble; the 5 nibbles form a 20-bit index.
 */
static inline int encode5_board(board_t b, const int pos[][2]) {
    int idx = 0;
    for (int i = 0; i < 5; i++) {
        int y = pos[i][0], x = pos[i][1];
        int shift = y * 16 + (3 - x) * 4;
        int val = (int)((b >> shift) & 0xF);
        idx = idx * 16 + val;
    }
    return idx;
}

/* ─── Tile Downgrading ────────────────────────────────────────── */

/*
 * Get the maximum nibble value on the board.
 */
static inline int board_max_nibble(board_t b) {
    int max_val = 0;
    uint64_t tmp = b;
    for (int i = 0; i < 16; i++) {
        int nibble = (int)(tmp & 0xF);
        if (nibble > max_val) max_val = nibble;
        tmp >>= 4;
    }
    return max_val;
}

/*
 * Subtract 1 from every non-zero nibble in the board.
 * This effectively divides all tile values by 2.
 * Used when max_nibble > 15 to keep indices in the valid LUT range.
 */
static inline board_t board_downgrade(board_t b) {
    board_t ones = 0;
    for (int i = 0; i < 16; i++) {
        if ((b >> (i * 4)) & 0xF)
            ones |= (1ULL << (i * 4));
    }
    return b - ones;
}

/* ─── Network Evaluation ──────────────────────────────────────── */

/*
 * Evaluate a board position using the n-tuple network.
 * Applies tile downgrading if the max nibble exceeds 15.
 * Returns: sum of all activated weight entries.
 * Also sets *n_active to the number of features activated (for distributed LR).
 */
static float net_evaluate_ex(const ntuple_net_t *net, board_t b, int *n_active) {
    /* Auto-downgrade if max tile exceeds representable range */
    int max_nib = board_max_nibble(b);
    while (max_nib > 15) {
        b = board_downgrade(b);
        max_nib--;
    }

    float total = 0;
    int active = 0;
    for (int t = 0; t < net->n_base; t++) {
        const float *w = net->weights[t];
        for (int s = 0; s < net->n_sym[t]; s++) {
            total += w[encode6_board(b, net->syms[t][s].pos)];
            active++;
        }
    }

    /* Redundant encoding: sum contributions from 5-tuple sub-features */
    if (net->use_redundant) {
        for (int st = 0; st < net->n_sub_tuples; st++) {
            const float *w = net->sub_weights[st];
            for (int s = 0; s < net->n_sub_sym[st]; s++) {
                total += w[encode5_board(b, net->sub_syms[st][s].pos)];
                active++;
            }
        }
    }

    if (n_active) *n_active = active;
    return total;
}

static inline float net_evaluate(const ntuple_net_t *net, board_t b) {
    return net_evaluate_ex(net, b, NULL);
}

/* ─── Network Update ──────────────────────────────────────────── */

/*
 * Update weights using forward TD(0).
 * Applies gradient clipping (delta clamped to [-1000, +1000]).
 * Uses distributed LR: effective_lr = base_lr / n_active_features.
 * With TC-learning enabled, each weight has an adaptive LR based on
 * temporal coherence of its updates.
 */
static void net_update(ntuple_net_t *net, board_t b, float delta, float base_lr,
                       const training_config_t *cfg, int max_tile_log2) {
    /* Gradient clipping */
    if (delta > 1000.0f) delta = 1000.0f;
    if (delta < -1000.0f) delta = -1000.0f;

    /* Auto-downgrade for update too */
    int max_nib = board_max_nibble(b);
    while (max_nib > 15) {
        b = board_downgrade(b);
        max_nib--;
    }

    /* Count active features for distributed LR */
    int n_active = 0;
    for (int t = 0; t < net->n_base; t++)
        n_active += net->n_sym[t];
    if (net->use_redundant) {
        for (int st = 0; st < net->n_sub_tuples; st++)
            n_active += net->n_sub_sym[st];
    }
    if (n_active < 1) n_active = 1;

    /* Multistage LR adjustment */
    float lr = base_lr;
    if (cfg) lr = multistage_get_lr(cfg, base_lr, max_tile_log2);

    /* Direct LR per weight (matches Szubert's proven approach) */
    float dist_lr = lr;

    if (!net->use_tc) {
        float adj = dist_lr * delta;
        for (int t = 0; t < net->n_base; t++) {
            float *w = net->weights[t];
            for (int s = 0; s < net->n_sym[t]; s++)
                w[encode6_board(b, net->syms[t][s].pos)] += adj;
        }
    } else {
        /* TC-learning: per-weight adaptive LR */
        float abs_delta = fabsf(delta);
        float decay = 0.9995f;
        for (int t = 0; t < net->n_base; t++) {
            float *w = net->weights[t];
            float *ts = net->tc_sum[t];
            float *ta = net->tc_abs[t];
            for (int s = 0; s < net->n_sym[t]; s++) {
                int idx = encode6_board(b, net->syms[t][s].pos);
                ts[idx] = ts[idx] * decay + delta;
                ta[idx] = ta[idx] * decay + abs_delta;
                float ratio = (ta[idx] > 1e-6f)
                    ? (fabsf(ts[idx]) / ta[idx])
                    : 1.0f;
                w[idx] += dist_lr * ratio * delta;
            }
        }
    }

    /* Redundant encoding: update 5-tuple sub-feature weights */
    if (net->use_redundant) {
        if (!net->use_tc) {
            float adj = dist_lr * delta;
            for (int st = 0; st < net->n_sub_tuples; st++) {
                float *w = net->sub_weights[st];
                for (int s = 0; s < net->n_sub_sym[st]; s++)
                    w[encode5_board(b, net->sub_syms[st][s].pos)] += adj;
            }
        } else {
            float abs_delta = fabsf(delta);
            float decay = 0.9995f;
            for (int st = 0; st < net->n_sub_tuples; st++) {
                float *w = net->sub_weights[st];
                float *ts = net->sub_tc_sum[st];
                float *ta = net->sub_tc_abs[st];
                for (int s = 0; s < net->n_sub_sym[st]; s++) {
                    int idx = encode5_board(b, net->sub_syms[st][s].pos);
                    ts[idx] = ts[idx] * decay + delta;
                    ta[idx] = ta[idx] * decay + abs_delta;
                    float ratio = (ta[idx] > 1e-6f)
                        ? (fabsf(ts[idx]) / ta[idx])
                        : 1.0f;
                    w[idx] += dist_lr * ratio * delta;
                }
            }
        }
    }
}

/* ─── Network Init / Free ─────────────────────────────────────── */

static void net_init(ntuple_net_t *net, int use_tc, int use_redundant) {
    net->n_base = n_tuples;
    net->use_tc = use_tc;
    net->use_redundant = use_redundant;
    net->n_sub_tuples = 0;
    net->sub_tuple_size = SUB_TUPLE_SIZE;

    for (int t = 0; t < n_tuples; t++) {
        net->weights[t] = (float *)calloc(LUT_SIZE, sizeof(float));
        if (!net->weights[t]) { fprintf(stderr, "OOM allocating weights\n"); exit(1); }
        if (use_tc) {
            net->tc_abs[t] = (float *)calloc(LUT_SIZE, sizeof(float));
            net->tc_sum[t] = (float *)calloc(LUT_SIZE, sizeof(float));
            if (!net->tc_abs[t] || !net->tc_sum[t]) {
                fprintf(stderr, "OOM allocating TC tables\n"); exit(1);
            }
        } else {
            net->tc_abs[t] = NULL;
            net->tc_sum[t] = NULL;
        }
        generate_symmetries(net, t);
    }

    /* Initialize redundant 5-tuple sub-features */
    if (use_redundant) {
        int st_idx = 0;
        for (int t = 0; t < n_tuples; t++) {
            /* Generate C(6,5)=6 sub-tuples by omitting one position each */
            for (int omit = 0; omit < TUPLE_SIZE; omit++) {
                int sub_pos[SUB_TUPLE_SIZE][2];
                int k = 0;
                for (int p = 0; p < TUPLE_SIZE; p++) {
                    if (p == omit) continue;
                    sub_pos[k][0] = active_tuples[t][p][0];
                    sub_pos[k][1] = active_tuples[t][p][1];
                    k++;
                }
                net->sub_weights[st_idx] = (float *)calloc(SUB_LUT_SIZE, sizeof(float));
                if (!net->sub_weights[st_idx]) {
                    fprintf(stderr, "OOM allocating sub-tuple weights\n"); exit(1);
                }
                if (use_tc) {
                    net->sub_tc_abs[st_idx] = (float *)calloc(SUB_LUT_SIZE, sizeof(float));
                    net->sub_tc_sum[st_idx] = (float *)calloc(SUB_LUT_SIZE, sizeof(float));
                    if (!net->sub_tc_abs[st_idx] || !net->sub_tc_sum[st_idx]) {
                        fprintf(stderr, "OOM allocating sub-tuple TC tables\n"); exit(1);
                    }
                } else {
                    net->sub_tc_abs[st_idx] = NULL;
                    net->sub_tc_sum[st_idx] = NULL;
                }
                generate_sub_symmetries(net, st_idx, sub_pos);
                st_idx++;
            }
        }
        net->n_sub_tuples = st_idx;
        printf("[Redundant] Initialized %d sub-tuples (%d-pos), "
               "%.0f MB additional memory\n",
               st_idx, SUB_TUPLE_SIZE,
               (double)st_idx * SUB_LUT_SIZE * sizeof(float) / 1048576.0);
    }
}

static void net_free(ntuple_net_t *net) {
    for (int t = 0; t < net->n_base; t++) {
        free(net->weights[t]);
        if (net->tc_abs[t]) free(net->tc_abs[t]);
        if (net->tc_sum[t]) free(net->tc_sum[t]);
    }
    if (net->use_redundant) {
        for (int st = 0; st < net->n_sub_tuples; st++) {
            free(net->sub_weights[st]);
            if (net->sub_tc_abs[st]) free(net->sub_tc_abs[st]);
            if (net->sub_tc_sum[st]) free(net->sub_tc_sum[st]);
        }
    }
}

/* ─── Save / Load (backward-compatible binary format) ─────────── */

static void net_save(const ntuple_net_t *net, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) { perror("net_save"); return; }
    int n = net->n_base;
    fwrite(&n, sizeof(int), 1, f);
    for (int t = 0; t < n; t++) {
        int size = LUT_SIZE;
        fwrite(&size, sizeof(int), 1, f);
        fwrite(net->weights[t], sizeof(float), size, f);
    }
    /* TC tables with marker for backward compatibility */
    if (net->use_tc) {
        int marker = 0x5443; /* "TC" */
        fwrite(&marker, sizeof(int), 1, f);
        for (int t = 0; t < n; t++) {
            fwrite(net->tc_sum[t], sizeof(float), LUT_SIZE, f);
            fwrite(net->tc_abs[t], sizeof(float), LUT_SIZE, f);
        }
    }
    /* Redundant sub-tuple weights with marker */
    if (net->use_redundant) {
        int marker = 0x5244; /* "RD" */
        fwrite(&marker, sizeof(int), 1, f);
        int nsub = net->n_sub_tuples;
        fwrite(&nsub, sizeof(int), 1, f);
        for (int st = 0; st < nsub; st++) {
            int size = SUB_LUT_SIZE;
            fwrite(&size, sizeof(int), 1, f);
            fwrite(net->sub_weights[st], sizeof(float), size, f);
        }
        if (net->use_tc) {
            int tc_marker = 0x5254; /* "RT" = redundant TC */
            fwrite(&tc_marker, sizeof(int), 1, f);
            for (int st = 0; st < nsub; st++) {
                fwrite(net->sub_tc_sum[st], sizeof(float), SUB_LUT_SIZE, f);
                fwrite(net->sub_tc_abs[st], sizeof(float), SUB_LUT_SIZE, f);
            }
        }
    }
    fclose(f);
    /* Report file size */
    FILE *fs = fopen(path, "rb");
    if (fs) {
        fseek(fs, 0, SEEK_END);
        long sz = ftell(fs);
        fclose(fs);
        printf("Saved %s (%.1f MB)\n", path, sz / 1048576.0);
    }
}

static int net_load(ntuple_net_t *net, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    int n;
    if (fread(&n, sizeof(int), 1, f) != 1) { fclose(f); return 0; }
    if (n != net->n_base) {
        fprintf(stderr, "Incompatible checkpoint: %d vs %d tuples\n", n, net->n_base);
        fclose(f); return 0;
    }
    for (int t = 0; t < n; t++) {
        int size;
        if (fread(&size, sizeof(int), 1, f) != 1) { fclose(f); return 0; }
        if (size != LUT_SIZE) {
            fprintf(stderr, "Incompatible LUT size: %d vs %d\n", size, LUT_SIZE);
            fclose(f); return 0;
        }
        if (fread(net->weights[t], sizeof(float), size, f) != (size_t)size) {
            fclose(f); return 0;
        }
    }
    /* Try reading optional sections by marker */
    int marker = 0;
    int loaded_tc = 0, loaded_rd = 0;

    while (fread(&marker, sizeof(int), 1, f) == 1) {
        if (marker == 0x5443 && net->use_tc) {
            /* TC marker: load TC tables for main tuples */
            for (int t = 0; t < n; t++) {
                fread(net->tc_sum[t], sizeof(float), LUT_SIZE, f);
                fread(net->tc_abs[t], sizeof(float), LUT_SIZE, f);
            }
            loaded_tc = 1;
        } else if (marker == 0x5244 && net->use_redundant) {
            /* RD marker: load redundant sub-tuple weights */
            int nsub;
            if (fread(&nsub, sizeof(int), 1, f) != 1) break;
            if (nsub != net->n_sub_tuples) {
                fprintf(stderr, "Warning: sub-tuple count mismatch %d vs %d, skipping\n",
                        nsub, net->n_sub_tuples);
                /* Skip past the data */
                for (int st = 0; st < nsub; st++) {
                    int size;
                    if (fread(&size, sizeof(int), 1, f) != 1) break;
                    fseek(f, (long)size * sizeof(float), SEEK_CUR);
                }
                continue;
            }
            for (int st = 0; st < nsub; st++) {
                int size;
                if (fread(&size, sizeof(int), 1, f) != 1) break;
                if (size != SUB_LUT_SIZE) {
                    fprintf(stderr, "Warning: sub-LUT size mismatch %d vs %d\n",
                            size, SUB_LUT_SIZE);
                    fseek(f, (long)size * sizeof(float), SEEK_CUR);
                    continue;
                }
                fread(net->sub_weights[st], sizeof(float), size, f);
            }
            loaded_rd = 1;
        } else if (marker == 0x5254 && net->use_redundant && net->use_tc) {
            /* RT marker: load TC tables for sub-tuples */
            for (int st = 0; st < net->n_sub_tuples; st++) {
                fread(net->sub_tc_sum[st], sizeof(float), SUB_LUT_SIZE, f);
                fread(net->sub_tc_abs[st], sizeof(float), SUB_LUT_SIZE, f);
            }
        } else {
            /* Unknown marker, stop reading */
            break;
        }
    }

    printf("Loaded %s (weights%s%s)\n", path,
           loaded_tc ? " + TC" : "",
           loaded_rd ? " + redundant" : "");
    fclose(f);
    return 1;
}

/* ─── Transposition Table for Search ──────────────────────────── */

typedef struct {
    board_t key;
    float   value;
    int     depth;
    int     valid;
} tt_entry_t;

/* Per-thread transposition table to avoid contention */
typedef struct {
    tt_entry_t *table;
    int size;
} tt_t;

static void tt_init(tt_t *tt) {
    tt->size = TT_SIZE;
    tt->table = (tt_entry_t *)calloc(tt->size, sizeof(tt_entry_t));
    if (!tt->table) { fprintf(stderr, "OOM allocating TT\n"); exit(1); }
}

static void tt_free(tt_t *tt) {
    free(tt->table);
}

static void tt_clear(tt_t *tt) {
    memset(tt->table, 0, tt->size * sizeof(tt_entry_t));
}

static inline uint32_t tt_hash(board_t b) {
    /* Simple hash: XOR-fold the 64-bit board into 32 bits, then mask */
    uint64_t h = b ^ (b >> 32);
    h ^= (h >> 16);
    return (uint32_t)(h & TT_MASK);
}

static inline int tt_probe(tt_t *tt, board_t b, int depth, float *value) {
    uint32_t idx = tt_hash(b);
    tt_entry_t *e = &tt->table[idx];
    if (e->valid && e->key == b && e->depth >= depth) {
        *value = e->value;
        return 1;
    }
    return 0;
}

static inline void tt_store(tt_t *tt, board_t b, int depth, float value) {
    uint32_t idx = tt_hash(b);
    tt_entry_t *e = &tt->table[idx];
    e->key = b;
    e->value = value;
    e->depth = depth;
    e->valid = 1;
}

/* ─── Search (Expectimax with TT) ─────────────────────────────── */

static float search_chance(const ntuple_net_t *net, board_t b, int depth,
                           unsigned int *seed, tt_t *tt);

static float search_max(const ntuple_net_t *net, board_t b, int depth,
                        unsigned int *seed, tt_t *tt) {
    float best = -1e18f;
    int any = 0;
    for (int d = 0; d < 4; d++) {
        int score, moved;
        board_t after = do_move(b, d, &score, &moved);
        if (!moved) continue;
        any = 1;
        float v;
        if (depth <= 0) {
            v = (float)score + net_evaluate(net, after);
        } else {
            v = (float)score + search_chance(net, after, depth - 1, seed, tt);
        }
        if (v > best) best = v;
    }
    return any ? best : net_evaluate(net, b);
}

static float search_chance(const ntuple_net_t *net, board_t b, int depth,
                           unsigned int *seed, tt_t *tt) {
    /* Check TT */
    float cached;
    if (tt_probe(tt, b, depth, &cached)) return cached;

    /* Collect empty cells */
    int empty_pos[16], ne = 0;
    for (int i = 0; i < 16; i++) {
        if (((b >> (i * 4)) & 0xF) == 0)
            empty_pos[ne++] = i;
    }
    if (ne == 0) return net_evaluate(net, b);

    /* Sample up to 4 empty cells for deeper search */
    int sn = ne;
    int idx[16];
    for (int i = 0; i < ne; i++) idx[i] = i;
    if (ne > 4) {
        sn = 4;
        for (int i = 0; i < sn; i++) {
            int j = i + trand(seed) % (ne - i);
            int tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp;
        }
    }

    float total = 0;
    for (int i = 0; i < sn; i++) {
        int pos = empty_pos[idx[i]];
        /* Tile 2 (nibble=1), probability 0.9 */
        board_t b2 = b | ((uint64_t)1 << (pos * 4));
        total += 0.9f * search_max(net, b2, depth, seed, tt);
        /* Tile 4 (nibble=2), probability 0.1 */
        board_t b4 = b | ((uint64_t)2 << (pos * 4));
        total += 0.1f * search_max(net, b4, depth, seed, tt);
    }
    float result = total / sn;
    tt_store(tt, b, depth, result);
    return result;
}

/* ─── Best Action Selection ───────────────────────────────────── */

static int select_best(const ntuple_net_t *net, board_t state,
                       board_t *best_after, float *best_reward_out,
                       int search_depth, unsigned int *seed, tt_t *tt) {
    int best_action = -1;
    float best_value = -1e18f;
    *best_reward_out = 0;

    for (int d = 0; d < 4; d++) {
        int score, moved;
        board_t after = do_move(state, d, &score, &moved);
        if (!moved) continue;

        float value;
        if (search_depth <= 0) {
            value = (float)score + net_evaluate(net, after);
        } else {
            value = (float)score + search_chance(net, after, search_depth - 1, seed, tt);
        }

        if (value > best_value) {
            best_value = value;
            best_action = d;
            *best_after = after;
            *best_reward_out = (float)score;
        }
    }
    return best_action;
}

/* ─── Carousel Helpers (bitboard <-> grid conversion) ─────────── */

/*
 * Check if the board's max tile matches a carousel threshold and save if so.
 */
static void carousel_check_save(training_config_t *cfg, board_t b, int score_so_far) {
    if (!cfg->carousel) return;
    int max_nib = board_max_nibble(b);
    int grid[4][4];
    board_to_grid(b, grid);
    carousel_maybe_save(cfg, grid, score_so_far, max_nib);
}

/*
 * Get a carousel start position as a board_t.
 * Returns the saved score, or 0 if no position available.
 */
static int carousel_get_board(const training_config_t *cfg, board_t *b,
                              unsigned int *seed) {
    int grid[4][4];
    int score = carousel_get_position(cfg, grid, seed);
    *b = board_from_grid(grid);
    return score;
}

/* ─── Weight Promotion ───────────────────────────────────────── */

/*
 * Promote weights when max tile crosses a new threshold during training.
 * For each tuple and symmetry, LUT entries that contain new_max_nibble
 * in any position are initialized by copying weights from the corresponding
 * entry where that position has old_max_nibble instead.
 * Only copies if the target entry is still at zero (initial value).
 *
 * This bootstraps evaluation of boards with the new high tile using
 * knowledge learned about boards with the previous high tile.
 */
static void promote_weights(ntuple_net_t *net, int old_max_nibble, int new_max_nibble) {
    if (new_max_nibble > 15 || old_max_nibble > 15) return;
    if (new_max_nibble <= old_max_nibble) return;

    int promoted_6 = 0;

    /* Promote 6-tuple weights */
    for (int t = 0; t < net->n_base; t++) {
        float *w = net->weights[t];
        /*
         * Iterate over all LUT indices that have new_max_nibble in at least
         * one position. For a 6-tuple, the index is encoded as:
         *   idx = n0*16^5 + n1*16^4 + n2*16^3 + n3*16^2 + n4*16 + n5
         * We scan all 16^6 entries and for each one that contains
         * new_max_nibble, we compute the source index with old_max_nibble
         * in those positions.
         */
        int stride[TUPLE_SIZE];
        stride[TUPLE_SIZE - 1] = 1;
        for (int p = TUPLE_SIZE - 2; p >= 0; p--)
            stride[p] = stride[p + 1] * 16;

        for (int idx = 0; idx < LUT_SIZE; idx++) {
            /* Check if this index contains new_max_nibble in any position */
            int has_new = 0;
            int src_idx = idx;
            int tmp = idx;
            for (int p = 0; p < TUPLE_SIZE; p++) {
                int nibble = tmp / stride[p];
                tmp -= nibble * stride[p];
                if (nibble == new_max_nibble) {
                    has_new = 1;
                    src_idx -= (new_max_nibble - old_max_nibble) * stride[p];
                }
            }
            if (has_new && w[idx] == 0.0f && src_idx >= 0 && src_idx < LUT_SIZE) {
                w[idx] = w[src_idx];
                promoted_6++;
            }
        }
    }

    /* Promote 5-tuple sub-weights if redundant encoding is active */
    int promoted_5 = 0;
    if (net->use_redundant) {
        for (int st = 0; st < net->n_sub_tuples; st++) {
            float *w = net->sub_weights[st];
            int sub_stride[SUB_TUPLE_SIZE];
            sub_stride[SUB_TUPLE_SIZE - 1] = 1;
            for (int p = SUB_TUPLE_SIZE - 2; p >= 0; p--)
                sub_stride[p] = sub_stride[p + 1] * 16;

            for (int idx = 0; idx < SUB_LUT_SIZE; idx++) {
                int has_new = 0;
                int src_idx = idx;
                int tmp = idx;
                for (int p = 0; p < SUB_TUPLE_SIZE; p++) {
                    int nibble = tmp / sub_stride[p];
                    tmp -= nibble * sub_stride[p];
                    if (nibble == new_max_nibble) {
                        has_new = 1;
                        src_idx -= (new_max_nibble - old_max_nibble) * sub_stride[p];
                    }
                }
                if (has_new && w[idx] == 0.0f && src_idx >= 0 && src_idx < SUB_LUT_SIZE) {
                    w[idx] = w[src_idx];
                    promoted_5++;
                }
            }
        }
    }

    (void)promoted_6;
    (void)promoted_5;
}

/* ─── Global Shared State ─────────────────────────────────────── */

static ntuple_net_t shared_net;
static training_config_t train_cfg;
static atomic_int episodes_done = 0;
static atomic_int total_wins = 0;
static int total_episodes = 0;
static int train_search_depth = 0;
static float lr_start = 0.01f;
static float lr_end = 0.0005f;
static const char *global_save_dir = "checkpoints";

/* Per-episode CSV logging */
static FILE *csv_file = NULL;
static pthread_mutex_t csv_mutex = PTHREAD_MUTEX_INITIALIZER;

/* Checkpoint mutex */
static pthread_mutex_t ckpt_mutex = PTHREAD_MUTEX_INITIALIZER;

/* ─── Per-Thread Statistics ────────────────────────────────────── */

#define STATS_BUF_SIZE 100

typedef struct {
    int thread_id;
    int local_buf_idx;
    int local_scores[STATS_BUF_SIZE];
    int local_tiles[STATS_BUF_SIZE];
    long move_counts[4]; /* UP, RIGHT, DOWN, LEFT */
} thread_stats_t;

/* ─── Training Worker ─────────────────────────────────────────── */

static void *train_worker(void *arg) {
    thread_stats_t *stats = (thread_stats_t *)arg;
    unsigned int seed = (unsigned int)(time(NULL) + stats->thread_id * 7919);

    /* Per-thread transposition table for search */
    tt_t tt;
    int has_tt = (train_search_depth > 0);
    if (has_tt) {
        tt_init(&tt);
    }

    while (1) {
        int ep = atomic_fetch_add(&episodes_done, 1) + 1;
        if (ep > total_episodes) break;

        /* Compute LR for this episode (decayed) */
        float progress = (float)ep / (float)total_episodes;
        float base_lr = lr_start * powf(lr_end / lr_start, progress);

        /* Initialize board */
        board_t state = 0;
        int game_score = 0;
        int from_carousel = 0;

        /* Carousel: optionally start from a saved position */
        if (carousel_should_use(&train_cfg, ep)) {
            board_t carousel_board;
            int carousel_score = carousel_get_board(&train_cfg, &carousel_board, &seed);
            if (carousel_score > 0 || carousel_board != 0) {
                state = carousel_board;
                game_score = carousel_score;
                from_carousel = 1;
            }
        }

        if (!from_carousel) {
            state = board_add_random(state, &seed);
            state = board_add_random(state, &seed);
        }

        /* Track max tile for carousel saving */
        int prev_max_nib = board_max_nibble(state);

        /* Clear TT for each episode */
        if (has_tt) tt_clear(&tt);

        /* First move: select best action */
        board_t prev_after;
        float prev_reward;
        int action = select_best(&shared_net, state, &prev_after, &prev_reward,
                                 train_search_depth, &seed, has_tt ? &tt : NULL);
        if (action == -1) goto episode_end;

        stats->move_counts[action]++;
        state = board_add_random(prev_after, &seed);
        game_score += (int)prev_reward;

        /* Main game loop: forward TD(0) afterstate learning */
        int move_count = 0;
        int max_moves = 3000; /* Safety limit — normal games are 500-2000 moves */
        while (!board_game_over(state) && move_count < max_moves) {
            move_count++;
            board_t curr_after;
            float curr_reward;
            action = select_best(&shared_net, state, &curr_after, &curr_reward,
                                 train_search_depth, &seed, has_tt ? &tt : NULL);
            if (action == -1) break;

            stats->move_counts[action]++;

            /* TD(0) update: delta = r + V(s') - V(s_prev) */
            float v_prev = net_evaluate(&shared_net, prev_after);
            float v_curr = net_evaluate(&shared_net, curr_after);
            float delta = curr_reward + v_curr - v_prev;

            int max_tile_log2 = board_max_nibble(prev_after);
            net_update(&shared_net, prev_after, delta, base_lr, &train_cfg, max_tile_log2);

            /* Add random tile and advance */
            state = board_add_random(curr_after, &seed);
            game_score += (int)curr_reward;
            prev_after = curr_after;

            /* Check if max tile crossed a threshold */
            int cur_max_nib = board_max_nibble(state);
            if (cur_max_nib > prev_max_nib) {
                /* Carousel: save position at threshold */
                carousel_check_save(&train_cfg, state, game_score);
                /* Weight promotion: copy weights for the new tile value */
                if (train_cfg.use_weight_promotion) {
                    promote_weights(&shared_net, prev_max_nib, cur_max_nib);
                }
                prev_max_nib = cur_max_nib;
            }
        }

        /* Terminal update: V(terminal) = 0, so delta = -V(s_last) */
        {
            float v_last = net_evaluate(&shared_net, prev_after);
            int max_tile_log2 = board_max_nibble(prev_after);
            net_update(&shared_net, prev_after, -v_last, base_lr, &train_cfg, max_tile_log2);
        }

episode_end:
        ;
        int max_tile = board_max_tile(state);
        if (max_tile >= 2048) atomic_fetch_add(&total_wins, 1);

        /* Store per-episode stats in circular buffer */
        int bi = stats->local_buf_idx % STATS_BUF_SIZE;
        stats->local_scores[bi] = game_score;
        stats->local_tiles[bi] = max_tile;
        stats->local_buf_idx++;

        /* CSV logging (thread-safe) */
        if (csv_file) {
            pthread_mutex_lock(&csv_mutex);
            fprintf(csv_file, "%d,%d,%d\n", ep, game_score, max_tile);
            fflush(csv_file);
            pthread_mutex_unlock(&csv_mutex);
        }
    }

    if (has_tt) tt_free(&tt);
    return NULL;
}

/* ─── Main Training Orchestrator ──────────────────────────────── */

static void train(int episodes, const char *save_dir, int use_tc, int use_redundant,
                  int n_threads, const char *csv_path) {
    /* Initialize network */
    net_init(&shared_net, use_tc, train_cfg.use_redundant);

    char path_latest[512], path_best[512];
    snprintf(path_latest, sizeof(path_latest), "%s/ntuple_latest.bin", save_dir);
    snprintf(path_best, sizeof(path_best), "%s/ntuple_best.bin", save_dir);

    /* Try loading existing checkpoint */
    if (net_load(&shared_net, path_latest)) {
        train_cfg.loaded_from_checkpoint = 1;
    }

    /* Apply optimistic initialization (skipped if checkpoint loaded) */
    config_apply_optimistic(&train_cfg, shared_net.weights, n_tuples, LUT_SIZE);

    /* Apply optimistic init to sub-tuples as well */
    if (shared_net.use_redundant) {
        config_apply_optimistic(&train_cfg, shared_net.sub_weights,
                                shared_net.n_sub_tuples, SUB_LUT_SIZE);
    }

    /* Open CSV file for per-episode logging */
    if (csv_path) {
        csv_file = fopen(csv_path, "a");
        if (csv_file) {
            fseek(csv_file, 0, SEEK_END);
            if (ftell(csv_file) == 0)
                fprintf(csv_file, "episode,score,max_tile\n");
            printf("Logging per-episode data to %s\n", csv_path);
        } else {
            fprintf(stderr, "Warning: could not open CSV file %s\n", csv_path);
        }
    }

    /* Print training configuration */
    print_config(&train_cfg);

    total_episodes = episodes;
    episodes_done = 0;
    total_wins = 0;
    float best_avg = 0;

    thread_stats_t *stats = (thread_stats_t *)calloc(n_threads, sizeof(thread_stats_t));
    pthread_t *threads = (pthread_t *)malloc(n_threads * sizeof(pthread_t));

    time_t start_time = time(NULL);

    /* Launch worker threads */
    for (int i = 0; i < n_threads; i++) {
        stats[i].thread_id = i;
        pthread_create(&threads[i], NULL, train_worker, &stats[i]);
    }

    /* Monitor and report from main thread */
    int last_reported = 0;
    while (1) {
        int done = atomic_load(&episodes_done);
        if (done >= episodes) break;

        int current_100 = (done / 100) * 100;
        if (current_100 > last_reported && current_100 > 0) {
            last_reported = current_100;

            /* Aggregate stats across all threads */
            double avg = 0;
            int tile_counts[20] = {0};
            int n_samples = 0;
            for (int t = 0; t < n_threads; t++) {
                int n = (stats[t].local_buf_idx < STATS_BUF_SIZE)
                    ? stats[t].local_buf_idx : STATS_BUF_SIZE;
                for (int i = 0; i < n; i++) {
                    avg += stats[t].local_scores[i];
                    int tv = stats[t].local_tiles[i];
                    int lt = 0;
                    while (tv > 1) { tv >>= 1; lt++; }
                    if (lt < 20) tile_counts[lt]++;
                    n_samples++;
                }
            }
            if (n_samples > 0) avg /= n_samples;

            time_t elapsed = time(NULL) - start_time;
            int wins = atomic_load(&total_wins);

            /* Compute current LR */
            float progress = (float)done / (float)total_episodes;
            float cur_lr = lr_start * powf(lr_end / lr_start, progress);

            printf("\n============================================================\n");
            printf("N-Tuple Episode %d/%d | Time: %lds | %d threads\n",
                   done, episodes, (long)elapsed, n_threads);
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
            printf("  LR:           %f\n", cur_lr);
            printf("  Ep/s:         %.1f\n",
                   (float)done / (elapsed > 0 ? (float)elapsed : 1.0f));

            /* Move distribution */
            long total_moves[4] = {0};
            for (int t = 0; t < n_threads; t++)
                for (int d = 0; d < 4; d++)
                    total_moves[d] += stats[t].move_counts[d];
            long sum_moves = total_moves[0] + total_moves[1] +
                             total_moves[2] + total_moves[3];
            if (sum_moves > 0) {
                printf("  Moves:        UP=%.1f%% RIGHT=%.1f%% DOWN=%.1f%% LEFT=%.1f%%\n",
                       100.0 * total_moves[0] / sum_moves,
                       100.0 * total_moves[1] / sum_moves,
                       100.0 * total_moves[2] / sum_moves,
                       100.0 * total_moves[3] / sum_moves);
            }

            /* Config summary line */
            printf("  Config:       OPT=%s MS=%s CR=%s TC=%s WP=%s RD=%s depth=%d-ply\n",
                   train_cfg.optimistic ? "ON" : "OFF",
                   train_cfg.multistage ? "ON" : "OFF",
                   train_cfg.carousel ? "ON" : "OFF",
                   shared_net.use_tc ? "ON" : "OFF",
                   train_cfg.use_weight_promotion ? "ON" : "OFF",
                   shared_net.use_redundant ? "ON" : "OFF",
                   1 + train_search_depth * 2);

            /* Save best model */
            if (avg > best_avg) {
                best_avg = avg;
                pthread_mutex_lock(&ckpt_mutex);
                net_save(&shared_net, path_best);
                pthread_mutex_unlock(&ckpt_mutex);
            }

            /* Periodic save every 1000 episodes */
            if (current_100 % 1000 == 0) {
                pthread_mutex_lock(&ckpt_mutex);
                net_save(&shared_net, path_latest);
                pthread_mutex_unlock(&ckpt_mutex);
            }

            /* Periodic checkpoint every 10k episodes for learning curves */
            if (current_100 % 10000 == 0) {
                char path_periodic[512];
                snprintf(path_periodic, sizeof(path_periodic),
                         "%s/ntuple_ep%d.bin", save_dir, current_100);
                pthread_mutex_lock(&ckpt_mutex);
                net_save(&shared_net, path_periodic);
                pthread_mutex_unlock(&ckpt_mutex);
            }
        }

        /* Sleep briefly to avoid busy-waiting */
        struct timespec ts = {0, 100000000}; /* 100ms */
        nanosleep(&ts, NULL);
    }

    /* Join all threads */
    for (int i = 0; i < n_threads; i++)
        pthread_join(threads[i], NULL);

    /* Final save */
    net_save(&shared_net, path_latest);
    printf("\nTraining complete! Best avg score: %.0f\n", best_avg);

    if (csv_file) {
        fclose(csv_file);
        csv_file = NULL;
        printf("Per-episode CSV data finalized.\n");
    }

    free(stats);
    free(threads);
    net_free(&shared_net);
}

/* ─── Directory Creation Helper ───────────────────────────────── */

static void ensure_dir(const char *path) {
    struct stat st;
    if (stat(path, &st) != 0) {
        mkdir(path, 0755);
    }
}

/* ─── Main ────────────────────────────────────────────────────── */

int main(int argc, char **argv) {
    setbuf(stdout, NULL);

    int episodes = 50000;
    const char *save_dir = "checkpoints";
    int use_tc = 0;
    int use_redundant = 0;
    int n_threads = 1;
    const char *csv_path = NULL;

    /* Initialize training config with defaults (all techniques off) */
    config_init(&train_cfg);

    /* Parse command-line arguments */
    for (int i = 1; i < argc; i++) {
        /* Check training_config.h flags first */
        if (config_parse_arg(&train_cfg, argc, argv, &i))
            continue;

        if (strcmp(argv[i], "--episodes") == 0 && i + 1 < argc)
            episodes = atoi(argv[++i]);
        else if (strcmp(argv[i], "--save-dir") == 0 && i + 1 < argc)
            save_dir = argv[++i];
        else if (strcmp(argv[i], "--depth") == 0 && i + 1 < argc)
            train_search_depth = atoi(argv[++i]);
        else if (strcmp(argv[i], "--redundant") == 0)
            use_redundant = 1;
        else if (strcmp(argv[i], "--tc") == 0)
            use_tc = 1;
        else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc)
            n_threads = atoi(argv[++i]);
        else if (strcmp(argv[i], "--log-csv") == 0 && i + 1 < argc)
            csv_path = argv[++i];
        else if (strcmp(argv[i], "--lr-start") == 0 && i + 1 < argc)
            lr_start = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--lr-end") == 0 && i + 1 < argc)
            lr_end = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--tdl2048") == 0) {
            n_tuples = 4;
            active_tuples = TDL2048_TUPLES;
            lr_start = 0.1f;
            lr_end = 0.001f;
            printf("[TDL2048+] Using 4x6patt configuration (32 features)\n");
        }
        else {
            fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            fprintf(stderr, "Usage: %s [OPTIONS]\n", argv[0]);
            fprintf(stderr, "  --episodes N       Total training episodes (default: 50000)\n");
            fprintf(stderr, "  --threads N        Number of worker threads (default: 1)\n");
            fprintf(stderr, "  --depth 0|1|2      Search depth: 1/3/5-ply (default: 0)\n");
            fprintf(stderr, "  --tc               Enable TC-learning\n");
            fprintf(stderr, "  --optimistic       Enable optimistic initialization\n");
            fprintf(stderr, "  --multistage       Enable multistage LR\n");
            fprintf(stderr, "  --carousel         Enable carousel shaping\n");
            fprintf(stderr, "  --weight-promotion Enable weight promotion across stages\n");
            fprintf(stderr, "  --tdl2048          Use TDL2048+ 4x6patt configuration\n");
            fprintf(stderr, "  --redundant        Enable redundant 5-tuple sub-features\n");
            fprintf(stderr, "  --lr-start F       Starting LR (default: 0.01)\n");
            fprintf(stderr, "  --lr-end F         Ending LR (default: 0.0005)\n");
            fprintf(stderr, "  --log-csv PATH     Per-episode CSV log file\n");
            fprintf(stderr, "  --save-dir DIR     Checkpoint directory (default: checkpoints)\n");
            return 1;
        }
    }

    global_save_dir = save_dir;

    /* Ensure output directories exist */
    ensure_dir(save_dir);
    if (csv_path) {
        /* Ensure parent directory of CSV exists */
        char csv_dir[512];
        strncpy(csv_dir, csv_path, sizeof(csv_dir) - 1);
        csv_dir[sizeof(csv_dir) - 1] = '\0';
        char *slash = strrchr(csv_dir, '/');
        if (slash) {
            *slash = '\0';
            ensure_dir(csv_dir);
        }
    }

    printf("N-Tuple Training for 2048 (State-of-the-Art)\n");
    printf("Episodes: %d | Tuples: %d x %d-pos (%s) | Search: %d-ply\n",
           episodes, n_tuples, TUPLE_SIZE,
           n_tuples == 4 ? "TDL2048+" : "extended",
           1 + train_search_depth * 2);
    printf("TC: %s | Threads: %d | LR: %.6f -> %.6f\n",
           use_tc ? "ON" : "OFF", n_threads, lr_start, lr_end);
    printf("TT size: %d entries (%lu MB)\n",
           TT_SIZE, (unsigned long)(TT_SIZE * sizeof(tt_entry_t) / 1048576));

    /* Build move lookup tables from engine_bitboard.h */
    build_move_tables();

    /* Run training */
    train(episodes, save_dir, use_tc, use_redundant, n_threads, csv_path);

    return 0;
}
