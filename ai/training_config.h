/*
 * training_config.h — Advanced TDL2048+ training techniques
 *
 * Implements five state-of-the-art techniques from TDL2048+:
 *   1. Optimistic Initialization (--optimistic)
 *   2. Multistage Training with stage-aware LR (--multistage)
 *   3. Carousel Shaping with position replay (--carousel)
 *   4. Weight Promotion across stages (--weight-promotion)
 *   5. Redundant Encoding with 5-tuple sub-features (--redundant)
 *
 * Include this header in ntuple_train.c and call the appropriate
 * functions to integrate these techniques into the training loop.
 */

#ifndef TRAINING_CONFIG_H
#define TRAINING_CONFIG_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ─── Constants ────────────────────────────────────────────────── */

#define OPTIMISTIC_INIT_VALUE  320000.0f

/* Multistage thresholds (log2 tile values) */
#define STAGE_THRESHOLD_0  13   /* 8192  = 2^13 */
#define STAGE_THRESHOLD_1  14   /* 16384 = 2^14 */
#define STAGE_THRESHOLD_2  15   /* 32768 = 2^15 */
#define MAX_STAGES          3

/* Carousel buffer */
#define CAROUSEL_BUFFER_SIZE     8192
#define CAROUSEL_SAVE_INTERVAL   1     /* save eligible positions every episode */
#define CAROUSEL_USE_INTERVAL    5     /* use a saved position every N episodes */

/* Carousel save thresholds: log2 tile values at which to snapshot boards */
#define CAROUSEL_NUM_THRESHOLDS  4
static const int CAROUSEL_THRESHOLDS[CAROUSEL_NUM_THRESHOLDS] = {
    10, /* 1024  */
    11, /* 2048  */
    12, /* 4096  */
    13, /* 8192  */
};

/* ─── Structs ──────────────────────────────────────────────────── */

/*
 * Multistage configuration.
 * Each stage covers play from one tile threshold to the next.
 * The LR multiplier scales the base learning rate for that stage.
 */
typedef struct {
    int    threshold_log2;   /* max tile (log2) that triggers promotion */
    float  lr_multiplier;    /* multiply base LR by this in this stage  */
} stage_config_t;

/*
 * Carousel entry: a saved board position from training.
 */
typedef struct {
    int  grid[4][4];         /* board state (log2 tile values)          */
    int  score_so_far;       /* cumulative score when position was saved */
    int  max_tile_log2;      /* max tile on board when saved            */
} carousel_entry_t;

/*
 * Carousel buffer: circular buffer of saved positions.
 */
typedef struct {
    carousel_entry_t entries[CAROUSEL_BUFFER_SIZE];
    int  head;               /* next write position                     */
    int  count;              /* total entries stored (up to buffer size) */
    int  use_interval;       /* start from saved pos every N episodes   */
} carousel_buffer_t;

/*
 * Master training configuration.
 */
typedef struct {
    /* Optimistic initialization */
    int   optimistic;            /* --optimistic flag                   */
    float optimistic_value;      /* initial weight value (default 320k) */

    /* Multistage training */
    int   multistage;            /* --multistage flag                   */
    int   num_stages;
    stage_config_t stages[MAX_STAGES];
    int   current_stage;         /* runtime: active stage index         */

    /* Carousel shaping */
    int   carousel;              /* --carousel flag                     */
    carousel_buffer_t carousel_buf;

    /* Weight promotion */
    int   use_weight_promotion;  /* --weight-promotion flag             */

    /* Redundant encoding (5-tuple sub-features) */
    int   use_redundant;         /* --redundant flag                    */

    /* Tracking */
    int   loaded_from_checkpoint; /* set to 1 if weights came from file */
} training_config_t;

/* ─── Default Initialization ───────────────────────────────────── */

/*
 * Initialize config with sensible defaults. All techniques disabled.
 */
static void config_init(training_config_t *cfg) {
    memset(cfg, 0, sizeof(training_config_t));

    cfg->optimistic       = 0;
    cfg->optimistic_value = OPTIMISTIC_INIT_VALUE;

    cfg->multistage       = 0;
    cfg->num_stages       = MAX_STAGES;
    cfg->current_stage    = 0;

    /* Stage 0: start -> 8192. Standard LR. */
    cfg->stages[0].threshold_log2 = STAGE_THRESHOLD_0;
    cfg->stages[0].lr_multiplier  = 1.0f;

    /* Stage 1: 8192 -> 16384. Reduce LR for fine-tuning. */
    cfg->stages[1].threshold_log2 = STAGE_THRESHOLD_1;
    cfg->stages[1].lr_multiplier  = 0.5f;

    /* Stage 2: 16384 -> 32768. Even smaller LR for endgame. */
    cfg->stages[2].threshold_log2 = STAGE_THRESHOLD_2;
    cfg->stages[2].lr_multiplier  = 0.25f;

    cfg->carousel = 0;
    cfg->carousel_buf.head = 0;
    cfg->carousel_buf.count = 0;
    cfg->carousel_buf.use_interval = CAROUSEL_USE_INTERVAL;

    cfg->use_weight_promotion = 0;
    cfg->use_redundant = 0;

    cfg->loaded_from_checkpoint = 0;
}

/* ─── 1. Optimistic Initialization ─────────────────────────────── */

/*
 * Apply optimistic initialization to all weight tables.
 * Call AFTER net_init() and ONLY if no checkpoint was loaded.
 *
 * Rationale: unvisited states start with high value estimates,
 * encouraging the agent to explore them. As states are visited,
 * TD updates bring values down to accurate estimates.
 *
 * Parameters:
 *   weights   - array of weight table pointers (one per tuple)
 *   n_tuples  - number of tuples
 *   lut_size  - entries per weight table
 *   value     - optimistic initial value (typically 320000)
 */
static void apply_optimistic_init(float **weights, int n_tuples, int lut_size,
                                  float value) {
    for (int t = 0; t < n_tuples; t++) {
        for (int i = 0; i < lut_size; i++) {
            weights[t][i] = value;
        }
    }
}

/*
 * Convenience: apply optimistic init using config, respecting
 * the checkpoint-loaded flag.
 */
static void config_apply_optimistic(training_config_t *cfg,
                                    float **weights, int n_tuples,
                                    int lut_size) {
    if (!cfg->optimistic) return;
    if (cfg->loaded_from_checkpoint) {
        printf("[Optimistic] Skipped: weights loaded from checkpoint\n");
        return;
    }
    apply_optimistic_init(weights, n_tuples, lut_size, cfg->optimistic_value);
    printf("[Optimistic] Initialized %d tables x %d entries to %.0f\n",
           n_tuples, lut_size, cfg->optimistic_value);
}

/* ─── 2. Multistage Training ──────────────────────────────────── */

/*
 * Determine the current stage based on the max tile on the board.
 * Returns the stage index (0 to num_stages-1), or num_stages if
 * all thresholds have been passed (final/endgame stage).
 *
 * Parameters:
 *   max_tile_log2 - log2 of the highest tile on the current board
 */
static int multistage_get_stage(const training_config_t *cfg,
                                int max_tile_log2) {
    for (int s = 0; s < cfg->num_stages; s++) {
        if (max_tile_log2 < cfg->stages[s].threshold_log2)
            return s;
    }
    return cfg->num_stages - 1; /* stay in last stage */
}

/*
 * Get the effective learning rate for the current board state.
 * Applies the stage-specific LR multiplier to the base LR.
 *
 * Parameters:
 *   cfg          - training configuration
 *   base_lr      - the base learning rate (from schedule)
 *   max_tile_log2 - log2 of current max tile
 *
 * Returns: adjusted learning rate
 */
static float multistage_get_lr(const training_config_t *cfg,
                               float base_lr, int max_tile_log2) {
    if (!cfg->multistage) return base_lr;

    int stage = multistage_get_stage(cfg, max_tile_log2);
    return base_lr * cfg->stages[stage].lr_multiplier;
}

/*
 * Get the max tile (log2) from a grid. Utility for stage detection.
 */
static int grid_max_tile_log2(const int g[4][4]) {
    int m = 0;
    for (int y = 0; y < 4; y++)
        for (int x = 0; x < 4; x++)
            if (g[y][x] > m) m = g[y][x];
    return m;
}

/* ─── 3. Carousel Shaping ─────────────────────────────────────── */

/*
 * Save a board position to the carousel buffer if the max tile
 * matches one of the save thresholds.
 *
 * Call this during training whenever max_tile changes.
 *
 * Parameters:
 *   cfg          - training configuration (carousel_buf is modified)
 *   grid         - current board state (log2 values)
 *   score_so_far - cumulative game score at this point
 *   max_tile_log2 - current max tile (log2)
 */
static void carousel_maybe_save(training_config_t *cfg,
                                const int grid[4][4],
                                int score_so_far,
                                int max_tile_log2) {
    if (!cfg->carousel) return;

    /* Check if max_tile matches a save threshold */
    int dominated = 0;
    for (int i = 0; i < CAROUSEL_NUM_THRESHOLDS; i++) {
        if (max_tile_log2 == CAROUSEL_THRESHOLDS[i]) {
            dominated = 1;
            break;
        }
    }
    if (!dominated) return;

    carousel_buffer_t *buf = &cfg->carousel_buf;
    carousel_entry_t *entry = &buf->entries[buf->head % CAROUSEL_BUFFER_SIZE];

    memcpy(entry->grid, grid, sizeof(int) * 16);
    entry->score_so_far  = score_so_far;
    entry->max_tile_log2 = max_tile_log2;

    buf->head = (buf->head + 1) % CAROUSEL_BUFFER_SIZE;
    if (buf->count < CAROUSEL_BUFFER_SIZE)
        buf->count++;
}

/*
 * Decide whether to start a game from a saved position.
 * Returns 1 if a carousel position should be used, 0 otherwise.
 *
 * Parameters:
 *   cfg     - training configuration
 *   episode - current episode number (1-indexed)
 */
static int carousel_should_use(const training_config_t *cfg, int episode) {
    if (!cfg->carousel) return 0;
    if (cfg->carousel_buf.count == 0) return 0;
    return (episode % cfg->carousel_buf.use_interval) == 0;
}

/*
 * Retrieve a random saved position from the carousel buffer.
 * Copies the board state into dst_grid and returns the saved score.
 *
 * Parameters:
 *   cfg       - training configuration
 *   dst_grid  - output: board state to play from
 *   seed      - per-thread RNG seed
 *
 * Returns: the score_so_far associated with the saved position.
 */
static int carousel_get_position(const training_config_t *cfg,
                                 int dst_grid[4][4],
                                 unsigned int *seed) {
    const carousel_buffer_t *buf = &cfg->carousel_buf;
    if (buf->count == 0) return 0;

    /* Simple LCG random using the provided seed */
    *seed = *seed * 1103515245 + 12345;
    int idx = (int)((*seed >> 16) & 0x7FFF) % buf->count;

    const carousel_entry_t *entry = &buf->entries[idx];
    memcpy(dst_grid, entry->grid, sizeof(int) * 16);
    return entry->score_so_far;
}

/*
 * Get carousel buffer statistics for reporting.
 */
static void carousel_stats(const training_config_t *cfg,
                           int *out_count, int *out_capacity) {
    *out_count    = cfg->carousel_buf.count;
    *out_capacity = CAROUSEL_BUFFER_SIZE;
}

/* ─── Configuration Printing ──────────────────────────────────── */

/*
 * Print the full training configuration to stdout.
 */
static void print_config(const training_config_t *cfg) {
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║            TDL2048+ Training Configuration              ║\n");
    printf("╠══════════════════════════════════════════════════════════╣\n");

    /* Optimistic init */
    printf("║ Optimistic Init:  %-38s ║\n",
           cfg->optimistic ? "ENABLED" : "disabled");
    if (cfg->optimistic) {
        char buf[40];
        snprintf(buf, sizeof(buf), "%.0f", cfg->optimistic_value);
        printf("║   Init value:     %-38s ║\n", buf);
        printf("║   From checkpoint:%-38s ║\n",
               cfg->loaded_from_checkpoint ? "yes (init skipped)" : "no (init applied)");
    }

    /* Multistage */
    printf("║ Multistage:       %-38s ║\n",
           cfg->multistage ? "ENABLED" : "disabled");
    if (cfg->multistage) {
        for (int s = 0; s < cfg->num_stages; s++) {
            char buf[60];
            int tile = 1 << cfg->stages[s].threshold_log2;
            if (s == 0)
                snprintf(buf, sizeof(buf), "stage %d: start -> %d (LR x%.2f)",
                         s, tile, cfg->stages[s].lr_multiplier);
            else {
                int prev_tile = 1 << cfg->stages[s-1].threshold_log2;
                snprintf(buf, sizeof(buf), "stage %d: %d -> %d (LR x%.2f)",
                         s, prev_tile, tile, cfg->stages[s].lr_multiplier);
            }
            printf("║   %-54s ║\n", buf);
        }
    }

    /* Carousel */
    printf("║ Carousel Shaping: %-38s ║\n",
           cfg->carousel ? "ENABLED" : "disabled");
    if (cfg->carousel) {
        char buf[60];
        snprintf(buf, sizeof(buf), "buffer=%d, use every %d episodes",
                 CAROUSEL_BUFFER_SIZE, cfg->carousel_buf.use_interval);
        printf("║   %-54s ║\n", buf);

        char thresh[60] = "save at tiles: ";
        for (int i = 0; i < CAROUSEL_NUM_THRESHOLDS; i++) {
            char tmp[10];
            snprintf(tmp, sizeof(tmp), "%s%d",
                     i > 0 ? "," : "", 1 << CAROUSEL_THRESHOLDS[i]);
            strcat(thresh, tmp);
        }
        printf("║   %-54s ║\n", thresh);
    }

    /* Weight promotion */
    printf("║ Weight Promotion: %-38s ║\n",
           cfg->use_weight_promotion ? "ENABLED" : "disabled");

    /* Redundant encoding */
    printf("║ Redundant Enc:    %-38s ║\n",
           cfg->use_redundant ? "ENABLED" : "disabled");

    printf("╚══════════════════════════════════════════════════════════╝\n\n");
}

/* ─── CLI Argument Parsing Helper ─────────────────────────────── */

/*
 * Parse training_config flags from argv.
 * Call this in your argument parsing loop.
 *
 * Returns 1 if the argument was consumed, 0 if not recognized.
 *
 * Recognized flags:
 *   --optimistic          Enable optimistic initialization
 *   --optimistic-value N  Set optimistic init value (default 320000)
 *   --multistage          Enable multistage training
 *   --carousel            Enable carousel shaping
 *   --carousel-interval N Use saved position every N episodes (default 5)
 *   --weight-promotion    Enable weight promotion across stages
 *   --redundant           Enable redundant 5-tuple sub-features
 */
static int config_parse_arg(training_config_t *cfg, int argc, char **argv,
                            int *i) {
    if (strcmp(argv[*i], "--optimistic") == 0) {
        cfg->optimistic = 1;
        return 1;
    }
    if (strcmp(argv[*i], "--optimistic-value") == 0 && *i + 1 < argc) {
        cfg->optimistic_value = (float)atof(argv[++(*i)]);
        return 1;
    }
    if (strcmp(argv[*i], "--multistage") == 0) {
        cfg->multistage = 1;
        return 1;
    }
    if (strcmp(argv[*i], "--carousel") == 0) {
        cfg->carousel = 1;
        return 1;
    }
    if (strcmp(argv[*i], "--carousel-interval") == 0 && *i + 1 < argc) {
        cfg->carousel_buf.use_interval = atoi(argv[++(*i)]);
        return 1;
    }
    if (strcmp(argv[*i], "--weight-promotion") == 0) {
        cfg->use_weight_promotion = 1;
        return 1;
    }
    if (strcmp(argv[*i], "--redundant") == 0) {
        cfg->use_redundant = 1;
        return 1;
    }
    return 0;
}

#endif /* TRAINING_CONFIG_H */
