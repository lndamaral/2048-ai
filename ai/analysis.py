#!/usr/bin/env python3
"""
Análise e visualização dos logs de treinamento e relatórios de jogo do 2048.
Gera gráficos comparativos dos agentes DQN, Expectimax e N-Tuple.
"""

import re
import json
import os
import glob
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Configuração ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHARTS_DIR = os.path.join(BASE_DIR, "charts")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(CHARTS_DIR, exist_ok=True)

DPI = 150
TILE_KEYS = [128, 256, 512, 1024, 2048, 4096, 8192]

# Paleta de cores consistente
COLORS = {
    "ntuple_training": "#2196F3",
    "ntuple_training_v1_12k": "#1565C0",
    "ntuple_training_v2_failed": "#E53935",
    "ntuple_training_overflow": "#FF9800",
    "training": "#9C27B0",          # DQN
    "ntuple_training_5k": "#4CAF50",
    "ntuple_training_v1_continued": "#00BCD4",
    # Agents
    "dqn": "#9C27B0",
    "expectimax": "#FF9800",
    "ntuple": "#2196F3",
}

AGENT_LABELS = {
    "dqn": "DQN",
    "expectimax": "Expectimax",
    "ntuple": "N-Tuple",
}

RUN_LABELS = {
    "ntuple_training": "N-Tuple 500k (1-ply + 3-ply)",
    "ntuple_training_v1_12k": "N-Tuple v1 12k (3-ply)",
    "ntuple_training_v2_failed": "N-Tuple v2 (failed)",
    "ntuple_training_overflow": "N-Tuple overflow",
    "training": "DQN",
    "ntuple_training_5k": "N-Tuple 5k (continued)",
    "ntuple_training_v1_continued": "N-Tuple v1 continued",
}

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 8,
    "figure.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
})


# ── Parser de logs ────────────────────────────────────────────────────────────

def parse_log(filepath):
    """Analisa um arquivo de log de treinamento e retorna lista de checkpoints."""
    if not os.path.exists(filepath):
        return []
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    entries = []
    # Match episode blocks
    pattern = re.compile(
        r"(?:N-Tuple )?Epis[óo]dio\s+([\d]+)/([\d]+)\s*\|.*?\n"
        r"={10,}\n"
        r"(.*?)(?=\n={10,}|\nN-Tuple salvo|\nModelo salvo|\Z)",
        re.DOTALL
    )

    for m in re.finditer(
        r"={10,}\n"
        r"(?:N-Tuple )?Epis[óo]dio\s+([\d]+)/([\d]+)\s*\|.*?Tempo:\s*([\d]+)s.*?\n"
        r"={10,}\n"
        r"(.*?)(?=\n={5,}|\nN-Tuple salvo|\nModelo salvo|\Z)",
        text, re.DOTALL
    ):
        ep = int(m.group(1))
        total = int(m.group(2))
        time_s = int(m.group(3))
        block = m.group(4)

        entry = {"episode": ep, "total": total, "time_s": time_s}

        # Score médio
        sm = re.search(r"Score m[ée]dio:\s+([\d]+)", block)
        if sm:
            entry["score_avg"] = int(sm.group(1))

        # Max tiles dict
        tm = re.search(r"Max tiles:\s+\{([^}]+)\}", block)
        if tm:
            tiles = {}
            for pair in re.findall(r"(\d+):\s*(\d+)", tm.group(1)):
                tiles[int(pair[0])] = int(pair[1])
            entry["max_tiles"] = tiles

        # 2048+ count
        wm = re.search(r"2048[\+]?.*?:\s+(\d+)x", block)
        if wm:
            entry["wins_2048"] = int(wm.group(1))

        # LR
        lrm = re.search(r"LR:\s+([\d.]+)", block)
        if lrm:
            entry["lr"] = float(lrm.group(1))

        # Epsilon (DQN)
        em = re.search(r"Epsilon:\s+([\d.]+)", block)
        if em:
            entry["epsilon"] = float(em.group(1))

        # Loss (DQN)
        lm = re.search(r"Loss m[ée]dia:\s+([\d.]+)", block)
        if lm:
            entry["loss"] = float(lm.group(1))

        entries.append(entry)

    return entries


def detect_stage_split(filepath):
    """Detecta onde muda de 1-ply para 3-ply no log principal."""
    if not os.path.exists(filepath):
        return None
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if "3-ply" in line and i > 10:
            # Return the line number where 3-ply starts
            return i
    return None


# ── Carregar dados ────────────────────────────────────────────────────────────

LOG_FILES = {
    "ntuple_training": os.path.join(BASE_DIR, "ntuple_training.log"),
    "ntuple_training_v1_12k": os.path.join(BASE_DIR, "ntuple_training_v1_12k.log"),
    "ntuple_training_v2_failed": os.path.join(BASE_DIR, "ntuple_training_v2_failed.log"),
    "ntuple_training_overflow": os.path.join(BASE_DIR, "ntuple_training_overflow.log"),
    "training": os.path.join(BASE_DIR, "training.log"),
    "ntuple_training_5k": os.path.join(BASE_DIR, "ntuple_training_5k.log"),
    "ntuple_training_v1_continued": os.path.join(BASE_DIR, "ntuple_training_v1_continued.log"),
}

print("Analyzing training logs...")
all_runs = {}
for name, path in LOG_FILES.items():
    data = parse_log(path)
    if data:
        all_runs[name] = data
        print(f"  {name}: {len(data)} checkpoints")
    else:
        print(f"  {name}: empty or not found")

print(f"\nLoading JSON reports from {REPORTS_DIR}...")
reports = []
for f in sorted(glob.glob(os.path.join(REPORTS_DIR, "*.json"))):
    with open(f, "r") as fp:
        reports.append(json.load(fp))
print(f"  {len(reports)} reports loaded")


# ── Chart 1: Score médio ao longo dos episódios ──────────────────────────────

def chart_score_evolution():
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, data in sorted(all_runs.items()):
        eps = [d["episode"] for d in data if "score_avg" in d]
        scores = [d["score_avg"] for d in data if "score_avg" in d]
        if not eps:
            continue
        color = COLORS.get(name, "#666666")
        label = RUN_LABELS.get(name, name)
        ax.plot(eps, scores, color=color, label=label, linewidth=1.2, alpha=0.85)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Average Score")
    ax.set_title("Average Score Evolution per Episode — All Training Runs")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    path = os.path.join(CHARTS_DIR, "chart_score_evolution.png")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Chart 2: Distribuição de tiles ao longo do treinamento ────────────────────

def chart_tile_distribution():
    # Use the main ntuple_training run (largest)
    run_name = "ntuple_training"
    if run_name not in all_runs:
        run_name = next(iter(all_runs), None)
    if not run_name:
        return
    data = all_runs[run_name]
    data_with_tiles = [d for d in data if "max_tiles" in d]
    if not data_with_tiles:
        return

    # Sample at most 30 epochs for readability
    step = max(1, len(data_with_tiles) // 30)
    sampled = data_with_tiles[::step]

    eps = [d["episode"] for d in sampled]
    tile_counts = {t: [] for t in TILE_KEYS}
    for d in sampled:
        tiles = d["max_tiles"]
        total = sum(tiles.values())
        for t in TILE_KEYS:
            tile_counts[t].append(tiles.get(t, 0) / max(total, 1) * 100)

    fig, ax = plt.subplots(figsize=(14, 6))
    cmap = plt.cm.YlOrRd
    colors_tiles = [cmap(0.15 + 0.85 * i / (len(TILE_KEYS) - 1)) for i in range(len(TILE_KEYS))]

    bottom = np.zeros(len(sampled))
    x = np.arange(len(sampled))
    width = 0.8
    for i, t in enumerate(TILE_KEYS):
        vals = np.array(tile_counts[t])
        ax.bar(x, vals, width, bottom=bottom, label=str(t), color=colors_tiles[i])
        bottom += vals

    ax.set_xticks(x[::max(1, len(x)//15)])
    ax.set_xticklabels([str(eps[i]) for i in range(0, len(eps), max(1, len(x)//15))],
                       rotation=45, ha="right", fontsize=7)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Distribution (%)")
    ax.set_title(f"Max Tile Distribution Throughout Training — {RUN_LABELS.get(run_name, run_name)}")
    ax.legend(title="Tile", loc="upper left", framealpha=0.9)
    fig.tight_layout()
    path = os.path.join(CHARTS_DIR, "chart_tile_distribution.png")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Chart 3: Taxa de 2048+ ───────────────────────────────────────────────────

def chart_2048_rate():
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, data in sorted(all_runs.items()):
        entries = [d for d in data if "wins_2048" in d and "max_tiles" in d]
        if not entries:
            continue
        eps = [d["episode"] for d in entries]
        rates = []
        for d in entries:
            total = sum(d["max_tiles"].values())
            rate = (d["wins_2048"] - (entries[0]["wins_2048"] if entries[0] != d else 0))
            # Compute the rate per batch: tiles >= 2048 / total games in this batch
            tiles = d["max_tiles"]
            wins_in_batch = sum(v for k, v in tiles.items() if k >= 2048)
            rate_pct = wins_in_batch / max(total, 1) * 100
            rates.append(rate_pct)

        color = COLORS.get(name, "#666666")
        label = RUN_LABELS.get(name, name)
        ax.plot(eps, rates, color=color, label=label, linewidth=1.2, alpha=0.85)

    ax.set_xlabel("Episode")
    ax.set_ylabel("2048+ Achievement Rate (%)")
    ax.set_title("2048+ Tile Achievement Rate Throughout Training")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_ylim(0, 105)
    fig.tight_layout()
    path = os.path.join(CHARTS_DIR, "chart_2048_rate.png")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Chart 4: Comparação de agentes ───────────────────────────────────────────

def chart_agent_comparison():
    if not reports:
        return
    agent_data = defaultdict(lambda: {"scores": [], "max_tiles": [], "wins": 0, "count": 0})
    for r in reports:
        a = r["agent"]
        agent_data[a]["scores"].append(r["score"])
        agent_data[a]["max_tiles"].append(r["max_tile"])
        agent_data[a]["wins"] += 1 if r.get("won", False) else 0
        agent_data[a]["count"] += 1

    agents_order = ["dqn", "expectimax", "ntuple"]
    agents_present = [a for a in agents_order if a in agent_data]
    if not agents_present:
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Score médio
    ax = axes[0]
    means = [np.mean(agent_data[a]["scores"]) for a in agents_present]
    bars = ax.bar([AGENT_LABELS[a] for a in agents_present], means,
                  color=[COLORS[a] for a in agents_present], edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Average Score")
    ax.set_title("Average Score")
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                f"{val:.0f}", ha="center", va="bottom", fontsize=9)

    # Max tile médio
    ax = axes[1]
    means_tile = [np.mean(agent_data[a]["max_tiles"]) for a in agents_present]
    bars = ax.bar([AGENT_LABELS[a] for a in agents_present], means_tile,
                  color=[COLORS[a] for a in agents_present], edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Average Max Tile")
    ax.set_title("Average Max Tile")
    for bar, val in zip(bars, means_tile):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                f"{val:.0f}", ha="center", va="bottom", fontsize=9)

    # Win rate (2048+)
    ax = axes[2]
    win_rates = [agent_data[a]["wins"] / agent_data[a]["count"] * 100 for a in agents_present]
    bars = ax.bar([AGENT_LABELS[a] for a in agents_present], win_rates,
                  color=[COLORS[a] for a in agents_present], edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Win Rate (%)")
    ax.set_title("Win Rate (2048+)")
    ax.set_ylim(0, 105)
    for bar, val in zip(bars, win_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Agent Comparison — DQN vs Expectimax vs N-Tuple", fontsize=14, y=1.02)
    fig.tight_layout()
    path = os.path.join(CHARTS_DIR, "chart_agent_comparison.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Chart 5: Velocidade (ms/move) ────────────────────────────────────────────

def chart_speed_comparison():
    if not reports:
        return
    agent_speeds = defaultdict(list)
    for r in reports:
        if r.get("duration_ms") and r.get("moves") and r["moves"] > 0:
            ms_per_move = r["duration_ms"] / r["moves"]
            agent_speeds[r["agent"]].append(ms_per_move)

    agents_order = ["dqn", "expectimax", "ntuple"]
    agents_present = [a for a in agents_order if a in agent_speeds]
    if not agents_present:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    means = [np.mean(agent_speeds[a]) for a in agents_present]
    bars = ax.bar([AGENT_LABELS[a] for a in agents_present], means,
                  color=[COLORS[a] for a in agents_present], edgecolor="white", linewidth=0.5)
    ax.set_ylabel("ms / move")
    ax.set_title("Decision Speed per Agent (ms/move)")
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f}", ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    path = os.path.join(CHARTS_DIR, "chart_speed_comparison.png")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Chart 6: Tempo de treinamento vs performance ─────────────────────────────

def chart_training_time_vs_performance():
    fig, ax = plt.subplots(figsize=(8, 6))

    points = []
    # N-Tuple runs: use last entry's time and score
    for name, data in all_runs.items():
        entries_with_score = [d for d in data if "score_avg" in d]
        if not entries_with_score:
            continue
        last = entries_with_score[-1]
        time_min = last.get("time_s", 0) / 60
        score = last["score_avg"]
        agent_type = "ntuple" if "ntuple" in name else "dqn"
        points.append((time_min, score, name, agent_type))

    for time_min, score, name, agent_type in points:
        color = COLORS.get(name, COLORS.get(agent_type, "#666"))
        label = RUN_LABELS.get(name, name)
        ax.scatter(time_min, score, c=color, s=100, zorder=5, edgecolors="black", linewidth=0.5)
        ax.annotate(label, (time_min, score), textcoords="offset points",
                    xytext=(8, 5), fontsize=7, alpha=0.9)

    ax.set_xlabel("Training Time (minutes)")
    ax.set_ylabel("Final Average Score")
    ax.set_title("Training Time vs Final Performance")
    fig.tight_layout()
    path = os.path.join(CHARTS_DIR, "chart_training_time_vs_performance.png")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Chart 7: Análise dos relatórios JSON ─────────────────────────────────────

def chart_reports_analysis():
    if not reports:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Score distribution histogram
    ax = axes[0]
    for agent in ["dqn", "expectimax", "ntuple"]:
        scores = [r["score"] for r in reports if r["agent"] == agent]
        if scores:
            ax.hist(scores, bins=15, alpha=0.6, label=AGENT_LABELS[agent],
                    color=COLORS[agent], edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Score")
    ax.set_ylabel("Frequency")
    ax.set_title("Score Distribution — Game Reports")
    ax.legend(framealpha=0.9)

    # Max tile distribution
    ax = axes[1]
    tile_counts_by_agent = {}
    all_tiles_set = set()
    for agent in ["dqn", "expectimax", "ntuple"]:
        tiles = [r["max_tile"] for r in reports if r["agent"] == agent]
        if tiles:
            counts = defaultdict(int)
            for t in tiles:
                counts[t] += 1
            tile_counts_by_agent[agent] = counts
            all_tiles_set.update(counts.keys())

    if all_tiles_set:
        all_tiles_sorted = sorted(all_tiles_set)
        x = np.arange(len(all_tiles_sorted))
        width = 0.25
        for i, agent in enumerate(["dqn", "expectimax", "ntuple"]):
            if agent in tile_counts_by_agent:
                vals = [tile_counts_by_agent[agent].get(t, 0) for t in all_tiles_sorted]
                ax.bar(x + i * width, vals, width, label=AGENT_LABELS[agent],
                       color=COLORS[agent], edgecolor="white", linewidth=0.5)
        ax.set_xticks(x + width)
        ax.set_xticklabels([str(t) for t in all_tiles_sorted])
        ax.set_xlabel("Max Tile")
        ax.set_ylabel("Count")
        ax.set_title("Max Tile Distribution — Game Reports")
        ax.legend(framealpha=0.9)

    fig.suptitle("Game Reports Analysis", fontsize=14, y=1.02)
    fig.tight_layout()
    path = os.path.join(CHARTS_DIR, "chart_reports_analysis.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Chart 8: Comparação Stage 1 (1-ply) vs Stage 2 (3-ply) ──────────────────

def parse_log_with_stages(filepath):
    """Parse the main ntuple log splitting into Stage 1 (1-ply) and Stage 2 (3-ply)."""
    if not os.path.exists(filepath):
        return [], []
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    # Split at the 3-ply header
    parts = re.split(r"Treinamento conclu[ií]do!.*?\n", text)
    stage1_text = parts[0] if len(parts) >= 1 else ""
    stage2_text = parts[1] if len(parts) >= 2 else ""

    def extract_entries(txt):
        entries = []
        for m in re.finditer(
            r"={10,}\n"
            r"(?:N-Tuple )?Epis[óo]dio\s+([\d]+)/([\d]+)\s*\|.*?Tempo:\s*([\d]+)s.*?\n"
            r"={10,}\n"
            r"(.*?)(?=\n={5,}|\nN-Tuple salvo|\nModelo salvo|\Z)",
            txt, re.DOTALL
        ):
            ep = int(m.group(1))
            block = m.group(4)
            entry = {"episode": ep}
            sm = re.search(r"Score m[ée]dio:\s+([\d]+)", block)
            if sm:
                entry["score_avg"] = int(sm.group(1))
            tm = re.search(r"Max tiles:\s+\{([^}]+)\}", block)
            if tm:
                tiles = {}
                for pair in re.findall(r"(\d+):\s*(\d+)", tm.group(1)):
                    tiles[int(pair[0])] = int(pair[1])
                entry["max_tiles"] = tiles
            entries.append(entry)
        return entries

    return extract_entries(stage1_text), extract_entries(stage2_text)


def chart_stage_comparison():
    main_log = os.path.join(BASE_DIR, "ntuple_training.log")
    stage1, stage2 = parse_log_with_stages(main_log)

    if not stage1 and not stage2:
        print("  chart_stage_comparison: no data")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Score evolution — use sequential index for x-axis since episodes reset
    ax = axes[0]
    if stage1:
        s1_with_score = [d for d in stage1 if "score_avg" in d]
        ax.plot([d["episode"] for d in s1_with_score],
                [d["score_avg"] for d in s1_with_score],
                color="#2196F3", label="Stage 1 (1-ply, 500k)", linewidth=1.2)
    if stage2:
        s2_with_score = [d for d in stage2 if "score_avg" in d]
        ax.plot([d["episode"] for d in s2_with_score],
                [d["score_avg"] for d in s2_with_score],
                color="#E53935", label="Stage 2 (3-ply, 5M)", linewidth=1.2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Average Score")
    ax.set_title("Average Score — 1-ply vs 3-ply")
    ax.legend(framealpha=0.9)

    # 2048+ rate
    ax = axes[1]
    for stage_data, label, color in [(stage1, "Stage 1 (1-ply, 500k)", "#2196F3"),
                                      (stage2, "Stage 2 (3-ply, 5M)", "#E53935")]:
        entries = [d for d in stage_data if "max_tiles" in d]
        if not entries:
            continue
        eps = [d["episode"] for d in entries]
        rates = []
        for d in entries:
            total = sum(d["max_tiles"].values())
            wins = sum(v for k, v in d["max_tiles"].items() if k >= 2048)
            rates.append(wins / max(total, 1) * 100)
        ax.plot(eps, rates, color=color, label=label, linewidth=1.2)

    ax.set_xlabel("Episode")
    ax.set_ylabel("2048+ Achievement Rate (%)")
    ax.set_title("2048+ Achievement Rate — 1-ply vs 3-ply")
    ax.legend(framealpha=0.9)

    fig.suptitle("Comparison: Stage 1 (1-ply) vs Stage 2 (3-ply)", fontsize=14, y=1.02)
    fig.tight_layout()
    path = os.path.join(CHARTS_DIR, "chart_stage_comparison.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Executar todos os gráficos ────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Generating charts...")
    print("=" * 60 + "\n")

    chart_score_evolution()
    chart_tile_distribution()
    chart_2048_rate()
    chart_agent_comparison()
    chart_speed_comparison()
    chart_training_time_vs_performance()
    chart_reports_analysis()
    chart_stage_comparison()

    print(f"\nAll charts saved to: {CHARTS_DIR}/")
    print("Done!")
