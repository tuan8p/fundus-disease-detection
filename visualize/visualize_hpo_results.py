
import argparse
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.gridspec as gridspec

matplotlib.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.titleweight": "bold",
    "axes.labelsize":   11,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "figure.dpi":       150,
    "savefig.dpi":      150,
    "savefig.bbox":     "tight",
})

# ── Dữ liệu HPO ───────────────────────────────────────────────────────────────

RESULTS = [
    # shard, label, val_qwk, lr, backbone_lr_scale, weight_decay, t0, warmup
    (0, "sw0_lr5e5_t6",            0.92549, 5e-5, 0.10, 1e-4,  6, 2),
    (0, "sw0_lr5e5_t10",           0.92676, 5e-5, 0.05, 1e-4, 10, 2),
    (0, "sw0_lr5e5_wd2e4_t8",      0.92985, 5e-5, 0.10, 2e-4,  8, 2),
    (1, "sw1_lr3e5_t8",            0.91004, 3e-5, 0.10, 2e-4,  8, 2),
    (1, "sw1_lr8e5_t6_wd2e4",      0.92024, 8e-5, 0.08, 2e-4,  6, 2),
    (1, "sw1_lr6e5_t8",            0.91059, 6e-5, 0.08, 2e-4,  8, 2),
    (2, "sw2_wu3_lr5e5_wd3e4",     0.92526, 5e-5, 0.10, 3e-4,  8, 3),
    (2, "sw2_wu3_lr4e5",           0.91972, 4e-5, 0.05, 1e-4,  9, 3),
    (2, "sw2_wu3_wd5e4",           0.92461, 3e-5, 0.10, 5e-4,  9, 3),
    (3, "sw3_lr1e5_blr001",        0.88017, 1e-5, 0.01, 1e-4,  8, 3),
    (3, "sw3_lr1e5_wd5e4",         0.92028, 1e-5, 0.10, 5e-4,  8, 2),
    (3, "sw3_lr5e6_blr002",        0.88184, 5e-6, 0.02, 1e-4,  9, 3),
    (4, "sw4_lr2e5_blr005",        0.90667, 2e-5, 0.05, 1e-4,  9, 3),
    (4, "sw4_lr2e5_blr020",        0.91317, 2e-5, 0.20, 5e-5, 10, 2),
    (4, "sw4_lr5e6_wd5e5",         0.90141, 5e-6, 0.10, 5e-5,  9, 3),
    (5, "sw5_lr1e4_t8",            0.93623, 1e-4, 0.10, 1e-4,  8, 2),
    (5, "sw5_lr1e4_blr015",        0.94353, 1e-4, 0.15, 5e-5,  8, 2),
    (5, "sw5_lr1e4_wd3e4",         0.93757, 1e-4, 0.08, 3e-4,  8, 2),
    (6, "sw6_lr1e4_wd5e4",         0.93251, 1e-4, 0.10, 5e-4,  8, 2),
    (6, "sw6_lr7e5_t10",           0.93176, 7e-5, 0.10, 1e-4, 10, 2),
    (6, "sw6_lr1e4_blr001_wd1e4",  0.92192, 1e-4, 0.01, 1e-4,  8, 2),
    (7, "sw7_b1_091_lr5e5",        0.93081, 5e-5, 0.10, 1e-4,  8, 2),
    (7, "sw7_b2_9995_lr5e5",       0.92872, 5e-5, 0.10, 1e-4,  8, 2),
    (7, "sw7_b1_092_b2_9999",      0.92946, 6e-5, 0.10, 1e-4,  9, 2),
    (8, "sw8_blr002_lr5e5",        0.92153, 5e-5, 0.02, 1e-4,  8, 2),
    (8, "sw8_blr015_lr5e5",        0.92699, 5e-5, 0.15, 1e-4,  8, 2),
    (8, "sw8_blr005_lr8e5",        0.91844, 8e-5, 0.05, 1e-4,  7, 2),
]

SHARD_LABELS = {
    0: "S0: LR=5e-5",
    1: "S1: LR=3-8e-5",
    2: "S2: warmup=3",
    3: "S3: LR≤1e-5",
    4: "S4: LR=2e-5/5e-6",
    5: "S5: LR=1e-4 ★",
    6: "S6: LR=1e-4/7e-5",
    7: "S7: β variations",
    8: "S8: blr boundary",
}

SHARD_COLORS = [
    "#4A7CC3", "#5BA08A", "#D4874E", "#C05050",
    "#9B6BB5", "#2E7D32", "#00838F", "#795548", "#607D8B",
]


def _save(fig, path):
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Figure 1: Overview bar chart ──────────────────────────────────────────────

def fig_overview_bar(output_dir: str):
    shards  = [r[0] for r in RESULTS]
    labels  = [r[1] for r in RESULTS]
    qwks    = [r[2] for r in RESULTS]
    colors  = [SHARD_COLORS[s] for s in shards]

    sorted_idx = np.argsort(qwks)[::-1]
    s_labels   = [labels[i] for i in sorted_idx]
    s_qwks     = [qwks[i]   for i in sorted_idx]
    s_colors   = [colors[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(16, 6))
    x = np.arange(len(s_labels))
    bars = ax.bar(x, s_qwks, color=s_colors, width=0.7, zorder=2)

    # Highlight best
    bars[0].set_edgecolor("#000")
    bars[0].set_linewidth(1.5)
    ax.annotate(
        f"Best\n{s_qwks[0]:.4f}",
        xy=(0, s_qwks[0]),
        xytext=(1.5, s_qwks[0] + 0.003),
        arrowprops=dict(arrowstyle="->", color="#333", lw=1),
        fontsize=9, color="#333",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(s_labels, rotation=55, ha="right", fontsize=8)
    ax.set_ylabel("Validation QWK")
    ax.set_title("HPO Results — SwinV2-Base-384 (27 cases, sorted by QWK)")
    ax.set_ylim(0.86, 0.960)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.01))
    ax.grid(axis="y", alpha=0.3, zorder=1)

    # Legend
    patches = [mpatches.Patch(color=SHARD_COLORS[s], label=SHARD_LABELS[s])
               for s in range(9)]
    ax.legend(handles=patches, fontsize=8, ncol=3,
              loc="lower right", framealpha=0.85)

    ax.axhline(y=np.mean(s_qwks), color="gray", linestyle="--",
               alpha=0.6, linewidth=1,
               label=f"Mean = {np.mean(s_qwks):.4f}")

    _save(fig, os.path.join(output_dir, "hpo_overview_bar.png"))


# ── Figure 2: Boxplot per shard ───────────────────────────────────────────────

def fig_shard_boxplot(output_dir: str):
    from collections import defaultdict
    shard_qwks = defaultdict(list)
    for r in RESULTS:
        shard_qwks[r[0]].append(r[2])

    fig, ax = plt.subplots(figsize=(11, 5))

    data_by_shard = [shard_qwks[s] for s in range(9)]
    bp = ax.boxplot(
        data_by_shard,
        patch_artist=True,
        widths=0.5,
        medianprops=dict(color="white", linewidth=2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        flierprops=dict(marker="o", markersize=5, alpha=0.6),
    )
    for patch, color in zip(bp["boxes"], SHARD_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    for flier, color in zip(bp["fliers"], SHARD_COLORS):
        flier.set_markerfacecolor(color)

    # Overlay scatter (jitter)
    for i, (vals, color) in enumerate(zip(data_by_shard, SHARD_COLORS), start=1):
        jitter = np.random.default_rng(i).uniform(-0.12, 0.12, len(vals))
        ax.scatter([i + j for j in jitter], vals, color=color,
                   edgecolors="white", s=45, zorder=3, linewidth=0.8)

    ax.set_xticks(range(1, 10))
    ax.set_xticklabels([SHARD_LABELS[s] for s in range(9)],
                       rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Validation QWK")
    ax.set_title("QWK Distribution by Shard")
    ax.set_ylim(0.86, 0.960)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.01))
    ax.grid(axis="y", alpha=0.3)

    _save(fig, os.path.join(output_dir, "hpo_shard_boxplot.png"))


# ── Figure 3: Hyperparameter heatmap (top 10) ────────────────────────────────

def fig_top10_heatmap(output_dir: str):
    top10 = sorted(RESULTS, key=lambda r: r[2], reverse=True)[:10]

    case_names = [r[1] for r in top10]
    qwks       = [r[2] for r in top10]
    lrs        = [r[3] for r in top10]
    blrs       = [r[4] for r in top10]
    wds        = [r[5] for r in top10]
    t0s        = [r[6] for r in top10]
    wus        = [r[7] for r in top10]

    # Normalize mỗi HP vào [0, 1] để heatmap
    def norm(arr):
        a = np.array(arr, dtype=float)
        mn, mx = a.min(), a.max()
        return (a - mn) / (mx - mn + 1e-12)

    matrix = np.vstack([
        norm(np.log10(lrs)),
        norm(blrs),
        norm(np.log10(wds)),
        norm(t0s),
        norm(wus),
        norm(qwks),
    ])

    row_labels = ["LR (log)", "Backbone LR scale", "Weight decay (log)",
                  "T_0", "Warmup epochs", "val QWK ★"]

    fig, ax = plt.subplots(figsize=(13, 4.5))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)

    ax.set_xticks(range(10))
    ax.set_xticklabels(case_names, rotation=40, ha="right", fontsize=9)
    ax.set_yticks(range(6))
    ax.set_yticklabels(row_labels, fontsize=10)
    ax.set_title("Top 10 Cases — Hyperparameter Heatmap (normalized 0→1)")

    # Annotate raw values
    raw_display = [
        [f"{v:.0e}" for v in lrs],
        [f"{v:.2f}" for v in blrs],
        [f"{v:.0e}" for v in wds],
        [f"{v:.0f}" for v in t0s],
        [f"{v:.0f}" for v in wus],
        [f"{v:.4f}" for v in qwks],
    ]
    for row_i in range(6):
        for col_i in range(10):
            val = matrix[row_i, col_i]
            text_color = "black" if 0.3 < val < 0.75 else "white" if val < 0.3 else "black"
            ax.text(col_i, row_i, raw_display[row_i][col_i],
                    ha="center", va="center",
                    fontsize=7.5, color=text_color)

    plt.colorbar(im, ax=ax, label="Normalized value", shrink=0.8, pad=0.01)
    plt.tight_layout()
    _save(fig, os.path.join(output_dir, "hpo_top10_heatmap.png"))


# ── Figure 4: LR vs QWK scatter (bubble = blr) ───────────────────────────────

def fig_lr_analysis(output_dir: str):
    lrs   = np.array([r[3] for r in RESULTS])
    qwks  = np.array([r[2] for r in RESULTS])
    blrs  = np.array([r[4] for r in RESULTS])
    shards = [r[0] for r in RESULTS]
    colors = [SHARD_COLORS[s] for s in shards]

    bubble_size = (blrs / blrs.max()) * 600 + 40

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Left: LR (log) vs QWK ─────────────────────────────────────────────────
    ax = axes[0]
    sc = ax.scatter(np.log10(lrs), qwks, s=bubble_size,
                    c=colors, alpha=0.8, edgecolors="white", linewidth=0.8, zorder=3)
    ax.set_xlabel("Learning Rate (log₁₀ scale)")
    ax.set_ylabel("Validation QWK")
    ax.set_title("LR vs val QWK\n(bubble size ∝ BACKBONE_LR_SCALE)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"1e{int(v)}" if v == int(v) else f"10^{v:.1f}"
    ))
    ax.grid(alpha=0.3)

    # Annotate best
    best_idx = int(np.argmax(qwks))
    ax.annotate(
        RESULTS[best_idx][1],
        xy=(np.log10(lrs[best_idx]), qwks[best_idx]),
        xytext=(np.log10(lrs[best_idx]) - 0.4, qwks[best_idx] - 0.006),
        arrowprops=dict(arrowstyle="->", color="#333", lw=1),
        fontsize=8,
    )

    # ── Right: BACKBONE_LR_SCALE vs QWK ───────────────────────────────────────
    ax = axes[1]
    norm_lr = Normalize(vmin=np.log10(lrs.min()), vmax=np.log10(lrs.max()))
    cmap = plt.cm.plasma
    sc2 = ax.scatter(blrs, qwks, s=100,
                     c=np.log10(lrs), cmap=cmap, norm=norm_lr,
                     edgecolors="white", linewidth=0.8, zorder=3, alpha=0.85)
    plt.colorbar(ScalarMappable(norm=norm_lr, cmap=cmap),
                 ax=ax, label="LR (log₁₀)", shrink=0.85)
    ax.set_xlabel("BACKBONE_LR_SCALE")
    ax.set_ylabel("Validation QWK")
    ax.set_title("Backbone LR Scale vs val QWK\n(color = learning rate)")
    ax.grid(alpha=0.3)
    ax.annotate(
        RESULTS[best_idx][1],
        xy=(blrs[best_idx], qwks[best_idx]),
        xytext=(blrs[best_idx] + 0.01, qwks[best_idx] - 0.006),
        arrowprops=dict(arrowstyle="->", color="#333", lw=1),
        fontsize=8,
    )

    # Shared legend (shard colors)
    patches = [mpatches.Patch(color=SHARD_COLORS[s], label=f"Shard {s}")
               for s in range(9)]
    fig.legend(handles=patches, fontsize=8, ncol=3,
               loc="lower center", bbox_to_anchor=(0.5, -0.08), framealpha=0.85)
    plt.tight_layout()
    _save(fig, os.path.join(output_dir, "hpo_lr_analysis.png"))


# ── Figure 5: Summary table (publication-ready) ───────────────────────────────

def fig_summary_table(output_dir: str):
    top10 = sorted(RESULTS, key=lambda r: r[2], reverse=True)[:10]

    col_labels = ["Rank", "Case", "Shard", "val QWK", "LR", "Backbone LR", "WD", "T₀", "Warmup"]
    rows = []
    for rank, r in enumerate(top10, 1):
        rows.append([
            str(rank),
            r[1],
            str(r[0]),
            f"{r[2]:.4f}",
            f"{r[3]:.0e}",
            f"{r[4]:.2f}",
            f"{r[5]:.0e}",
            str(r[6]),
            str(r[7]),
        ])

    n_rows = len(rows)
    n_cols = len(col_labels)
    col_widths = [0.06, 0.26, 0.07, 0.09, 0.08, 0.10, 0.08, 0.07, 0.09]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis("off")

    # Draw header
    header_y = 0.93
    x_pos = 0.0
    for w, label in zip(col_widths, col_labels):
        ax.text(x_pos + w / 2, header_y, label,
                transform=ax.transAxes,
                ha="center", va="center",
                fontsize=9.5, fontweight="bold", color="white",
                bbox=dict(boxstyle="square,pad=0.3",
                          facecolor="#2E4057", edgecolor="none"))
        x_pos += w

    # Draw rows
    row_height = 0.079
    for row_i, row in enumerate(rows):
        y = header_y - (row_i + 1) * row_height - 0.005
        bg = "#E8F5E9" if row_i == 0 else ("#F5F5F5" if row_i % 2 == 0 else "white")
        ax.add_patch(plt.Rectangle(
            (0, y - 0.025), 1, row_height,
            transform=ax.transAxes,
            facecolor=bg, edgecolor="#DDDDDD", linewidth=0.5,
        ))
        x_pos = 0.0
        for col_i, (w, cell) in enumerate(zip(col_widths, row)):
            color = "#1B5E20" if row_i == 0 else "#212121"
            weight = "bold" if row_i == 0 else "normal"
            ax.text(x_pos + w / 2, y + 0.01, cell,
                    transform=ax.transAxes,
                    ha="center", va="center",
                    fontsize=8.8, color=color, fontweight=weight)
            x_pos += w

    ax.set_title("Top 10 Configurations — SwinV2-Base-384 HPO",
                 fontsize=13, fontweight="bold", pad=14, loc="left")
    ax.text(0.99, 1.01, "Model: swinv2_base_384 | freeze=none | aug=v1 | epochs=15",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, color="#666666")

    _save(fig, os.path.join(output_dir, "hpo_summary_table.png"))


# ── Figure 6: All-in-one dashboard (1 figure cho báo cáo) ────────────────────

def fig_dashboard(output_dir: str):
    from collections import defaultdict

    fig = plt.figure(figsize=(18, 11))
    fig.suptitle(
        "SwinV2-Base-384 — HPO Results Dashboard\n"
        "27 cases · 9 shards · FREEZE=none · AUG=v1 · EPOCHS=15",
        fontsize=14, fontweight="bold", y=0.98,
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

    shards_list  = [r[0] for r in RESULTS]
    labels_list  = [r[1] for r in RESULTS]
    qwks_list    = [r[2] for r in RESULTS]
    lrs_list     = np.array([r[3] for r in RESULTS])
    blrs_list    = np.array([r[4] for r in RESULTS])
    wds_list     = np.array([r[5] for r in RESULTS])
    colors_list  = [SHARD_COLORS[s] for s in shards_list]

    # ── (A) Overview bar (top row, span 2) ────────────────────────────────────
    ax_bar = fig.add_subplot(gs[0, :2])
    sorted_idx = np.argsort(qwks_list)[::-1]
    s_qwks  = [qwks_list[i]   for i in sorted_idx]
    s_clrs  = [colors_list[i] for i in sorted_idx]
    s_lbls  = [labels_list[i] for i in sorted_idx]
    x = np.arange(len(s_lbls))
    bars = ax_bar.bar(x, s_qwks, color=s_clrs, width=0.7, zorder=2)
    bars[0].set_edgecolor("black"); bars[0].set_linewidth(1.5)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(s_lbls, rotation=55, ha="right", fontsize=7)
    ax_bar.set_ylabel("val QWK")
    ax_bar.set_title("(A) All 27 cases ranked by val QWK")
    ax_bar.set_ylim(0.86, 0.960)
    ax_bar.yaxis.set_major_locator(mticker.MultipleLocator(0.01))
    ax_bar.grid(axis="y", alpha=0.25, zorder=1)
    ax_bar.axhline(np.mean(qwks_list), color="gray", ls="--", lw=1, alpha=0.7,
                   label=f"mean={np.mean(qwks_list):.4f}")
    ax_bar.legend(fontsize=8)

    # ── (B) Boxplot per shard ─────────────────────────────────────────────────
    ax_box = fig.add_subplot(gs[0, 2])
    shard_qwks = defaultdict(list)
    for r in RESULTS:
        shard_qwks[r[0]].append(r[2])
    bp = ax_box.boxplot(
        [shard_qwks[s] for s in range(9)],
        patch_artist=True, widths=0.5, vert=False,
        medianprops=dict(color="white", lw=2),
    )
    for patch, color in zip(bp["boxes"], SHARD_COLORS):
        patch.set_facecolor(color); patch.set_alpha(0.85)
    ax_box.set_yticks(range(1, 10))
    ax_box.set_yticklabels([f"S{s}" for s in range(9)], fontsize=9)
    ax_box.set_xlabel("val QWK")
    ax_box.set_title("(B) QWK by shard")
    ax_box.set_xlim(0.86, 0.960)
    ax_box.grid(axis="x", alpha=0.25)

    # ── (C) LR vs QWK scatter ─────────────────────────────────────────────────
    ax_sc1 = fig.add_subplot(gs[1, 0])
    ax_sc1.scatter(np.log10(lrs_list), qwks_list,
                   s=(blrs_list / blrs_list.max()) * 400 + 30,
                   c=colors_list, alpha=0.85,
                   edgecolors="white", lw=0.7, zorder=3)
    best_i = int(np.argmax(qwks_list))
    ax_sc1.annotate("★ Best",
                    xy=(np.log10(lrs_list[best_i]), qwks_list[best_i]),
                    xytext=(-4.6, 0.937),
                    arrowprops=dict(arrowstyle="->", lw=0.9),
                    fontsize=7.5)
    ax_sc1.set_xlabel("LR (log₁₀)")
    ax_sc1.set_ylabel("val QWK")
    ax_sc1.set_title("(C) LR vs val QWK\n(size ∝ backbone LR scale)")
    ax_sc1.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"1e{int(v)}"))
    ax_sc1.grid(alpha=0.25)

    # ── (D) Backbone LR scale vs QWK ─────────────────────────────────────────
    ax_sc2 = fig.add_subplot(gs[1, 1])
    norm_lr = Normalize(vmin=np.log10(lrs_list.min()), vmax=np.log10(lrs_list.max()))
    sc2 = ax_sc2.scatter(blrs_list, qwks_list, s=80,
                         c=np.log10(lrs_list), cmap="plasma", norm=norm_lr,
                         edgecolors="white", lw=0.7, alpha=0.85, zorder=3)
    plt.colorbar(ScalarMappable(norm=norm_lr, cmap="plasma"),
                 ax=ax_sc2, label="LR (log₁₀)", shrink=0.85, pad=0.01)
    ax_sc2.set_xlabel("Backbone LR scale")
    ax_sc2.set_ylabel("val QWK")
    ax_sc2.set_title("(D) Backbone LR scale vs val QWK\n(color = LR)")
    ax_sc2.grid(alpha=0.25)

    # ── (E) Best 5 vs worst 5 summary ────────────────────────────────────────
    ax_cmp = fig.add_subplot(gs[1, 2])
    top5   = sorted(RESULTS, key=lambda r: r[2], reverse=True)[:5]
    bot5   = sorted(RESULTS, key=lambda r: r[2])[:5]
    grp_lbls = [r[1].replace("sw", "").replace("_", "\n") for r in top5] + \
               [r[1].replace("sw", "").replace("_", "\n") for r in bot5]
    grp_qwks = [r[2] for r in top5] + [r[2] for r in bot5]
    grp_clrs = ["#2E7D32"] * 5 + ["#C62828"] * 5
    ypos = np.arange(10)
    ax_cmp.barh(ypos, grp_qwks, color=grp_clrs, height=0.65, zorder=2)
    ax_cmp.set_yticks(ypos)
    ax_cmp.set_yticklabels(grp_lbls, fontsize=7)
    ax_cmp.set_xlabel("val QWK")
    ax_cmp.set_title("(E) Top 5 vs Bottom 5")
    ax_cmp.set_xlim(0.86, 0.960)
    ax_cmp.axvline(np.mean(qwks_list), color="gray", ls="--", lw=1, alpha=0.7)
    ax_cmp.axhline(4.5, color="#999", lw=0.8, ls=":")
    ax_cmp.grid(axis="x", alpha=0.25, zorder=1)
    top_patch = mpatches.Patch(color="#2E7D32", label="Top 5")
    bot_patch = mpatches.Patch(color="#C62828", label="Bottom 5")
    ax_cmp.legend(handles=[top_patch, bot_patch], fontsize=8, loc="lower right")

    # Shard color legend (bottom)
    patches = [mpatches.Patch(color=SHARD_COLORS[s], label=SHARD_LABELS[s])
               for s in range(9)]
    fig.legend(handles=patches, fontsize=8, ncol=5,
               loc="lower center", bbox_to_anchor=(0.5, -0.04),
               framealpha=0.9, title="Shard legend", title_fontsize=8)

    _save(fig, os.path.join(output_dir, "hpo_dashboard.png"))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="./hpo_figures",
                        help="Thư mục lưu figure (mặc định: ./hpo_figures)")
    parser.add_argument("--dpi", type=int, default=150,
                        help="DPI khi lưu file (mặc định: 150, báo cáo: 300)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    matplotlib.rcParams["figure.dpi"]  = args.dpi
    matplotlib.rcParams["savefig.dpi"] = args.dpi

    print(f"Saving figures to: {os.path.abspath(args.output_dir)}")
    print(f"DPI: {args.dpi}\n")

    fig_overview_bar(args.output_dir)
    fig_shard_boxplot(args.output_dir)
    fig_top10_heatmap(args.output_dir)
    fig_lr_analysis(args.output_dir)
    fig_summary_table(args.output_dir)
    fig_dashboard(args.output_dir)

    print("\nDone! All 6 figures saved.")
    print("  hpo_overview_bar.png     — Bar chart 27 cases xếp hạng")
    print("  hpo_shard_boxplot.png    — Boxplot QWK theo shard")
    print("  hpo_top10_heatmap.png    — Heatmap hyperparameter top 10")
    print("  hpo_lr_analysis.png      — Scatter LR vs QWK")
    print("  hpo_summary_table.png    — Bảng tóm tắt top 10")
    print("  hpo_dashboard.png        — Dashboard tổng hợp (dùng cho báo cáo)")


if __name__ == "__main__":
    main()
