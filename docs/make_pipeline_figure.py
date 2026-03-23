#!/usr/bin/env python3
"""Generate the SE-CCM pipeline figure for README / paper.

Usage:
    python docs/make_pipeline_figure.py            # -> docs/pipeline.pdf + .png
    python docs/make_pipeline_figure.py --dark      # dark-mode variant

Edit colours, labels, or layout directly in this script and re-run.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# =====================================================================
# Configurable palette — edit here to restyle
# =====================================================================
LIGHT = dict(
    bg="#FFFFFF",
    # Stage boxes
    stage1="#4C72B0",   # steel blue    – Data Generation
    stage2="#55A868",   # sage green    – Embedding & CCM
    stage3="#C44E52",   # muted red     – Surrogate Testing
    stage4="#8172B2",   # muted purple  – Statistical Decision
    stage5="#CCB974",   # sand          – Evaluation
    # Annotation / detail boxes
    detail="#F5F5F5",
    detail_edge="#C0C0C0",
    # Input/output boxes
    io_input="#DAE3F0",
    io_input_edge="#8FAAC8",
    io_output="#F2EDDA",
    io_output_edge="#C8BC8A",
    # Text
    text_stage="#FFFFFF",
    text_detail="#333333",
    text_arrow="#444444",
    # Arrows
    arrow="#555555",
    arrow_highlight="#C44E52",
    # Title
    title="#222222",
)

DARK = dict(
    bg="#1E1E1E",
    stage1="#5B8BD0",
    stage2="#6DC47E",
    stage3="#E06060",
    stage4="#9B8BD8",
    stage5="#D4C87A",
    detail="#2A2A2A",
    detail_edge="#555555",
    io_input="#2A3548",
    io_input_edge="#5B8BD0",
    io_output="#3A3828",
    io_output_edge="#D4C87A",
    text_stage="#FFFFFF",
    text_detail="#CCCCCC",
    text_arrow="#AAAAAA",
    arrow="#888888",
    arrow_highlight="#E06060",
    title="#EEEEEE",
)


def draw_pipeline(palette, save_prefix="docs/pipeline"):
    """Draw the full SE-CCM pipeline figure."""
    C = palette

    fig, ax = plt.subplots(figsize=(15, 7.2))
    fig.patch.set_facecolor(C["bg"])
    ax.set_facecolor(C["bg"])
    ax.set_xlim(-0.3, 14.8)
    ax.set_ylim(-0.8, 7.2)
    ax.axis("off")

    # ── Helper: rounded box ──────────────────────────────────────
    def rounded_box(x, y, w, h, fc, ec, lw=1.2, alpha=1.0, zorder=2):
        box = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.15",
            facecolor=fc, edgecolor=ec, linewidth=lw,
            alpha=alpha, zorder=zorder,
        )
        ax.add_patch(box)

    # ── Helper: centered text ────────────────────────────────────
    def ctext(x, y, txt, **kw):
        defaults = dict(ha="center", va="center", zorder=5)
        defaults.update(kw)
        ax.text(x, y, txt, **defaults)

    # ── Helper: arrow ────────────────────────────────────────────
    def arrow(x1, y1, x2, y2, label=None, highlight=False,
              connectionstyle="arc3,rad=0.0", label_above=True):
        color = C["arrow_highlight"] if highlight else C["arrow"]
        style = "Simple,tail_width=1.0,head_width=6,head_length=4.5"
        arr = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle=style, connectionstyle=connectionstyle,
            color=color, linewidth=1.2, zorder=6,
        )
        ax.add_patch(arr)
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            dy = 0.2 if label_above else -0.2
            va = "bottom" if label_above else "top"
            ax.text(mx, my + dy, label, ha="center", va=va,
                    fontsize=7.5, color=C["text_arrow"],
                    fontstyle="italic", zorder=7)

    # ── Layout constants ─────────────────────────────────────────
    sw, sh = 2.3, 0.80              # stage box size
    row_stage = 5.2                 # y of stage boxes
    row_detail = 2.2                # y-bottom of detail boxes
    dw = 2.3                        # detail box width
    dh = [1.35, 1.35, 1.65, 1.35, 1.35]  # per-stage detail height
    xs = [0.3, 3.1, 5.9, 8.7, 11.5]      # x of each stage

    # ── Title ────────────────────────────────────────────────────
    ctext(7.4, 6.9,
          "SE-CCM: Surrogate-Enhanced Convergent Cross Mapping — Pipeline",
          fontsize=13.5, fontweight="bold", color=C["title"])

    # ── Stage boxes ──────────────────────────────────────────────
    stage_labels = [
        ("1. Data\nGeneration", "stage1"),
        ("2. Embedding\n& CCM", "stage2"),
        ("3. Surrogate\nTesting", "stage3"),
        ("4. Statistical\nDecision", "stage4"),
        ("5. Evaluation", "stage5"),
    ]
    for i, (label, ckey) in enumerate(stage_labels):
        rounded_box(xs[i], row_stage, sw, sh, C[ckey], "white", lw=1.5, zorder=3)
        ctext(xs[i] + sw / 2, row_stage + sh / 2, label,
              fontsize=10, fontweight="bold", color=C["text_stage"])

    # ── Arrows between stages ────────────────────────────────────
    for i in range(4):
        arrow(xs[i] + sw + 0.05, row_stage + sh / 2,
              xs[i + 1] - 0.05, row_stage + sh / 2)

    # ── Surrogate feedback loop (3 -> 2) ─────────────────────────
    loop_y = row_stage + sh + 0.45
    ax.annotate(
        "", xy=(xs[1] + sw - 0.2, row_stage + sh + 0.08),
        xytext=(xs[2] + 0.2, row_stage + sh + 0.08),
        arrowprops=dict(
            arrowstyle="-|>", color=C["arrow_highlight"],
            lw=1.5, connectionstyle="arc3,rad=-0.35",
        ),
        zorder=6,
    )
    ctext((xs[1] + sw + xs[2]) / 2, loop_y,
          "repeat \u00d7 n_surrogates",
          fontsize=7.5, color=C["arrow_highlight"],
          fontstyle="italic", fontweight="medium")

    # ── Detail boxes ─────────────────────────────────────────────
    detail_contents = [
        [  # Stage 1
            "Topology: ER / WS / Ring",
            "8 dynamical systems",
            "Coupling \u03b5, noise \u03c3_obs, \u03c3_dyn",
            "\u2192 data (T, N) + adj (N, N)",
        ],
        [  # Stage 2
            "\u03c4 : mutual-info first minimum",
            "E : simplex prediction max",
            "Delay embed  x \u2192 M\u2093",
            "\u03c1 = CCM cross-predict y",
        ],
        [  # Stage 3
            "Generate surrogates of y:",
            "  iAAFT       | spectrum + amp.",
            "  Cycle-phase  | oscillatory",
            "  Twin         | recurrence",
            "  + 9 more (12 uni + 2 multi)",
        ],
        [  # Stage 4
            "p = rank(\u03c1_obs vs {\u03c1_surr})",
            "z = (\u03c1_obs \u2212 \u03bc) / \u03c3",
            "BH-FDR network correction",
            "Effect gate: \u03c1 \u2265 0.3",
        ],
        [  # Stage 5
            "AUROC  (raw \u03c1)",
            "AUROC  (z-score)",
            "AUPRC  (z-score)",
            "\u0394AUROC = surr. \u2212 raw",
        ],
    ]

    for i, lines in enumerate(detail_contents):
        bx, by, bw, bh = xs[i], row_detail, dw, dh[i]
        rounded_box(bx, by, bw, bh, C["detail"], C["detail_edge"], lw=0.8)

        line_spacing = bh / (len(lines) + 0.8)
        for j, txt in enumerate(lines):
            ax.text(
                bx + 0.18, by + bh - (j + 0.7) * line_spacing,
                txt, ha="left", va="center",
                fontsize=7.8, color=C["text_detail"],
                fontfamily="monospace", zorder=3,
            )

    # ── Vertical arrows: stage -> detail ─────────────────────────
    for i in range(5):
        arrow(xs[i] + sw / 2, row_stage - 0.05,
              xs[i] + sw / 2, row_detail + dh[i] + 0.08)

    # ── Input box (bottom-left) ──────────────────────────────────
    io_w, io_h = 4.2, 0.65
    inp_x, inp_y = 0.3, 0.0
    rounded_box(inp_x, inp_y, io_w, io_h,
                C["io_input"], C["io_input_edge"], lw=1.0)
    ctext(inp_x + io_w / 2, inp_y + io_h / 2 + 0.12,
          "Input", fontsize=9, fontweight="bold", color=C["text_detail"])
    ctext(inp_x + io_w / 2, inp_y + io_h / 2 - 0.12,
          "Multivariate time series (T, N)  +  adjacency (N, N)",
          fontsize=7.5, color=C["text_detail"])

    # ── Output box (bottom-right) ────────────────────────────────
    out_x = 10.3
    rounded_box(out_x, inp_y, io_w, io_h,
                C["io_output"], C["io_output_edge"], lw=1.0)
    ctext(out_x + io_w / 2, inp_y + io_h / 2 + 0.12,
          "Output", fontsize=9, fontweight="bold", color=C["text_detail"])
    ctext(out_x + io_w / 2, inp_y + io_h / 2 - 0.12,
          "Detected edges, p-values, z-scores, AUROC, AUPRC, \u0394AUROC",
          fontsize=7.5, color=C["text_detail"])

    # ── Arrows: input -> stage 1 detail, stage 5 detail -> output
    arrow(inp_x + io_w / 2, inp_y + io_h + 0.05,
          xs[0] + sw / 2, row_detail - 0.08)
    arrow(xs[4] + sw / 2, row_detail - 0.08,
          out_x + io_w / 2, inp_y + io_h + 0.05)

    # ── Save ─────────────────────────────────────────────────────
    fig.tight_layout(pad=0.2)
    for ext in ("pdf", "png"):
        path = f"{save_prefix}.{ext}"
        fig.savefig(path, dpi=300, facecolor=C["bg"],
                    bbox_inches="tight", pad_inches=0.12)
        print(f"  Saved: {path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate SE-CCM pipeline figure")
    parser.add_argument("--dark", action="store_true", help="Dark-mode variant")
    parser.add_argument("--output", default="docs/pipeline",
                        help="Output path prefix (without extension)")
    args = parser.parse_args()

    palette = DARK if args.dark else LIGHT
    save_prefix = args.output
    if args.dark and args.output == "docs/pipeline":
        save_prefix = "docs/pipeline_dark"

    os.makedirs(os.path.dirname(save_prefix) or ".", exist_ok=True)
    draw_pipeline(palette, save_prefix)
    print("Done.")


if __name__ == "__main__":
    main()
