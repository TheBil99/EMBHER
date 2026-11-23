"""Draw docs/schematic.png (numpy + matplotlib only): python docs/make_schematic.py"""
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

NAVY, RED, GOLD, BG, INK, MUTE = "#0B2A52", "#C8434B", "#F5B800", "#F0F2F7", "#222B3A", "#5B6577"
BLUE = LinearSegmentedColormap.from_list("blue", ["#E3F1FD", "#74C8F4", "#1E9BF0", "#0B4FD1"])
GREY = LinearSegmentedColormap.from_list("navy", ["#FFFFFF", "#9FB0CC", NAVY])
CELL = LinearSegmentedColormap.from_list("dose", ["#E8ECF4", "#7C93B8", NAVY], N=3)

rng = np.random.default_rng(3)
n = 8
# toy data behind the icons: genotypes -> GRM, embedding -> whitened kernel
G = rng.binomial(2, rng.uniform(0.15, 0.5, 14), (n, 14)).astype(float)
Zg = (G - G.mean(0)) / (G.std(0) + 1e-9)
K_G = Zg @ Zg.T / G.shape[1]
Z = rng.normal(size=(n, 2)) @ rng.normal(size=(2, 12)) + 0.5 * rng.normal(size=(n, 12))
U, S, _ = np.linalg.svd((Z - Z.mean(0)) / Z.std(0), full_matrices=False)
W = U[:, :3] * np.sqrt(n)
K_E = W @ W.T / 3

fig, ax = plt.subplots(figsize=(13, 6.2), facecolor=BG)
ax.set_facecolor(BG)


def tiles(data, x, y, w, h, cmap, vmin=None, vmax=None):
    """A rounded card holding a heat-map of `data` (row 0 on top)."""
    ax.add_patch(FancyBboxPatch((x - 0.07, y - 0.07), w + 0.14, h + 0.14,
                                boxstyle="round,pad=0,rounding_size=0.09", fc="white", ec=NAVY, lw=1.7, zorder=1))
    xs, ys = np.linspace(x, x + w, data.shape[1] + 1), np.linspace(y, y + h, data.shape[0] + 1)
    ax.pcolormesh(xs, ys, data[::-1], cmap=cmap, vmin=vmin, vmax=vmax, edgecolors="white", linewidth=1.0, zorder=2)


def caption(cx, y, title, mono):
    ax.text(cx, y, title, ha="center", va="top", fontsize=10.5, weight="bold", color=INK)
    ax.text(cx, y - 0.3, mono, ha="center", va="top", fontsize=8.5, family="monospace", color=MUTE)


def arrow(p0, p1, color, rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=17, lw=2.4, color=color,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0, zorder=3))


def pill(x, y, text, color):
    ax.text(x, y, text, fontsize=8.5, family="monospace", color=color, va="center",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, lw=1.3))


def bell(x, y, w, h, h2, color):
    t = np.linspace(-3.2, 3.2, 240)
    px = x + (t + 3.2) / 6.4 * w
    pdf = lambda s: np.exp(-0.5 * (t / s) ** 2) / (s * np.sqrt(2 * np.pi))
    inner = h2 * pdf(np.sqrt(h2))
    ax.fill_between(px, y, y + h * inner / pdf(1).max(), color=color, alpha=0.9, lw=0, zorder=2)
    ax.plot(px, y + h * pdf(1) / pdf(1).max(), color=NAVY, lw=1.6, ls=(0, (4, 3)), zorder=3)
    ax.plot([x, x + w], [y, y], color=NAVY, lw=1.6, zorder=3)
    for tick in (-2, -1, 0, 1, 2):
        tx = x + (tick + 3.2) / 6.4 * w
        ax.plot([tx, tx], [y - 0.05, y + 0.05], color=NAVY, lw=1.3, zorder=3)


LA, LB, CY = 4.35, 1.75, 3.05          # lane centres, donors centre
ax.text(0.35, 5.95, "How much expression variance do genotypes vs. frozen model embeddings explain?",
        fontsize=13, weight="bold", color=NAVY, va="center")

# --- donors: two haplotypes carrying variants
ax.text(1.45, CY + 0.95, "Donors", ha="center", fontsize=10.5, weight="bold", color=INK)
for k, yy in enumerate((CY + 0.3, CY - 0.25)):
    ax.add_patch(FancyBboxPatch((0.35, yy), 2.2, 0.36, boxstyle="round,pad=0,rounding_size=0.12",
                                fc="white", ec=NAVY, lw=1.7, zorder=1))
    for xv in np.sort(rng.uniform(0.55, 2.35, 5)):
        ax.plot([xv, xv], [yy + 0.06, yy + 0.30], color=RED if rng.random() < 0.5 else NAVY, lw=2.4, zorder=2)
ax.text(1.45, CY - 0.55, "2 haplotypes each,\nwith their variants", ha="center", va="top", fontsize=8.5, color=MUTE)

# --- lane A: genotype
pill(3.3, LA + 0.95, "embher.heritability", NAVY)
tiles(G, 3.3, LA - 0.55, 1.7, 1.1, CELL, 0, 2)
caption(4.15, LA - 0.72, "genotype dosages", "read_bed")
tiles(K_G, 6.7, LA - 0.55, 1.1, 1.1, GREY, K_G.min(), K_G.max())
caption(7.25, LA - 0.72, "genotype kernel", "build_grm")
arrow((2.75, CY + 0.35), (3.15, LA - 0.1), NAVY, rad=-0.25)
arrow((5.25, LA), (6.5, LA), NAVY)
arrow((8.05, LA), (9.75, LA), NAVY)
ax.text(8.9, LA + 0.12, "fit_reml\nhe_fit", ha="center", va="bottom", fontsize=8.5, family="monospace", color=MUTE)

# --- lane B: model -> embedding -> kernel
pill(3.3, LB + 0.95, "embher.embedding  →  embher.heritability", RED)
heights = [0.95, 0.75, 0.55, 0.38, 0.38, 0.38, 0.38, 0.55, 0.75, 0.95]
ax.add_patch(FancyBboxPatch((3.23, LB - 0.62), 2.28, 1.24, boxstyle="round,pad=0,rounding_size=0.09",
                            fc="white", ec=NAVY, lw=1.7, zorder=1))
for i, hh in enumerate(heights):
    last = i == len(heights) - 1
    ax.add_patch(FancyBboxPatch((3.36 + i * 0.21, LB - hh / 2), 0.15, hh, boxstyle="round,pad=0,rounding_size=0.05",
                                fc=BLUE(0.25 + 0.75 * i / 9), ec=GOLD if last else "none", lw=2.4, zorder=2))
caption(4.37, LB - 0.72, "Borzoi (frozen)", "per-layer embeddings")
tiles(Z, 6.0, LB - 0.55, 1.2, 1.1, BLUE, Z.min(), Z.max())
caption(6.6, LB - 0.72, "embeddings", "pool over positions")
tiles(K_E, 7.95, LB - 0.55, 1.1, 1.1, BLUE, K_E.min(), K_E.max())
caption(8.5, LB - 0.72, "embedding kernel", "whiten_embeddings")
arrow((2.75, CY - 0.35), (3.1, LB + 0.05), GOLD, rad=0.25)
arrow((5.6, LB), (5.85, LB), GOLD)
arrow((7.3, LB), (7.8, LB), NAVY)
arrow((9.2, LB), (9.75, LB), NAVY)

# --- heritability panels (phenotype variance, dashed; explained share, filled)
PX, PW, PH = 9.95, 2.85, 1.05
bell(PX, LA - 0.5, PW, PH, 0.45, NAVY)
ax.text(PX + PW / 2, LA - 0.72, r"$h^2_{\mathrm{G}}$ genotypes", ha="center", va="top", fontsize=12, color=INK)
bell(PX, LB - 0.5, PW, PH, 0.36, RED)
ax.text(PX + PW / 2, LB - 0.72, r"$h^2_{\mathrm{E}}$ embeddings", ha="center", va="top", fontsize=12, color=INK)
ax.text(PX + PW / 2, 0.55, "dashed: total variance of the phenotype\nfilled: share explained", ha="center",
        va="center", fontsize=8.5, color=MUTE)

ax.text(0.35, 0.2, "Same phenotype, donors and covariates in both lanes; shuffling the donors of Z gives the null.",
        fontsize=9, color=MUTE, va="center")

ax.set_xlim(0, 13)
ax.set_ylim(0, 6.2)
ax.set_aspect("equal")
ax.axis("off")
out = Path(__file__).with_name("schematic.png")
fig.savefig(out, dpi=200, facecolor=BG, bbox_inches="tight", pad_inches=0.15)
print("wrote", out)
