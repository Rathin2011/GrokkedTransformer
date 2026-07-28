"""N x D heatmap using d_G and d_M instead of raw accuracy or the broken d_rel ratio."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D_VALUES = [500, 800, 1200, 2000]

rows = []  # (D, step, d_G, d_M, accuracy)
for D in D_VALUES:
    data = json.load(open(f"dGdM_D{D}.json"))
    for row in data:
        rows.append((D, row["step"], row["d_G_mean"], row["d_M_mean"], row["accuracy"]))

steps_all = sorted(set(r[1] for r in rows if r[0] == 500))
grid_dG = np.full((len(D_VALUES), len(steps_all)), np.nan)
grid_dM = np.full((len(D_VALUES), len(steps_all)), np.nan)
step_index = {s: i for i, s in enumerate(steps_all)}

for D, step, dG, dM, acc in rows:
    if step not in step_index:
        continue
    di = D_VALUES.index(D)
    si = step_index[step]
    grid_dG[di, si] = dG
    grid_dM[di, si] = dM

fig, axes = plt.subplots(1, 2, figsize=(16, 4.5))

for ax, grid, title, cmap in (
    (axes[0], grid_dG, "d_G = 1 - p(t*)  (low = G-like)", "viridis_r"),
    (axes[1], grid_dM, "d_M = TV(model, uniform)  (low = M-like)", "viridis_r"),
):
    im = ax.imshow(grid, aspect="auto", origin="lower", cmap=cmap, vmin=0, vmax=1,
                   extent=[0, len(steps_all), -0.5, len(D_VALUES) - 0.5])
    ax.set_yticks(range(len(D_VALUES)))
    ax.set_yticklabels([str(d) for d in D_VALUES])
    tick_idx = list(range(0, len(steps_all), max(1, len(steps_all)//8)))
    ax.set_xticks([i + 0.5 for i in tick_idx])
    ax.set_xticklabels([f"{steps_all[i]//1000}k" for i in tick_idx], rotation=45)
    ax.set_xlabel("Training steps N")
    ax.set_ylabel("Task diversity D (num_entities)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)

fig.suptitle("N x D: d_G / d_M (separate distances, not a forced M-vs-G ratio)")
fig.tight_layout()
fig.savefig("nd_dGdM_heatmap.png", dpi=150)
print("Wrote nd_dGdM_heatmap.png")

# Quick classification summary
print("\n--- 3-way classification (thresholds: small=0.3) ---")
for D, step, dG, dM, acc in rows:
    if step != steps_all[-1] and step != 5000:
        continue
    if dG < 0.3 and dM < 0.3:
        label = "BOTH (rare)"
    elif dG < 0.3:
        label = "G-like"
    elif dM < 0.3:
        label = "M-like"
    else:
        label = "NEITHER"
    print(f"D={D} step={step}: d_G={dG:.3f} d_M={dM:.3f} acc={acc:.3f} -> {label}")
