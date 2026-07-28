"""Build the N x D (training steps x num_entities) heatmap for the
composition task, colored by test_inferred_iid accuracy."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from eval_qa import eval_file

D_VALUES = [500, 800, 1200, 2000]
DIR_FMT = "outputs/composition.{D}.{R}.3.6_wd0.1_L8_pilot"

rows = []  # (D, step, test_inferred_iid, test_inferred_ood)
for D in D_VALUES:
    R = D // 10
    d = DIR_FMT.format(D=D, R=R)
    for checkpoint_name, type_accs in eval_file(d):
        step = int(checkpoint_name.split("-")[1])
        acc_by_type = dict(type_accs)
        rows.append((D, step, acc_by_type.get("test_inferred_iid", float("nan")),
                     acc_by_type.get("test_inferred_ood", float("nan"))))

steps_all = sorted(set(r[1] for r in rows if r[0] == 500))  # D=500 has the densest checkpoint set
grid_iid = np.full((len(D_VALUES), len(steps_all)), np.nan)
grid_ood = np.full((len(D_VALUES), len(steps_all)), np.nan)
step_index = {s: i for i, s in enumerate(steps_all)}

for D, step, iid, ood in rows:
    if step not in step_index:
        continue
    di = D_VALUES.index(D)
    si = step_index[step]
    grid_iid[di, si] = iid
    grid_ood[di, si] = ood

fig, axes = plt.subplots(1, 2, figsize=(16, 4.5))

for ax, grid, title in ((axes[0], grid_iid, "test_inferred_iid"), (axes[1], grid_ood, "test_inferred_ood")):
    im = ax.imshow(grid, aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=1,
                   extent=[0, len(steps_all), -0.5, len(D_VALUES) - 0.5])
    ax.set_yticks(range(len(D_VALUES)))
    ax.set_yticklabels([str(d) for d in D_VALUES])
    tick_idx = list(range(0, len(steps_all), max(1, len(steps_all)//8)))
    ax.set_xticks([i + 0.5 for i in tick_idx])
    ax.set_xticklabels([f"{steps_all[i]//1000}k" for i in tick_idx], rotation=45)
    ax.set_xlabel("Training steps N")
    ax.set_ylabel("Task diversity D (num_entities)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="accuracy")

fig.suptitle("N x D phase diagram: composition.*.*.3.6, n_layer=8, phi=3.6")
fig.tight_layout()
fig.savefig("nd_phase_diagram.png", dpi=150)
print("Wrote nd_phase_diagram.png")
print("Grid shape:", grid_iid.shape, "D values:", D_VALUES, "N steps:", len(steps_all))
