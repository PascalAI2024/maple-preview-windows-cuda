"""Draw the published benchmark results as one chart.

Usage:
  python scripts/plot-benchmarks.py

Writes docs/img/maple-benchmarks.png. Every number is read from the committed
result.json files, nothing is typed in:

  remote-results/local-controlled-20260809/result.json  RTX 4080 SUPER, MMQ on/off, A/B/B/A
  remote-results/vast-crossarch-20260808/sm86/result.json   RTX 3080
  remote-results/vast-crossarch-20260808/sm120/result.json  RTX 5060 Ti

Bars are means; whiskers are one standard deviation as recorded in each file.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "remote-results"
OUT = ROOT / "docs" / "img" / "maple-benchmarks.png"

OFF = "#a0aec0"
ON = "#2b6cb0"


def load(rel: str) -> dict:
    with (RESULTS / rel).open(encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    local = load("local-controlled-20260809/result.json")
    sm86 = load("vast-crossarch-20260808/sm86/result.json")
    sm120 = load("vast-crossarch-20260808/sm120/result.json")

    # (label, off bar, on bar), each bar is (mean, stddev) or None
    def local_pair(metric: str):
        block = local[metric]
        return (
            (block["fallback"]["mean_ts"], block["fallback"]["stddev_ts"]),
            (block["mmq"]["mean_ts"], block["mmq"]["stddev_ts"]),
        )

    def single(result: dict, metric: str):
        return (result[f"{metric}_avg_ts"], result[f"{metric}_stddev_ts"])

    gpus = [
        (local["gpu"].replace("NVIDIA GeForce ", "") + "\nsm_89\ndriver " + local["driver_version"], "pp512", "tg128", True),
        (sm86["gpu_name"].replace("NVIDIA GeForce ", "") + "\nsm_86\ndriver " + sm86["driver_version"], None, None, False),
        (sm120["gpu_name"].replace("NVIDIA GeForce ", "") + "\nsm_120\ndriver " + sm120["driver_version"], None, None, False),
    ]
    results = [local, sm86, sm120]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, metric, title in (
        (axes[0], "pp512", "Prompt processing, pp512"),
        (axes[1], "tg128", "Token generation, tg128"),
    ):
        width = 0.38
        for index, (label, _, _, has_pair) in enumerate(gpus):
            if has_pair:
                off, on = local_pair(metric)
                ax.bar(index - width / 2, off[0], width, yerr=off[1], color=OFF, error_kw={"capsize": 3, "elinewidth": 0.9})
                ax.bar(index + width / 2, on[0], width, yerr=on[1], color=ON, error_kw={"capsize": 3, "elinewidth": 0.9})
                for x, bar in ((index - width / 2, off), (index + width / 2, on)):
                    ax.annotate(f"{bar[0]:,.0f}", (x, bar[0] + bar[1]), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5)
            else:
                value = single(results[index], metric)
                ax.bar(index, value[0], width, yerr=value[1], color=ON, error_kw={"capsize": 3, "elinewidth": 0.9})
                ax.annotate(f"{value[0]:,.0f}", (index, value[0] + value[1]), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5)
        ax.set_xticks(range(len(gpus)), [g[0] for g in gpus], fontsize=8.5)
        ax.set_ylabel("tokens per second (mean, whiskers one std dev)")
        ax.set_title(title, loc="left", fontsize=10.5)
        ax.grid(axis="y", color="#e2e8f0", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    top_pp = max(local["pp512"]["mmq"]["mean_ts"], sm86["pp512_avg_ts"], sm120["pp512_avg_ts"])
    axes[0].set_ylim(0, top_pp * 1.45)
    axes[1].set_ylim(0, max(local["tg128"]["fallback"]["mean_ts"], local["tg128"]["mmq"]["mean_ts"]) * 1.15)

    handles = [plt.Rectangle((0, 0), 1, 1, color=OFF), plt.Rectangle((0, 0), 1, 1, color=ON)]
    axes[0].legend(handles, ["TQ2_0 MMQ path disabled", "patched build, TQ2_0 MMQ path enabled"], frameon=False, loc="upper left", fontsize=8.5)

    fig.suptitle(
        "Maple-Preview TQ2_0, llama.cpp fork with this repo's patches (model: maple-tq2_0.gguf, 5.1 GiB)",
        x=0.01,
        ha="left",
        fontsize=11,
    )
    fig.text(
        0.01,
        0.005,
        "Source: remote-results/local-controlled-20260809 (A/B/B/A, 10 samples per bar, 2026-08-09) and "
        "remote-results/vast-crossarch-20260808 (rented hosts, 2026-08-08).\n"
        "The rented-host bars have no MMQ-disabled run and are not like-for-like with the 4080 SUPER (different driver and host).",
        fontsize=7.5,
        color="#4a5568",
        va="bottom",
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160)
    plt.close(fig)
    print(f"wrote {OUT}")
    print(
        "pp512 on 4080 SUPER: "
        f"{local['pp512']['fallback']['mean_ts']:.0f} -> {local['pp512']['mmq']['mean_ts']:.0f} t/s "
        f"({local['pp512']['speedup_mean']:.2f}x)"
    )


if __name__ == "__main__":
    main()
