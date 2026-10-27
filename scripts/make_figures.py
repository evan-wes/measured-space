"""Draw the audit's publication figures from the outputs of ``scripts/run_audit.py``.

Figures, each 3.4 in wide (one column of a two-column paper), as PDF and PNG:

- ``audit_error_vs_range``: absolute numeric error against distance from the camera (the farthest referent's
  median auto depth), per depth source; binned medians with interquartile bands, pooled over the numeric
  question types (no per-item points, so the figure carries aggregates only). Every source in the table is
  drawn: those in :data:`SOURCE_STYLE` with their style, any other in a neutral grey with its own name, after a
  printed warning.
- ``audit_coverage``: empirical against nominal coverage of the out-of-scene intervals (feature model after
  conformal calibration, its raw Gaussian interval, and the constant-sigma baseline), and 90 % coverage by
  terciles of the predicted sigma; datasets pooled, plus ``audit_coverage_<dataset>`` per dataset.
- ``audit_risk_yield``: error rate among accepted labels against the accepted share, for rankings by the feature
  model, the baselines and at random, with the certified gate's operating point.
- ``audit_reliability``: reliability diagram of categorical p(correct), feature model against constant.

Inputs: per-item tables in ``<data root>/outputs/audit/`` (``--audit``) and aggregate tables in ``results/audit/``
(``--results``). Outputs: the folder ``--figures``, required with no default so a run never overwrites the
committed figures (``results/figures``) by accident, and a copy in each ``--copy-dir`` (none by default; the
per-item audit folder's ``figures/`` was written by default before).

Usage::

    python scripts/make_figures.py --figures results/figures
    python scripts/make_figures.py --figures results/figures --copy-dir <data root>/outputs/audit/figures
    python scripts/make_figures.py --audit <audit folder> --results <aggregate folder> --figures <scratch folder>
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Iterable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator  # noqa: E402

from measured_space.config import REPO_ROOT, data_root  # noqa: E402

WIDTH = 3.4                                                   # inches: one column of a two-column paper
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"          # categorical slots 1-3 (validated all-pairs)
INK, MUTED, GRID, BASELINE = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"
# Depth sources coloured by class: monocular (MoGe-2), device sensor (iPad LiDAR), reference (laser scan in
# ARKitScenes, rendered depth in the synthetic warehouse; the two never share a panel). Markers tell them apart too.
# Box sources: the two CuTR variants share ARKitScenes panels with the three depth sources, so they take the two
# remaining palette colours (ink, grey); Sparse4D (Smart Spaces, never in a panel with iPad LiDAR) takes aqua with
# its own marker.
SOURCE_STYLE = {"moge": (BLUE, "o", "MoGe-2 (monocular)"), "lidar": (AQUA, "s", "iPad LiDAR"),
                "laser": (ORANGE, "^", "Laser scan"), "gt_depth": (ORANGE, "D", "Rendered depth"),
                "cutr_sam3": (INK, "v", "CuTR + SAM 3 names"), "cutr_clip": (MUTED, "P", "CuTR + CLIP names"),
                "sparse4d": (AQUA, "X", "Sparse4D (multi-camera)")}
DEFAULT_SOURCE_STYLE = (MUTED, "*")   # colour and marker of a source missing from SOURCE_STYLE (labelled by name)
DATASET_NAME = {"arkitscenes": "ARKitScenes", "smartspaces": "Smart Spaces"}
METHOD_STYLE = {"sigma": (BLUE, "o", "Feature model, conformal"),
                "sigma_raw": (AQUA, "s", "Feature model, as fitted"),
                "const": (ORANGE, "^", "Constant σ, conformal")}
RANKING_STYLE = {"sigma": (BLUE, "-", "Feature model"), "const": (ORANGE, "-", "Constant σ"),
                 "prop": (AQUA, "-", "σ ∝ value")}


def style() -> None:
    """Set matplotlib defaults for small, clean paper figures."""
    plt.rcParams.update({
        "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "legend.fontsize": 6, "font.family": "sans-serif", "axes.linewidth": 0.6, "axes.edgecolor": MUTED,
        "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK,
        "ytick.labelcolor": INK, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
        "ytick.major.size": 2.5, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.5, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "lines.linewidth": 1.2, "lines.markersize": 3.5, "savefig.dpi": 300,
        "pdf.fonttype": 42, "figure.dpi": 150})


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse command-line arguments.

    Parameters
    ----------
    argv : list of str or None
        Arguments, or ``None`` for ``sys.argv[1:]``.

    Returns
    -------
    argparse.Namespace
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--audit", type=Path, default=None, help="per-item audit outputs (default <data root>/outputs/audit)")
    parser.add_argument("--results", type=Path, default=REPO_ROOT / "results" / "audit", help="aggregate tables")
    parser.add_argument("--figures", type=Path, required=True,
                        help="figure folder; required (no default). The committed figures are in "
                             f"{REPO_ROOT / 'results' / 'figures'}")
    parser.add_argument("--copy-dir", type=Path, action="append", default=[],
                        help="also save every figure into this folder (repeatable; none by default)")
    return parser.parse_args(argv)


def source_style(source: str) -> tuple[str, str, str]:
    """Colour, marker and legend label of a depth or box source.

    Parameters
    ----------
    source : str
        Source name, as in the pair table's ``depth_source``.

    Returns
    -------
    tuple of str
        :data:`SOURCE_STYLE`'s entry; for a source missing from it, :data:`DEFAULT_SOURCE_STYLE` labelled with
        the source's own name, after a warning on standard error.
    """
    if source in SOURCE_STYLE:
        return SOURCE_STYLE[source]
    print(f"warning: no style for source {source!r} in make_figures.SOURCE_STYLE; drawn in neutral grey",
          file=sys.stderr)
    return DEFAULT_SOURCE_STYLE[0], DEFAULT_SOURCE_STYLE[1], source


def ordered_sources(sources: Iterable[str]) -> list[str]:
    """Sources in drawing order: those in :data:`SOURCE_STYLE` in its order, then the others by name.

    Parameters
    ----------
    sources : iterable of str
        Source names (repeats allowed).

    Returns
    -------
    list of str
        Each source once.
    """
    present = set(sources)
    return [s for s in SOURCE_STYLE if s in present] + sorted(present - set(SOURCE_STYLE))


def save(fig: Figure, name: str, folders: list[Path]) -> list[Path]:
    """Save a figure as PDF and PNG into each folder.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure.
    name : str
        File stem.
    folders : list of pathlib.Path
        Output folders (created if needed).

    Returns
    -------
    list of pathlib.Path
        Files written.
    """
    written = []
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)
        for ext in ("pdf", "png"):
            path = folder / f"{name}.{ext}"
            fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
            written.append(path)
    plt.close(fig)
    return written


def plain_log_axis(ax: Axes, which: str) -> None:
    """Label a log axis with plain numbers (0.05, 0.1, 0.2, ...) instead of powers of ten.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The panel.
    which : str
        ``"x"`` or ``"y"``.
    """
    axis = ax.xaxis if which == "x" else ax.yaxis
    axis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    axis.set_minor_locator(LogLocator(base=10, subs=(3.0, 4.0, 6.0, 7.0, 8.0, 9.0)))
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_formatter(FuncFormatter(lambda v, _: ""))


def shared_legend(fig: Figure, ax: Axes, ncol: int) -> None:
    """One legend for the whole figure, centred below the panels.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure.
    ax : matplotlib.axes.Axes
        A panel holding every labelled artist.
    ncol : int
        Legend columns.
    """
    handles, labels = ax.get_legend_handles_labels()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()                      # type: ignore[attr-defined]
    bottom = min(a.get_tightbbox(renderer).transformed(fig.transFigure.inverted()).y0 for a in fig.axes)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, bottom - 0.01), ncol=ncol,
               handlelength=1.6, columnspacing=1.2)


def panels(n: int, height: float, sharey: bool = True) -> tuple[Figure, list[Axes]]:
    """A row of ``n`` panels, 3.4 in wide in total.

    Parameters
    ----------
    n : int
        Number of panels (>= 1).
    height : float
        Figure height (in).
    sharey : bool, default True
        Share the y axis between panels.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure.
    axes : list of matplotlib.axes.Axes
        The panels, left to right.
    """
    fig, axes = plt.subplots(1, n, figsize=(WIDTH, height), sharey=sharey, squeeze=False,
                             gridspec_kw={"wspace": 0.12 if sharey else 0.45})
    return fig, list(axes[0])


def binned(x: np.ndarray, y: np.ndarray, n_bins: int) -> pd.DataFrame:
    """Quantiles of ``y`` in equal-count bins of ``x``.

    Parameters
    ----------
    x : numpy.ndarray, shape (n,)
        Binning variable.
    y : numpy.ndarray, shape (n,)
        Values summarised per bin.
    n_bins : int
        Number of bins.

    Returns
    -------
    pandas.DataFrame
        Per bin: ``x`` (median), ``p25``, ``p50``, ``p75`` of ``y`` and ``n``.
    """
    order = np.argsort(x)
    rows = []
    for idx in np.array_split(order, n_bins):
        if len(idx):
            q = np.quantile(y[idx], [0.25, 0.5, 0.75])
            rows.append({"x": float(np.median(x[idx])), "p25": q[0], "p50": q[1], "p75": q[2], "n": len(idx)})
    return pd.DataFrame(rows)


def fig_error_vs_range(pairs: pd.DataFrame) -> Figure:
    """Absolute numeric error against camera range, per dataset and depth source.

    Parameters
    ----------
    pairs : pandas.DataFrame
        Per-item pair table (``dataset, depth_source, answer_type, matched, log_range, abs_e``).

    Returns
    -------
    matplotlib.figure.Figure
        One panel per dataset; median ``|e|`` per equal-count range bin with an interquartile band, one line
        per source (:func:`ordered_sources`, styled by :func:`source_style`).
    """
    num = pairs[(pairs["answer_type"] == "number") & pairs["matched"].astype(bool)
                & np.isfinite(pairs["log_range"]) & np.isfinite(pairs["abs_e"])]
    datasets = [d for d in DATASET_NAME if d in set(num["dataset"])] or sorted(num["dataset"].unique())
    fig, axes = panels(len(datasets), 2.2)
    for ax, dataset in zip(axes, datasets):
        d = num[num["dataset"] == dataset]
        for source in ordered_sources(d["depth_source"]):
            g = d[d["depth_source"] == source]
            colour, marker, label = source_style(source)
            b = binned(np.exp(g["log_range"].to_numpy(float)), g["abs_e"].to_numpy(float),
                       int(np.clip(len(g) // 40, 3, 8)))
            ax.fill_between(b["x"], b["p25"], b["p75"], color=colour, alpha=0.12, linewidth=0)
            ax.plot(b["x"], b["p50"], color=colour, marker=marker, label=label, markeredgecolor="white",
                    markeredgewidth=0.5)
        ax.set_yscale("log")
        plain_log_axis(ax, "y")
        ax.set_title(DATASET_NAME.get(dataset, dataset), color=INK)
        ax.set_xlabel("Distance from camera (m)")
        ax.legend(loc="upper left", handlelength=1.6)
    axes[0].set_ylabel("|auto − oracle| (m)")
    return fig


def fig_coverage(coverage: pd.DataFrame, dataset: str = "all") -> Figure:
    """Empirical against nominal coverage, and 90 % coverage by predicted-sigma tercile.

    Parameters
    ----------
    coverage : pandas.DataFrame
        The ``coverage`` aggregate table.
    dataset : str, default "all"
        Dataset to plot (``"all"`` pools them).

    Returns
    -------
    matplotlib.figure.Figure
        Two panels with Wilson 95 % intervals.
    """
    cov = coverage[coverage["dataset"] == dataset]
    fig, (left, right) = panels(2, 1.8, sharey=False)
    overall = cov[cov["by"] == "all"]
    left.plot([0.45, 1.0], [0.45, 1.0], color=MUTED, linewidth=0.8, zorder=1)
    for method, (colour, marker, label) in METHOD_STYLE.items():
        g = overall[overall["method"] == method].sort_values("level")
        left.errorbar(g["level"], g["coverage"], yerr=[g["coverage"] - g["lo"], g["hi"] - g["coverage"]],
                      color=colour, marker=marker, label=label, capsize=0, elinewidth=0.8,
                      markeredgecolor="white", markeredgewidth=0.5, zorder=3)
    left.set_xlim(0.45, 1.0)
    left.set_ylim(0.35, 1.0)
    left.set_xlabel("Nominal coverage")
    left.set_ylabel("Empirical coverage")
    shared_legend(fig, left, 3)
    terc = cov[(cov["by"] == "sigma_tercile") & (cov["level"] == 0.9)]
    order = ["low", "mid", "high"]
    right.axhline(0.9, color=MUTED, linewidth=0.8, zorder=1)
    for j, (method, (colour, marker, _)) in enumerate(METHOD_STYLE.items()):
        g = terc[terc["method"] == method].set_index("group").reindex(order)
        x = np.arange(3) + (j - 1) * 0.18
        right.errorbar(x, g["coverage"], yerr=[g["coverage"] - g["lo"], g["hi"] - g["coverage"]], color=colour,
                       marker=marker, linestyle="none", capsize=0, elinewidth=0.8, markeredgecolor="white",
                       markeredgewidth=0.5, zorder=3)
    right.set_xticks(range(3), ["Low", "Mid", "High"])
    right.set_xlabel("Predicted σ tercile")
    right.set_ylabel("Coverage at nominal 90 %")
    right.grid(axis="x", visible=False)
    return fig


def fig_risk_yield(curves: pd.DataFrame, gate: pd.DataFrame) -> Figure:
    """Error rate among accepted numeric labels against the accepted share, per dataset.

    Parameters
    ----------
    curves : pandas.DataFrame
        The ``risk_yield`` aggregate table.
    gate : pandas.DataFrame
        The ``gate`` aggregate table (for the certified operating point and epsilon).

    Returns
    -------
    matplotlib.figure.Figure
        One panel per dataset; random order and the ideal order in grey; the certified gate as a filled
        marker with its Wilson interval; the target epsilon as a hairline.
    """
    num = curves[curves["answer_kind"] == "numeric"]
    datasets = [d for d in DATASET_NAME if d in set(num["dataset"])] or sorted(num["dataset"].unique())
    fig, axes = panels(len(datasets), 2.1)
    for ax, dataset in zip(axes, datasets):
        d = num[num["dataset"] == dataset]
        for ranking, colour, ls, label, lw in (("random", MUTED, "-", "Random order", 0.8),
                                               ("ideal", BASELINE, ":", "Ideal order", 1.0)):
            g = d[d["ranking"] == ranking]
            ax.plot(g["yield"], g["risk"], color=colour, linestyle=ls, linewidth=lw, label=label, zorder=1)
        for ranking, (colour, ls, label) in RANKING_STYLE.items():
            g = d[d["ranking"] == ranking]
            if len(g):
                ax.plot(g["yield"], g["risk"], color=colour, linestyle=ls, label=label, zorder=2)
        op = gate[(gate["dataset"] == dataset) & (gate["answer_kind"] == "numeric") & (gate["kind"] == "sigma")
                  & (gate["tau_scale"] == 1.0)]
        if len(op) and op["accept_n"].iloc[0] > 0:
            r = op.iloc[0]
            ax.errorbar([r["accept_yield"]], [r["accept_err"]], yerr=[[r["accept_err"] - r["accept_lo"]],
                                                                       [r["accept_hi"] - r["accept_err"]]],
                        color=BLUE, marker="o", markersize=4.5, markeredgecolor="white", markeredgewidth=0.6,
                        elinewidth=0.8, capsize=0, label="Certified gate", zorder=4)
        if len(op):
            ax.axhline(op["epsilon"].iloc[0], color=INK, linewidth=0.6, linestyle=(0, (4, 3)), zorder=1)
            ax.annotate(f"ε = {op['epsilon'].iloc[0]:g}", (1.0, op["epsilon"].iloc[0]), xytext=(-1, -2),
                        textcoords="offset points", ha="right", va="top", fontsize=6, color=INK)
        ax.set_xlim(0, 1.02)
        ax.set_ylim(bottom=0)
        ax.set_title(DATASET_NAME.get(dataset, dataset), color=INK)
        ax.set_xlabel("Accepted share of labels")
    axes[0].set_ylabel("Wrong labels among accepted")
    shared_legend(fig, axes[-1], 3)
    return fig


def fig_reliability(bins: pd.DataFrame, min_n: int = 10) -> Figure:
    """Reliability diagram of categorical p(correct), per dataset (categorical types pooled).

    Parameters
    ----------
    bins : pandas.DataFrame
        The ``reliability`` aggregate table.
    min_n : int, default 10
        Bins with fewer items are not drawn (their intervals span most of the axis).

    Returns
    -------
    matplotlib.figure.Figure
        One panel per dataset; observed share correct (Wilson 95 %) against mean predicted p per bin.
    """
    b = bins[(bins["qtype"] == "all") & (bins["n"] >= min_n)]
    datasets = [d for d in DATASET_NAME if d in set(b["dataset"])] or sorted(b["dataset"].unique())
    fig, axes = panels(len(datasets), 1.9)
    styles = {"logistic": (BLUE, "o", "Feature model"), "const": (ORANGE, "^", "Constant p")}
    for ax, dataset in zip(axes, datasets):
        d = b[b["dataset"] == dataset]
        ax.plot([0, 1], [0, 1], color=MUTED, linewidth=0.8, zorder=1)
        for kind, (colour, marker, label) in styles.items():
            g = d[d["kind"] == kind]
            ax.errorbar(g["p_mean"], g["observed"], yerr=[g["observed"] - g["lo"], g["hi"] - g["observed"]],
                        color=colour, marker=marker, linestyle="-", linewidth=0.8, label=label, capsize=0,
                        elinewidth=0.8,
                        markeredgecolor="white", markeredgewidth=0.5, zorder=3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
        ax.set_title(DATASET_NAME.get(dataset, dataset), color=INK)
        ax.set_xlabel("Predicted P(label correct)")
    axes[0].set_ylabel("Observed share correct")
    shared_legend(fig, axes[0], 2)
    return fig


def main(argv: list[str] | None = None) -> int:
    """Draw every figure whose inputs exist.

    Parameters
    ----------
    argv : list of str or None, optional
        Arguments, or ``None`` for ``sys.argv[1:]``.

    Returns
    -------
    int
        Process exit status (0).
    """
    args = parse_args(argv)
    audit = args.audit or data_root() / "outputs" / "audit"
    folders = [args.figures, *args.copy_dir]
    style()
    jobs: list[tuple[str, Path, Callable[[], Figure]]] = [
        ("audit_error_vs_range", audit / "pairs.csv", lambda: fig_error_vs_range(pd.read_csv(audit / "pairs.csv"))),
        ("audit_coverage", args.results / "coverage.csv",
         lambda: fig_coverage(pd.read_csv(args.results / "coverage.csv"))),
        ("audit_risk_yield", args.results / "risk_yield.csv",
         lambda: fig_risk_yield(pd.read_csv(args.results / "risk_yield.csv"), pd.read_csv(args.results / "gate.csv"))),
        ("audit_reliability", args.results / "reliability.csv",
         lambda: fig_reliability(pd.read_csv(args.results / "reliability.csv"))),
    ]
    if (args.results / "coverage.csv").exists():
        coverage = pd.read_csv(args.results / "coverage.csv")
        for dataset in sorted(set(coverage["dataset"]) - {"all"}):
            jobs.append((f"audit_coverage_{dataset}", args.results / "coverage.csv",
                         lambda d=dataset: fig_coverage(coverage, d)))
    for name, needed, draw in jobs:
        if not needed.exists():
            print(f"skip {name}: {needed} not found")
            continue
        for path in save(draw(), name, folders):
            print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
