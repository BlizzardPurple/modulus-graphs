"""Quick modulus graph viewer.

Usage:
    python plot_modulus.py                      # show all examples
    python plot_modulus.py --list               # list example names
    python plot_modulus.py nested               # show one example
    python plot_modulus.py nested --save        # save PNG instead of showing
    python plot_modulus.py --expr "abs(x-1)+abs(x-2)+abs(x-3)" --k 6
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

DEFAULT_RANGE = (-10, 10)

# name: (title, function of x, optional horizontal line k, optional (xmin, xmax))
EXAMPLES = {
    "basic": ("y = |x - 2|", lambda x: np.abs(x - 2), 3),
    "nested": ("y = ||x - 11| - 8|  (line y = 8 -> 3 solutions)",
               lambda x: np.abs(np.abs(x - 11) - 8), 8, (-10, 30)),
    "sum3": ("y = |x-1| + |x-2| + |x-3|  (line y = 6)",
             lambda x: np.abs(x - 1) + np.abs(x - 2) + np.abs(x - 3), 6),
    "sum2": ("y = |x-3| + |x+2|  (flat between -2 and 3)",
             lambda x: np.abs(x - 3) + np.abs(x + 2), 5),
    "diff": ("y = |x-3| - |x+2|  (range -5 to 5)",
             lambda x: np.abs(x - 3) - np.abs(x + 2), None),
    "quad": ("y = |x^2 - 4|  (line y = 5)",
             lambda x: np.abs(x ** 2 - 4), 5),
}


def solutions(X, y, k):
    """Approximate x where y crosses k (sign changes of y - k)."""
    d = y - k
    idx = np.where(np.diff(np.sign(d)) != 0)[0]
    sols = [X[i] - d[i] * (X[i + 1] - X[i]) / (d[i + 1] - d[i]) for i in idx]
    # flat segments lying exactly on k are not sign changes; report them separately
    flat = np.isclose(d, 0, atol=1e-9)
    return sols, flat


def draw(ax, title, f, k, xr=None):
    xmin, xmax = xr or DEFAULT_RANGE
    X = np.linspace(xmin, xmax, 4001)
    y = f(X)
    ax.plot(X, y, lw=2)
    ax.axhline(0, color="gray", lw=0.8)
    ax.axvline(0, color="gray", lw=0.8)
    if k is not None:
        ax.axhline(k, color="red", ls="--", lw=1, label=f"y = {k}")
        sols, flat = solutions(X, y, k)
        ax.scatter(sols, [k] * len(sols), color="red", zorder=5)
        for s in sols:
            ax.annotate(f"{s:.2f}", (s, k), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=8)
        if flat.sum() > 5:  # a real flat stretch, not isolated crossings
            ax.plot(X, np.where(flat, y, np.nan), color="red", lw=4, alpha=0.5)
        ax.legend(loc="upper center")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    lo, hi = y.min(), y.max()
    ax.set_ylim(min(lo, 0) - 1, min(hi, max(3 * (k or 0), 15)) + 2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("names", nargs="*", help="example names")
    p.add_argument("--list", action="store_true")
    p.add_argument("--save", action="store_true", help="save PNG to ./output")
    p.add_argument("--expr", help="custom expr in x, e.g. 'abs(x-1)+abs(x-2)'")
    p.add_argument("--k", type=float, help="horizontal line y = k")
    a = p.parse_args()

    if a.list:
        print("\n".join(EXAMPLES))
        return

    if a.expr:
        # local-only tool: eval limited to numpy + abs
        env = {"__builtins__": {}, "abs": np.abs, "np": np}
        items = [(a.expr, lambda x: eval(a.expr, {**env, "x": x}), a.k, None)]
    else:
        names = a.names or list(EXAMPLES)
        bad = [n for n in names if n not in EXAMPLES]
        if bad:
            p.error(f"unknown example(s): {bad}. Use --list")
        items = [EXAMPLES[n] for n in names]

    cols = min(2, len(items))
    rows = -(-len(items) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 4 * rows),
                             squeeze=False)
    for ax, item in zip(axes.flat, items):
        title, f, k, *rest = item
        draw(ax, title, f, k, rest[0] if rest else None)
    for ax in list(axes.flat)[len(items):]:
        ax.axis("off")
    fig.tight_layout()

    if a.save:
        out = Path(__file__).parent / "output"
        out.mkdir(exist_ok=True)
        path = out / ("_".join(a.names) if a.names else "all")
        path = path.with_suffix(".png") if not a.expr else out / "custom.png"
        fig.savefig(path, dpi=130)
        print(f"saved {path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
