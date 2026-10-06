"""Följ en träning live: grafer över hur AI:n blir bättre, uppdateras varje sekund.

    python dashboard.py                          # den senaste körningen i runs/
    python dashboard.py runs/live-20261006-1830

Läser runs/<namn>/progress.jsonl, som train.py fyller på efter varje uppdatering av nätverket.
Fönstret kan stängas och öppnas igen när som helst - träningen påverkas inte.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt

from labyrint.agent import latest_run
from labyrint.autopilot import BASELINE_SECONDS

# the game's side-panel colours
BG, AX_BG, GRID = "#241d18", "#2e2620", "#4a3e34"
TEXT, DIM = "#f0e2c8", "#aa967d"
ACCENT, GOOD, BAD, BLUE = "#e8b054", "#82c878", "#e66e5a", "#7fb4d8"
STALE_SECONDS = 60.0   # no new data for this long: the training has probably stopped

# (title, explanation, [(logger key, legend label, colour, line style, scale)], options)
PANELS = [
    ("Andel i mål", "hur ofta kulan når FINISH (%)", [
        ("rollout/success_rate", "träning (hälften startar mitt i banan)", ACCENT, "-", 100.0),
        ("eval/success_rate", "prov från START", GOOD, "o-", 100.0)], {"ylim": (-3, 103)}),
    ("Tid till mål", "snittid på proven från START (sekunder)", [
        ("eval/mean_time", "AI:n", GOOD, "o-", 1.0)], {"baseline": BASELINE_SECONDS}),
    ("Belöning per omgång", "poängen AI:n försöker få så hög som möjligt", [
        ("rollout/ep_rew_mean", "", ACCENT, "-", 1.0)], {}),
    ("Förlust (value loss)", "hur fel AI:n gissar sin framtida belöning", [
        ("train/value_loss", "", BAD, "-", 1.0)], {"log": True}),
    ("Utforskning", "slumpbrus på rattarna (std) - minskar när den blir säker", [
        ("train/std", "", BLUE, "-", 1.0)], {}),
    ("Förutsägelse", "explained variance - 1 = förutser belöningen perfekt", [
        ("train/explained_variance", "", BLUE, "-", 1.0)], {"ylim": (-1.0, 1.05), "fit_bottom": True}),
]


class ProgressLog:
    """Reads progress.jsonl incrementally while the training keeps appending to it."""

    def __init__(self, path: Path):
        self.path = path
        self.rows: list[dict] = []
        self.pos = 0
        self.last_new = time.monotonic()

    def poll(self) -> bool:
        """Read the lines added since last time. True if there were any."""
        try:
            if self.path.stat().st_size < self.pos:   # a fresh run started in the same folder
                self.rows, self.pos = [], 0
            with self.path.open("rb") as f:
                f.seek(self.pos)
                data = f.read()
        except FileNotFoundError:
            return False
        end = data.rfind(b"\n") + 1                   # a half-written last line waits for next time
        if end == 0:
            return False
        self.pos += end
        self.rows += [json.loads(line) for line in data[:end].splitlines() if line.strip()]
        self.last_new = time.monotonic()
        return True

    def series(self, key: str, scale: float = 1.0) -> tuple[list[float], list[float]]:
        pts = [(r["step"] / 1e6, r[key] * scale) for r in self.rows if key in r and math.isfinite(r[key])]
        return [p[0] for p in pts], [p[1] for p in pts]

    def last(self, key: str) -> float | None:
        return next((r[key] for r in reversed(self.rows) if key in r), None)


def sv(x: float, decimals: int = 1) -> str:
    """A number the Swedish way: decimal comma, space as thousands separator."""
    return f"{x:,.{decimals}f}".replace(",", " ").replace(".", ",")


def status_line(log: ProgressLog, run: Path) -> str:
    if not log.rows:
        return f"Väntar på träningen i {run} ..."
    steps = log.last("time/total_timesteps") or log.rows[-1]["step"]
    target = log.last("time/target_timesteps") or steps
    parts = [run.name, f"{sv(steps / 1e6)} av {sv(target / 1e6, 0)} M steg ({100 * steps / target:.0f} %)"]
    if (fps := log.last("time/fps")) is not None:
        parts.append(f"{sv(fps, 0)} steg/s")
    if steps >= target:
        parts.append("KLAR")
    elif time.monotonic() - log.last_new > STALE_SECONDS:
        parts.append("ingen ny data - har träningen stannat?")
    return "   ·   ".join(parts)


def draw(fig, axes, log: ProgressLog, run: Path) -> None:
    x_max = max(0.1, log.rows[-1]["step"] / 1e6 if log.rows else 0.0)
    for i, (ax, (title, explanation, series, opts)) in enumerate(zip(axes.flat, PANELS)):
        ax.clear()
        ax.set_facecolor(AX_BG)
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.tick_params(colors=DIM, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(GRID)
        ax.set_title(title, loc="left", color=TEXT, fontsize=12, fontweight="bold", pad=16)
        ax.text(0, 1.02, explanation, transform=ax.transAxes, color=DIM, fontsize=8)
        if i >= 3:
            ax.set_xlabel("miljoner träningssteg", color=DIM, fontsize=8)
        labelled = False
        for key, label, color, style, scale in series:
            xs, ys = log.series(key, scale)
            if xs:
                ax.plot(xs, ys, style, color=color, label=label or None, linewidth=1.6, markersize=3.5)
                labelled |= bool(label)
        if "baseline" in opts:
            ax.axhline(opts["baseline"], color=DIM, linestyle="--", linewidth=1.2,
                       label=f"autopiloten ({opts['baseline']:.0f} s)")
            ax.set_ylim(bottom=0)
            labelled = True
        if "ylim" in opts:
            low, high = opts["ylim"]
            if opts.get("fit_bottom"):   # only go as low as the data does
                low = max(low, ax.get_ylim()[0])
            ax.set_ylim(low, high)
        ax.set_xlim(0, x_max * 1.02)
        if opts.get("log") and ax.lines:
            ax.set_yscale("log")
        if labelled:
            ax.legend(fontsize=7.5, facecolor=AX_BG, edgecolor=GRID, labelcolor=TEXT, loc="best")
    fig.suptitle(status_line(log, run), color=TEXT, fontsize=11)


def main() -> None:
    ap = argparse.ArgumentParser(description="Följ en träning live.")
    ap.add_argument("run", nargs="?", default=None, help="mappen under runs/ (standard: den senaste)")
    args = ap.parse_args()
    run = Path(args.run) if args.run else latest_run()
    if run is None:
        raise SystemExit("Hittar ingen träning i runs/. Starta en med: python train.py")

    log = ProgressLog(run / "progress.jsonl")
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), facecolor=BG, layout="constrained")
    fig.canvas.manager.set_window_title("Labyrint – träningen")
    log.poll()
    draw(fig, axes, log, run)
    plt.show(block=False)

    win = getattr(fig.canvas.manager, "window", None)
    if hasattr(win, "winfo_screenwidth"):   # Tk: put the window on the right, next to the game
        win.update_idletasks()
        win.geometry(f"+{max(0, win.winfo_screenwidth() - win.winfo_width() - 10)}+40")

    while plt.fignum_exists(fig.number):
        if log.poll():
            draw(fig, axes, log, run)
        else:
            fig.suptitle(status_line(log, run), color=TEXT, fontsize=11)
        plt.pause(1.0)


if __name__ == "__main__":
    main()
