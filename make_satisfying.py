"""CLI: rendert die Satisfying-Simulationen als TikTok-Videos.

  python3 make_satisfying.py                          # alle Episoden, beide Formate
  python3 make_satisfying.py s01_ring_battle -f vertical
  python3 make_satisfying.py --preview                # Standbilder inkl. TikTok-Oberfläche
  python3 make_satisfying.py --search ring            # spannende Seeds für ein neues Ring-Battle suchen
"""
import argparse
import os
from multiprocessing import Pool

from satisfying import flow, hypno, marble_render, maze, minimal, orbit
from satisfying.episodes import EPISODES, FLOW, HYPNO, MARBLE, MAZE, MINIMAL, ORBIT
from satisfying.scene import LAYOUTS, render
from satisfying.sim import Elimination, RingEscape

OUT = "output"


def job(args):
    name, fmt, preview = args
    spec = dict({**EPISODES, **HYPNO, **FLOW, **MAZE}[name], id=name)
    rend = maze.render if name in MAZE else flow.render if name in FLOW else hypno.render if name in HYPNO else render
    sim = spec["make"]().run()
    total = sim.total
    if preview:
        f = sim.finish_t or total
        times = [0.0, 3.0, 12.0, total * 0.45, total * 0.7, f - 1.5, f + 0.1, f + 0.6, total - 0.1]
        rend(sim, spec, fmt, None, preview_dir=os.path.join(OUT, "preview"), preview_times=times)
        return f"{name} [{fmt}] Vorschau ({total:.1f}s)"
    out = os.path.join(OUT, f"{name}_{fmt}.mp4")
    rend(sim, spec, fmt, out)
    return f"{out} ({total:.1f}s)"


def search(kind, n=400):
    for s in range(n):
        if kind == "ring":
            q = RingEscape(s).run(max_t=60).quality()
            if q and 44 <= q["finish"] <= 52 and q["margin"] == 1 and q["comeback"] and q["changes"] >= 3:
                print(s, q)
        else:
            q = Elimination(s, gap_deg=14).run(max_t=60).quality()
            if q and 44 <= q["finish"] <= 52 and 3 <= q["first"] <= 10 and q["min_gap"] >= 3:
                print(s, q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episodes", nargs="*", default=["v03_spirale_vermehrung"])
    ap.add_argument("-f", "--format", choices=[*LAYOUTS, "all"], default="all")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--search", choices=["ring", "elim"])
    ap.add_argument("-j", "--jobs", type=int, default=os.cpu_count())
    a = ap.parse_args()
    if a.search:
        return search(a.search)
    for name in [n for n in a.episodes if n in MINIMAL]:  # eigener Prozess-Pool, nur Hochformat
        spec = dict(MINIMAL[name], id=name)
        sim = spec["make"]().run()
        os.makedirs(OUT, exist_ok=True)
        if a.preview:
            ts = [0.0, 5.0, 12.0, 20.0, 26.0, 30.0, 34.0, 38.0, 42.0, 45.0, round(sim.t - 3, 1), round(sim.t - 0.1, 1)]
            minimal.render(sim, spec, None, preview_dir=os.path.join(OUT, "preview"), preview_times=ts)
            print(f"{name} Vorschau ({sim.t:.1f}s)")
        else:
            out = os.path.join(OUT, f"{name}_vertical.mp4")
            minimal.render(sim, spec, out, jobs=a.jobs)
            print(f"{out} ({sim.t:.1f}s)")
    for name in [n for n in a.episodes if n in ORBIT]:  # eigener Prozess-Pool, nur Hochformat
        spec = dict(ORBIT[name], id=name)
        sim = spec["make"]().run()
        os.makedirs(OUT, exist_ok=True)
        if a.preview:
            ts = sorted({0.0, 4.0, 8.0} | {round(t + dt_, 1) for t, k, _ in sim.events if k in ("open", "calm") for dt_ in (0.6, 2.5)}
                        | {round(sim.t - 0.1, 1)})
            orbit.render(sim, spec, None, preview_dir=os.path.join(OUT, "preview"), preview_times=ts)
            print(f"{name} Vorschau ({sim.t:.1f}s)")
        else:
            out = os.path.join(OUT, f"{name}_vertical.mp4")
            orbit.render(sim, spec, out, jobs=a.jobs)
            print(f"{out} ({sim.t:.1f}s)")
    for name in [n for n in a.episodes if n in MARBLE]:  # eigener Prozess-Pool, nur Hochformat
        spec = dict(MARBLE[name], id=name)
        sim = spec["make"]().run()
        os.makedirs(OUT, exist_ok=True)
        if a.preview:
            ts = sorted({0.0, 3.0, 6.0, 9.0} | {round(t, 1) for t, k, _ in sim.events if k in ("tramp", "seesaw", "loop", "fire", "goal")}
                        | {round(sim.t - 3, 1), round(sim.t - 0.1, 1)})
            marble_render.render(sim, spec, "vertical", None, preview_dir=os.path.join(OUT, "preview"), preview_times=ts)
            print(f"{name} Vorschau ({sim.t:.1f}s)")
        else:
            out = os.path.join(OUT, f"{name}_vertical.mp4")
            marble_render.render(sim, spec, "vertical", out, jobs=a.jobs)
            print(f"{out} ({sim.t:.1f}s)")
    a.episodes = [n for n in a.episodes if n not in MARBLE and n not in ORBIT and n not in MINIMAL]
    if not a.episodes:
        return
    fmts = list(LAYOUTS) if a.format == "all" else [a.format]
    os.makedirs(OUT, exist_ok=True)
    jobs = [(n, f, a.preview) for n in a.episodes for f in fmts]
    with Pool(min(a.jobs, len(jobs))) as p:
        for msg in p.imap_unordered(job, jobs):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
