"""CLI: rendert die Satisfying-Simulationen als TikTok-Videos.

  python3 make_satisfying.py                          # alle Episoden, beide Formate
  python3 make_satisfying.py s01_ring_battle -f vertical
  python3 make_satisfying.py --preview                # Standbilder inkl. TikTok-Oberfläche
  python3 make_satisfying.py --search ring            # spannende Seeds für ein neues Ring-Battle suchen
"""
import argparse
import os
from multiprocessing import Pool

from satisfying.episodes import EPISODES
from satisfying.scene import LAYOUTS, render
from satisfying.sim import Elimination, RingEscape

OUT = "output"


def job(args):
    name, fmt, preview = args
    spec = dict(EPISODES[name], id=name)
    sim = spec["make"]().run()
    total = sim.total
    if preview:
        f = sim.finish_t
        times = [0.0, 3.0, 12.0, total * 0.45, total * 0.7, f - 1.5, f + 0.8, f + 2.5, total - 0.5]
        render(sim, spec, fmt, None, preview_dir=os.path.join(OUT, "preview"), preview_times=times)
        return f"{name} [{fmt}] Vorschau ({total:.1f}s)"
    out = os.path.join(OUT, f"{name}_{fmt}.mp4")
    render(sim, spec, fmt, out)
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
    ap.add_argument("episodes", nargs="*", default=list(EPISODES))
    ap.add_argument("-f", "--format", choices=[*LAYOUTS, "all"], default="all")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--search", choices=["ring", "elim"])
    ap.add_argument("-j", "--jobs", type=int, default=os.cpu_count())
    a = ap.parse_args()
    if a.search:
        return search(a.search)
    fmts = list(LAYOUTS) if a.format == "all" else [a.format]
    os.makedirs(OUT, exist_ok=True)
    jobs = [(n, f, a.preview) for n in a.episodes for f in fmts]
    with Pool(min(a.jobs, len(jobs))) as p:
        for msg in p.imap_unordered(job, jobs):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
