"""CLI: rendert Episoden als TikTok-Videos.

Beispiele:
  python3 make_video.py                       # alle Episoden, beide Formate
  python3 make_video.py ep01_loewen -f vertical
  python3 make_video.py ep01_loewen --preview  # nur Standbilder + TikTok-UI-Overlay
"""
import argparse
import importlib
import os
import pkgutil
from multiprocessing import Pool

import episodes
from tiktok.render import FORMATS, Episode, render_episode

OUT = "output"


def load(name):
    return importlib.import_module(f"episodes.{name}").EPISODE


def job(args):
    name, fmt, preview = args
    spec = load(name)
    if preview:
        total = Episode(spec, fmt).total
        times = [0.0] + [round(total * k / 8, 1) for k in range(1, 8)] + [round(total - 0.2, 1)]
        render_episode(spec, fmt, None, preview_dir=os.path.join(OUT, "preview"), preview_times=times)
        return f"{name} [{fmt}] Vorschau ({total:.1f}s)"
    out = os.path.join(OUT, f"{name}_{fmt}.mp4")
    total = render_episode(spec, fmt, out)
    return f"{out} ({total:.1f}s)"


def main():
    names = sorted(m.name for m in pkgutil.iter_modules(episodes.__path__))
    ap = argparse.ArgumentParser()
    ap.add_argument("episodes", nargs="*", default=names)
    ap.add_argument("-f", "--format", choices=[*FORMATS, "all"], default="all")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("-j", "--jobs", type=int, default=os.cpu_count())
    a = ap.parse_args()
    fmts = list(FORMATS) if a.format == "all" else [a.format]
    os.makedirs(OUT, exist_ok=True)
    jobs = [(n, f, a.preview) for n in a.episodes for f in fmts]
    with Pool(min(a.jobs, len(jobs))) as p:
        for msg in p.imap_unordered(job, jobs):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
