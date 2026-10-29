"""Sorting Algorithm Race: sechs Sortierverfahren, dieselben Daten, dasselbe Tempo (Operationen pro Sekunde).

Jedes Verfahren läuft als Generator und meldet jede Operation (Vergleich, Tausch, Schreiben). Alle Panels führen
pro Sekunde gleich viele Operationen aus – wer weniger Operationen braucht, ist zuerst fertig.
"""
import colorsys
import math
import os
import random
import subprocess
import tempfile
import wave
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from tiktok.render import tiktok_ui_mock
from .marble_render import reverb

FPS = 60
W1, H1 = 1080, 1920
SR = 44100
F_BOLD = "/usr/share/fonts/opentype/inter/InterDisplay-Black.otf"
F_SEMI = "/usr/share/fonts/opentype/inter/Inter-SemiBold.otf"


# ------------------------------------------------------------------ Algorithmen (Generatoren über eine Liste)

def bubble(a):
    n = len(a)
    for i in range(n):
        swapped = False
        for j in range(n - 1 - i):
            yield "cmp", j, j + 1
            if a[j] > a[j + 1]:
                a[j], a[j + 1] = a[j + 1], a[j]
                swapped = True
                yield "swap", j, j + 1
        if not swapped:
            return


def quick(a, lo=0, hi=None):
    if hi is None:
        hi = len(a) - 1
    if lo >= hi:
        return
    p = a[(lo + hi) // 2]
    i, j = lo, hi
    while i <= j:
        while True:
            yield "cmp", i, (lo + hi) // 2
            if a[i] < p:
                i += 1
            else:
                break
        while True:
            yield "cmp", j, (lo + hi) // 2
            if a[j] > p:
                j -= 1
            else:
                break
        if i <= j:
            a[i], a[j] = a[j], a[i]
            yield "swap", i, j
            i += 1
            j -= 1
    yield from quick(a, lo, j)
    yield from quick(a, i, hi)


def merge(a, lo=0, hi=None):
    if hi is None:
        hi = len(a)
    if hi - lo <= 1:
        return
    mid = (lo + hi) // 2
    yield from merge(a, lo, mid)
    yield from merge(a, mid, hi)
    left, right = a[lo:mid], a[mid:hi]
    i = j = 0
    k = lo
    while i < len(left) and j < len(right):
        yield "cmp", lo + i, mid + j
        if left[i] <= right[j]:
            a[k] = left[i]
            i += 1
        else:
            a[k] = right[j]
            j += 1
        yield "set", k, k
        k += 1
    for v in left[i:] + right[j:]:
        a[k] = v
        yield "set", k, k
        k += 1


def heap(a):
    n = len(a)

    def sift(start, end):
        root = start
        while 2 * root + 1 <= end:
            child = 2 * root + 1
            if child + 1 <= end:
                yield "cmp", child, child + 1
                if a[child] < a[child + 1]:
                    child += 1
            yield "cmp", root, child
            if a[root] < a[child]:
                a[root], a[child] = a[child], a[root]
                yield "swap", root, child
                root = child
            else:
                return

    for start in range((n - 2) // 2, -1, -1):
        yield from sift(start, n - 1)
    for end in range(n - 1, 0, -1):
        a[0], a[end] = a[end], a[0]
        yield "swap", 0, end
        yield from sift(0, end - 1)


def shell(a):
    n = len(a)
    gap = n // 2
    while gap > 0:
        for i in range(gap, n):
            j = i
            while j >= gap:
                yield "cmp", j - gap, j
                if a[j - gap] > a[j]:
                    a[j - gap], a[j] = a[j], a[j - gap]
                    yield "swap", j - gap, j
                    j -= gap
                else:
                    break
        gap //= 2


def radix(a, base=4):
    m = max(a)
    exp = 1
    n = len(a)
    while m // exp > 0:
        buckets = [[] for _ in range(base)]
        for i in range(n):
            yield "cmp", i, i      # Lesen
            buckets[(a[i] // exp) % base].append(a[i])
        k = 0
        for b in buckets:
            for v in b:
                a[k] = v
                yield "set", k, k
                k += 1
        exp *= base


def insertion(a):
    for i in range(1, len(a)):
        j = i
        while j > 0:
            yield "cmp", j - 1, j
            if a[j - 1] > a[j]:
                a[j - 1], a[j] = a[j], a[j - 1]
                yield "swap", j - 1, j
                j -= 1
            else:
                break


ALGOS = {"QUICK SORT": quick, "MERGE SORT": merge, "HEAP SORT": heap, "SHELL SORT": shell, "RADIX SORT": radix,
         "BUBBLE SORT": bubble, "INSERTION SORT": insertion}


# ------------------------------------------------------------------ Rennen

class SortRace:
    def __init__(self, seed, names=("QUICK SORT", "MERGE SORT", "HEAP SORT", "SHELL SORT", "RADIX SORT", "BUBBLE SORT"),
                 n=48, ops_per_s=None, finish_last=44.0, total=54.0, lead=3.2):
        rng = random.Random(seed)
        data = list(range(n))
        rng.shuffle(data)
        self.n, self.names, self.total, self.lead = n, list(names), total, lead
        self.runs = []
        for nm in names:
            a = list(data)
            states, ops = [np.array(a, np.uint8)], [None]
            for op in ALGOS[nm](a):
                states.append(np.array(a, np.uint8))
                ops.append(op)
            assert a == sorted(a), nm
            self.runs.append((np.stack(states), ops))
        counts = [len(o) - 1 for _, o in self.runs]
        self.counts = counts
        # Tempo: der langsamste Teilnehmer, der noch ankommen soll (nicht Bubble), ist bei finish_last fertig
        fin = sorted(c for c, nm in zip(counts, names) if nm != "BUBBLE SORT")
        self.rate = ops_per_s or fin[-1] / (finish_last - lead)
        self.finish = [lead + c / self.rate for c in counts]
        order = sorted(range(len(names)), key=lambda i: self.finish[i])
        self.place = {i: (k + 1 if self.finish[i] <= total - 1 else None) for k, i in enumerate(order)}
        self.n_frames = int(total * FPS)
        self.events = []
        for i in range(len(names)):
            ops = self.runs[i][1]
            for k in range(1, len(ops)):
                t = lead + k / self.rate
                if t > total:
                    break
                kind, x, y = ops[k]
                self.events.append((t, "op", {"p": i, "v": int(self.runs[i][0][k][x])}))
            if self.finish[i] <= total - 1:
                self.events.append((self.finish[i], "finish", {"p": i, "place": self.place[i]}))
        self.events.sort(key=lambda e: e[0])

    def state(self, i, t):
        states, ops = self.runs[i]
        k = int(max(0.0, t - self.lead) * self.rate)
        k = min(k, len(states) - 1)
        return states[k], (ops[k] if k > 0 and k < len(states) - 1 else None)


# ------------------------------------------------------------------ Darstellung

def hue_col(v, n, light=0.0):
    r, g, b = colorsys.hsv_to_rgb(0.83 * v / max(1, n - 1), 0.78 - 0.4 * light, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


MEDAL = {1: (255, 205, 60), 2: (210, 220, 235), 3: (225, 145, 80)}


class RaceRenderer:
    def __init__(self, race):
        self.r = race
        self.f_title = ImageFont.truetype(F_BOLD, 27)
        self.f_small = ImageFont.truetype(F_SEMI, 19)
        self.f_badge = ImageFont.truetype(F_BOLD, 26)
        self.f_time = ImageFont.truetype(F_BOLD, 40)
        yy, xx = np.mgrid[0:H1, 0:W1].astype(np.float32)
        vig = 0.5 + 0.5 * np.exp(-(((xx - W1 / 2) / W1) ** 2 + ((yy - H1 * 0.45) / H1) ** 2) * 3.0)
        self.base = (np.array((10, 9, 20), np.float32) * vig[..., None]).astype(np.uint8)
        # Panels: 2 Spalten x 3 Reihen im sicheren Bereich
        self.panels = []
        x0s, w = (100, 528), 412
        ys, h = (318, 698, 1078), 362
        for k in range(len(race.names)):
            self.panels.append((x0s[k % 2], ys[k // 2], w, h))

    def frame(self, fi):
        r = self.r
        t = (fi + 1) / FPS
        im = Image.fromarray(self.base.copy())
        glow = Image.new("RGB", (W1, H1))
        d = ImageDraw.Draw(im, "RGBA")
        dg = ImageDraw.Draw(glow)
        n = r.n
        for i, (px, py, pw, ph) in enumerate(self.panels):
            done = t >= r.finish[i]
            place = r.place.get(i) if done else None
            border = MEDAL.get(place, (120, 255, 160)) if done else (70, 70, 100)
            d.rounded_rectangle([px, py, px + pw, py + ph], radius=20, fill=(16, 15, 30, 255), outline=border + (255,),
                                width=3 if done else 2)
            if done:
                dg.rounded_rectangle([px, py, px + pw, py + ph], radius=20, outline=border, width=6)
            d.text((px + 18, py + 26), r.names[i], font=self.f_title, fill=(240, 240, 255, 255), anchor="lm")
            # Operationen-Zähler rechts im Titel
            k = min(r.counts[i], int(max(0.0, t - r.lead) * r.rate))
            d.text((px + pw - 16, py + 27), f"{k} ops", font=self.f_small, fill=(160, 160, 190, 255), anchor="rm")
            arr, op = r.state(i, t)
            bx0, by0, bx1, by1 = px + 14, py + 52, px + pw - 14, py + ph - 14
            bw = (bx1 - bx0) / n
            hl = set()
            if op is not None and not done:
                hl = {op[1], op[2]}
            sweep = (t - r.finish[i]) / 0.9 if done else -1
            for j in range(n):
                v = int(arr[j])
                hgt = (by1 - by0) * (0.08 + 0.92 * (v + 1) / n)
                x0 = bx0 + j * bw
                col = hue_col(v, n)
                if j in hl:
                    col = (255, 255, 255)
                if 0 <= sweep < 1.3 and abs(j / n - sweep) < 0.08:
                    col = (255, 255, 255)
                d.rounded_rectangle([x0 + 1, by1 - hgt, x0 + bw - 1, by1], radius=2, fill=col + (255,))
                if j in hl or (0 <= sweep < 1.3 and abs(j / n - sweep) < 0.08):
                    dg.rectangle([x0, by1 - hgt, x0 + bw, by1], fill=(200, 200, 220))
            if done:
                e = min(1.0, (t - r.finish[i]) / 0.35)
                txt = (f"{place}{['ST', 'ND', 'RD'][place - 1] if place <= 3 else 'TH'}  " if place else "") + \
                      f"{r.finish[i] - r.lead:.1f}s"
                tw = d.textlength(txt, font=self.f_badge) + 30
                cx, cy = px + pw / 2, py + ph / 2
                s_ = 0.6 + 0.4 * e
                d.rounded_rectangle([cx - tw / 2 * s_, cy - 26 * s_, cx + tw / 2 * s_, cy + 26 * s_], radius=26,
                                    fill=(8, 8, 16, int(225 * e)), outline=border + (int(255 * e),), width=3)
                if e > 0.5:
                    d.text((cx, cy), txt, font=self.f_badge, fill=border + (255,), anchor="mm")
        # Glow
        g = glow.resize((W1 // 4, H1 // 4), Image.BILINEAR).filter(ImageFilter.GaussianBlur(4)).resize((W1, H1))
        arr = np.asarray(im, np.float32) + np.asarray(g, np.float32) * 0.7
        im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        d = ImageDraw.Draw(im, "RGBA")
        # Zeit oben
        tt = max(0.0, t - r.lead)
        d.rounded_rectangle([W1 / 2 - 130 - 30, 236, W1 / 2 + 130 - 30, 300], radius=18, fill=(0, 0, 0, 170),
                            outline=(255, 255, 255, 70), width=2)
        d.text((W1 / 2 - 30, 268), f"{tt:.1f}s", font=self.f_time, fill=(255, 255, 255, 255), anchor="mm")
        # Hook
        a = min(1.0, t / 0.25) * min(1.0, max(0.0, (r.lead + 0.2 - t) / 0.5))
        self.banner(d, [("Same data. Same speed.", 54, (255, 255, 255)), ("Which algorithm sorts first?", 40,
                                                                          (255, 215, 110)),
                        ("Pick one!", 36, (190, 190, 215))], a, 880)
        # Ende: Bubble Sort ist noch nicht fertig
        slow = [i for i in range(len(r.names)) if r.finish[i] > r.total - 1]
        if slow and t > r.total - 6.5:
            a = min(1.0, (t - (r.total - 6.5)) / 0.6)
            px, py, pw, ph = self.panels[slow[0]]
            cx = px + pw / 2
            for k, (txt, sz) in enumerate((("STILL", 30), ("SORTING...", 30))):
                d.text((cx, py + ph / 2 - 18 + k * 36), txt, font=ImageFont.truetype(F_BOLD, sz),
                       fill=(255, 120, 120, int(255 * a)), anchor="mm", stroke_width=4, stroke_fill=(0, 0, 0, int(220 * a)))
        return im

    @staticmethod
    def banner(d, lines, alpha, cy):
        if alpha <= 0.01:
            return
        fonts = [ImageFont.truetype(F_BOLD, sz) for _, sz, _ in lines]
        ws = [d.textlength(txt, font=fo) for (txt, _, _), fo in zip(lines, fonts)]
        H = sum(sz for _, sz, _ in lines) * 1.3 + 44
        W = min(860, max(ws) + 70)
        cx = 520
        d.rounded_rectangle([cx - W / 2, cy - H / 2, cx + W / 2, cy + H / 2], radius=28,
                            fill=(6, 6, 14, int(235 * alpha)), outline=(255, 255, 255, int(70 * alpha)), width=2)
        yy = cy - H / 2 + 22
        for (txt, sz, col), fo in zip(lines, fonts):
            d.text((cx, yy), txt, font=fo, fill=col + (int(255 * alpha),), anchor="ma")
            yy += sz * 1.3


# ------------------------------------------------------------------ Klang: weiche Töne nach Balkenhöhe

_cache = {}


def blip(f):
    key = round(f)
    if key not in _cache:
        n = int(SR * 0.09)
        t = np.arange(n) / SR
        s = np.sin(2 * np.pi * f * t) * np.exp(-t * 45) * np.minimum(t / 0.003, 1)
        _cache[key] = s * 0.25
    return _cache[key]


def build_audio(race, path):
    from .viral import soft_tone
    n = int(SR * race.total)
    out = np.zeros(n + SR * 3)
    last = {}

    def put(t, s, vol):
        i = int(t * SR)
        e = min(len(out), i + len(s))
        if i < e:
            out[i:e] += s[: e - i] * vol

    penta = [0, 2, 4, 7, 9]
    for t, kind, d in race.events:
        if kind == "op":
            p = d["p"]
            if t - last.get(p, -1) < 0.07:   # höchstens ~14 Töne pro Sekunde und Panel
                continue
            last[p] = t
            v = d["v"]
            semi = penta[v % 5] + 12 * (v * 2 // race.n)
            put(t, blip(330.0 * 2 ** (semi / 12)), 0.22)
        elif kind == "finish":
            for j, f in enumerate((523.25, 659.25, 783.99, 1046.5)):
                put(t + j * 0.07, soft_tone(f), 0.5 if d["place"] == 1 else 0.32)
    out = out[:n]
    tt = np.arange(n) / SR
    mix_ = out + reverb(out, 1.2) * 0.2
    mix_ *= np.clip((race.total - tt) / 1.2, 0, 1)
    peak = np.max(np.abs(mix_)) or 1
    mix_ = np.tanh(mix_ / peak * 1.2) * 0.9
    st = np.stack([mix_, mix_], axis=1)
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((st * 32767).astype(np.int16).tobytes())


# ------------------------------------------------------------------ Ausgabe

_R = None


def _chunk(args):
    i0, i1, out = args
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W1}x{H1}", "-r", str(FPS),
           "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-maxrate", "12M", "-bufsize", "24M",
           "-pix_fmt", "yuv420p", "-g", "120", out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(i0, i1):
        proc.stdin.write(_R.frame(i).tobytes())
    proc.stdin.close()
    proc.wait()
    return out


def render(race, name, out_path, preview_dir=None, preview_times=(), jobs=4):
    global _R
    _R = RaceRenderer(race)
    n = race.n_frames
    if preview_dir:
        os.makedirs(preview_dir, exist_ok=True)
        for pt in preview_times:
            fr = _R.frame(min(int(pt * FPS), n - 1))
            base = os.path.join(preview_dir, f"{name}_vertical_{pt:05.1f}")
            fr.save(base + ".png")
            tiktok_ui_mock(fr, "vertical").save(base + "_ui.png")
        return
    with tempfile.TemporaryDirectory() as tmp:
        bounds = [n * j // jobs for j in range(jobs + 1)]
        parts = [(bounds[j], bounds[j + 1], os.path.join(tmp, f"p{j}.mp4")) for j in range(jobs)]
        with Pool(jobs) as p:
            p.map(_chunk, parts)
        lst = os.path.join(tmp, "list.txt")
        with open(lst, "w") as f:
            for _, _, pth in parts:
                f.write(f"file '{pth}'\n")
        wav = os.path.join(tmp, "a.wav")
        build_audio(race, wav)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-i", wav, "-c:v", "copy",
                        "-c:a", "aac", "-b:a", "192k", "-shortest", "-af", "loudnorm=I=-14:TP=-1.5:LRA=11",
                        "-movflags", "+faststart", out_path], check=True)


RACES = {"r01_sorting_race": lambda: SortRace(SEED_RACE)}
SEED_RACE = 9

if __name__ == "__main__":
    import sys
    preview = "--preview" in sys.argv
    names = [a for a in sys.argv[1:] if not a.startswith("--")] or list(RACES)
    for name in names:
        race = RACES[name]()
        print(name, "ops", dict(zip(race.names, race.counts)), "Ziel",
              [round(f - race.lead, 1) for f in race.finish])
        if preview:
            ts = [1.5, 6.0, 12.0, 18.0, 24.0, 30.0, 36.0, 42.0, 48.0, race.total - 0.5]
            render(race, name, None, preview_dir="output/preview", preview_times=ts)
        else:
            out = f"output/{name}_vertical.mp4"
            render(race, name, out)
            print(out)
