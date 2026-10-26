"""Töne für Satisfying-Videos: Marimba-Melodie bei jedem Aufprall, Glas, Whomp, Fanfare."""
import wave

import numpy as np

from tiktok.audio import whoosh

SR = 44100

# Akkordfolge Am - F - C - G, jede Note ein Aufprall -> es entsteht eine Melodie
_A3 = 220.0


def _hz(semi):
    return _A3 * 2 ** (semi / 12)


_CHORDS = [(0, 3, 7), (-4, 0, 3), (3, 7, 10), (-2, 2, 5)]
_PATTERN = [0, 1, 2, 3, 4, 3, 2, 1]


def melody_note(i):
    chord = _CHORDS[(i // len(_PATTERN)) % len(_CHORDS)]
    step = _PATTERN[i % len(_PATTERN)]
    semi = chord[step % 3] + 12 * (step // 3) + 12
    return _hz(semi)


_cache = {}


def marimba(f):
    if f not in _cache:
        n = int(SR * 0.7)
        t = np.arange(n) / SR
        s = np.sin(2 * np.pi * f * t) * np.exp(-t * 6) + 0.25 * np.sin(2 * np.pi * f * 4 * t) * np.exp(-t * 30) \
            + 0.12 * np.sin(2 * np.pi * f * 2 * t) * np.exp(-t * 12)
        s *= np.minimum(t / 0.003, 1)
        _cache[f] = s * 0.5
    return _cache[f]


def glass(rng):
    n = int(SR * 0.6)
    t = np.arange(n) / SR
    noise = np.diff(rng.standard_normal(n + 1)) * np.exp(-t * 14) * 0.25
    pings = sum(np.sin(2 * np.pi * rng.uniform(2200, 4200) * t) * np.exp(-t * rng.uniform(8, 16)) * 0.12 for _ in range(4))
    return noise + pings


def whomp():
    n = int(SR * 0.6)
    t = np.arange(n) / SR
    f = 500 * np.exp(-t * 5) + 70
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 4) * 0.7


def fanfare():
    out = np.zeros(int(SR * 2.6))
    for k, semi in enumerate((3, 7, 10, 15, 19)):  # C-E-G-C-E über A3
        f = _hz(semi + 12)
        s = marimba(f) * 1.3
        i = int(k * 0.09 * SR)
        out[i:i + len(s)] += s[: len(out) - i]
    t = np.arange(len(out)) / SR
    chord = sum(np.sin(2 * np.pi * _hz(x + 12) * t) for x in (3, 7, 10)) * np.exp(-t * 1.2) * 0.18
    return out + chord * np.minimum(t / 0.4, 1)


def build(total, events, path, seed=3):
    rng = np.random.default_rng(seed)
    mix = np.zeros(int(SR * (total + 0.5)))
    n_note = 0
    for t, kind, d in events:
        if kind == "note":
            s = marimba(melody_note(n_note))
            n_note += 1
        elif kind == "break":
            g, m = glass(rng), marimba(melody_note(n_note) * 2) * 0.6
            s = np.zeros(max(len(g), len(m)))
            s[: len(g)] += g
            s[: len(m)] += m
        elif kind == "pop":
            s = marimba(melody_note(n_note) * 2) * 0.5
            n_note += 1
        elif kind == "elim":
            s = whomp()
        elif kind == "finish":
            s = fanfare()
        elif kind == "whoosh":
            s = whoosh(rng) * 0.6
        else:
            continue
        i = int(t * SR)
        j = min(len(mix), i + len(s))
        mix[i:j] += s[: j - i]
    mix = mix[: int(SR * total)]
    peak = np.max(np.abs(mix)) or 1
    mix = np.tanh(mix / peak * 2.2) * 0.9
    st = np.stack([mix, mix], axis=1)
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((st * 32767).astype(np.int16).tobytes())
