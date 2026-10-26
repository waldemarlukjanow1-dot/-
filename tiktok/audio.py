"""Synthetischer Soundtrack: Spannungs-Drone, Herzschlag, Ticks, Whooshes, Hits."""
import wave

import numpy as np

SR = 44100


def _env(n, attack, decay):
    t = np.arange(n) / SR
    return np.minimum(t / max(attack, 1e-4), 1.0) * np.exp(-t / decay)


def whoosh(rng, dur=0.7):
    n = int(SR * dur)
    noise = rng.standard_normal(n)
    t = np.linspace(0, 1, n)
    # gleitender Bandpass über zwei Tiefpässe mit wechselnder Grenzfrequenz
    lp = np.convolve(noise, np.ones(6) / 6, mode="same")
    lp2 = np.convolve(noise, np.ones(40) / 40, mode="same")
    mix = lp * t + lp2 * (1 - t)
    env = np.sin(np.pi * t) ** 2
    return mix * env * 0.9


def hit(rng, dur=1.2):
    n = int(SR * dur)
    t = np.arange(n) / SR
    freq = 40 + 110 * np.exp(-t * 18)
    body = np.sin(2 * np.pi * np.cumsum(freq) / SR) * np.exp(-t * 3.2)
    click = rng.standard_normal(n) * np.exp(-t * 60) * 0.5
    return (body * 1.1 + click) * 0.9


def tick(rng, high=False):
    n = int(SR * 0.08)
    t = np.arange(n) / SR
    f = 2400 if high else 1700
    return (np.sin(2 * np.pi * f * t) * np.exp(-t * 90) + rng.standard_normal(n) * np.exp(-t * 200) * 0.3) * 0.5


def heartbeat():
    n = int(SR * 0.6)
    t = np.arange(n) / SR
    thump = lambda d: np.sin(2 * np.pi * 55 * (t - d)) * np.exp(-np.maximum(t - d, 0) * 14) * (t >= d)
    return (thump(0) + 0.7 * thump(0.22)) * 0.9


def ding():
    n = int(SR * 1.6)
    t = np.arange(n) / SR
    return sum(np.sin(2 * np.pi * f * t) * a for f, a in ((880, 0.5), (1320, 0.3), (1760, 0.15))) * np.exp(-t * 2.5) * 0.7


def buzzer():
    n = int(SR * 0.45)
    t = np.arange(n) / SR
    sq = np.sign(np.sin(2 * np.pi * 110 * t)) + 0.5 * np.sign(np.sin(2 * np.pi * 116 * t))
    return sq * _env(n, 0.005, 0.25) * 0.22


def riser(rng, dur=2.0):
    n = int(SR * dur)
    t = np.linspace(0, 1, n)
    f = 200 + 900 * t ** 2
    tone = np.sin(2 * np.pi * np.cumsum(f) / SR) * 0.25
    noise = np.convolve(rng.standard_normal(n), np.ones(5) / 5, mode="same") * 0.4
    return (tone + noise) * t ** 2


def drone(total, tension):
    """Dunkle Hintergrund-Fläche. tension(t)->0..1 hebt Lautstärke/Puls an."""
    n = int(SR * total)
    t = np.arange(n) / SR
    out = np.zeros(n)
    for f, a in ((55, 0.35), (82.4, 0.22), (110.3, 0.15), (164.8, 0.07), (54.6, 0.2)):
        out += a * np.sin(2 * np.pi * f * t + np.sin(2 * np.pi * 0.1 * t) * 2)
    lfo = 0.7 + 0.3 * np.sin(2 * np.pi * 0.25 * t)
    ten = np.array([tension(x) for x in t[:: SR // 100]])
    ten = np.interp(t, t[:: SR // 100], ten)
    pulse = 0.5 + 0.5 * np.sign(np.sin(2 * np.pi * 2 * t))
    out = out * lfo * (0.5 + 0.5 * ten) + 0.08 * ten * pulse * np.sin(2 * np.pi * 220 * t)
    fade = np.minimum(1, np.minimum(t / 0.3, (total - t) / 0.6))
    return out * 0.32 * np.clip(fade, 0, 1)


def build_track(total, events, tension, path, seed=7):
    """events: Liste (zeit, name). Schreibt eine 16-bit Stereo-WAV."""
    rng = np.random.default_rng(seed)
    mix = drone(total, tension)
    n = len(mix)
    gens = {
        "whoosh": lambda: whoosh(rng), "hit": lambda: hit(rng), "tick": lambda: tick(rng),
        "tick_hi": lambda: tick(rng, True), "heart": heartbeat, "ding": ding, "buzzer": buzzer,
        "riser": lambda: riser(rng), "pop": lambda: tick(rng, True) * 0.5,
    }
    for at, name in events:
        s = gens[name]()
        i = int(at * SR)
        if name == "riser":
            i -= len(s)  # Riser endet genau am Ereignis
        if i < 0:
            s, i = s[-i:], 0
        j = min(n, i + len(s))
        mix[i:j] += s[: j - i]
    peak = np.max(np.abs(mix)) or 1
    mix = np.tanh(mix / peak * 1.4) * 0.85
    stereo = np.stack([mix, mix], axis=1)
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((stereo * 32767).astype(np.int16).tobytes())
