"""Neon-Renderer für die Satisfying-Simulationen (Hoch- und Querformat, TikTok-Safe-Zones)."""
import math
import os
import random
import subprocess
import tempfile

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from tiktok import fx
from tiktok.render import YELLOW, RED, back_out, composite, ease, font, paste_center, text_sprite, tiktok_ui_mock, wrap
from . import sound
from .sim import ARENA, FPS, hue_rgb

# Alles Wichtige liegt in "safe": rechts frei für Like/Kommentar/Teilen, oben für "Für dich",
# unten für Username/Beschreibung. Im Querformat stehen Titel links und Punkte rechts neben der Arena.
LAYOUTS = {
    "vertical": dict(W=1080, H=1920, safe=(60, 230, 920, 1420), arena=(490, 880, 365),
                     title=(490, 245, 840, 64, 2), sub=(490, 448, 48), board="row", board_y=1345,
                     panel=(60, 920)),
    "landscape": dict(W=1920, H=1080, safe=(90, 50, 1650, 1020), arena=(870, 540, 410),
                      title=(270, 150, 340, 54, 4), sub=(270, None, 38), board="col", board_y=250,
                      panel=(1310, 1640)),
}


class Scene:
    def __init__(self, sim, spec, fmt):
        self.sim, self.spec, self.fmt_name = sim, spec, fmt
        L = self.L = LAYOUTS[fmt]
        self.W, self.H = L["W"], L["H"]
        self.cx, self.cy, ar = L["arena"]
        self.s = ar / ARENA
        yy, xx = np.mgrid[0:self.H, 0:self.W].astype(np.float32)
        d = np.sqrt((xx - self.cx) ** 2 + (yy - self.cy) ** 2) / (ar * 2.2)
        g = np.clip(1 - d, 0, 1)[..., None] ** 1.5
        self.bg = (np.array([8, 6, 16], np.float32) + g * np.array([22, 16, 48], np.float32))
        self.ev = sim.events
        self.crown = fx.emoji_image("👑", 46 if fmt == "vertical" else 40)
        self._spr = {}

    # ------------------------------------------------------------ Hilfen

    def P(self, x, y):
        return self.cx + x * self.s, self.cy + y * self.s

    def events_of(self, kind, t, max_age=1e9):
        return [(et, d) for et, k, d in self.ev if k == kind and et <= t and t - et < max_age]

    def sprite(self, key, *a, **kw):
        if key not in self._spr:
            self._spr[key] = text_sprite(*a, **kw)
        return self._spr[key]

    def ball(self, d, x, y, r, col, outline=(255, 255, 255)):
        X, Y = self.P(x, y)
        R = r * self.s
        d.ellipse([X - R, Y - R, X + R, Y + R], fill=col, outline=outline, width=max(2, int(R * 0.12)))
        h = R * 0.35
        d.ellipse([X - R * 0.45 - h / 2, Y - R * 0.45 - h / 2, X - R * 0.45 + h / 2, Y - R * 0.45 + h / 2],
                  fill=tuple(min(255, c + 90) for c in col))

    def trail(self, d, fi, idx, n=7):
        for k in range(n, 0, -1):
            j = fi - k * 1
            if j < 0:
                continue
            x, y, r, alive, col = self.sim.frames[j]["balls"][idx]
            f = (1 - k / (n + 1)) * 0.55
            X, Y = self.P(x, y)
            R = r * self.s * (0.6 + 0.4 * (1 - k / n))
            d.ellipse([X - R, Y - R, X + R, Y + R], fill=tuple(int(c * f) for c in col))

    def particles(self, d, t, t0, origin, n, seed, colors, speed=(250, 520), life=1.4, size=6, ring=None):
        age = t - t0
        if age < 0 or age > life:
            return
        rng = random.Random(seed)
        fade = 1 - age / life
        for i in range(n):
            if ring is not None:
                R, gap, half = ring
                a = gap + half + (2 * math.pi - 2 * half) * (i + 0.5) / n
                ox, oy = math.cos(a) * R, math.sin(a) * R
                va = a + rng.uniform(-0.3, 0.3)
            else:
                ox, oy = origin
                a = va = rng.uniform(0, 2 * math.pi)
            sp = rng.uniform(*speed)
            x = ox + math.cos(va) * sp * age
            y = oy + math.sin(va) * sp * age + 0.5 * 900 * age ** 2
            col = colors[i % len(colors)]
            col = tuple(int(c * fade) for c in col)
            X, Y = self.P(x, y)
            if ring is not None:  # Ringsplitter als rotierende Linien
                L = 2 * math.pi * R / n * 0.42 * self.s
                rot = a + math.pi / 2 + age * rng.uniform(-8, 8)
                d.line([X - math.cos(rot) * L, Y - math.sin(rot) * L, X + math.cos(rot) * L, Y + math.sin(rot) * L],
                       fill=col, width=max(3, int(5 * self.s)))
            else:
                S = size * self.s * (0.5 + rng.random())
                d.ellipse([X - S, Y - S, X + S, Y + S], fill=col)

    def confetti(self, img, t, t0):
        age = t - t0
        if age < 0 or age > 4.5 or t >= self.teaser_start():
            return
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        rng = random.Random(42)
        cols = [(255, 60, 70), (50, 140, 255), (40, 230, 110), (255, 215, 30), (190, 80, 255), (255, 255, 255)]
        ar = self.L["arena"][2]
        for i in range(150):
            x0 = self.cx + rng.uniform(-ar, ar)
            y0 = self.cy - ar * rng.uniform(0.3, 1.1)
            vx, vy = rng.uniform(-260, 260), rng.uniform(-700, -150)
            x = x0 + vx * age
            y = y0 + vy * age + 0.5 * 800 * age ** 2
            rot = age * rng.uniform(-9, 9) + rng.random() * 6
            s1, s2 = rng.uniform(7, 13), rng.uniform(3, 6)
            pts = [(x + math.cos(rot) * a - math.sin(rot) * b, y + math.sin(rot) * a + math.cos(rot) * b)
                   for a, b in ((-s1, -s2), (s1, -s2), (s1, s2), (-s1, s2))]
            d.polygon(pts, fill=cols[i % len(cols)] + (int(255 * min(1, (4.5 - age) / 1)),))
        img.alpha_composite(layer)

    # ------------------------------------------------------------ Texte

    def title(self, img, t):
        x, y, maxw, size, max_lines = self.L["title"]
        text = self.spec["title"]
        while True:
            lines = wrap(text, font(size), maxw, stroke=8)
            if len(lines) <= max_lines or size < 28:
                break
            size -= 2
        pulse = 1 + 0.02 * math.sin(t * 4)
        for i, ln in enumerate(lines):
            col = YELLOW if i == len(lines) - 1 else (255, 255, 255)
            spr = self.sprite(("title", ln, size, col), ln, size, fill=col, stroke=8)
            paste_center(img, spr, x, y + size * 0.62 + i * size * 1.1, scale=pulse)
        self.title_bottom = y + len(lines) * size * 1.1

    def sub(self, img, t, text, col=(255, 255, 255), pulse=False, since=None):
        if not text:
            return
        x, y, size = self.L["sub"]
        if y is None:
            y = self.title_bottom + 50
        maxw = self.L["title"][2]
        lines = wrap(text, font(size), maxw, stroke=6) if self.fmt_name == "landscape" else [text]
        sc = 1.0
        if pulse:
            sc = 1 + 0.06 * math.sin(t * 9)
        if since is not None:
            sc *= 1 + 0.3 * (1 - ease((t - since) / 0.2))
        for i, ln in enumerate(lines):
            spr = self.sprite(("sub", ln, size, col), ln, size, fill=col, stroke=6)
            fit = min(1.0, (maxw + 20) / spr.width)
            paste_center(img, spr, x, y + i * size * 1.15, scale=sc * fit)

    def teaser_start(self):
        return self.sim.finish_t + self.sim.outro - 2.7

    def overlay_winner(self, img, t, t0, text, col):
        if t < t0 or t >= self.teaser_start():
            return
        spr = self.sprite(("win", text), text, 92, fill=col, stroke=12)
        maxw = (self.L["safe"][2] - self.L["safe"][0]) - 40 if self.fmt_name == "vertical" else self.L["arena"][2] * 2.1
        sc = min(1.0, maxw / spr.width) * back_out((t - t0) / 0.45)
        paste_center(img, spr, self.cx, self.cy, scale=max(0.01, sc), angle=-4)

    def teaser(self, img, t, t0, big, line):
        if t < t0:
            return
        a = ease((t - t0) / 0.4)
        img.alpha_composite(Image.new("RGBA", img.size, (0, 0, 0, int(200 * a))))
        ar = self.L["arena"][2]
        k = self.sprite(("tk",), "NÄCHSTES VIDEO", 44, fill=(255, 255, 255), stroke=6)
        paste_center(img, k, self.cx, self.cy - ar * 0.55, alpha=a)
        b = self.sprite(("tb", big), big, 150, fill=YELLOW, stroke=14)
        paste_center(img, b, self.cx, self.cy - ar * 0.15, scale=max(0.01, back_out((t - t0 - 0.1) / 0.45)))
        maxw = (self.L["safe"][2] - self.L["safe"][0]) - 40 if self.fmt_name == "vertical" else ar * 2
        l = self.sprite(("tl", line), line, 56, fill=(255, 255, 255), stroke=8)
        paste_center(img, l, self.cx, self.cy + ar * 0.3, scale=min(1, maxw / l.width), alpha=ease((t - t0 - 0.3) / 0.3))
        f = self.sprite(("tf",), "FOLGE MIR, UM ES ZU SEHEN!", 48, fill=YELLOW, stroke=7)
        paste_center(img, f, self.cx, self.cy + ar * 0.62, scale=min(1, maxw / f.width) * (1 + 0.04 * math.sin(t * 7)),
                     alpha=ease((t - t0 - 0.6) / 0.3))

    def board(self, img, t, items):
        """items: [(farbe, text oder None, hervorheben, ausgeschieden, pop)]"""
        n = len(items)
        d = ImageDraw.Draw(img)
        if self.L["board"] == "row":
            x0, x1 = self.L["panel"]
            gap = 14
            w = (x1 - x0 - gap * (n - 1)) / n
            h = 86
            cy = self.L["board_y"]
            pos = [(x0 + i * (w + gap) + w / 2, cy) for i in range(n)]
        else:
            x0, x1 = self.L["panel"]
            w, h, gap = x1 - x0, 78, 18
            pos = [(x0 + w / 2, self.L["board_y"] + i * (h + gap) + h / 2) for i in range(n)]
        for (col, txt, hl, out, pop), (px, py) in zip(items, pos):
            spr = Image.new("RGBA", (int(w) + 8, h + 8), (0, 0, 0, 0))
            sd = ImageDraw.Draw(spr)
            sd.rounded_rectangle([4, 4, w + 4, h + 4], radius=h // 2, fill=(18, 16, 30, 230),
                                 outline=YELLOW if hl else col, width=7 if hl else 4)
            r = h * 0.3
            bx = 4 + h / 2 if txt is not None else 4 + w / 2
            sd.ellipse([bx - r, 4 + h / 2 - r, bx + r, 4 + h / 2 + r], fill=col, outline=(255, 255, 255), width=3)
            if txt is not None:
                sd.text((4 + (h + w) / 2, 4 + h / 2), txt, font=font(h * 0.5), fill=(255, 255, 255), anchor="mm")
            if out:
                dim = Image.new("RGBA", spr.size, (0, 0, 0, 150))
                spr = Image.composite(Image.alpha_composite(spr, dim), spr, spr.getchannel("A"))
                sd = ImageDraw.Draw(spr)
                q = h * 0.28
                for s in (1, -1):
                    sd.line([bx - q, 4 + h / 2 - q * s, bx + q, 4 + h / 2 + q * s], fill=RED, width=8)
            paste_center(img, spr, px, py, scale=1 + 0.3 * pop)
            if hl:
                paste_center(img, self.crown, px, py - h * 0.72 if self.L["board"] == "row" else py, scale=1)
        return pos

    # ------------------------------------------------------------ Frame

    def frame(self, fi):
        t = fi / FPS
        geo = Image.new("RGB", (self.W, self.H), (0, 0, 0))
        d = ImageDraw.Draw(geo)
        self.draw_geo(d, fi, t)
        arr = np.asarray(geo, np.float32)
        small = geo.resize((self.W // 4, self.H // 4), Image.BILINEAR).filter(ImageFilter.GaussianBlur(5))
        glow = np.asarray(small.resize((self.W, self.H), Image.BILINEAR), np.float32)
        out = np.clip(self.bg + arr + glow * 1.6, 0, 255).astype(np.uint8)
        img = Image.fromarray(out, "RGB").convert("RGBA")
        self.title(img, t)
        self.draw_ui(img, fi, t)
        return img.convert("RGB")


class RingScene(Scene):
    def draw_geo(self, d, fi, t):
        sim, fr = self.sim, self.sim.frames[fi]
        for k in range(fr["k"], len(sim.rings)):
            ring = sim.rings[k]
            R = ring["R"] * self.s
            g = sim.gap(ring, t)
            a0, a1 = math.degrees(g + ring["half"]), math.degrees(g - ring["half"]) + 360
            inner = k == fr["k"]
            col = ring["color"] if inner else tuple(int(c * 0.75) for c in ring["color"])
            w = max(4, int((7 if inner else 4) * self.s))
            d.arc([self.cx - R, self.cy - R, self.cx + R, self.cy + R], a0, a1, fill=col, width=w)
        for et, ev in self.events_of("break", t, 1.5):
            ring = sim.rings[ev["ring"]]
            self.particles(d, t, et, None, 30, ev["ring"], [ring["color"]], ring=(ring["R"], ev["gap"], ring["half"]))
        for i in range(len(fr["balls"])):
            self.trail(d, fi, i)
        for x, y, r, alive, col in fr["balls"]:
            self.ball(d, x, y, r, col)

    def draw_ui(self, img, fi, t):
        sim, fr = self.sim, self.sim.frames[fi]
        n = len(sim.rings)
        left = n - fr["k"]
        fin = sim.finish_t
        for et, ev in self.events_of("break", t, 0.8):  # "+1" über der Kugel
            x, y = sim.frames[min(int(et * FPS), len(sim.frames) - 1)]["balls"][ev["ball"]][:2]
            X, Y = self.P(x, y)
            age = t - et
            col = sim.balls[ev["ball"]].color
            spr = self.sprite(("p1", col), "+1", 46, fill=col, stroke=6)
            paste_center(img, spr, X, Y - 40 - age * 90, alpha=1 - age / 0.8)
        if fin is None or t < fin:
            if t < 7:
                self.sub(img, t, "Schreib deine Farbe in die Kommentare!")
            elif left == 1:
                self.sub(img, t, "LETZTER RING!", RED, pulse=True)
            elif left <= 3:
                self.sub(img, t, f"NUR NOCH {left} RINGE!", RED, pulse=True)
            else:
                self.sub(img, t, f"NOCH {left} RINGE")
        score = fr["score"]
        m = max(score)
        unique = score.count(m) == 1 and m > 0
        items = []
        for i, b in enumerate(sim.balls):
            last = [et for et, ev in self.events_of("break", t, 0.3) if ev["ball"] == i]
            pop = 1 - ease((t - last[-1]) / 0.3) if last else 0
            items.append((b.color, str(score[i]), unique and score[i] == m, False, pop))
        self.board(img, t, items)
        if fin is not None and t >= fin:
            w = sim.winner()
            self.confetti(img, t, fin)
            self.overlay_winner(img, t, fin, f"{sim.balls[w].name} GEWINNT!", sim.balls[w].color)
            self.teaser(img, t, fin + sim.outro - 2.7, *self.spec["teaser"])


class GrowScene(Scene):
    def draw_geo(self, d, fi, t):
        sim, fr = self.sim, self.sim.frames[fi]
        R = ARENA * self.s + 6
        fin = sim.finish_t
        exploded = fin is not None and t >= fin
        d.ellipse([self.cx - R, self.cy - R, self.cx + R, self.cy + R], outline=(235, 235, 255), width=max(5, int(6 * self.s)))
        if not exploded:
            for k in range(26, 0, -2):  # Spur aus Umrissen
                j = fi - k
                if j < 0:
                    continue
                f = sim.frames[j]
                X, Y = self.P(f["x"], f["y"])
                r = f["r"] * self.s
                col = tuple(int(c * (1 - k / 28) * 0.8) for c in hue_rgb(f["hue"]))
                d.ellipse([X - r, Y - r, X + r, Y + r], outline=col, width=3)
            X, Y = self.P(fr["x"], fr["y"])
            r = fr["r"] * self.s
            d.ellipse([X - r, Y - r, X + r, Y + r], fill=hue_rgb(fr["hue"], 0.8, 0.9), outline=(255, 255, 255),
                      width=max(3, int(r * 0.05)))
        else:
            cols = [hue_rgb(fr["hue"] + k * 0.08) for k in range(8)]
            self.particles(d, t, fin, (fr["x"], fr["y"]), 160, 7, cols, speed=(150, 900), life=2.5, size=10)

    def draw_ui(self, img, fi, t):
        sim, fr = self.sim, self.sim.frames[fi]
        fin = sim.finish_t
        if fin is not None and fin <= t < fin + 0.18:
            img.alpha_composite(Image.new("RGBA", img.size, (255, 255, 255, int(220 * (1 - (t - fin) / 0.18)))))
        if fin is None or t < fin:
            if t < 6:
                self.sub(img, t, "Jeder Aufprall macht ihn größer…")
            elif fr["pct"] > 0.82:
                self.sub(img, t, "GLEICH IST ES SO WEIT!", RED, pulse=True)
            elif 14 < t < 22:
                self.sub(img, t, "Warte bis zum Ende…")
        self.progress(img, fr)
        if fin is not None and t >= fin:
            self.confetti(img, t, fin)
            self.overlay_winner(img, t, fin + 0.15, "GESCHAFFT!", YELLOW)
            if fin + 0.6 <= t < self.teaser_start():
                s = self.sprite(("cnt", fr["bounces"]), f"{fr['bounces']} AUFPRALLE", 58, fill=(255, 255, 255), stroke=8)
                paste_center(img, s, self.cx, self.cy + 110, scale=back_out((t - fin - 0.6) / 0.4))
            self.teaser(img, t, fin + sim.outro - 2.7, *self.spec["teaser"])

    def progress(self, img, fr):
        d = ImageDraw.Draw(img)
        pct = max(0.0, min(1.0, fr["pct"]))
        x0, x1 = self.L["panel"]
        if self.L["board"] == "row":
            y = self.L["board_y"] - 22
            h = 44
            bar = (x0 + 250, y, x1, y + h)
            label_pos = (x0 + 120, y + h / 2)
        else:
            y = self.L["board_y"] + 60
            h = 40
            bar = (x0, y, x1, y + h)
            label_pos = ((x0 + x1) / 2, y - 50)
        txt = f"{fr['bounces']} HITS"
        spr = self.sprite(("hits", txt), txt, 40, fill=(255, 255, 255), stroke=6)
        paste_center(img, spr, *label_pos)
        d.rounded_rectangle(bar, radius=h // 2, fill=(30, 26, 48), outline=(255, 255, 255), width=3)
        fw = (bar[2] - bar[0] - 8) * pct
        if fw > h - 8:
            d.rounded_rectangle([bar[0] + 4, bar[1] + 4, bar[0] + 4 + fw, bar[3] - 4], radius=(h - 8) // 2, fill=hue_rgb(fr["hue"]))
        pt = f"{int(pct * 100)}%"
        d.text(((bar[0] + bar[2]) / 2, (bar[1] + bar[3]) / 2), pt, font=font(h * 0.62), fill=(255, 255, 255),
               anchor="mm", stroke_width=4, stroke_fill=(0, 0, 0))


class ElimScene(Scene):
    def draw_geo(self, d, fi, t):
        sim, fr = self.sim, self.sim.frames[fi]
        R = ARENA * self.s + 8
        g = sim.gap(t)
        a0, a1 = math.degrees(g + sim.half), math.degrees(g - sim.half) + 360
        d.arc([self.cx - R, self.cy - R, self.cx + R, self.cy + R], a0, a1, fill=(120, 230, 255), width=max(8, int(12 * self.s)))
        for et, ev in self.events_of("elim", t, 1.4):
            x, y = sim.frames[min(int(et * FPS), len(sim.frames) - 1)]["balls"][ev["ball"]][:2]
            self.particles(d, t, et, (x, y), 40, ev["ball"], [sim.balls[ev["ball"]].color], speed=(200, 600), life=1.2)
        for i, (x, y, r, alive, col) in enumerate(fr["balls"]):
            if alive:
                self.trail(d, fi, i)
        for x, y, r, alive, col in fr["balls"]:
            if alive:
                self.ball(d, x, y, r, col)
            else:
                X, Y = self.P(x, y)
                if Y < self.H + 60:  # fällt aus dem Bild
                    self.ball(d, x, y, r, tuple(int(c * 0.45) for c in col), outline=(110, 110, 110))

    def draw_ui(self, img, fi, t):
        sim, fr = self.sim, self.sim.frames[fi]
        fin = sim.finish_t
        alive = [i for i, b in enumerate(fr["balls"]) if b[3]]
        recent = self.events_of("elim", t, 1.8)
        if fin is None or t < fin:
            if recent:
                et, ev = recent[-1]
                b = sim.balls[ev["ball"]]
                self.sub(img, t, f"{b.name} IST RAUS!", b.color, since=et)
            elif t < 7:
                self.sub(img, t, "Schreib deine Farbe in die Kommentare!")
            elif len(alive) == 2:
                a, b = (sim.balls[i].name for i in alive)
                self.sub(img, t, f"FINALE: {a} GEGEN {b}!", RED, pulse=True)
            else:
                self.sub(img, t, f"NOCH {len(alive)} IM SPIEL")
        items = []
        for i, b in enumerate(sim.balls):
            out_t = [et for et, ev in self.events_of("elim", t) if ev["ball"] == i]
            pop = 1 - ease((t - out_t[0]) / 0.3) if out_t else 0
            win = fin is not None and t >= fin and i in alive
            txt = b.name if self.L["board"] == "col" else None
            items.append((b.color, txt, win, bool(out_t), pop))
        self.board(img, t, items)
        if fin is not None and t >= fin:
            w = alive[0]
            self.confetti(img, t, fin)
            self.overlay_winner(img, t, fin, f"{sim.balls[w].name} GEWINNT!", sim.balls[w].color)
            self.teaser(img, t, fin + sim.outro - 2.7, *self.spec["teaser"])


SCENES = {"ring": RingScene, "grow": GrowScene, "elim": ElimScene}


def render(sim, spec, fmt, out_path, preview_dir=None, preview_times=()):
    scene = SCENES[spec["kind"]](sim, spec, fmt)
    n = len(sim.frames)
    if preview_dir:
        os.makedirs(preview_dir, exist_ok=True)
        for pt in preview_times:
            fi = min(int(pt * FPS), n - 1)
            fr = scene.frame(fi)
            base = os.path.join(preview_dir, f"{spec['id']}_{fmt}_{pt:05.1f}")
            fr.save(base + ".png")
            tiktok_ui_mock(fr, fmt).save(base + "_ui.png")
        return
    events = list(sim.events)
    if sim.finish_t is not None:
        events.append((sim.finish_t + sim.outro - 2.7, "whoosh", {}))
    with tempfile.TemporaryDirectory() as tmp:
        wav = os.path.join(tmp, "a.wav")
        sound.build(n / FPS, events, wav)
        cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{scene.W}x{scene.H}",
               "-r", str(FPS), "-i", "-", "-i", wav, "-c:v", "libx264", "-preset", "medium", "-crf", "20",
               "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", out_path]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        for fi in range(n):
            proc.stdin.write(scene.frame(fi).tobytes())
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError("ffmpeg fehlgeschlagen")
