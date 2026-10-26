"""Renderer für Tür-Rätsel-Videos (TikTok).

Ein Episode-Skript (siehe episodes/) beschreibt Türen + eine Liste von Segmenten.
Der Renderer erzeugt daraus Bilder (Pillow/numpy), einen Soundtrack und muxt alles
mit ffmpeg zu einer MP4. Alle wichtigen Elemente liegen innerhalb der TikTok-Safe-Zone,
damit weder Like-/Kommentar-Buttons rechts noch Beschreibung/Username unten etwas verdecken.
"""
import math
import os
import subprocess
import tempfile

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from . import audio, fx

FPS = 30
FONT_BLACK = "/usr/share/fonts/opentype/inter/Inter-Black.otf"
FONT_BOLD = "/usr/share/fonts/opentype/inter/Inter-ExtraBold.otf"
YELLOW = (255, 214, 0)
RED = (235, 40, 40)
GREEN = (40, 210, 90)

# safe = Bereich, in dem nichts von der TikTok-Oberfläche liegt (x0, y0, x1, y1).
# Hochformat: oben "Folge ich | Für dich", rechts Like/Kommentar/Teilen-Leiste,
# unten Username + Beschreibung + Musik. Querformat: TikTok zeigt das Video als Band
# in der Bildschirmmitte, die Button-Leiste ragt rechts ins untere Video hinein.
FORMATS = {
    "vertical": dict(W=1080, H=1920, safe=(60, 230, 920, 1420), series_y=250, title_y=310,
                     title_size=64, series_size=32, doors_box=(80, 520, 900, 1200),
                     cap_y=1365, cap_size=78),
    "landscape": dict(W=1920, H=1080, safe=(90, 50, 1650, 1020), series_y=62, title_y=110,
                      title_size=58, series_size=28, doors_box=(170, 285, 1570, 870),
                      cap_y=962, cap_size=68),
}


def ease(p):
    p = min(max(p, 0.0), 1.0)
    return p * p * (3 - 2 * p)


def back_out(p):
    p = min(max(p, 0.0), 1.0)
    c = 1.9
    return 1 + (c + 1) * (p - 1) ** 3 + c * (p - 1) ** 2


_font_cache = {}


def font(size, path=FONT_BLACK):
    key = (path, int(size))
    if key not in _font_cache:
        _font_cache[key] = ImageFont.truetype(path, int(size))
    return _font_cache[key]


def perspective_coeffs(dst, src):
    """Koeffizienten für Image.PERSPECTIVE: Ausgabe-Punkte dst -> Quell-Punkte src."""
    m = []
    for (x, y), (u, v) in zip(dst, src):
        m.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        m.append([0, 0, 0, x, y, 1, -v * x, -v * y])
    A = np.array(m, float)
    B = np.array(src, float).reshape(8)
    return np.linalg.solve(A, B).tolist()


def text_sprite(text, size, fill=(255, 255, 255), stroke=8, stroke_fill=(0, 0, 0), path=FONT_BLACK, shadow=True):
    f = font(size, path)
    l, t, r, b = f.getbbox(text, stroke_width=stroke)
    pad = stroke + 12
    im = Image.new("RGBA", (r - l + pad * 2, b - t + pad * 2), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    if shadow:
        sh = Image.new("RGBA", im.size, (0, 0, 0, 0))
        ImageDraw.Draw(sh).text((pad - l + 4, pad - t + 6), text, font=f, fill=(0, 0, 0, 170),
                                stroke_width=stroke, stroke_fill=(0, 0, 0, 170))
        im = Image.alpha_composite(im, sh.filter(ImageFilter.GaussianBlur(5)))
        d = ImageDraw.Draw(im)
    d.text((pad - l, pad - t), text, font=f, fill=fill, stroke_width=stroke, stroke_fill=stroke_fill)
    return im


def composite(base, sprite, x, y):
    """alpha_composite, das Sprites am Bildrand sauber beschneidet."""
    x, y = int(x), int(y)
    cx0, cy0 = max(0, -x), max(0, -y)
    cx1, cy1 = min(sprite.width, base.width - x), min(sprite.height, base.height - y)
    if cx1 <= cx0 or cy1 <= cy0:
        return
    base.alpha_composite(sprite.crop((cx0, cy0, cx1, cy1)), (x + cx0, y + cy0))


def paste_center(base, sprite, cx, cy, scale=1.0, alpha=1.0, angle=0.0):
    if scale != 1.0:
        sprite = sprite.resize((max(1, int(sprite.width * scale)), max(1, int(sprite.height * scale))), Image.BICUBIC)
    if angle:
        sprite = sprite.rotate(angle, resample=Image.BICUBIC, expand=True)
    if alpha < 1.0:
        a = sprite.getchannel("A").point(lambda v: int(v * alpha))
        sprite = sprite.copy()
        sprite.putalpha(a)
    composite(base, sprite, cx - sprite.width / 2, cy - sprite.height / 2)


def wrap(text, f, max_w, stroke=0):
    words, lines, cur = text.split(), [], ""
    for w in words:
        test = (cur + " " + w).strip()
        if f.getlength(test) + stroke * 2 <= max_w or not cur:
            cur = test
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


# ------------------------------------------------------------------------ Türen

class Door:
    def __init__(self, idx, spec, rect, seed):
        self.idx, self.spec = idx, spec
        self.x, self.y, self.w, self.h = rect
        self.ft = max(6, int(self.w * 0.07))
        self.color = tuple(spec.get("color", (150, 30, 30)))
        self.sprite = self._leaf(spec.get("sign"))
        self.frame = self._frame()
        self.interior = fx.Interior(spec.get("inside", "dark"), self.w, self.h, seed=seed, icon=spec.get("icon"))
        self.guard = fx.emoji_image(spec["guard"], int(self.w * 0.78)) if spec.get("guard") else None
        self.open_ev, self.mark_ev, self.label_t, self.sign_ev = [], [], None, []

    def _leaf(self, sign):
        w, h = self.w, self.h
        c = np.array(self.color, np.float32)
        grad = np.linspace(1.08, 0.82, h)[:, None, None]
        arr = np.clip(c[None, None, :] * grad * np.ones((h, w, 1)), 0, 255).astype(np.uint8)
        im = Image.fromarray(arr, "RGB").convert("RGBA")
        d = ImageDraw.Draw(im)
        dark = tuple(int(v * 0.55) for v in self.color)
        light = tuple(min(255, int(v * 1.25 + 20)) for v in self.color)
        m = w * 0.12
        pw = (w - 3 * m) / 2
        rows = [(0.06, 0.22), (0.30, 0.62), (0.68, 0.94)]
        for c0 in range(2):
            for r0, r1 in rows:
                x0, x1 = m + c0 * (pw + m), m + c0 * (pw + m) + pw
                y0, y1 = h * r0, h * r1
                d.rectangle([x0, y0, x1, y1], outline=dark, width=max(2, int(w * 0.02)))
                d.rectangle([x0 + 4, y0 + 4, x1 - 4, y1 - 4], outline=light, width=max(1, int(w * 0.01)))
        # Griff
        kx, ky = w * 0.84, h * 0.53
        d.rounded_rectangle([kx - w * 0.035, ky - h * 0.05, kx + w * 0.035, ky + h * 0.05], radius=4, fill=(190, 190, 195), outline=(90, 90, 95))
        d.rounded_rectangle([kx - w * 0.17, ky - h * 0.012, kx + w * 0.02, ky + h * 0.012], radius=4, fill=(225, 225, 230), outline=(90, 90, 95))
        d.rectangle([0, 0, w - 1, h - 1], outline=(20, 15, 10), width=2)
        if sign:
            self._sign(im, sign)
        return im

    def _sign(self, im, sign):
        w, h = self.w, self.h
        d = ImageDraw.Draw(im)
        pw, x0 = w * 0.86, w * 0.07
        size = int(w * 0.13)
        while True:
            f = font(size, FONT_BOLD)
            lines = wrap(sign, f, pw - w * 0.08)
            lh = size * 1.12
            if len(lines) * lh <= h * 0.42 or size < 12:
                break
            size -= 1
        ph = len(lines) * lh + w * 0.1
        y0 = h * 0.08
        d.rounded_rectangle([x0, y0, x0 + pw, y0 + ph], radius=int(w * 0.04), fill=(245, 238, 220), outline=(40, 30, 20), width=3)
        for i, ln in enumerate(lines):
            d.text((w / 2, y0 + w * 0.05 + i * lh + lh / 2), ln, font=f, fill=(25, 20, 15), anchor="mm")

    def _frame(self):
        w, h, ft = self.w, self.h, self.ft
        pad = int(w * 0.25)
        W, H = w + 2 * ft + 2 * pad, h + ft + 2 * pad
        sh = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        ImageDraw.Draw(sh).rectangle([pad + 10, pad + 14, pad + w + 2 * ft + 10, pad + ft + h + 4], fill=(0, 0, 0, 200))
        sh = sh.filter(ImageFilter.GaussianBlur(pad / 3))
        d = ImageDraw.Draw(sh)
        d.rectangle([pad, pad, pad + w + 2 * ft, pad + ft + h], fill=(232, 226, 214), outline=(120, 110, 100), width=2)
        d.rectangle([pad + ft - 2, pad + ft - 2, pad + ft + w + 1, pad + ft + h], fill=(10, 8, 6))
        self.frame_off = (self.x - ft - pad, self.y - ft - pad)
        return sh

    # ---- Zustand

    def openness(self, t):
        state, te = 0, None
        for at, v in self.open_ev:
            if at <= t:
                state, te = v, at
        if te is None:
            return 0.0
        p = ease((t - te) / 0.7)
        return p if state else 1 - p

    def mark(self, t):
        cur = None
        for at, kind in self.mark_ev:
            if at <= t:
                cur = (at, kind)
        return cur

    # ---- Zeichnen

    def draw(self, img, t, highlight=False, glow=None):
        x, y, w, h = self.x, self.y, self.w, self.h
        if glow is not None:
            col, strength = glow
            g = Image.new("RGBA", (w * 2, h * 2), (0, 0, 0, 0))
            ImageDraw.Draw(g).rounded_rectangle([w * 0.4, h * 0.25, w * 1.6, h * 1.75], radius=int(w * 0.2), fill=col + (int(255 * strength),))
            g = g.filter(ImageFilter.GaussianBlur(w * 0.18))
            composite(img, g, x - w * 0.5, y - h * 0.5)
        composite(img, self.frame, *self.frame_off)
        a = self.openness(t)
        if a > 0.01:
            img.paste(self.interior.frame(t), (x, y))
            rx = w * (1 - 0.8 * a)
            dy = h * 0.07 * a
            quad = [(0, 0), (0, h), (rx, h + dy), (rx, -dy)]
            ox, oy = 0, -int(dy) - 1
            out_w, out_h = int(w) + 1, int(h + 2 * dy) + 2
            dst = [(qx - ox, qy - oy) for qx, qy in quad]
            src = [(0, 0), (0, h), (w, h), (w, 0)]
            leaf = self.sprite.transform((out_w, out_h), Image.PERSPECTIVE, perspective_coeffs(dst, src), Image.BICUBIC)
            shade = Image.new("RGBA", leaf.size, (0, 0, 0, int(140 * a)))
            leaf = Image.composite(Image.alpha_composite(leaf, shade), leaf, leaf.getchannel("A"))
            img.alpha_composite(leaf, (x + ox, y + oy))
        else:
            img.alpha_composite(self.sprite, (x, y))
        if highlight:
            p = 0.5 + 0.5 * math.sin(t * 8)
            d = ImageDraw.Draw(img)
            m = self.ft + 6
            d.rectangle([x - m, y - m, x + w + m, y + h + 3], outline=YELLOW + (int(150 + 105 * p),), width=max(4, int(w * 0.03)))
        if self.guard is not None:
            g = self.guard
            img.alpha_composite(g, (int(x + (w - g.width) / 2), int(y + h + h * 0.04 - g.height)))

    def draw_badge(self, img, fmt_scale):
        d = 0.34 * self.w
        cx, cy = self.x + self.w / 2, self.y - self.ft - d * 0.75
        dr = ImageDraw.Draw(img)
        dr.ellipse([cx - d / 2, cy - d / 2, cx + d / 2, cy + d / 2], fill=(20, 20, 24), outline=self.color, width=max(4, int(d * 0.09)))
        dr.text((cx, cy + 1), str(self.spec.get("number", self.idx + 1)), font=font(d * 0.6), fill=(255, 255, 255), anchor="mm")

    def draw_label(self, img, t):
        if self.label_t is None or t < self.label_t:
            return
        p = back_out((t - self.label_t) / 0.35)
        text = self.spec.get("label", "")
        size = self.w * 0.17
        f = font(size)
        while f.getlength(text) > self.w * 1.12 and size > 14:
            size -= 1
            f = font(size)
        tw = f.getlength(text)
        bw, bh = tw + size * 0.9, size * 1.55
        spr = Image.new("RGBA", (int(bw) + 4, int(bh) + 4), (0, 0, 0, 0))
        dd = ImageDraw.Draw(spr)
        col = tuple(self.spec.get("label_color", (20, 20, 24)))
        dd.rounded_rectangle([2, 2, bw, bh], radius=int(bh / 2), fill=col + (235,), outline=(255, 255, 255), width=3)
        dd.text((bw / 2 + 2, bh / 2 + 2), text, font=f, fill=(255, 255, 255), anchor="mm")
        paste_center(img, spr, self.x + self.w / 2, self.y + self.h + bh * 0.85, scale=max(0.01, p))

    def draw_mark(self, img, t):
        m = self.mark(t)
        if not m or m[1] is None:
            return
        at, kind = m
        p = back_out((t - at) / 0.3)
        x, y, w, h = self.x, self.y, self.w, self.h
        cx, cy = x + w / 2, y + h / 2
        s = max(0.01, p)
        layer = Image.new("RGBA", (w + 40, h + 40), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        if kind == "x":
            dim = Image.new("RGBA", (w, h), (0, 0, 0, int(120 * min(1, (t - at) / 0.3))))
            img.alpha_composite(dim, (x, y))
            lw = int(w * 0.11)
            for (a0, b0, a1, b1) in ((20 + w * 0.08, 20 + h * 0.2, 20 + w * 0.92, 20 + h * 0.8), (20 + w * 0.92, 20 + h * 0.2, 20 + w * 0.08, 20 + h * 0.8)):
                d.line([a0, b0, a1, b1], fill=(0, 0, 0, 255), width=lw + 10)
            for (a0, b0, a1, b1) in ((20 + w * 0.08, 20 + h * 0.2, 20 + w * 0.92, 20 + h * 0.8), (20 + w * 0.92, 20 + h * 0.2, 20 + w * 0.08, 20 + h * 0.8)):
                d.line([a0, b0, a1, b1], fill=RED + (255,), width=lw)
        else:
            r = w * 0.36
            c0, c1 = (w + 40) / 2, (h + 40) / 2
            d.ellipse([c0 - r - 5, c1 - r - 5, c0 + r + 5, c1 + r + 5], fill=(0, 0, 0, 255))
            d.ellipse([c0 - r, c1 - r, c0 + r, c1 + r], fill=GREEN + (255,), outline=(255, 255, 255), width=5)
            d.line([c0 - r * 0.45, c1 + r * 0.02, c0 - r * 0.1, c1 + r * 0.38, c0 + r * 0.5, c1 - r * 0.35],
                   fill=(255, 255, 255), width=int(r * 0.2), joint="curve")
        paste_center(img, layer, cx, cy, scale=s)


# --------------------------------------------------------------------- Episode

class Episode:
    def __init__(self, spec, fmt_name):
        self.spec = spec
        self.fmt = FORMATS[fmt_name]
        self.fmt_name = fmt_name
        F = self.fmt
        self.W, self.H = F["W"], F["H"]
        self.doors = self._layout()
        floor_y = self.doors[0].y + self.doors[0].h
        cx = (F["safe"][0] + F["safe"][2]) / 2
        self.bg = fx.room_background(self.W, self.H, cx, floor_y, wall=spec.get("wall", (70, 48, 38)), seed=3)
        self._compile()

    def _layout(self):
        bx0, by0, bx1, by1 = self.fmt["doors_box"]
        specs = self.spec["doors"]
        n = len(specs)
        bw, bh = bx1 - bx0, by1 - by0
        h = bh * 0.74
        w = h / 2.05
        unit = 1.14  # Tür + Rahmen
        if n * w * unit + (n - 1) * 0.1 * w > bw:
            w = bw / (n * unit + (n - 1) * 0.1)
            h = w * 2.05
        gap = 0 if n == 1 else min((bw - n * w * unit) / (n - 1), w * 0.9)
        total = n * w * unit + (n - 1) * gap
        x = bx0 + (bw - total) / 2 + w * (unit - 1) / 2
        y = by0 + bh * 0.13 + (bh * 0.74 - h) / 2
        doors = []
        for i, s in enumerate(specs):
            doors.append(Door(i, s, (int(x), int(y), int(w), int(h)), seed=i * 7 + 1))
            x += w * unit + gap
        return doors

    # ---- Timeline

    def _compile(self):
        t = 0.0
        self.segments, self.sfx, self.words = [], [], []
        self.teaser_t = None
        for seg in self.spec["segments"]:
            text = seg.get("text", "")
            n_words = len(text.split())
            dur = seg.get("dur") or (seg["countdown"] if "countdown" in seg else max(1.4, n_words / 3.0 + 0.45))
            s = dict(seg, start=t, end=t + dur)
            self.segments.append(s)
            for name in seg.get("sfx", []):
                self.sfx.append((t, name))
            if seg.get("riser"):
                self.sfx.append((t, "riser"))
            for i in seg.get("open", []):
                self.doors[i].open_ev.append((t + 0.1, 1))
                self.sfx.append((t + 0.1, "whoosh"))
            for i in seg.get("close", []):
                self.doors[i].open_ev.append((t, 0))
                self.sfx.append((t, "whoosh"))
            for i, kind in seg.get("mark", {}).items():
                self.doors[int(i)].mark_ev.append((t + 0.15, kind))
                if kind == "x":
                    self.sfx += [(t + 0.15, "hit"), (t + 0.15, "buzzer")]
                elif kind == "check":
                    self.sfx += [(t + 0.15, "hit"), (t + 0.15, "ding")]
            for i in seg.get("label", []):
                self.doors[i].label_t = t + 0.25
                self.sfx.append((t + 0.25, "pop"))
            if "stamp" in seg:
                self.sfx.append((t + 0.05, "hit"))
            if "countdown" in seg:
                for k in range(int(seg["countdown"])):
                    self.sfx.append((t + k, "tick_hi" if seg["countdown"] - k <= 3 else "tick"))
                    self.sfx.append((t + k + 0.5, "heart"))
            if seg.get("teaser") and self.teaser_t is None:
                self.teaser_t = t
                self.sfx.append((t, "whoosh"))
            # Wort-Timing für Untertitel
            words = text.split()
            if words:
                weights = [len(w) + 3 for w in words]
                tot = sum(weights)
                cur = t + 0.05
                span = dur - 0.25
                group, gw = [], 0
                maxc = 16 if self.fmt_name == "vertical" else 26
                groups = []
                for wd in words:
                    if group and (len(group) >= 3 or gw + len(wd) > maxc):
                        groups.append(group)
                        group, gw = [], 0
                    group.append(wd)
                    gw += len(wd) + 1
                groups.append(group)
                k = 0
                for gi, g in enumerate(groups):
                    for j, wd in enumerate(g):
                        d = span * weights[k] / tot
                        self.words.append(dict(word=wd, start=cur, end=cur + d, group=(len(self.segments), gi), pos=j, all=g))
                        cur += d
                        k += 1
            t += dur
        self.total = t

    def seg_at(self, t):
        for s in self.segments:
            if s["start"] <= t < s["end"]:
                return s
        return self.segments[-1]

    def tension(self, t):
        s = self.seg_at(t)
        if "countdown" in s:
            return 1.0
        return s.get("tension", 0.35)

    # ---- Kamera

    def cam_anchor(self):
        by0, by1 = self.fmt["doors_box"][1], self.fmt["doors_box"][3]
        return self.safe_cx(), (by0 + by1) / 2

    def cam_target(self, seg):
        """Zoom um die Mitte der Safe-Zone mit leichtem Schwenk zur Tür,
        damit auch die äußeren Türen nie unter die TikTok-Buttons rutschen."""
        sx, sy = self.cam_anchor()
        f = seg.get("focus")
        if f is None:
            return 1.0, sx, sy
        d = self.doors[f]
        return 1.07, sx + 0.12 * (d.x + d.w / 2 - sx), sy

    def camera(self, t):
        idx = self.segments.index(self.seg_at(t))
        seg = self.segments[idx]
        prev = self.cam_target(self.segments[idx - 1]) if idx > 0 else self.cam_target(seg)
        cur = self.cam_target(seg)
        p = ease((t - seg["start"]) / 0.6)
        z = prev[0] + (cur[0] - prev[0]) * p
        cx = prev[1] + (cur[1] - prev[1]) * p
        cy = prev[2] + (cur[2] - prev[2]) * p
        z *= 1.0 + 0.012 * (0.5 + 0.5 * math.sin(t * 0.7))
        return z, cx, cy

    # ---- Frame

    def render(self, t):
        W, H = self.W, self.H
        seg = self.seg_at(t)
        flick = 1.0 - (0.18 if (math.sin(t * 7.3) > 0.97 or math.sin(t * 2.1 + 1) > 0.995) else 0) + 0.02 * math.sin(t * 13)
        scene = Image.fromarray(np.clip(self.bg * flick, 0, 255).astype(np.uint8), "RGB").convert("RGBA")
        hl = seg.get("highlight", [])
        for d in self.doors:
            m = d.mark(t)
            glow = None
            if m and m[1] == "check":
                glow = (GREEN, 0.55 + 0.25 * math.sin(t * 5))
            elif d.openness(t) > 0.2 and d.interior.kind in ("fire", "light", "gas"):
                col = {"fire": (255, 110, 20), "light": (255, 220, 120), "gas": (90, 255, 60)}[d.interior.kind]
                glow = (col, 0.45 * d.openness(t))
            d.draw(scene, t, highlight=d.idx in hl, glow=glow)
        for d in self.doors:
            d.draw_badge(scene, 1)
            d.draw_label(scene, t)
            d.draw_mark(scene, t)

        # Kamera-Zoom (nur Szene, nicht UI)
        z, cx, cy = self.camera(t)
        if abs(z - 1) > 1e-3:
            sx, sy = self.cam_anchor()
            cw, ch = W / z, H / z
            x0 = min(max(cx - sx / z, 0), W - cw)
            y0 = min(max(cy - sy / z, 0), H - ch)
            scene = scene.resize((W, H), Image.BILINEAR, box=(x0, y0, x0 + cw, y0 + ch))

        dim = 0.0
        if "countdown" in seg:
            dim = 0.5 * ease((t - seg["start"]) / 0.4)
        if self.teaser_t is not None and t >= self.teaser_t:
            dim = 0.86 * ease((t - self.teaser_t) / 0.5)
        if dim > 0:
            scene.alpha_composite(Image.new("RGBA", (W, H), (0, 0, 0, int(255 * dim))))

        self.draw_header(scene, t)
        if "countdown" in seg:
            self.draw_countdown(scene, t, seg)
        if self.teaser_t is not None and t >= self.teaser_t:
            self.draw_teaser(scene, t, seg.get("teaser") or self.spec.get("teaser", {}))
        if "stamp" in seg:
            self.draw_stamp(scene, t, seg)
        self.draw_caption(scene, t)
        return scene.convert("RGB")

    def safe_cx(self):
        s = self.fmt["safe"]
        return (s[0] + s[2]) / 2

    def draw_header(self, img, t):
        F = self.fmt
        cx = self.safe_cx()
        series = self.spec.get("series", "")
        if series:
            f = font(F["series_size"])
            tw = f.getlength(series)
            bw, bh = tw + F["series_size"] * 1.2, F["series_size"] * 1.6
            spr = Image.new("RGBA", (int(bw) + 2, int(bh) + 2), (0, 0, 0, 0))
            d = ImageDraw.Draw(spr)
            d.rounded_rectangle([0, 0, bw, bh], radius=int(bh / 2), fill=RED + (255,))
            d.text((bw / 2, bh / 2), series, font=f, fill=(255, 255, 255), anchor="mm")
            img.alpha_composite(spr, (int(cx - bw / 2), F["series_y"]))
        title = self.spec.get("title", "")
        safe_w = F["safe"][2] - F["safe"][0]
        size = F["title_size"]
        while True:
            lines = wrap(title, font(size), safe_w - 30, stroke=8)
            if len(lines) <= (2 if self.fmt_name == "vertical" else 1) or size < 30:
                break
            size -= 2
        y = F["title_y"] + F["series_size"] * 0.9
        pulse = 1 + 0.025 * math.sin(t * 4)
        for i, ln in enumerate(lines):
            col = YELLOW if i == len(lines) - 1 and len(lines) > 1 else (255, 255, 255)
            spr = text_sprite(ln, size, fill=col, stroke=8)
            paste_center(img, spr, cx, y + i * size * 1.08 + size * 0.6, scale=pulse)

    def draw_countdown(self, img, t, seg):
        bx0, by0, bx1, by1 = self.fmt["doors_box"]
        cx, cy = self.safe_cx(), (by0 + by1) / 2
        r = (by1 - by0) * 0.22
        el = t - seg["start"]
        total = seg["countdown"]
        remain = max(0.0, total - el)
        num = int(math.ceil(remain - 1e-6)) if remain > 0 else 0
        frac_in = el - math.floor(el)
        sc = 1 + 0.18 * (1 - ease(frac_in / 0.25))
        layer = Image.new("RGBA", (int(r * 2.6), int(r * 2.6)), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        c = r * 1.3
        d.ellipse([c - r, c - r, c + r, c + r], fill=(10, 10, 14, 220), outline=(255, 255, 255, 90), width=6)
        col = RED if num <= 3 else YELLOW
        ang = 360 * remain / total
        d.arc([c - r + 8, c - r + 8, c + r - 8, c + r - 8], start=-90, end=-90 + ang, fill=col, width=int(r * 0.13))
        d.text((c, c), str(num), font=font(r * 1.05), fill=(255, 255, 255), anchor="mm")
        paste_center(img, layer, cx, cy, scale=sc)
        hint = seg.get("hint")
        if hint:
            spr = text_sprite(hint, self.fmt["cap_size"] * 0.62, fill=(255, 255, 255), stroke=6)
            paste_center(img, spr, cx, cy - r * 1.55)

    def draw_stamp(self, img, t, seg):
        text, col = seg["stamp"][0], tuple(seg["stamp"][1])
        bx0, by0, bx1, by1 = self.fmt["doors_box"]
        cx, cy = self.safe_cx(), by1 - (by1 - by0) * 0.02
        p = (t - seg["start"]) / 0.18
        sc = 1.7 - 0.7 * ease(p)
        size = self.fmt["title_size"] * 1.25
        spr = text_sprite(text, size, fill=col, stroke=10, stroke_fill=(255, 255, 255))
        safe_w = self.fmt["safe"][2] - self.fmt["safe"][0]
        sc *= min(1.0, (safe_w - 40) / spr.width)
        paste_center(img, spr, cx, cy, scale=sc, alpha=min(1, p * 2), angle=-6)

    def draw_teaser(self, img, t, teaser):
        el = t - self.teaser_t
        bx0, by0, bx1, by1 = self.fmt["doors_box"]
        cx = self.safe_cx()
        F = self.fmt
        if teaser.get("kicker"):
            spr = text_sprite(teaser["kicker"], F["cap_size"] * 0.7, fill=(255, 255, 255), stroke=6)
            paste_center(img, spr, cx, by0 + (by1 - by0) * 0.08, alpha=ease(el / 0.4))
        big = text_sprite(teaser.get("big", "TEIL 2"), F["title_size"] * 2.0, fill=YELLOW, stroke=12)
        paste_center(img, big, cx, by0 + (by1 - by0) * 0.27, scale=max(0.01, back_out((el - 0.15) / 0.45)) * (1 + 0.03 * math.sin(t * 5)))
        # drei Mystery-Türen
        n = 3
        dh = (by1 - by0) * 0.42
        dw = dh / 2.05
        gap = dw * 0.45
        x = cx - (n * dw + (n - 1) * gap) / 2
        y = by1 - dh - (by1 - by0) * 0.06
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        for i in range(n):
            p = ease((el - 0.4 - i * 0.18) / 0.35)
            if p <= 0:
                x += dw + gap
                continue
            a = int(255 * p)
            glow = 0.5 + 0.5 * math.sin(t * 4 + i)
            d.rectangle([x - 8, y - 8, x + dw + 8, y + dh], outline=(255, 255, 255, int(a * (0.35 + 0.4 * glow))), width=5)
            d.rectangle([x, y, x + dw, y + dh], fill=(30, 26, 34, a))
            d.text((x + dw / 2, y + dh / 2), "?", font=font(dw * 0.75), fill=(255, 255, 255, a), anchor="mm")
            x += dw + gap
        img.alpha_composite(layer)

    def draw_caption(self, img, t):
        cur = None
        for w in self.words:
            if w["start"] <= t < w["end"]:
                cur = w
                break
        if cur is None:
            # zwischen Wörtern: letztes Wort des Segments stehen lassen
            seg = self.seg_at(t)
            past = [w for w in self.words if w["end"] <= t and w["start"] >= seg["start"] - 1e-6]
            if not past:
                return
            cur = past[-1]
        F = self.fmt
        size = F["cap_size"]
        words = [w.upper() for w in cur["all"]]
        f = font(size)
        safe_w = F["safe"][2] - F["safe"][0] - 40
        sp = f.getlength(" ")
        widths = [f.getlength(w) for w in words]
        total = sum(widths) + sp * (len(words) - 1) + 20
        scale = min(1.0, safe_w / total)
        # Gruppe als Sprite
        pad = 30
        spr = Image.new("RGBA", (int(total + pad * 2), int(size * 1.6 + pad)), (0, 0, 0, 0))
        sh = Image.new("RGBA", spr.size, (0, 0, 0, 0))
        d, dsh = ImageDraw.Draw(spr), ImageDraw.Draw(sh)
        x = pad
        for i, w in enumerate(words):
            col = YELLOW if i == cur["pos"] else (255, 255, 255)
            dsh.text((x + 4, pad + 6), w, font=f, fill=(0, 0, 0, 180), stroke_width=10, stroke_fill=(0, 0, 0, 180))
            d.text((x, pad), w, font=f, fill=col, stroke_width=10, stroke_fill=(0, 0, 0))
            x += widths[i] + sp
        spr = Image.alpha_composite(sh.filter(ImageFilter.GaussianBlur(5)), spr)
        first = [w for w in self.words if w["group"] == cur["group"]][0]
        pop = 1 + 0.14 * (1 - ease((t - first["start"]) / 0.12))
        paste_center(img, spr, self.safe_cx(), F["cap_y"], scale=scale * pop)


# --------------------------------------------------------------- Ausgabe

def tiktok_ui_mock(frame, fmt_name):
    """Legt eine angenäherte TikTok-Oberfläche über ein Frame (nur für die Vorschau)."""
    phone = Image.new("RGBA", (1080, 1920), (0, 0, 0, 255))
    if fmt_name == "vertical":
        phone.paste(frame, (0, 0))
    else:
        f = frame.resize((1080, int(1080 * frame.height / frame.width)), Image.BICUBIC)
        phone.paste(f, (0, (1920 - f.height) // 2))
    ov = Image.new("RGBA", phone.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    w = (255, 255, 255, 200)
    d.text((540, 130), "Folge ich   Für dich", font=font(40, FONT_BOLD), fill=w, anchor="mm")
    x = 1010
    d.ellipse([x - 50, 860, x + 50, 960], fill=(200, 200, 200, 200))
    for i, lab in enumerate(["♥", "…", "★", "➦"]):
        y = 1040 + i * 165
        d.ellipse([x - 45, y - 45, x + 45, y + 45], fill=(255, 255, 255, 170))
        d.text((x, y + 75), "12K", font=font(30, FONT_BOLD), fill=w, anchor="mm")
    d.ellipse([x - 50, 1700, x + 50, 1800], fill=(60, 60, 60, 220))
    d.rectangle([30, 1500, 880, 1545], fill=(255, 255, 255, 170))
    d.rectangle([30, 1570, 820, 1610], fill=(255, 255, 255, 120))
    d.rectangle([30, 1635, 700, 1675], fill=(255, 255, 255, 120))
    d.rectangle([0, 1760, 1080, 1920], fill=(20, 20, 20, 235))
    return Image.alpha_composite(phone, ov).convert("RGB")


def render_episode(spec, fmt_name, out_path, preview_dir=None, preview_times=()):
    ep = Episode(spec, fmt_name)
    if preview_dir:
        os.makedirs(preview_dir, exist_ok=True)
        for pt in preview_times:
            fr = ep.render(pt)
            fr.save(os.path.join(preview_dir, f"{spec['id']}_{fmt_name}_{pt:05.1f}.png"))
            tiktok_ui_mock(fr, fmt_name).save(os.path.join(preview_dir, f"{spec['id']}_{fmt_name}_{pt:05.1f}_ui.png"))
        return ep.total
    with tempfile.TemporaryDirectory() as tmp:
        wav = os.path.join(tmp, "a.wav")
        audio.build_track(ep.total, ep.sfx, ep.tension, wav)
        cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{ep.W}x{ep.H}",
               "-r", str(FPS), "-i", "-", "-i", wav, "-c:v", "libx264", "-preset", "medium", "-crf", "20",
               "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", out_path]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        n = int(round(ep.total * FPS))
        for i in range(n):
            proc.stdin.write(ep.render(i / FPS).tobytes())
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError("ffmpeg fehlgeschlagen")
    return ep.total
