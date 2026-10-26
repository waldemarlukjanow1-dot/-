"""Prozedurale Effekte: Rauschen, Raum-Hintergrund und animierte Tür-Innenräume."""
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

EMOJI_FONT = "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf"


def periodic_noise(h, w, scale, seed, octaves=3):
    """Glattes, in beiden Achsen kachelbares Rauschen (0..1) via FFT-Filter."""
    rng = np.random.default_rng(seed)
    out = np.zeros((h, w), np.float32)
    amp, total = 1.0, 0.0
    fy = np.fft.fftfreq(h)[:, None]
    fx_ = np.fft.fftfreq(w)[None, :]
    rad = np.sqrt(fy ** 2 + fx_ ** 2)
    for o in range(octaves):
        s = scale / (2 ** o)
        filt = np.exp(-(rad * s) ** 2)
        layer = np.real(np.fft.ifft2(np.fft.fft2(rng.standard_normal((h, w))) * filt))
        layer = (layer - layer.min()) / (np.ptp(layer) + 1e-6)
        out += amp * layer
        total += amp
        amp *= 0.5
    return out / total


def palette_map(v, stops):
    """v (0..1) -> RGB anhand von Farbstopps [(pos, (r,g,b)), ...]."""
    pos = np.array([s[0] for s in stops])
    out = np.empty(v.shape + (3,), np.float32)
    for c in range(3):
        out[..., c] = np.interp(v, pos, [s[1][c] for s in stops])
    return out


def emoji_image(char, size):
    font = ImageFont.truetype(EMOJI_FONT, 109)
    im = Image.new("RGBA", (136, 128), (0, 0, 0, 0))
    ImageDraw.Draw(im).text((0, 0), char, font=font, embedded_color=True)
    bbox = im.getbbox() or (0, 0, 136, 128)
    im = im.crop(bbox)
    r = size / max(im.size)
    return im.resize((max(1, int(im.width * r)), max(1, int(im.height * r))), Image.LANCZOS)


# --------------------------------------------------------------------------- Raum

def room_background(W, H, cx, floor_y, wall=(70, 48, 38), seed=1):
    """Dunkler Raum: Putzwand mit Licht von oben + Fliesenboden in Perspektive."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    wall_c = np.array(wall, np.float32)
    plaster = periodic_noise(H // 4, W // 4, 6, seed)
    plaster = np.array(Image.fromarray((plaster * 255).astype(np.uint8)).resize((W, H), Image.BICUBIC),
                       np.float32) / 255
    # Lichtkegel von oben-mittig
    d = np.sqrt(((xx - cx) / (W * 0.55)) ** 2 + ((yy - H * 0.05) / (H * 0.75)) ** 2)
    light = np.clip(1.25 - d, 0.15, 1.2)
    img = wall_c[None, None, :] * (0.65 + 0.5 * plaster[..., None]) * light[..., None]

    # Fliesenboden (Schachbrett, perspektivisch)
    fy = yy - floor_y
    mask = fy > 0
    z = 260.0 / np.maximum(fy, 1)
    u = (xx - cx) * z / 90.0
    v = z * 3.0
    checker = ((np.floor(u) + np.floor(v)) % 2).astype(np.float32)
    tile_a, tile_b = np.array([58, 44, 36], np.float32), np.array([150, 122, 92], np.float32)
    floor = tile_a * (1 - checker[..., None]) + tile_b * checker[..., None]
    shade = np.clip(fy / (H - floor_y + 1), 0, 1) ** 0.6
    floor = floor * (0.35 + 0.65 * shade[..., None]) * np.clip(1.3 - np.abs(xx - cx) / W, 0.3, 1)[..., None]
    img = np.where(mask[..., None], floor, img)
    # Sockelleiste
    band = (yy > floor_y - 10) & (yy <= floor_y + 4)
    img[band] = img[band] * 0.35

    # Vignette
    vd = np.sqrt(((xx - W / 2) / (W * 0.7)) ** 2 + ((yy - H / 2) / (H * 0.7)) ** 2)
    img *= np.clip(1.25 - vd ** 2, 0.25, 1)[..., None]
    return np.clip(img, 0, 255).astype(np.float32)


# --------------------------------------------------------------------- Innenräume

class Interior:
    """Animierter Raum hinter einer Tür. kind: fire, water, gas, dark, light, void."""

    def __init__(self, kind, w, h, seed=0, icon=None):
        self.kind, self.w, self.h = kind, w, h
        self.n1 = periodic_noise(h, w, 18, seed + 11)
        self.n2 = periodic_noise(h, w, 9, seed + 23)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        self.yn, self.xn = yy / h, xx / w
        edge = np.minimum(np.minimum(self.xn, 1 - self.xn) * 2.2, 1) * np.minimum(np.minimum(self.yn, 1 - self.yn) * 4, 1)
        self.edge = (0.45 + 0.55 * np.clip(edge, 0, 1))[..., None]
        rng = np.random.default_rng(seed + 5)
        self.eyes = [(rng.uniform(0.15, 0.85), rng.uniform(0.25, 0.8), rng.uniform(0, 6)) for _ in range(4)]
        self.stars = rng.uniform(0, 1, (70, 3))
        self.icon = emoji_image(icon, int(w * 0.62)) if icon else None

    def _scroll(self, n, dy, dx=0.0):
        return np.roll(np.roll(n, int(dy) % self.h, axis=0), int(dx) % self.w, axis=1)

    def frame(self, t):
        k, h, w = self.kind, self.h, self.w
        if k == "fire":
            a = self._scroll(self.n1, t * h * 0.9)
            b = self._scroll(self.n2, t * h * 1.6, math.sin(t * 2) * 6)
            f = (0.65 * a + 0.55 * b) * (self.yn ** 0.9) * 1.75 + 0.06 * math.sin(t * 17)
            col = palette_map(np.clip(f - 0.1, 0, 1), [(0, (15, 2, 0)), (0.35, (150, 20, 0)), (0.6, (240, 90, 0)),
                                                         (0.82, (255, 190, 40)), (1, (255, 250, 210))])
        elif k == "water":
            a = self._scroll(self.n1, 0, t * 25)
            c = 0.5 + 0.5 * np.sin(a * 22 + t * 2.5 + self._scroll(self.n2, t * 15) * 9)
            base = palette_map(self.yn, [(0, (10, 60, 110)), (1, (0, 15, 45))])
            col = base + (c ** 6)[..., None] * np.array([90, 200, 255], np.float32) * 0.8
        elif k == "gas":
            a = self._scroll(self.n1, t * 8, t * 30)
            b = self._scroll(self.n2, -t * 12, -t * 20)
            f = 0.6 * a + 0.5 * b
            col = palette_map(np.clip(f, 0, 1), [(0, (5, 25, 5)), (0.5, (40, 140, 30)), (0.8, (150, 255, 60)), (1, (220, 255, 160))])
        elif k == "light":
            dx, dy = self.xn - 0.5, self.yn - 0.45
            r = np.sqrt(dx ** 2 + (dy * 0.6) ** 2)
            ang = np.arctan2(dy, dx)
            rays = 0.5 + 0.5 * np.sin(ang * 9 + t * 0.8)
            g = np.clip(1.25 - r * 2.0, 0, 1) * (0.85 + 0.15 * math.sin(t * 3)) + rays * 0.25 * np.clip(1 - r, 0, 1)
            col = palette_map(np.clip(g, 0, 1), [(0, (120, 80, 20)), (0.5, (255, 200, 90)), (1, (255, 255, 240))])
        elif k == "void":
            dx, dy = self.xn - 0.5, self.yn - 0.5
            r = np.sqrt(dx ** 2 + dy ** 2)
            sw = 0.5 + 0.5 * np.sin(np.arctan2(dy, dx) * 3 + r * 25 - t * 3)
            g = np.clip(r * 1.6, 0, 1) * (0.4 + 0.3 * sw)
            col = palette_map(g, [(0, (0, 0, 0)), (0.6, (25, 10, 60)), (1, (70, 30, 120))])
        else:  # dark
            f = self._scroll(self.n1, 0, t * 10) * 0.25
            col = palette_map(np.clip(f + self.yn * 0.15, 0, 1), [(0, (2, 2, 4)), (1, (40, 30, 30))])
        col = col * self.edge
        im = Image.fromarray(np.clip(col, 0, 255).astype(np.uint8), "RGB")
        if k == "dark":
            d = ImageDraw.Draw(im)
            for ex, ey, ph in self.eyes:
                blink = (math.sin(t * 0.9 + ph * 3) + 1) / 2
                if blink < 0.08:
                    continue
                px, py, s = ex * w, ey * h, w * 0.035
                for off in (-s * 1.6, s * 1.6):
                    d.ellipse([px + off - s, py - s * 0.6, px + off + s, py + s * 0.6], fill=(255, 200, 40))
        if k == "void":
            d = ImageDraw.Draw(im)
            for sx, sy, ph in self.stars:
                b = int(120 + 135 * (0.5 + 0.5 * math.sin(t * 4 + ph * 10)))
                d.point((sx * w, sy * h), fill=(b, b, b))
        if self.icon is not None:
            ic = self.icon
            if k == "dark":
                arr = np.array(ic).astype(np.float32)
                arr[..., :3] *= 0.55 + 0.1 * math.sin(t * 2)
                ic = Image.fromarray(arr.astype(np.uint8), "RGBA")
            im.paste(ic, ((w - ic.width) // 2, int(h * 0.9) - ic.height + int(4 * math.sin(t * 3))), ic)
        return im
