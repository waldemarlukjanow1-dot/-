"""How many pieces? – realistische Früchte fallen auf einen Tisch und zerbrechen in Würfel-Blöcke.

Eigener kleiner 3D-Renderer (numba):
  * ganze Frucht: Raymarching einer SDF-Form mit prozeduraler Schale (Glanz, Randlicht, Struktur)
  * Bruchstücke: die Frucht wird in Voxel (Würfel, 1 Einheit) zerlegt und per Voronoi in genau N Teile aufgeteilt.
    Jeder Teil ist ein Starrkörper (Schwerkraft, Drehung, Aufprall auf dem Tisch). Schnittflächen zeigen das
    Fruchtfleisch (Kerngehäuse, Segmente, Kerne ...). Sichtbare Voxel werden als kleine Blöcke mit Fase gezeichnet.
  * Tisch mit weichen Schatten aller Teile.
"""
import math
import os
import random
import subprocess
import tempfile
import wave
from multiprocessing import Pool

import numpy as np
from numba import njit
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import cKDTree

from tiktok.render import tiktok_ui_mock
from .marble_render import reverb

FPS = 60
W1, H1 = 1080, 1920
SR = 44100
F_BOLD = "/usr/share/fonts/opentype/inter/InterDisplay-Black.otf"
F_SEMI = "/usr/share/fonts/opentype/inter/Inter-SemiBold.otf"

APPLE, ORANGE, KIWI, STRAWBERRY, MELON = 0, 1, 2, 3, 4

# Licht (Welt, y oben): Hauptlicht von links oben vorne
LIGHT = np.array([-0.45, 0.8, 0.4])
LIGHT = LIGHT / np.linalg.norm(LIGHT)


# ------------------------------------------------------------------ Rauschen

@njit(cache=True)
def _hash(ix, iy, iz):
    h = (ix * 374761393 + iy * 668265263 + iz * 2147483647) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    return ((h ^ (h >> 16)) & 0xFFFF) / 65535.0


@njit(cache=True)
def vnoise(x, y, z):
    ix, iy, iz = math.floor(x), math.floor(y), math.floor(z)
    fx, fy, fz = x - ix, y - iy, z - iz
    ix, iy, iz = int(ix), int(iy), int(iz)
    ux, uy, uz = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy), fz * fz * (3 - 2 * fz)
    v000 = _hash(ix, iy, iz)
    v100 = _hash(ix + 1, iy, iz)
    v010 = _hash(ix, iy + 1, iz)
    v110 = _hash(ix + 1, iy + 1, iz)
    v001 = _hash(ix, iy, iz + 1)
    v101 = _hash(ix + 1, iy, iz + 1)
    v011 = _hash(ix, iy + 1, iz + 1)
    v111 = _hash(ix + 1, iy + 1, iz + 1)
    a = v000 + (v100 - v000) * ux
    b = v010 + (v110 - v010) * ux
    c = v001 + (v101 - v001) * ux
    d = v011 + (v111 - v011) * ux
    e = a + (b - a) * uy
    f = c + (d - c) * uy
    return e + (f - e) * uz


@njit(cache=True)
def fbm(x, y, z, octaves):
    s, a, f = 0.0, 0.5, 1.0
    for _ in range(octaves):
        s += a * vnoise(x * f, y * f, z * f)
        f *= 2.03
        a *= 0.5
    return s


@njit(cache=True)
def smoothstep(a, b, x):
    t = min(1.0, max(0.0, (x - a) / (b - a)))
    return t * t * (3 - 2 * t)


@njit(cache=True)
def mix3(c0, c1, t):
    return (c0[0] + (c1[0] - c0[0]) * t, c0[1] + (c1[1] - c0[1]) * t, c0[2] + (c1[2] - c0[2]) * t)


# ------------------------------------------------------------------ Formen (lokale Koordinaten, y oben)

@njit(cache=True)
def sdf(fid, x, y, z):
    if fid == APPLE:
        r = 58.0
        d = math.sqrt(x * x + (y * 1.08) ** 2 + z * z) - r
        rxz2 = x * x + z * z
        d += 13.0 * math.exp(-rxz2 / (2 * 13.0 ** 2)) * smoothstep(10, 50, y)        # Stielmulde
        d += 8.0 * math.exp(-rxz2 / (2 * 11.0 ** 2)) * smoothstep(-10, -50, y)       # Blütenmulde
        d += 1.2 * math.sin(math.atan2(z, x) * 5) * smoothstep(-20, -55, y)           # leichte Beulen unten
        # Stiel
        t = min(1.0, max(0.0, (y - 30.0) / 38.0))
        sx, sy, sz = 2.5 * t * t, 30.0 + 38.0 * t, 1.5 * t
        ds = math.sqrt((x - sx) ** 2 + (y - sy) ** 2 + (z - sz) ** 2) - 2.4
        return min(d * 0.85, ds)
    if fid == ORANGE:
        return (math.sqrt(x * x + (y * 1.05) ** 2 + z * z) - 56.0) * 0.95
    if fid == KIWI:
        return (math.sqrt((x / 64.0) ** 2 + (y / 47.0) ** 2 + (z / 49.0) ** 2) - 1.0) * 46.0
    if fid == STRAWBERRY:
        u = (y + 52.0) / 92.0                       # 0 Spitze unten, 1 oben
        if u < 0:
            return math.sqrt(x * x + z * z + (y + 52.0) ** 2) - 0.5
        rad = 46.0 * (min(u, 1.0) ** 0.6) * (1.0 - 0.25 * max(0.0, u - 0.78) / 0.22)
        d = math.sqrt(x * x + z * z) - rad
        d = max(d, y - 41.0 + 0.0065 * (x * x + z * z))     # gewölbte Schulter
        # Kelchblätter: flacher Stern oben
        ang = math.atan2(z, x)
        rr = math.sqrt(x * x + z * z)
        star = 7.0 + 33.0 * max(0.0, math.cos(ang * 2.5)) ** 2.2
        dl = max(rr - star, abs(y - 41.5 + 0.0065 * rr * rr + rr * 0.05) - 1.0)
        # kleiner Stiel
        t = min(1.0, max(0.0, (y - 38.0) / 14.0))
        ds = math.sqrt((x - 1.5 * t) ** 2 + (y - 38.0 - 14.0 * t) ** 2 + z * z) - 1.8
        return min(d * 0.8, min(dl * 0.8, ds))
    # Wassermelone
    return (math.sqrt((x / 86.0) ** 2 + (y / 62.0) ** 2 + (z / 63.0) ** 2) - 1.0) * 61.0


@njit(cache=True)
def skin(fid, x, y, z):
    """Schalenfarbe + Glanz + Bump (r, g, b, gloss, bump)."""
    if fid == APPLE:
        if y > 32.0 and x * x + z * z < 16.0:  # Stiel
            n = fbm(x * 0.3, y * 0.3, z * 0.3, 2)
            return 95 + 30 * n, 60 + 20 * n, 30, 0.1, 0.0
        ang = math.atan2(z, x)
        streak = fbm(ang * 6.0, y * 0.04, 0.5, 4)
        top = smoothstep(25, 55, y)
        red = mix3((150.0, 12.0, 22.0), (205.0, 35.0, 30.0), streak)
        col = mix3(red, (205.0, 175.0, 60.0), top * 0.55 + 0.25 * smoothstep(0.62, 0.8, fbm(x * 0.05, y * 0.05, z * 0.05, 3)))
        dn = vnoise(x * 1.1, y * 1.1, z * 1.1)
        if dn > 0.82:
            col = mix3(col, (225.0, 200.0, 140.0), min(1.0, (dn - 0.82) * 7.0) * 0.45)
        return col[0], col[1], col[2], 0.85, 0.0
    if fid == ORANGE:
        n = fbm(x * 0.06, y * 0.06, z * 0.06, 3)
        col = mix3((240.0, 110.0, 5.0), (255.0, 150.0, 20.0), n)
        b = vnoise(x * 0.9, y * 0.9, z * 0.9)
        return col[0], col[1], col[2], 0.45, (b - 0.5) * 0.9
    if fid == KIWI:
        n = fbm(x * 0.08, y * 0.08, z * 0.08, 4)
        f = vnoise(x * 2.2, y * 2.2, z * 2.2)
        col = mix3((92.0, 70.0, 38.0), (140.0, 112.0, 62.0), n * 0.6 + f * 0.4)
        return col[0], col[1], col[2], 0.0, (f - 0.5) * 1.4
    if fid == STRAWBERRY:
        rr = math.sqrt(x * x + z * z)
        ang0 = math.atan2(z, x)
        star = 7.0 + 33.0 * max(0.0, math.cos(ang0 * 2.5)) ** 2.2
        if y > 28.0 and ((abs(y - 41.5 + 0.0065 * rr * rr + rr * 0.05) < 1.9 and rr < star + 0.8) or (rr < 3.0 and y > 40)):
            n = fbm(x * 0.15, y * 0.15, z * 0.15, 3)
            return 30 + 25 * n, 85 + 45 * n, 25 + 10 * n, 0.3, 0.0
        ang = math.atan2(z, x)
        u = ang * 9.0 / math.pi
        v = y * 0.13
        cu, cv = u - math.floor(u) - 0.5, v - math.floor(v) - 0.5
        if (int(math.floor(v)) & 1) == 1:
            cu = (u + 0.5) - math.floor(u + 0.5) - 0.5
        seed = cu * cu * 4.0 + cv * cv * 4.0
        n = fbm(x * 0.05, y * 0.05, z * 0.05, 3)
        col = mix3((185.0, 10.0, 30.0), (235.0, 40.0, 45.0), n)
        col = mix3(col, (240.0, 220.0, 120.0), 0.6 * smoothstep(-20, 30, y) * 0.0)
        if seed < 0.06:
            return 235, 205, 90, 0.5, -0.6
        return col[0], col[1], col[2], 0.95, (seed - 0.3) * 0.5
    # Melone: dunkle und helle Streifen entlang der Längsachse
    ang = math.atan2(z, y)
    w = math.sin(ang * 9.0 + 1.6 * fbm(x * 0.04, y * 0.04, z * 0.04, 3) * 3.0)
    n = fbm(x * 0.12, y * 0.12, z * 0.12, 3)
    col = mix3((20.0, 70.0, 25.0), (110.0, 160.0, 70.0), smoothstep(0.1, 0.5, w) * 0.8 + n * 0.2)
    return col[0], col[1], col[2], 0.6, 0.0


@njit(cache=True)
def flesh(fid, x, y, z, depth):
    """Fruchtfleisch an einem inneren Punkt (depth = Abstand zur Oberfläche)."""
    if fid == APPLE:
        rxz = math.sqrt(x * x + z * z)
        ang = math.atan2(z, x)
        core = 15.0 * (1.0 + 0.28 * math.cos(ang * 5)) * smoothstep(48, 20, abs(y))
        n = fbm(x * 0.15, y * 0.15, z * 0.15, 2)
        col = mix3((246.0, 228.0, 172.0), (236.0, 214.0, 150.0), n)
        if depth < 2.5:
            col = mix3((200.0, 40.0, 40.0), col, depth / 2.5)
        if rxz < core:
            col = mix3(col, (225.0, 205.0, 150.0), 0.7)
            # Kerne
            sa = ang * 5.0 / (2 * math.pi)
            if abs(sa - math.floor(sa) - 0.5) < 0.18 and 5.0 < rxz < 11.0 and abs(y) < 10.0:
                return 90, 55, 25, 0.6
        return col[0], col[1], col[2], 0.45
    if fid == ORANGE:
        if depth < 3.5:
            return 250, 130, 20, 0.3
        if depth < 7.5:
            return 248, 238, 215, 0.1
        ang = math.atan2(z, y)
        seg = ang * 11.0 / (2 * math.pi)
        fs = abs(seg - math.floor(seg) - 0.5)
        rxz = math.sqrt(y * y + z * z)
        if fs > 0.465 or rxz < 4.5:
            return 250, 235, 200, 0.3
        n = vnoise(rxz * 0.6, ang * 30.0, x * 0.25)
        col = mix3((245.0, 120.0, 10.0), (255.0, 170.0, 50.0), n)
        return col[0], col[1], col[2], 0.9
    if fid == KIWI:
        if depth < 1.6:
            return 110, 80, 40, 0.1
        ryz = math.sqrt(y * y + z * z) / (1.0 - (x / 66.0) ** 2 + 0.05) ** 0.5
        ang = math.atan2(z, y)
        if ryz < 9.0:
            return 245, 240, 200, 0.5
        if 12.0 < ryz < 19.0 and _hash(int(ang * 22), int(ryz * 0.6), int(x * 0.25)) > 0.55:
            return 25, 20, 10, 0.7
        n = vnoise(ang * 25.0, ryz * 0.2, x * 0.1)
        col = mix3((95.0, 170.0, 25.0), (160.0, 215.0, 70.0), n * smoothstep(40, 10, ryz) * 0.7 + 0.15)
        return col[0], col[1], col[2], 0.8
    if fid == STRAWBERRY:
        return 235, 90, 100, 0.8
    if depth < 2.5:
        return 25, 80, 30, 0.4
    if depth < 6.5:
        return 215, 235, 175, 0.3
    if depth < 9.0:
        return 245, 150, 140, 0.6
    if _hash(int(x * 0.18), int(y * 0.18), int(z * 0.18)) > 0.985 and depth > 16:
        return 20, 15, 15, 0.8
    n = fbm(x * 0.07, y * 0.07, z * 0.07, 3)
    col = mix3((215.0, 25.0, 45.0), (240.0, 60.0, 70.0), n)
    return col[0], col[1], col[2], 0.85


# ------------------------------------------------------------------ Kamera

class Camera:
    def __init__(self, pos=(0.0, 380.0, 640.0), target=(0.0, 60.0, 0.0), f=2050.0, cx=520.0, cy=900.0):
        self.pos = np.array(pos, np.float64)
        fwd = np.array(target, np.float64) - self.pos
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, np.array([0.0, 1.0, 0.0]))
        right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        self.fwd, self.right, self.up, self.f, self.cx, self.cy = fwd, right, up, f, cx, cy

    def params(self):
        return np.concatenate([self.pos, self.right, self.up, self.fwd, [self.f, self.cx, self.cy]])


# ------------------------------------------------------------------ Hintergrund + Tisch

@njit(cache=True)
def render_background(img, floor_xz, cam, W, H):
    px, py, pz = cam[0], cam[1], cam[2]
    for j in range(H):
        for i in range(W):
            dx = (i - cam[13]) / cam[12]
            dy = -(j - cam[14]) / cam[12]
            rx = cam[3] * dx + cam[6] * dy + cam[9]
            ry = cam[4] * dx + cam[7] * dy + cam[10]
            rz = cam[5] * dx + cam[8] * dy + cam[11]
            if ry < -1e-4:
                t = -py / ry
                X, Z = px + rx * t, pz + rz * t
                floor_xz[j, i, 0] = X
                floor_xz[j, i, 1] = Z
                d = math.sqrt(X * X + (Z - 20.0) ** 2)
                spot = math.exp(-(d / 430.0) ** 2)
                g = fbm(X * 0.01, 0.0, Z * 0.03, 4)
                base = 12.0 + 98.0 * spot
                img[j, i, 0] = base * (1.0 + 0.10 * (g - 0.5)) + 6
                img[j, i, 1] = base * (0.96 + 0.10 * (g - 0.5)) + 3
                img[j, i, 2] = base * (0.92 + 0.08 * (g - 0.5))
            else:
                floor_xz[j, i, 0] = 1e9
                floor_xz[j, i, 1] = 1e9
                # Wand geht weich in den Tisch über (Studio-Hohlkehle)
                v = min(1.0, max(0.0, ry * 6.0))
                glow = math.exp(-((i - cam[13]) / 600.0) ** 2)
                img[j, i, 0] = 12 + 8 * glow * (1 - v)
                img[j, i, 1] = 12 + 8 * glow * (1 - v)
                img[j, i, 2] = 16 + 9 * glow * (1 - v)


# Schatten: Belegung in Tischkoordinaten (X, Z)
SH_X0, SH_Z0, SH_RES, SH_W, SH_H = -500.0, -360.0, 2.0, 500, 420


@njit(cache=True)
def shadow_splat(buf, pts, lx, ly, lz):
    for k in range(pts.shape[0]):
        x, y, z = pts[k, 0], pts[k, 1], pts[k, 2]
        t = y / ly
        X = x - lx * t
        Z = z - lz * t
        i = int((X - SH_X0) / SH_RES)
        j = int((Z - SH_Z0) / SH_RES)
        if 0 <= i < SH_W and 0 <= j < SH_H:
            buf[j, i] += 1.0 / (1.0 + 0.012 * y)


@njit(cache=True)
def shade_floor(img, floor_xz, sh, W, H):
    for j in range(H):
        for i in range(W):
            X = floor_xz[j, i, 0]
            if X > 1e8:
                continue
            Z = floor_xz[j, i, 1]
            fi = (X - SH_X0) / SH_RES
            fj = (Z - SH_Z0) / SH_RES
            i0, j0 = int(fi), int(fj)
            if 0 <= i0 < SH_W - 1 and 0 <= j0 < SH_H - 1:
                ax, ay = fi - i0, fj - j0
                s = (sh[j0, i0] * (1 - ax) * (1 - ay) + sh[j0, i0 + 1] * ax * (1 - ay) +
                     sh[j0 + 1, i0] * (1 - ax) * ay + sh[j0 + 1, i0 + 1] * ax * ay)
                k = 1.0 - 0.72 * s
                img[j, i, 0] *= k
                img[j, i, 1] *= k
                img[j, i, 2] *= k


def blur_shadow(buf):
    """Belegung -> weicher Schatten (0..1)."""
    s = 1.0 - np.exp(-buf * 0.9)
    for r in (3, 3, 2):
        k = np.ones(2 * r + 1) / (2 * r + 1)
        s = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 0, s)
        s = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 1, s)
    return s.astype(np.float64)


# ------------------------------------------------------------------ Beleuchtung

@njit(cache=True)
def light(r, g, b, gloss, nx, ny, nz, vx, vy, vz, lx, ly, lz):
    ndl = nx * lx + ny * ly + nz * lz
    wrap = max(0.0, (ndl + 0.3) / 1.3)
    hx, hy, hz = lx - vx, ly - vy, lz - vz
    hl = math.sqrt(hx * hx + hy * hy + hz * hz) + 1e-9
    ndh = max(0.0, (nx * hx + ny * hy + nz * hz) / hl)
    spec = gloss * (ndh ** (20.0 + 90.0 * gloss)) * 1.1 * (ndl > 0)
    sky = 0.5 + 0.5 * ny
    ndv = max(0.0, -(nx * vx + ny * vy + nz * vz))
    rim = (1.0 - ndv) ** 3 * 0.35 * max(0.0, -nz * 0.4 + ny * 0.3 + 0.5)
    amb = 0.22 + 0.16 * sky
    k = amb + 0.95 * wrap
    return (r * k + 255 * (spec + rim * 0.5), g * k + 255 * (spec + rim * 0.5), b * k + 255 * (spec * 0.95 + rim * 0.55))


# ------------------------------------------------------------------ Raymarching der ganzen Frucht

@njit(cache=True)
def raymarch(img, zb, fid, R, T, cam, bx0, by0, bx1, by1, brad, lx, ly, lz):
    cx, cy, cz = cam[0], cam[1], cam[2]
    for j in range(max(0, by0), min(img.shape[0], by1)):
        for i in range(max(0, bx0), min(img.shape[1], bx1)):
            acc0 = acc1 = acc2 = 0.0
            hits = 0
            zmin = 1e18
            for s in range(4):
                sx = i + 0.25 + 0.5 * (s & 1)
                sy = j + 0.25 + 0.5 * (s >> 1)
                dx = (sx - cam[13]) / cam[12]
                dy = -(sy - cam[14]) / cam[12]
                rx = cam[3] * dx + cam[6] * dy + cam[9]
                ry = cam[4] * dx + cam[7] * dy + cam[10]
                rz = cam[5] * dx + cam[8] * dy + cam[11]
                rl = math.sqrt(rx * rx + ry * ry + rz * rz)
                rx, ry, rz = rx / rl, ry / rl, rz / rl
                # in lokale Koordinaten
                ox, oy, oz = cx - T[0], cy - T[1], cz - T[2]
                lox = R[0, 0] * ox + R[1, 0] * oy + R[2, 0] * oz
                loy = R[0, 1] * ox + R[1, 1] * oy + R[2, 1] * oz
                loz = R[0, 2] * ox + R[1, 2] * oy + R[2, 2] * oz
                ldx = R[0, 0] * rx + R[1, 0] * ry + R[2, 0] * rz
                ldy = R[0, 1] * rx + R[1, 1] * ry + R[2, 1] * rz
                ldz = R[0, 2] * rx + R[1, 2] * ry + R[2, 2] * rz
                b = lox * ldx + loy * ldy + loz * ldz
                c = lox * lox + loy * loy + loz * loz - brad * brad
                disc = b * b - c
                if disc < 0:
                    continue
                t = max(0.0, -b - math.sqrt(disc))
                t_end = -b + math.sqrt(disc)
                hit = False
                for _ in range(140):
                    px, py, pz = lox + ldx * t, loy + ldy * t, loz + ldz * t
                    d = sdf(fid, px, py, pz)
                    if d < 0.04:
                        hit = True
                        break
                    t += max(d, 0.05)
                    if t > t_end:
                        break
                if not hit:
                    continue
                e = 0.25
                gx = sdf(fid, px + e, py, pz) - sdf(fid, px - e, py, pz)
                gy = sdf(fid, px, py + e, pz) - sdf(fid, px, py - e, pz)
                gz = sdf(fid, px, py, pz + e) - sdf(fid, px, py, pz - e)
                cr, cg, cb, gloss, bump = skin(fid, px, py, pz)
                if bump != 0.0:
                    q = 0.9
                    gx += bump * (vnoise(px * q + 7, py * q, pz * q) - 0.5) * 0.8
                    gy += bump * (vnoise(px * q, py * q + 3, pz * q) - 0.5) * 0.8
                    gz += bump * (vnoise(px * q, py * q, pz * q + 5) - 0.5) * 0.8
                gl = math.sqrt(gx * gx + gy * gy + gz * gz) + 1e-9
                gx, gy, gz = gx / gl, gy / gl, gz / gl
                # Normale in Welt
                nx = R[0, 0] * gx + R[0, 1] * gy + R[0, 2] * gz
                ny = R[1, 0] * gx + R[1, 1] * gy + R[1, 2] * gz
                nz = R[2, 0] * gx + R[2, 1] * gy + R[2, 2] * gz
                # einfache Umgebungsverdeckung
                ao = 0.0
                for k in range(1, 4):
                    hh = 2.5 * k
                    ao += (hh - sdf(fid, px + gx * hh, py + gy * hh, pz + gz * hh)) / hh
                aof = max(0.45, 1.0 - 0.25 * ao)
                wy = T[1] + R[1, 0] * px + R[1, 1] * py + R[1, 2] * pz
                ground = 0.75 + 0.25 * min(1.0, wy / 40.0)   # unten zum Tisch hin etwas dunkler
                o0, o1, o2 = light(cr, cg, cb, gloss, nx, ny, nz, rx, ry, rz, lx, ly, lz)
                acc0 += o0 * aof * ground
                acc1 += o1 * aof * ground
                acc2 += o2 * aof * ground
                hits += 1
                zmin = min(zmin, t)
            if hits > 0:
                a = hits / 4.0
                img[j, i, 0] = img[j, i, 0] * (1 - a) + acc0 / 4.0
                img[j, i, 1] = img[j, i, 1] * (1 - a) + acc1 / 4.0
                img[j, i, 2] = img[j, i, 2] * (1 - a) + acc2 / 4.0
                zb[j, i] = zmin


# ------------------------------------------------------------------ Voxel-Blöcke zeichnen

@njit(cache=True)
def splat_voxels(img, zb, P, Nrm, Col, lab, Rs, Ts, Coms, cam, lx, ly, lz, vsize):
    W, H = img.shape[1], img.shape[0]
    for k in range(P.shape[0]):
        c = lab[k]
        R = Rs[c]
        qx, qy, qz = P[k, 0] - Coms[c, 0], P[k, 1] - Coms[c, 1], P[k, 2] - Coms[c, 2]
        wx = R[0, 0] * qx + R[0, 1] * qy + R[0, 2] * qz + Ts[c, 0]
        wy = R[1, 0] * qx + R[1, 1] * qy + R[1, 2] * qz + Ts[c, 1]
        wz = R[2, 0] * qx + R[2, 1] * qy + R[2, 2] * qz + Ts[c, 2]
        vx, vy, vz = wx - cam[0], wy - cam[1], wz - cam[2]
        zc = vx * cam[9] + vy * cam[10] + vz * cam[11]
        if zc < 5:
            continue
        sx = cam[13] + cam[12] * (vx * cam[3] + vy * cam[4] + vz * cam[5]) / zc
        sy = cam[14] - cam[12] * (vx * cam[6] + vy * cam[7] + vz * cam[8]) / zc
        size = cam[12] * vsize / zc
        half = size * 0.5
        x0, x1 = int(sx - half + 0.5), int(sx + half + 0.5)
        y0, y1 = int(sy - half + 0.5), int(sy + half + 0.5)
        if x1 <= x0:
            x1 = x0 + 1
        if y1 <= y0:
            y1 = y0 + 1
        if x1 < 0 or y1 < 0 or x0 >= W or y0 >= H:
            continue
        nx0, ny0, nz0 = Nrm[k, 0], Nrm[k, 1], Nrm[k, 2]
        nx = R[0, 0] * nx0 + R[0, 1] * ny0 + R[0, 2] * nz0
        ny = R[1, 0] * nx0 + R[1, 1] * ny0 + R[1, 2] * nz0
        nz = R[2, 0] * nx0 + R[2, 1] * ny0 + R[2, 2] * nz0
        vl = math.sqrt(vx * vx + vy * vy + vz * vz)
        o0, o1, o2 = light(Col[k, 0], Col[k, 1], Col[k, 2], Col[k, 3], nx, ny, nz, vx / vl, vy / vl, vz / vl,
                           lx, ly, lz)
        ground = 0.78 + 0.22 * min(1.0, max(0.0, wy) / 30.0)
        o0 *= ground
        o1 *= ground
        o2 *= ground
        bevel = (x1 - x0) >= 3
        for y in range(max(0, y0), min(H, y1)):
            for x in range(max(0, x0), min(W, x1)):
                if zc < zb[y, x]:
                    zb[y, x] = zc
                    m = 1.0
                    if bevel:
                        if x == x0 or y == y0:
                            m = 1.05
                        elif x == x1 - 1 or y == y1 - 1:
                            m = 0.88
                    img[y, x, 0] = o0 * m
                    img[y, x, 1] = o1 * m
                    img[y, x, 2] = o2 * m


# ------------------------------------------------------------------ Frucht in Voxel zerlegen

BOX = {APPLE: (62, -66, 72, 62), ORANGE: (58, -56, 56, 58), KIWI: (66, -49, 49, 51), STRAWBERRY: (50, -54, 42, 50),
       MELON: (88, -64, 64, 65)}     # (halbe Breite x, y min, y max, halbe Tiefe z)


@njit(cache=True)
def sdf_points(fid, P):
    out = np.empty(P.shape[0])
    for k in range(P.shape[0]):
        out[k] = sdf(fid, P[k, 0], P[k, 1], P[k, 2])
    return out


@njit(cache=True)
def voxel_attr(fid, P, depth, is_skin, cutn, Nrm, Col):
    for k in range(P.shape[0]):
        x, y, z = P[k, 0], P[k, 1], P[k, 2]
        if is_skin[k]:
            r, g, b, gl, bump = skin(fid, x, y, z)
            e = 0.5
            gx = sdf(fid, x + e, y, z) - sdf(fid, x - e, y, z)
            gy = sdf(fid, x, y + e, z) - sdf(fid, x, y - e, z)
            gz = sdf(fid, x, y, z + e) - sdf(fid, x, y, z - e)
            if bump != 0.0:
                gx += bump * (vnoise(x * 0.9 + 7, y * 0.9, z * 0.9) - 0.5) * 0.8
                gy += bump * (vnoise(x * 0.9, y * 0.9 + 3, z * 0.9) - 0.5) * 0.8
                gz += bump * (vnoise(x * 0.9, y * 0.9, z * 0.9 + 5) - 0.5) * 0.8
        else:
            r, g, b, gl = flesh(fid, x, y, z, depth[k])
            gx, gy, gz = cutn[k, 0], cutn[k, 1], cutn[k, 2]
        L = math.sqrt(gx * gx + gy * gy + gz * gz) + 1e-9
        Nrm[k, 0], Nrm[k, 1], Nrm[k, 2] = gx / L, gy / L, gz / L
        Col[k, 0], Col[k, 1], Col[k, 2], Col[k, 3] = r, g, b, gl


class Fruit:
    """Voxel-Version einer Frucht, aufgeteilt in genau n_pieces Teile."""

    def __init__(self, fid, n_pieces, seed):
        self.fid, self.n = fid, n_pieces
        hx, y0, y1, hz = BOX[fid]
        xs, ys, zs = np.arange(-hx, hx + 1), np.arange(y0, y1 + 1), np.arange(-hz, hz + 1)
        G = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), -1).reshape(-1, 3).astype(np.float64)
        d = sdf_points(fid, G).reshape(len(xs), len(ys), len(zs))
        inside = d < 0
        rng = np.random.default_rng(seed)
        idx = np.argwhere(inside)
        P = G.reshape(len(xs), len(ys), len(zs), 3)[inside]
        assert len(P) >= n_pieces, (len(P), n_pieces)
        # Teile: Voronoi aus n Saatpunkten (Saat = Voxel -> jedes Teil hat mindestens einen Voxel)
        if n_pieces == 1:
            labv = np.zeros(len(P), np.int64)
        elif n_pieces == 2:
            labv = (P[:, 0] + 0.15 * P[:, 2] > 0).astype(np.int64)
        else:
            seeds = P[rng.choice(len(P), n_pieces, replace=False)]
            _, labv = cKDTree(seeds).query(P, workers=-1)
        lab = np.full(inside.shape, -1, np.int64)
        lab[inside] = labv
        # freiliegende Voxel (Nachbar gehört zu anderem Teil oder ist außen) + Schnitt-Normale
        cutn = np.zeros(inside.shape + (3,))
        exposed = np.zeros(inside.shape, bool)
        for ax in range(3):
            for s in (-1, 1):
                nb = np.roll(lab, -s, axis=ax)
                diff = nb != lab
                exposed |= diff
                cutn[..., ax] += np.where(diff, s, 0)
        exposed &= inside
        depth = -d
        is_skin = (depth < 1.8) & inside
        keep = exposed[inside]
        self.P = P[keep]
        self.lab = labv[keep].astype(np.int64)
        sk = is_skin[inside][keep]
        dp = depth[inside][keep]
        cn = cutn[inside][keep]
        self.Nrm = np.zeros((len(self.P), 3))
        self.Col = np.zeros((len(self.P), 4))
        voxel_attr(fid, self.P, dp, sk, cn, self.Nrm, self.Col)
        # Schwerpunkt und Größe jedes Teils (aus allen Voxeln)
        cnt = np.bincount(labv, minlength=n_pieces).astype(np.float64)
        self.count = cnt
        self.com = np.stack([np.bincount(labv, weights=P[:, a], minlength=n_pieces) for a in range(3)], 1) / cnt[:, None]
        self.size = np.cbrt(cnt)
        self.skin_pts = P[is_skin[inside]]
        self.n_vox = len(P)


# ------------------------------------------------------------------ Starrkörper-Physik der Teile

@njit(cache=True)
def world_points(P, lab, Rs, Ts, Coms, out, step):
    m = 0
    for k in range(0, P.shape[0], step):
        c = lab[k]
        R = Rs[c]
        qx, qy, qz = P[k, 0] - Coms[c, 0], P[k, 1] - Coms[c, 1], P[k, 2] - Coms[c, 2]
        out[m, 0] = R[0, 0] * qx + R[0, 1] * qy + R[0, 2] * qz + Ts[c, 0]
        out[m, 1] = R[1, 0] * qx + R[1, 1] * qy + R[1, 2] * qz + Ts[c, 1]
        out[m, 2] = R[2, 0] * qx + R[2, 1] * qy + R[2, 2] * qz + Ts[c, 2]
        m += 1
    return m


@njit(cache=True)
def chunk_min_y(P, lab, Rs, Ts, Coms, out):
    out[:] = 1e18
    for k in range(P.shape[0]):
        c = lab[k]
        R = Rs[c]
        qx, qy, qz = P[k, 0] - Coms[c, 0], P[k, 1] - Coms[c, 1], P[k, 2] - Coms[c, 2]
        y = R[1, 0] * qx + R[1, 1] * qy + R[1, 2] * qz + Ts[c, 1] - 0.5
        if y < out[c]:
            out[c] = y


def rot_from_axis(w, dt):
    """Rotationsmatrizen (N,3,3) aus Winkelgeschwindigkeiten (N,3) für Zeitschritt dt (Rodrigues)."""
    th = np.linalg.norm(w, axis=1) * dt
    ax = w / np.maximum(np.linalg.norm(w, axis=1), 1e-9)[:, None]
    x, y, z = ax[:, 0], ax[:, 1], ax[:, 2]
    c, s, C = np.cos(th), np.sin(th), 1 - np.cos(th)
    R = np.empty((len(w), 3, 3))
    R[:, 0, 0] = c + x * x * C
    R[:, 0, 1] = x * y * C - z * s
    R[:, 0, 2] = x * z * C + y * s
    R[:, 1, 0] = y * x * C + z * s
    R[:, 1, 1] = c + y * y * C
    R[:, 1, 2] = y * z * C - x * s
    R[:, 2, 0] = z * x * C - y * s
    R[:, 2, 1] = z * y * C + x * s
    R[:, 2, 2] = c + z * z * C
    return R


G_ACC = 2600.0


class Pieces:
    """Alle Teile als Starrkörper. Vor dem Aufprall bewegen sie sich gemeinsam (ganze Frucht)."""

    def __init__(self, fruit, seed, hover_y=255.0, burst=1.0):
        self.f = fruit
        n = fruit.n
        self.rng = np.random.default_rng(seed + 11)
        self.Rs = np.tile(np.eye(3), (n, 1, 1))
        self.Ts = fruit.com.copy()
        self.Ts[:, 1] += hover_y
        self.V = np.zeros((n, 3))
        self.W = np.zeros((n, 3))
        self.broken = False
        self.burst = burst
        self.miny = np.zeros(n)
        self.whole_R = np.eye(3)
        self.whole_T = np.array([0.0, hover_y, 0.0])
        self.whole_V = np.zeros(3)
        self.whole_W = np.array([0.15, 0.9, 0.25])
        self.impact_t = None
        self.bounces = 0

    def whole_pose(self):
        return self.whole_R, self.whole_T

    def _sync_whole(self):
        self.Rs[:] = self.whole_R
        self.Ts[:] = self.f.com @ self.whole_R.T + self.whole_T

    def step(self, t, dt, dropping):
        f = self.f
        if not self.broken:
            self.whole_R = rot_from_axis(self.whole_W[None], dt)[0] @ self.whole_R
            if dropping:
                self.whole_V[1] -= G_ACC * dt
                self.whole_T = self.whole_T + self.whole_V * dt
            self._sync_whole()
            chunk_min_y(f.P, f.lab, self.Rs, self.Ts, f.com, self.miny)
            low = self.miny.min()
            if low < 0:
                self.impact_t = t if self.impact_t is None else self.impact_t
                if f.n > 1:
                    self.whole_T[1] -= low
                    self._sync_whole()
                    self.break_apart()
                else:          # bleibt ganz: springt zurück
                    self.whole_T[1] -= low
                    self.whole_V[1] = -self.whole_V[1] * (0.42 if self.bounces == 0 else 0.3)
                    self.whole_V[0] += 25.0 if self.bounces == 0 else 0.0
                    self.whole_W *= 0.5
                    if self.bounces == 0:
                        self.whole_W += np.array([0.0, 0.0, -1.2])
                    self.bounces += 1
                    if abs(self.whole_V[1]) < 60:
                        self.whole_V[:] = 0
                        self.whole_W *= 0.85
                    self._sync_whole()
            return
        # Teile frei
        self.V[:, 1] -= G_ACC * dt
        self.Ts += self.V * dt
        moving = np.linalg.norm(self.W, axis=1) > 1e-3
        if moving.any():
            self.Rs[moving] = rot_from_axis(self.W[moving], dt) @ self.Rs[moving]
        chunk_min_y(f.P, f.lab, self.Rs, self.Ts, f.com, self.miny)
        hit = self.miny < 0
        if hit.any():
            self.Ts[hit, 1] -= self.miny[hit]
            vy = self.V[hit, 1]
            self.V[hit, 1] = np.where(vy < 0, -vy * 0.32, vy)
            self.V[hit, 0] *= 0.55
            self.V[hit, 2] *= 0.55
            self.W[hit] *= 0.62
            slow = hit.copy()
            slow[hit] = np.linalg.norm(self.V[hit], axis=1) < 45
            self.V[slow] = 0
            self.W[slow] *= 0.5

    def break_apart(self):
        f = self.f
        self.broken = True
        rng = self.rng
        n = f.n
        wc = f.com @ self.whole_R.T          # Lage relativ zum Fruchtzentrum (Welt)
        r = np.linalg.norm(wc, axis=1)
        dirs = wc / np.maximum(r, 1e-6)[:, None]
        dirs[:, 1] = np.abs(dirs[:, 1]) * 0.5 + 0.15
        dirs /= np.linalg.norm(dirs, axis=1)[:, None]
        small = 1.0 / (1.0 + f.size / 6.0)
        if n == 2:
            sp = np.full(n, 230.0)
            up = np.full(n, 260.0)
        else:
            sp = (130.0 + 380.0 * small) * rng.uniform(0.4, 1.15, n) * self.burst
            up = (160.0 + 380.0 * small) * rng.uniform(0.3, 1.1, n) * self.burst
        self.V = dirs * sp[:, None]
        self.V[:, 1] = up
        self.V += self.whole_V * np.array([0.0, -0.08, 0.0])
        if n == 2:   # Hälften kippen nach außen auf die Rundung, Schnittfläche zeigt nach oben
            axis = np.cross(np.array([0.0, 1.0, 0.0]), dirs * np.array([1.0, 0.0, 1.0]))
            self.W = axis * 4.2
        else:
            self.W = rng.normal(0, 1, (n, 3)) * (6.0 * small + 2.0)[:, None]


# ------------------------------------------------------------------ Szene / Bild

def box_blur(a, r):
    c = np.cumsum(np.pad(a, ((r + 1, r), (0, 0)), mode="edge"), 0)
    a = (c[2 * r + 1:] - c[:-2 * r - 1]) / (2 * r + 1)
    c = np.cumsum(np.pad(a, ((0, 0), (r + 1, r)), mode="edge"), 1)
    return (c[:, 2 * r + 1:] - c[:, :-2 * r - 1]) / (2 * r + 1)


class Stage:
    def __init__(self, cam=None):
        self.cam = cam or Camera()
        self.cp = self.cam.params()
        self.bg = np.zeros((H1, W1, 3))
        self.floor_xz = np.zeros((H1, W1, 2))
        render_background(self.bg, self.floor_xz, self.cp, W1, H1)
        self.tmp = np.zeros((2_000_000, 3))

    def render(self, pieces, whole):
        f = pieces.f
        img = self.bg.copy()
        zb = np.full((H1, W1), 1e18)
        # Schatten
        buf = np.zeros((SH_H, SH_W))
        step = max(1, len(f.P) // 250_000)
        m = world_points(f.P, f.lab, pieces.Rs, pieces.Ts, f.com, self.tmp, step)
        shadow_splat(buf, self.tmp[:m], LIGHT[0], LIGHT[1], LIGHT[2])
        sh = 1.0 - np.exp(-box_blur(box_blur(buf * step * 0.6, 4), 3))
        # Kontaktschatten direkt darunter
        buf2 = np.zeros((SH_H, SH_W))
        low = self.tmp[:m][self.tmp[:m, 1] < 25]
        shadow_splat(buf2, low, 0.0, 1.0, 0.0)
        sh = np.clip(sh + 0.6 * (1.0 - np.exp(-box_blur(buf2 * step * 0.8, 2))), 0, 1)
        shade_floor(img, self.floor_xz, sh, W1, H1)
        if whole:
            R, T = pieces.whole_pose()
            rad = {APPLE: 75, ORANGE: 62, KIWI: 70, STRAWBERRY: 62, MELON: 92}[f.fid]
            c = self.project(T)
            sz = self.cp[12] * rad / max(1.0, np.dot(T - self.cam.pos, self.cam.fwd)) + 6
            raymarch(img, zb, f.fid, R, T, self.cp, int(c[0] - sz), int(c[1] - sz), int(c[0] + sz), int(c[1] + sz),
                     float(rad), LIGHT[0], LIGHT[1], LIGHT[2])
        else:
            splat_voxels(img, zb, f.P, f.Nrm, f.Col, f.lab, pieces.Rs, pieces.Ts, f.com, self.cp,
                         LIGHT[0], LIGHT[1], LIGHT[2], 1.12)
        return img

    def project(self, p):
        v = p - self.cam.pos
        z = v @ self.cam.fwd
        return (self.cam.cx + self.cam.f * (v @ self.cam.right) / z, self.cam.cy - self.cam.f * (v @ self.cam.up) / z)


def tone(img):
    """Filmischer Look: weiche Schulter, leichte Vignette."""
    x = img / 255.0
    x = x / (1.0 + 0.35 * x)
    x = x * 1.25
    return np.clip(x ** (1 / 1.05) * 255, 0, 255).astype(np.uint8)


# ------------------------------------------------------------------ Runden, Texte, Ablauf

ROUND_LEN = 10.2
HOVER = 1.6


def fmt(n):
    return f"{n:,}"


class Round:
    def __init__(self, fid, n, name, seed, idx, n_rounds, burst=1.0):
        self.fid, self.n, self.name, self.seed, self.idx, self.n_rounds, self.burst = fid, n, name, seed, idx, n_rounds, burst

    def setup(self):
        self.fruit = Fruit(self.fid, self.n, self.seed)
        self.pieces = Pieces(self.fruit, self.seed, hover_y=170.0, burst=self.burst)
        self.t = 0.0

    def advance(self, t_to):
        dt = 1.0 / 120
        while self.t + 1e-9 < t_to:
            self.pieces.step(self.t, dt, self.t >= HOVER)
            self.t += dt

    def impact(self):
        return self.pieces.impact_t


F_CACHE = {}


def font(path, size):
    k = (path, size)
    if k not in F_CACHE:
        F_CACHE[k] = ImageFont.truetype(path, size)
    return F_CACHE[k]


def overlay(im, rd, t, imp, first, last):
    d = ImageDraw.Draw(im, "RGBA")
    cx = 520
    # Kopfzeile: Runde + Frucht + Frage
    d.text((cx, 248), f"ROUND {rd.idx + 1}/{rd.n_rounds}  ·  {rd.name}", font=font(F_SEMI, 30),
           fill=(235, 235, 245, 230), anchor="mm", stroke_width=3, stroke_fill=(0, 0, 0, 160))
    if imp is None or t < imp + 0.35:
        d.text((cx, 302), "HOW MANY PIECES?", font=font(F_BOLD, 58), fill=(255, 220, 90, 255), anchor="mm",
               stroke_width=5, stroke_fill=(0, 0, 0, 190))
    # Zähler
    cy = 1350
    if imp is None or t < imp + 0.3:
        txt, sub, col, sc = "?", "", (255, 255, 255), 1.0
    else:
        e = min(1.0, (t - imp - 0.3) / 1.6)
        k = 1 - (1 - e) ** 3
        val = int(round(rd.n * k)) if rd.n > 1 else 1
        txt = fmt(val)
        done = e >= 1.0
        sub = ("PIECE" if rd.n == 1 else "PIECES") if done else ""
        col = (255, 220, 90) if done else (255, 255, 255)
        pop = max(0.0, 1 - (t - imp - 1.9) / 0.35) if done and t - imp - 1.9 < 0.35 else 0.0
        sc = 1.0 + 0.25 * pop
    f = font(F_BOLD, int(110 * sc))
    w = d.textlength(txt, font=f)
    bw = max(260, w + 90)
    d.rounded_rectangle([cx - bw / 2, cy - 92, cx + bw / 2, cy + 92], radius=34, fill=(0, 0, 0, 165),
                        outline=(255, 220, 90, 170), width=3)
    d.text((cx, cy - 12), txt, font=f, fill=col + (255,), anchor="mm")
    if sub:
        d.text((cx, cy + 58), sub, font=font(F_BOLD, 34), fill=(255, 255, 255, 235), anchor="mm")
    # Hook zu Beginn, Abschluss am Ende
    if first and (imp is None or t < imp + 0.35):
        d.text((cx, 356), "Guess before it lands! Count your score", font=font(F_SEMI, 30),
               fill=(255, 255, 255, 235), anchor="mm", stroke_width=3, stroke_fill=(0, 0, 0, 170))
    if last and imp is not None and t > imp + 4.2:
        a = min(1.0, (t - imp - 4.2) / 0.5)
        d.rounded_rectangle([cx - 410, 650 - 80, cx + 410, 650 + 80], radius=30, fill=(0, 0, 0, int(200 * a)))
        d.text((cx, 620), "How many did you get right?", font=font(F_BOLD, 50), fill=(255, 255, 255, int(255 * a)),
               anchor="mm")
        d.text((cx, 685), "Comment your score!", font=font(F_SEMI, 36), fill=(255, 220, 90, int(255 * a)), anchor="mm")


def render_round(args):
    rd, out, first, last, preview = args
    rd.setup()
    st = Stage()
    n_fr = int(ROUND_LEN * FPS)
    times = preview if preview else range(n_fr)
    proc = None
    if not preview:
        cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W1}x{H1}", "-r", str(FPS),
               "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-maxrate", "14M", "-bufsize", "28M",
               "-pix_fmt", "yuv420p", "-g", "120", out]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    shots = []
    for fi in times:
        t = fi / FPS if not preview else fi
        rd.advance(t)
        whole = not rd.pieces.broken
        img = tone(st.render(rd.pieces, whole))
        # Ein-/Ausblenden zwischen den Runden
        fade = min(1.0, t / 0.25, (ROUND_LEN - t) / 0.25) if not first else min(1.0, (ROUND_LEN - t) / 0.25)
        if last:
            fade = min(1.0, t / 0.25) if t < 1 else min(1.0, (ROUND_LEN - t) / 0.8)
        im = Image.fromarray(img)
        overlay(im, rd, t, rd.impact(), first, last)
        if fade < 1.0:
            im = Image.fromarray((np.asarray(im, np.float32) * max(0.0, fade)).astype(np.uint8))
        if preview:
            shots.append(im)
        else:
            proc.stdin.write(im.tobytes())
    if proc:
        proc.stdin.close()
        proc.wait()
    return shots if preview else (out, rd.impact())


# ------------------------------------------------------------------ Klang

def noise_burst(dur, decay, lp, seed):
    rng = np.random.default_rng(seed)
    n = int(SR * dur)
    x = rng.standard_normal(n)
    if lp > 1:
        x = np.convolve(x, np.ones(lp) / lp, mode="same")
    t = np.arange(n) / SR
    return x * np.exp(-t * decay) * np.minimum(t / 0.002, 1)


def build_audio(rounds, impacts, path):
    from .viral import soft_tone
    total = ROUND_LEN * len(rounds)
    out = np.zeros(int(SR * (total + 3)))
    rng = np.random.default_rng(4)

    def put(t, s, vol):
        i = int(t * SR)
        e = min(len(out), i + len(s))
        if i < e:
            out[i:e] += s[: e - i] * vol

    for k, (rd, imp) in enumerate(zip(rounds, impacts)):
        t0 = k * ROUND_LEN
        # Fallgeräusch
        w = noise_burst(0.45, 0.0, 30, k)
        w *= np.linspace(0, 1, len(w)) ** 2
        put(t0 + HOVER, w, 0.25)
        if imp is None:
            continue
        ti = t0 + imp
        tt = np.arange(int(SR * 0.5)) / SR
        thud = np.sin(2 * np.pi * (55 + 60 * np.exp(-tt * 30)) * tt) * np.exp(-tt * 9)
        put(ti, thud, 0.9)
        n = rd.n
        if n == 1:
            for j, dt in enumerate((0.23, 0.38)):
                put(ti + dt, thud[: int(SR * 0.3)] * 0.5, 0.5 / (j + 1))
        elif n == 2:
            put(ti, noise_burst(0.12, 40, 3, 7), 0.7)
        elif n < 100:
            for j in range(min(n, 40)):
                put(ti + rng.uniform(0, 0.25) ** 1.5, noise_burst(0.06, 70, 2, j), 0.35)
        elif n < 50_000:
            for j in range(160):
                put(ti + rng.uniform(0, 0.7) ** 2, noise_burst(0.025, 160, 1, j), 0.22)
        else:
            hiss = noise_burst(2.2, 2.2, 1, 9)
            put(ti, hiss, 0.45)
        # Zähler-Klicks + Auflösung
        for j in range(18):
            u = j / 17
            tc = ti + 0.3 + 1.6 * (1 - (1 - u) ** (1 / 3)) * 0.98
            put(tc, noise_burst(0.012, 400, 1, 100 + j), 0.12)
        for j, f in enumerate((659.25, 783.99, 1046.5)):
            put(ti + 1.9 + j * 0.06, soft_tone(f), 0.4)
    out = out[: int(SR * total)]
    tt = np.arange(len(out)) / SR
    pad = (np.sin(2 * np.pi * 110 * tt) + 0.6 * np.sin(2 * np.pi * 164.8 * tt) + 0.4 * np.sin(2 * np.pi * 220 * tt))
    out = out + reverb(out, 0.8) * 0.12 + pad * 0.012 * np.minimum(1, tt / 2)
    out *= np.clip((total - tt) / 1.0, 0, 1)
    peak = np.max(np.abs(out)) or 1
    out = np.tanh(out / peak * 1.3) * 0.9
    st = np.stack([out, out], axis=1)
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((st * 32767).astype(np.int16).tobytes())


def make_video(rounds, out_path, jobs=4):
    with tempfile.TemporaryDirectory() as tmp:
        tasks = [(rd, os.path.join(tmp, f"r{k}.mp4"), k == 0, k == len(rounds) - 1, None)
                 for k, rd in enumerate(rounds)]
        with Pool(jobs) as p:
            res = p.map(render_round, tasks, chunksize=1)
        lst = os.path.join(tmp, "list.txt")
        with open(lst, "w") as f:
            for pth, _ in res:
                f.write(f"file '{pth}'\n")
        wav = os.path.join(tmp, "a.wav")
        build_audio(rounds, [imp for _, imp in res], wav)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-i", wav, "-c:v", "copy",
                        "-c:a", "aac", "-b:a", "192k", "-shortest", "-af", "loudnorm=I=-14:TP=-1.5:LRA=11",
                        "-movflags", "+faststart", out_path], check=True)


def fruit_rounds():
    spec = [(APPLE, 24, "APPLE", 1.0), (ORANGE, 2, "ORANGE", 1.0), (KIWI, 1000, "KIWI", 0.62),
            (STRAWBERRY, 1, "STRAWBERRY", 1.0), (MELON, 816392, "WATERMELON", 0.26)]
    return [Round(fid, n, name, 1, k, len(spec), burst) for k, (fid, n, name, burst) in enumerate(spec)]


if __name__ == "__main__":
    import sys
    rounds = fruit_rounds()
    if "--preview" in sys.argv:
        k = int(sys.argv[sys.argv.index("--preview") + 1])
        shots = render_round((rounds[k], None, k == 0, k == len(rounds) - 1, [0.8, 2.05, 2.6, 4.2, 9.0]))
        out = Image.new("RGB", (540 * len(shots), 960))
        for i, im in enumerate(shots):
            out.paste(im.resize((540, 960)), (i * 540, 0))
        out.save(f"output/preview/shatter_round{k}.png")
        tiktok_ui_mock(shots[-1], "vertical").save(f"output/preview/shatter_round{k}_ui.png")
    else:
        make_video(rounds, "output/f01_how_many_pieces_vertical.mp4")
        print("output/f01_how_many_pieces_vertical.mp4")
