"""Die große Murmelbahn: Bounce-Kreis -> Trichter -> Speedband -> Schanze -> Trampolin -> Halfpipe -> Wippe
-> Serpentinen -> Speedband -> Looping -> Katapult -> goldene Ziel-Schale.

Ruhig, rollend statt springend. Ab dem Trichter folgen die Bälle Schienen (Bahnphysik mit Schwerkraft,
Rollreibung, Speedbändern); Sprünge sind echte Wurfparabeln, deren Abwurf so berechnet ist, dass sie ihr Ziel
treffen. Welt-Koordinaten: x in [-420, 420], y nach unten, Start-Kreis um (0, 0).
"""
import math
import random

G = 900.0
R_BALL = 17.0
FPS = 60
SUB = 8
DT = 1.0 / (FPS * SUB)

BALL_COLORS = [(255, 122, 110), (100, 190, 255), (255, 208, 90)]


# ------------------------------------------------------------------ Bahnen

class Path:
    """Dicht abgetastete Mittellinie für den Ballmittelpunkt + Seite, auf der die Schiene liegt."""

    def __init__(self, name, drag=0.5, boosts=(), governor=None, bidir=False):
        self.name, self.drag, self.boosts, self.governor, self.bidir = name, drag, list(boosts), governor, bidir
        self.pts, self.nrm = [], []  # nrm: Richtung vom Ballmittelpunkt zur Schiene

    def _add(self, p, n):
        if self.pts and math.hypot(p[0] - self.pts[-1][0], p[1] - self.pts[-1][1]) < 1e-6:
            return
        self.pts.append(p)
        self.nrm.append(n)

    def line(self, p0, p1, side=None, step=3.0):
        L = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        tx, ty = (p1[0] - p0[0]) / L, (p1[1] - p0[1]) / L
        n = (-ty, tx) if (-ty, tx)[1] >= 0 else (ty, -tx)
        if side is not None:
            n = side
        k = max(1, int(L / step))
        for i in range(k + 1):
            u = i / k
            self._add((p0[0] + (p1[0] - p0[0]) * u, p0[1] + (p1[1] - p0[1]) * u), n)
        return self

    def arc(self, c, r, a0, a1, step=3.0):
        """Ball läuft innen im Bogen, Schiene liegt außen."""
        k = max(2, int(abs(a1 - a0) * r / step))
        for i in range(k + 1):
            a = a0 + (a1 - a0) * i / k
            self._add((c[0] + math.cos(a) * r, c[1] + math.sin(a) * r), (math.cos(a), math.sin(a)))
        return self

    def done(self):
        self.s = [0.0]
        for (x0, y0), (x1, y1) in zip(self.pts, self.pts[1:]):
            self.s.append(self.s[-1] + math.hypot(x1 - x0, y1 - y0))
        self.length = self.s[-1]
        return self

    def at(self, s):
        s = min(max(s, 0.0), self.length)
        lo, hi = 0, len(self.s) - 1
        while hi - lo > 1:
            m = (lo + hi) // 2
            if self.s[m] <= s:
                lo = m
            else:
                hi = m
        seg = self.s[hi] - self.s[lo] or 1e-9
        u = (s - self.s[lo]) / seg
        (x0, y0), (x1, y1) = self.pts[lo], self.pts[hi]
        tx, ty = (x1 - x0) / seg, (y1 - y0) / seg
        return (x0 + (x1 - x0) * u, y0 + (y1 - y0) * u), (tx, ty)

    def nearest_s(self, p):
        best, bs = 1e18, 0.0
        for i, (x, y) in enumerate(self.pts):
            d = (x - p[0]) ** 2 + (y - p[1]) ** 2
            if d < best:
                best, bs = d, self.s[i]
        return bs


# Start-Kreis
C0, R0 = (0.0, 0.0), 270.0
GAP_HALF = 0.2

# Trichter: zwei Schrägen zum Auslauf S
S = (0.0, 540.0)
FUNNEL = [((-410.0, 330.0), S), ((410.0, 330.0), S)]

# Trampolin
TRAMP = dict(x=-300.0, y=1180.0, half=75.0, depth=30.0)
# Wippe (rechte Seite oben in Ruhe)
SEESAW = dict(p=(-280.0, 1480.0), half=130.0, phi=math.radians(12), land_u=40.0, th=6.0)
# Katapult (Arm zeigt in Ruhe nach rechts)
DY = 520.0
CATA = dict(p=(-60.0, 2440.0 + DY), arm=150.0, rest=math.radians(350), wind=math.radians(362),
            release=math.radians(318), end=math.radians(250), slots=((120.0, 24.0), (154.0, 24.0), (137.0, 54.0)))  # Becher: 2 unten, 1 oben
# Ziel-Schale
BOWL = dict(c=(-245.0, 2650.0 + DY), r=150.0)


def build_paths():
    P = {}
    # Trichterschrägen
    for name, (p0, p1) in zip(("fl", "fr"), FUNNEL):
        P[name] = Path(name, drag=0.4).line(p0, p1).done()
    # A: Auslauf nach unten, Kurve nach rechts
    A = Path("A", drag=1.0).line(S, (0.0, 590.0), side=(-1.0, 0.0))
    A.arc((60.0, 590.0), 60.0, math.pi, math.pi / 2).line((60.0, 650.0), (340.0, 720.0))
    P["A"] = A.done()
    P["U1"] = Path("U1", drag=0.45).arc((340.0, 795.0), 75.0, -math.pi / 2, math.pi / 2).done()
    # B: nach links mit Speedband 1, Schanze am Ende
    B = Path("B", drag=1.0).line((340.0, 870.0), (-50.0, 950.0))  # Schanze am Ende
    B.arc((-50.0, 790.0), 160.0, math.pi / 2, math.pi / 2 + 0.36)
    P["B"] = B.done()
    # C: Halfpipe rechts, dann nach links
    C = Path("C", drag=0.9, bidir=True).arc((290.0, 1140.0), 110.0, 0.0, math.pi / 2)
    C.line((290.0, 1250.0), (-190.0, 1330.0))
    P["C"] = C.done()
    # D, U2, E, U3, F (Speedband 2), Looping, G, U4, H (zum Katapult)
    P["D"] = Path("D", drag=1.0).line((-40.0, 1620.0), (330.0, 1690.0)).done()
    P["D"].boosts = [(70.0, 250.0, 520.0)]
    P["U2"] = Path("U2", drag=0.45).arc((330.0, 1765.0), 75.0, -math.pi / 2, math.pi / 2).done()
    P["E"] = Path("E", drag=1.05).line((330.0, 1840.0), (-330.0, 1960.0)).done()
    P["U3"] = Path("U3", drag=0.45).arc((-330.0, 2035.0), 75.0, -math.pi / 2, -3 * math.pi / 2).done()
    # zusätzliche Serpentinen-Doppelkehre
    P["E2"] = Path("E2", drag=1.05).line((-330.0, 2110.0), (330.0, 2220.0)).done()
    P["U5"] = Path("U5", drag=0.45).arc((330.0, 2295.0), 75.0, -math.pi / 2, math.pi / 2).done()
    P["E3"] = Path("E3", drag=1.05).line((330.0, 2370.0), (-330.0, 2480.0)).done()
    P["U6"] = Path("U6", drag=0.45).arc((-330.0, 2555.0), 75.0, -math.pi / 2, -3 * math.pi / 2).done()
    dy = DY
    F = Path("F", drag=0.3).line((-330.0, 2110.0 + dy), (120.0, 2160.0 + dy)).line((120.0, 2160.0 + dy), (200.0, 2160.0 + dy))
    P["F"] = F.done()
    P["F"].boosts = [(60.0, 380.0, 760.0)]
    RL = 85.0
    P["LOOP"] = Path("LOOP", drag=0.12).arc((200.0, 2160.0 + dy - RL), RL, math.pi / 2, math.pi / 2 - 2 * math.pi).done()
    P["G"] = Path("G", drag=0.6).line((200.0, 2160.0 + dy), (330.0, 2190.0 + dy)).done()
    P["U4"] = Path("U4", drag=0.4).arc((330.0, 2265.0 + dy), 75.0, -math.pi / 2, math.pi / 2).done()
    H = Path("H", drag=1.0).line((330.0, 2340.0 + dy), (165.0, 2378.0 + dy))
    P["H"] = H.done()
    P["H"].governor = (P["H"].length - 160.0, 150.0, 2.0)
    return P


PATHS = build_paths()
NEXT = {"fl": "A", "fr": "A", "A": "U1", "U1": "B", "D": "U2", "U2": "E", "E": "U3", "U3": "E2", "E2": "U5", "U5": "E3", "E3": "U6", "U6": "F", "F": "LOOP",
        "LOOP": "G", "G": "U4", "U4": "H"}


def solve_dir_speed(p0, u, target):
    """Wurfgeschwindigkeit entlang Richtung u, damit die Parabel target trifft (y nach unten)."""
    dx, dy = target[0] - p0[0], target[1] - p0[1]
    ux, uy = u
    den = ux * ux * (dy - uy * dx / ux)
    if den <= 0:
        raise ValueError("Ziel nicht erreichbar")
    v = math.sqrt(0.5 * G * dx * dx / den)
    return v, dx / (ux * v)


def solve_time(p0, target, T):
    return ((target[0] - p0[0]) / T, (target[1] - p0[1] - 0.5 * G * T * T) / T)


# ------------------------------------------------------------------ Mechanik-Hilfen

def seesaw_point(phi, u, lift):
    px, py = SEESAW["p"]
    dx, dy = math.cos(phi), math.sin(phi)
    nx, ny = dy, -dx  # Normale nach oben
    return (px + u * dx + lift * nx, py + u * dy + lift * ny)


def tramp_target():
    return (TRAMP["x"], TRAMP["y"] - R_BALL - 3)


def ease(p):
    p = min(max(p, 0.0), 1.0)
    return p * p * (3 - 2 * p)


def back_out(p, c=1.4):
    p = min(max(p, 0.0), 1.0)
    return 1 + (c + 1) * (p - 1) ** 3 + c * (p - 1) ** 2


# ------------------------------------------------------------------ Simulation

class MBall:
    def __init__(self, i, x, y, vx, vy):
        self.i, self.x, self.y, self.vx, self.vy = i, x, y, vx, vy
        self.color = BALL_COLORS[i]
        self.state = "circle"
        self.path, self.s, self.v = None, 0.0, 0.0
        self.fly = None  # (p0, v0, t0, T, then)
        self.order = None
        self.data = {}
        self.last_note = -1.0


class MarbleSim:
    def __init__(self, seed, outro=5.5):
        self.seed = seed
        self.rng = random.Random(seed)
        rng = self.rng
        self.t = 0.0
        self.events = []
        self.frames = []
        self.balls = []
        for i in range(3):
            a = rng.uniform(0, 2 * math.pi)
            sp = rng.uniform(380, 520)
            self.balls.append(MBall(i, -90 + 90 * i + rng.uniform(-20, 20), rng.uniform(-120, -40),
                                    math.cos(a) * sp, math.sin(a) * sp))
        self.gap_phase = rng.uniform(0, 2 * math.pi)
        self.escaped = 0
        self.seesaw_ev = []  # (t_land, t_leave)
        self.tramp_ev = []
        self.cup = []  # Reihenfolge der Bälle im Katapult
        self.fire_t = None
        self.goal_n = 0
        self.finish_t = None
        self.outro = outro
        self.swing_dur = 0.6
        self._prep_launches()
        self._solve_swing()

    # Vorab: Abwurfgeschwindigkeiten der fest geplanten Sprünge
    def _prep_launches(self):
        B = PATHS["B"]
        pK, tK = B.at(B.length)
        self.v_kick, self.T_kick = solve_dir_speed(pK, tK, tramp_target())
        B.boosts = []
        B.governor = (B.length - 220.0, self.v_kick, 1.6)  # sanft auf Absprunggeschwindigkeit
        C = PATHS["C"]
        self.s_land_C = C.nearest_s((290.0 + 110 * math.cos(math.radians(70)), 1140.0 + 110 * math.sin(math.radians(70))))
        pC, tC = C.at(C.length)
        lift = R_BALL + SEESAW["th"]
        self.seesaw_target = seesaw_point(-SEESAW["phi"], SEESAW["land_u"], lift)
        self.v_drop_C, self.T_drop_C = solve_dir_speed(pC, tC, self.seesaw_target)
        C.governor = (C.length - 260.0, self.v_drop_C, 1.8)
        # Wippe gekippt -> Abwurf nach rechts unten auf D
        phi = SEESAW["phi"]
        self.seesaw_exit = seesaw_point(phi, SEESAW["half"], lift)
        self.v_seesaw, self.T_seesaw = solve_dir_speed(self.seesaw_exit, (math.cos(phi), math.sin(phi)), PATHS["D"].pts[0])

    def _landing_x(self, k):
        """Wo landet Becher-Platz k (auf Höhe der Schalenmitte), wenn beim Abwurfwinkel losgelassen wird?"""
        c = CATA
        d = (c["wind"] - c["release"]) / (c["wind"] - c["end"])
        om = -(c["wind"] - c["end"]) * 6 * d * (1 - d) / self.swing_dur
        a = c["release"]
        rho, lift = c["slots"][k]
        px, py = c["p"]
        x = px + rho * math.cos(a) + lift * math.sin(a)
        y = py + rho * math.sin(a) - lift * math.cos(a)
        vx, vy = -om * (y - py), om * (x - px)
        yt = BOWL["c"][1] + 40
        # y(t) = y + vy t + G/2 t^2 = yt
        A, Bq, Cq = 0.5 * G, vy, y - yt
        tt = (-Bq + math.sqrt(Bq * Bq - 4 * A * Cq)) / (2 * A)
        return x + vx * tt

    def _solve_swing(self):
        target = BOWL["c"][0]
        lo, hi = 0.3, 3.0
        for _ in range(50):  # längere Dauer -> langsamer -> landet näher
            mid = (lo + hi) / 2
            self.swing_dur = mid
            mx = sum(self._landing_x(k) for k in range(3)) / 3
            if mx < target:
                lo = mid
            else:
                hi = mid
        self.landings = [round(self._landing_x(k)) for k in range(3)]

    def ev(self, kind, **d):
        self.events.append((self.t, kind, d))

    def gap_center(self, t):
        return math.pi / 2 + 0.8 * math.sin(0.9 * t + self.gap_phase)

    # ---- Zustandswechsel

    def to_path(self, b, name, s=0.0, v=None):
        b.state, b.path, b.s = "rail", PATHS[name], s
        if v is not None:
            b.v = v

    def launch(self, b, p0, v0, T, then, kind="jump"):
        b.state = "fly"
        b.fly = (p0, v0, self.t, T, then)
        b.x, b.y = p0
        self.ev(kind, ball=b.i)

    def seesaw_phi(self, t):
        """Winkel der Wippe: -phi = rechte Seite oben (Ruhe), +phi = rechte Seite unten."""
        phi0 = SEESAW["phi"]
        ang = -phi0
        for tl, te in self.seesaw_ev:
            if t < tl:
                break
            ang = -phi0 + 2 * phi0 * back_out((t - tl) / 0.35, 1.2)
            if te is not None and t >= te:
                ang = phi0 - 2 * phi0 * back_out((t - te) / 0.6, 0.8)
        return ang

    def cata_angle(self, t):
        c = CATA
        if self.fire_t is None or t < self.fire_t:
            return c["rest"]
        e = t - self.fire_t
        if e < 0.7:  # ausholen
            return c["rest"] + (c["wind"] - c["rest"]) * ease(e / 0.7)
        e -= 0.95
        if e < 0:
            return c["wind"]
        sd = self.swing_dur
        if e < sd:  # Schwung (gegen den Uhrzeigersinn)
            return c["wind"] + (c["end"] - c["wind"]) * ease(e / sd)
        e -= sd
        if e < 0.5:
            return c["end"] + (c["end"] - c["wind"]) * 0.04 * math.sin(e / 0.5 * math.pi)
        e -= 0.5
        return c["end"] + (c["rest"] - 2 * math.pi - c["end"] + 2 * math.pi) * ease(e / 2.2)

    def cata_omega(self, t):
        h = 1e-4
        return (self.cata_angle(t + h) - self.cata_angle(t - h)) / (2 * h)

    def cup_pos(self, k, t):
        c = CATA
        a = self.cata_angle(t)
        px, py = c["p"]
        rho, lift = c["slots"][k]
        dx, dy = math.cos(a), math.sin(a)
        nx, ny = dy, -dx  # oberhalb des Arms (Arm zeigt nach rechts -> Normale nach oben)
        return (px + rho * dx + lift * nx, py + rho * dy + lift * ny)

    # ---- Schritt

    def step(self):
        dt, t = DT, self.t
        for b in self.balls:
            getattr(self, "st_" + b.state)(b, dt, t)
        self.collide_circle_balls()
        self.collide_bowl()
        self.t += dt

    def collide_circle_balls(self):
        cb = [b for b in self.balls if b.state == "circle"]
        for i in range(len(cb)):
            for j in range(i + 1, len(cb)):
                a, c = cb[i], cb[j]
                dx, dy = c.x - a.x, c.y - a.y
                d = math.hypot(dx, dy)
                if 0 < d < 2 * R_BALL:
                    nx, ny = dx / d, dy / d
                    rel = (a.vx - c.vx) * nx + (a.vy - c.vy) * ny
                    if rel > 0:
                        a.vx -= rel * nx
                        a.vy -= rel * ny
                        c.vx += rel * nx
                        c.vy += rel * ny
                    ov = (2 * R_BALL - d) / 2
                    a.x -= nx * ov
                    a.y -= ny * ov
                    c.x += nx * ov
                    c.y += ny * ov

    def st_circle(self, b, dt, t):
        rng = self.rng
        b.vy += G * dt
        b.x += b.vx * dt
        b.y += b.vy * dt
        d = math.hypot(b.x, b.y) or 1e-6
        ang = math.atan2(b.y, b.x)
        in_gap = abs((ang - self.gap_center(t) + math.pi) % (2 * math.pi) - math.pi) < GAP_HALF
        if in_gap and d > R0:
            b.state = "fall"
            b.order = self.escaped
            self.escaped += 1
            self.ev("escape", ball=b.i)
            return
        if not in_gap and d + R_BALL > R0:
            nx, ny = b.x / d, b.y / d
            b.x, b.y = nx * (R0 - R_BALL), ny * (R0 - R_BALL)
            vn = b.vx * nx + b.vy * ny
            if vn > 0:
                b.vx -= 2 * vn * nx
                b.vy -= 2 * vn * ny
                a = rng.uniform(-0.25, 0.25)
                cs, sn = math.cos(a), math.sin(a)
                b.vx, b.vy = b.vx * cs - b.vy * sn, b.vx * sn + b.vy * cs
                sp = math.hypot(b.vx, b.vy)
                k = min(max(sp, 380), 620) / sp
                b.vx *= k
                b.vy *= k
                self.ev("bounce", ball=b.i, strength=min(1.0, vn / 700), x=nx * R0, y=ny * R0)

    def st_fall(self, b, dt, t):
        b.vy += G * dt
        b.x += b.vx * dt
        b.y += b.vy * dt
        if abs(b.x) > 405 and b.vx * b.x > 0:  # seitliche Leitwand
            b.x = math.copysign(405, b.x)
            b.vx = -b.vx * 0.5
        name = "fl" if b.x < 0 else "fr"
        (x0, y0), (x1, y1) = FUNNEL[0] if name == "fl" else FUNNEL[1]
        yl = y0 + (b.x - x0) * (y1 - y0) / (x1 - x0)
        if b.y >= yl and abs(b.x) <= 410:
            path = PATHS[name]
            s = path.nearest_s((b.x, b.y))
            (px, py), (tx, ty) = path.at(s)
            b.x, b.y = px, py
            self.to_path(b, name, s, max(60.0, b.vx * tx + b.vy * ty))
            self.ev("land", ball=b.i, strength=0.5, x=px, y=py)

    def st_rail(self, b, dt, t):
        path = b.path
        (_, _), (tx, ty) = path.at(b.s)
        a = G * ty - path.drag * b.v
        for s0, s1, vt in path.boosts:
            if s0 <= b.s <= s1 and vt:
                a += 3.0 * (vt - b.v)
                if not b.data.get(("boost", path.name)):
                    b.data[("boost", path.name)] = True
                    self.ev("boost", ball=b.i)
        if path.governor and b.s >= path.governor[0]:
            gain = path.governor[2] if len(path.governor) > 2 else 4.0
            a = gain * (path.governor[1] - b.v)
        b.v += a * dt
        if not path.bidir and b.v < 40:
            b.v = 40.0
        b.s += b.v * dt
        if path.bidir and b.s < 0:
            b.s, b.v = 0.0, -b.v * 0.5
        (b.x, b.y), _ = path.at(b.s)
        if path.name == "BOWL":
            return
        if b.s >= path.length:
            self.rail_end(b, t)

    def rail_end(self, b, t):
        name = b.path.name
        p, tan = b.path.at(b.path.length)
        if name in NEXT:
            self.to_path(b, NEXT[name], 0.0)
            if NEXT[name] == "LOOP":
                self.ev("loop", ball=b.i)
            return
        if name == "B":  # Schanze -> Trampolin
            v = self.v_kick
            self.launch(b, p, (tan[0] * v, tan[1] * v), self.T_kick, "tramp")
        elif name == "C":  # Kante -> Wippe
            v = self.v_drop_C
            self.launch(b, p, (tan[0] * v, tan[1] * v), self.T_drop_C, "seesaw", kind="drop")
        elif name == "H":  # ins Katapult
            k = len(self.cup)
            self.cup.append(b.i)
            b.state = "cup_in"
            b.data["cup"] = (k, t, p)
            self.ev("cup", ball=b.i, k=k)

    def st_fly(self, b, dt, t):
        p0, v0, t0, T, then = b.fly
        e = t - t0
        b.x = p0[0] + v0[0] * e
        b.y = p0[1] + v0[1] * e + 0.5 * G * e * e
        b.vx, b.vy = v0[0], v0[1] + G * e
        if e < T:
            return
        if then == "tramp":
            b.state = "tramp"
            b.data["tramp_t"] = t
            self.tramp_ev.append(t)
            self.ev("tramp", ball=b.i)
        elif then == "C":
            C = PATHS["C"]
            (px, py), (tx, ty) = C.at(self.s_land_C)
            self.to_path(b, "C", self.s_land_C, b.vx * tx + b.vy * ty)
            self.ev("land", ball=b.i, strength=0.6, x=px, y=py)
        elif then == "seesaw":
            b.state = "seesaw"
            b.data["ss_t"] = t
            b.data["ss_u"] = SEESAW["land_u"]
            b.data["ss_v"] = 0.0
            self.seesaw_ev.append([t, None])
            self.ev("seesaw", ball=b.i)
        elif then == "D":
            D = PATHS["D"]
            (_, _), (tx, ty) = D.at(0)
            self.to_path(b, "D", 0.0, b.vx * tx + b.vy * ty)
            self.ev("land", ball=b.i, strength=0.5, x=b.x, y=b.y)
        elif then == "bowl":
            self.enter_bowl(b, t)

    def st_tramp(self, b, dt, t):
        e = t - b.data["tramp_t"]
        tx, ty = tramp_target()
        dur = 0.16
        b.x = tx
        b.y = ty + TRAMP["depth"] * math.sin(math.pi * min(e / dur, 1.0))
        if e >= dur:
            target = PATHS["C"].at(self.s_land_C)[0]
            v0 = solve_time((tx, ty), target, 1.25)
            self.launch(b, (tx, ty), v0, 1.25, "C", kind="boing")

    def st_seesaw(self, b, dt, t):
        lift = R_BALL + SEESAW["th"]
        tl = b.data["ss_t"]
        phi = self.seesaw_phi(t)
        u = b.data["ss_u"]
        if t - tl > 0.3:  # nach dem Kippen rollt der Ball zum rechten Ende
            L = SEESAW["half"] - SEESAW["land_u"]
            a = self.v_seesaw ** 2 / (2 * L)
            b.data["ss_v"] += a * dt
            u += b.data["ss_v"] * dt
            b.data["ss_u"] = u
        b.x, b.y = seesaw_point(phi, u, lift)
        if u >= SEESAW["half"]:
            for ev in self.seesaw_ev:
                if ev[0] == tl:
                    ev[1] = t
            d = (math.cos(phi), math.sin(phi))
            v = self.v_seesaw
            self.launch(b, (b.x, b.y), (d[0] * v, d[1] * v), self.T_seesaw, "D", kind="roll_off")

    def st_cup_in(self, b, dt, t):
        k, t0, p0 = b.data["cup"]
        e = (t - t0) / 0.4
        tgt = self.cup_pos(k, t)
        f = back_out(e, 1.1)
        b.x = p0[0] + (tgt[0] - p0[0]) * f
        b.y = p0[1] + (tgt[1] - p0[1]) * f - math.sin(min(e, 1) * math.pi) * 34
        if e >= 1:
            b.state = "cup"
            if len(self.cup) == 3 and self.fire_t is None:
                self.fire_t = t + 1.1
                self.ev("wind", at=self.fire_t)

    def st_cup(self, b, dt, t):
        k = b.data["cup"][0]
        b.x, b.y = self.cup_pos(k, t)
        if self.fire_t is not None and t >= self.fire_t + 0.95:
            a = self.cata_angle(t)
            if a <= CATA["release"]:
                om = self.cata_omega(t)
                px, py = CATA["p"]
                rx, ry = b.x - px, b.y - py
                v0 = (-om * ry, om * rx)  # Bahngeschwindigkeit des Löffels
                b.state = "free"
                b.vx, b.vy = v0
                if k == 0:
                    self.ev("fire")

    def st_free(self, b, dt, t):
        b.vy += G * dt
        b.x += b.vx * dt
        b.y += b.vy * dt
        cx, cy = BOWL["c"]
        rr = BOWL["r"] - R_BALL
        dx, dy = b.x - cx, b.y - cy
        if dy > 0 and math.hypot(dx, dy) >= rr and abs(dx) < BOWL["r"]:
            self.enter_bowl(b, t)

    def enter_bowl(self, b, t):
        if "BOWL" not in PATHS:
            cx, cy = BOWL["c"]
            PATHS["BOWL"] = Path("BOWL", drag=0.9, bidir=True).arc((cx, cy), BOWL["r"] - R_BALL, math.pi + 0.05, -0.05 + 0).done()
        bowl = PATHS["BOWL"]
        s = bowl.nearest_s((b.x, b.y))
        (_, _), (tx, ty) = bowl.at(s)
        self.to_path(b, "BOWL", s, b.vx * tx + b.vy * ty)
        b.state = "bowl"
        self.goal_n += 1
        self.ev("goal", ball=b.i, n=self.goal_n, x=b.x, y=b.y)

    def st_bowl(self, b, dt, t):
        path = b.path
        (_, _), (tx, ty) = path.at(b.s)
        b.v += (G * ty - path.drag * b.v) * dt
        b.s += b.v * dt
        if b.s < 0:
            b.s, b.v = 0.0, -b.v * 0.4
        if b.s > path.length:
            b.s, b.v = path.length, -b.v * 0.4
        (b.x, b.y), _ = path.at(b.s)

    def collide_bowl(self):
        bb = sorted([b for b in self.balls if b.state == "bowl"], key=lambda b: b.s)
        for a, c in zip(bb, bb[1:]):
            gap = c.s - a.s
            if gap < 2 * R_BALL:
                ov = (2 * R_BALL - gap) / 2
                a.s -= ov
                c.s += ov
                if a.v > c.v:
                    m = (a.v + c.v) / 2
                    a.v = m - (a.v - m) * 0.3
                    c.v = m - (c.v - m) * 0.3
        if self.finish_t is None and len(bb) == 3 and all(abs(b.v) < 6 for b in bb):
            self.finish_t = self.t
            self.ev("settled")

    def run(self, max_t=70.0):
        while True:
            for _ in range(SUB):
                self.step()
            fr = dict(t=self.t, balls=[(b.x, b.y, b.state, b.v if b.state in ("rail", "bowl") else math.hypot(b.vx, b.vy))
                                       for b in self.balls],
                      seesaw=self.seesaw_phi(self.t), cata=self.cata_angle(self.t))
            self.frames.append(fr)
            if self.finish_t is not None and self.t >= self.finish_t + self.outro:
                break
            if self.t > max_t:
                break
        return self
