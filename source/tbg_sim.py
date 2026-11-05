import math
import random
import sys
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import matplotlib
try:
    matplotlib.use("TkAgg")        # needed for the menu bar and dialogs
except Exception:
    pass
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Polygon, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap

from tbg_common import (ACT_COLORS, APP_AUTHOR, APP_DATE, APP_NAME, APP_URL, APP_VERSION, SRC_COLOR,
                        scene_sources, Scene,
                        act_range, default_value, load_scene, save_scene, scene_walls,
                        set_app_id, set_window_icon)

# ---- colours (edit here) ------------------------------------------------------
C_BG = "#B5B5B5"          # plot background
C_STREAM = "#9E9E9E"      # flow lines
C_SHOCK = "#242424"       # every shock
C_FAN_NEAR = "#525151"    # fan colour at the corner
C_FAN_FAR = "#B5B5B5"     # fan colour at its far end
C_SOLID = "black"         # closed bodies and the ground plane
C_SLIP = "#525151"        # slip lines
C_SHOCK_HI = "#FFFFFF"    # light half of every shock line (downstream side)
FAN_CMAP = LinearSegmentedColormap.from_list("fan", [C_FAN_NEAR, C_FAN_FAR])

PI = math.pi
XMIN, XMAX = -7.0, 8.0
YMIN, YMAX = -1.2, 6.2
GROUND_HALF = 60.0
MAX_GEN = 8
FAN_LEN = 1.8
STEP = 0.04
U_FLOW = np.array([-1.0, 0.0])


# =============================================================================
# Small helpers
# =============================================================================
def K(p):
    return (round(float(p[0]), 6), round(float(p[1]), 6))


def cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


def unit(v):
    n = math.hypot(v[0], v[1])
    return np.array([v[0] / n, v[1] / n])


def wrap(a):
    return (a + PI) % (2 * PI) - PI


def dirv(ang):
    return np.array([math.cos(ang), math.sin(ang)])


def deg(x):
    return math.degrees(x)


def ray_seg(o, d, a, b):
    e = b - a
    den = cross(d, e)
    if abs(den) < 1e-12:
        return None
    w = a - o
    return cross(w, e) / den, cross(w, d) / den


def exit_dist(o, d):
    ts = []
    if abs(d[0]) > 1e-12:
        ts.append(((XMAX + 1.0 if d[0] > 0 else XMIN - 1.0) - o[0]) / d[0])
    if abs(d[1]) > 1e-12:
        ts.append(((YMAX + 1.0 if d[1] > 0 else YMIN - 1.0) - o[1]) / d[1])
    ts = [t for t in ts if t > 0]
    return min(ts) if ts else 0.0


def seg_intersect(p, q, a, b, tol=1e-9):
    d, e = q - p, b - a
    den = cross(d, e)
    if abs(den) < 1e-12:
        return None
    w = a - p
    t, u = cross(w, e) / den, cross(w, d) / den
    if -tol <= t <= 1 + tol and -tol <= u <= 1 + tol:
        return t, u
    return None


# =============================================================================
# Geometry: walls as a planar graph
# =============================================================================
def base_walls():
    return [(np.array([-GROUND_HALF, 0.0]), np.array([GROUND_HALF, 0.0]))]


def add_wall(walls, p, q):
    p, q = np.array(p, float), np.array(q, float)
    if np.hypot(*(q - p)) < 1e-6:
        return walls
    ts = [0.0, 1.0]
    out = []
    for a, b in walls:
        r = seg_intersect(p, q, a, b)
        if r is None:
            out.append((a, b))
            continue
        t, u = r
        if 1e-7 < u < 1 - 1e-7:
            pt = p if t < 1e-7 else (q if t > 1 - 1e-7 else p + t * (q - p))
            out.append((a, pt.copy()))
            out.append((pt.copy(), b))
        else:
            out.append((a, b))
        if 1e-7 < t < 1 - 1e-7:
            ts.append(t)
    ts = sorted(set(round(t, 9) for t in ts))
    existing = {frozenset((K(a), K(b))) for a, b in out}
    for t0, t1 in zip(ts[:-1], ts[1:]):
        a = p + t0 * (q - p) if t0 > 0 else p
        b = p + t1 * (q - p) if t1 < 1 else q
        if frozenset((K(a), K(b))) not in existing:
            out.append((a.copy(), b.copy()))
    return out


def bounded_faces(walls):
    out = {}
    for a, b in walls:
        ka, kb = K(a), K(b)
        out.setdefault(ka, []).append((math.atan2(b[1] - a[1], b[0] - a[0]), kb))
        out.setdefault(kb, []).append((math.atan2(a[1] - b[1], a[0] - b[0]), ka))
    for k in out:
        out[k].sort()
    seen, faces = set(), []
    for ku in out:
        for _, kv in out[ku]:
            if (ku, kv) in seen:
                continue
            poly, h = [], (ku, kv)
            while h not in seen:
                seen.add(h)
                poly.append(h[0])
                u, v = h
                lst = out[v]
                idx = [i for i, (_, w) in enumerate(lst) if w == u][0]
                h = (v, lst[(idx - 1) % len(lst)][1])
            area = 0.5 * sum(poly[i][0] * poly[(i + 1) % len(poly)][1] -
                             poly[(i + 1) % len(poly)][0] * poly[i][1]
                             for i in range(len(poly)))
            if area > 1e-9:
                faces.append(poly)
    return faces


# =============================================================================
# Gas dynamics
# =============================================================================
def theta_from_beta(M, beta, g):
    num = 2.0 / np.tan(beta) * (M ** 2 * np.sin(beta) ** 2 - 1.0)
    den = M ** 2 * (g + np.cos(2.0 * beta)) + 2.0
    return np.arctan(num / den)


@lru_cache(maxsize=4096)
def _thmax(M, g):
    mu = math.asin(1.0 / M)
    betas = np.linspace(mu + 1e-9, PI / 2, 3000)
    th = theta_from_beta(M, betas, g)
    i = int(np.argmax(th))
    lo, hi = betas[max(i - 1, 0)], betas[min(i + 1, len(betas) - 1)]
    for _ in range(50):
        m1, m2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
        if theta_from_beta(M, m1, g) < theta_from_beta(M, m2, g):
            lo = m1
        else:
            hi = m2
    b = 0.5 * (lo + hi)
    return float(theta_from_beta(M, b, g)), float(b)


def theta_max(M, g):
    return _thmax(round(M, 8), round(g, 8))


def solve_beta(M, theta, g, strong=False):
    mu = math.asin(1.0 / M)
    th_max, b_max = theta_max(M, g)
    theta = min(theta, th_max)
    if strong:
        lo, hi = b_max, PI / 2
        for _ in range(70):
            mid = 0.5 * (lo + hi)
            if theta_from_beta(M, mid, g) > theta:
                lo = mid
            else:
                hi = mid
    else:
        lo, hi = mu, b_max
        for _ in range(70):
            mid = 0.5 * (lo + hi)
            if theta_from_beta(M, mid, g) < theta:
                lo = mid
            else:
                hi = mid
    return 0.5 * (lo + hi)


def p_ratio(M, beta, g):
    return 1 + 2 * g / (g + 1) * ((M * math.sin(beta)) ** 2 - 1)


def shock_ratios(M, beta, theta, g):
    Mn1 = M * math.sin(beta)
    rho = (g + 1) * Mn1 ** 2 / ((g - 1) * Mn1 ** 2 + 2)
    p = p_ratio(M, beta, g)
    Mn2 = math.sqrt((1 + (g - 1) / 2 * Mn1 ** 2) / (g * Mn1 ** 2 - (g - 1) / 2))
    M2 = Mn2 / math.sin(beta - theta)
    p0 = rho ** (g / (g - 1)) * ((g + 1) / (2 * g * Mn1 ** 2 - (g - 1))) ** (1 / (g - 1))
    return dict(Mn1=Mn1, M2=M2, p=p, rho=rho, T=p / rho, p0=p0)


def pm(M, g):
    a = math.sqrt((g + 1) / (g - 1))
    return a * math.atan(math.sqrt((g - 1) / (g + 1) * (M * M - 1))) - math.atan(math.sqrt(M * M - 1))


def nu_max(g):
    return 0.5 * PI * (math.sqrt((g + 1) / (g - 1)) - 1)


def inv_pm(nu, g):
    if nu >= nu_max(g) * 0.9995:
        return 40.0
    lo, hi = 1.0, 40.0
    for _ in range(70):
        mid = 0.5 * (lo + hi)
        if pm(mid, g) < nu:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@dataclass
class State:
    M: float
    phi: float
    p: float = 1.0
    rho: float = 1.0
    T: float = 1.0
    p0: float = 1.0


def oblique_state(S, beta, theta, s, g):
    r = shock_ratios(S.M, beta, theta, g)
    return State(r["M2"], S.phi + s * theta, S.p * r["p"], S.rho * r["rho"],
                 S.T * r["T"], S.p0 * r["p0"]), r


# =============================================================================
# Wave records
# =============================================================================
@dataclass
class Wave:
    id: str
    kind: str
    S1: State
    S2: State
    beta: float
    theta: float
    s: int
    gen: int
    ang: float = 0.0
    pts: list = field(default_factory=list)
    tail: list = field(default_factory=list)
    end: str = "view"
    cut: object = None
    child: object = None


@dataclass
class Fan:
    id: str
    origin: np.ndarray
    lines: list
    S1: State
    S2: State
    turn: float
    sigma: int = 0
    phi_out: float = 0.0
    capped: bool = False
    a0: float = 0.0
    rel: object = None
    ks: object = None
    Ms: object = None
    phis: object = None

    def lead_dir(self):
        return dirv(self.a0)

    def state_at(self, q):
        v = q - self.origin
        if float(v @ v) < 1e-12 or self.rel is None:
            return None
        sgn = 1.0 if self.rel[-1] >= 0 else -1.0
        x = sgn * self.rel
        if x[-1] < 1e-6:
            return None
        ar = sgn * wrap(math.atan2(v[1], v[0]) - self.a0)
        if ar < -1e-9 or ar > x[-1] + 1e-9:
            return None
        k = float(np.interp(ar, x, self.ks))
        return float(np.interp(k, self.ks, self.Ms)), float(np.interp(k, self.ks, self.phis))


# =============================================================================
# The flow solver (wave tracing)  -- unchanged from v3 except the `ground` flag
# =============================================================================
class Flow:
    def __init__(self, walls, M, g, ground=True, main=True):
        self.ground = ground
        self.main = main
        self.walls = [(np.array(a, float), np.array(b, float)) for a, b in walls]
        self.M, self.g = M, g
        self.S0 = State(M, PI)
        self.waves, self.fans, self.slips = [], [], []
        self.crossings, self.notes = [], []
        self.pending = []
        self.bow = None
        self.cnt = {}
        self.adj = {}
        self.polys = bounded_faces(self.walls)
        for i, (a, b) in enumerate(self.walls):
            self.adj.setdefault(K(a), []).append((i, 0))
            self.adj.setdefault(K(b), []).append((i, 1))
        self.deg = [(len(self.adj[K(a)]), len(self.adj[K(b)])) for a, b in self.walls]
        if main and M > 1.0 + 1e-9:
            self._primary()
            self._find_crossings()

    def _nid(self, prefix):
        self.cnt[prefix] = self.cnt.get(prefix, 0) + 1
        return f"{prefix}{self.cnt[prefix]}"

    def _solid(self, p):
        """True if p lies inside the ground (if present) or inside a closed body."""
        if self.ground and p[1] < -1e-9:
            return True
        for poly in self.polys:
            inside, n = False, len(poly)
            for i in range(n):
                x1, y1 = poly[i]
                x2, y2 = poly[(i + 1) % n]
                if (y1 > p[1]) != (y2 > p[1]) and \
                        p[0] < (x2 - x1) * (p[1] - y1) / (y2 - y1) + x1:
                    inside = not inside
            if inside:
                return True
        return False

    def first_hit(self, o, d, tmin=1e-7):
        best = None
        for i, (a, b) in enumerate(self.walls):
            r = ray_seg(o, d, a, b)
            if r is None:
                continue
            t, u = r
            if t <= tmin or u < -1e-9 or u > 1 + 1e-9:
                continue
            if (u < 1e-6 and self.deg[i][0] == 1) or (u > 1 - 1e-6 and self.deg[i][1] == 1):
                continue
            if best is None or t < best[0]:
                best = (t, i, u)
        return best

    def _adv_normal(self, c, r):
        return np.array([c[0] * math.cos(-r * PI / 2) - c[1] * math.sin(-r * PI / 2),
                         c[0] * math.sin(-r * PI / 2) + c[1] * math.cos(-r * PI / 2)])

    def _primary(self):
        M, g = self.M, self.g
        thm = theta_max(M, g)[0]
        cands = []
        for i, (a, b) in enumerate(self.walls):
            e = b - a
            L = float(np.hypot(*e))
            t = e / L
            if t[0] > 1e-9:
                t, A, B = -t, b, a
            else:
                A, B = a, b
            vertical = abs(t[0]) < 1e-9
            n1 = np.array([-t[1], t[0]])
            for sgn, n in ((+1, n1), (-1, -n1)):
                ndu = float(n @ U_FLOW)
                if ndu > -1e-9:
                    continue
                delta = math.asin(min(1.0, -ndu))
                if delta < 1e-6:
                    continue
                if self._solid((A + B) / 2 + n * 1e-4):
                    continue
                lead = (A + B) / 2 if vertical else A
                cands.append(dict(edge=i, A=A.copy(), B=B.copy(), t=t, n=n, delta=delta,
                                  s_n=sgn, L=L, vertical=vertical, lead=lead.copy(),
                                  cont=False))
        for g_ in cands:
            g_["succ"] = []
            nxt = self._next_edge(g_["B"], g_["edge"], g_["t"], g_["n"])
            if nxt is None or float(nxt[2] @ U_FLOW) <= 1e-6:
                continue
            nn = self._adv_normal(nxt[2], nxt[4])
            for f in cands:
                if f["edge"] == nxt[1] and float(np.hypot(*(f["n"] - nn))) < 1e-6:
                    f["cont"] = True
                    g_["succ"].append(f)

        def make_groups():
            grp = {}
            for f in cands:
                if f["cont"]:
                    continue
                if self.first_hit(f["lead"] + f["n"] * 1e-6, np.array([1.0, 0.0]), 1e-9) is not None:
                    self.notes.append(f"face at ({f['lead'][0]:.1f},{f['lead'][1]:.1f}) is shadowed "
                                      "by other geometry: not solved")
                    continue
                grp.setdefault(K(f["lead"]), []).append(f)
            return grp

        groups = make_groups()
        # FIX: a face that continues a wetted face is normally solved by that face's walk.
        # But a blunt face (turn > theta_max) becomes a bow shock and is never walked, so
        # whatever follows it (e.g. a chamfer on a cowl lip) would silently get no shock.
        # Which end of a vertical face is "B" depends on drawing order, so this also made
        # two identical cowls behave differently. Such faces now lead their own shock.
        for _ in range(10):
            freed = False
            for faces in groups.values():
                if any(f["delta"] > thm for f in faces):
                    for f in faces:
                        for c in f["succ"]:
                            if c["cont"]:
                                c["cont"] = False
                                freed = True
            if not freed:
                break
            self.notes = [n for n in self.notes if "is shadowed" not in n]
            groups = make_groups()
        for faces in groups.values():
            if any(f["delta"] > thm for f in faces):
                self._bow(faces)
                continue
            built = [(f, self._build_plans(f)) for f in faces]
            if any(f.get("detach") for f in faces):
                tot = max(f.get("detach", 0.0) for f in faces)
                self.notes.append(f"total compression turn {deg(tot):.1f} deg > theta_max: "
                                  "shocks coalesce into a detached bow shock")
                self._bow(faces)
            else:
                for f, plans in built:
                    for p in plans:
                        self._launch(p)
                    if f.get("local_bow"):
                        self._local_bow(f)
        self._drain()

    def _drain(self):
        guard = 0
        while self.pending and guard < 300:
            guard += 1
            w, hit = self.pending.pop(0)
            self._reflect(w, w.pts[-1].copy(), hit, w.gen)

    def _build_plans(self, f):
        M, g = self.M, self.g
        beta = solve_beta(M, f["delta"], g)
        S1, _ = oblique_state(self.S0, beta, f["delta"], f["s_n"], g)
        plans = [dict(origin=f["A"].copy(), ang=PI + f["s_n"] * beta, S1=self.S0, S2=S1,
                      beta=beta, theta=f["delta"], s=f["s_n"], kind="oblique", gen=0,
                      cut=None, prefix="S", parent=None)]
        self._walk(f, S1, plans)
        return plans

    def _next_edge(self, B, edge, t, n):
        alpha0 = math.atan2(-t[1], -t[0])
        r = 1 if cross(-t, n) > 0 else -1
        best = None
        for ei, which in self.adj[K(B)]:
            if ei == edge:
                continue
            a, b = self.walls[ei]
            other = b if which == 0 else a
            c = other - B
            dang = (r * (math.atan2(c[1], c[0]) - alpha0)) % (2 * PI)
            if dang < 1e-7 or dang > 2 * PI - 1e-7:
                continue
            if best is None or dang < best[0]:
                best = (dang, ei, unit(c), other, r)
        return best

    def _set_cut(self, plan, B, S, s_n):
        if plan is None or S.M <= 1.0 + 1e-6:
            return
        mu = math.asin(1.0 / S.M)
        plan["cut"] = (B.copy(), dirv(S.phi + s_n * mu))

    def _walk(self, f, S, plans):
        g = self.g
        edge, B, t, n, s_n = f["edge"], f["B"].copy(), f["t"], f["n"], f["s_n"]
        last = plans[-1]
        visited = set()
        thm0 = theta_max(self.M, g)[0]
        tot, had_fan = f["delta"], False
        for _ in range(2000):                      # (splines are many short pieces; loops are
                                                   #  stopped by `visited` below)
            nxt = self._next_edge(B, edge, t, n)
            if nxt is None:
                self._set_cut(last, B, S, s_n)
                break
            psi, ei, c, other, r = nxt
            if float(c @ U_FLOW) <= 1e-6:
                self.notes.append(f"corner at ({B[0]:.1f},{B[1]:.1f}): surface closes "
                                  "against the flow, waves handled by reflections only")
                break
            if c[1] < -1e-6 and (self.ground and other[1] <= 1e-6):
                # the surface runs down onto the ground: nothing further to solve along it.
                # (A downward slope that stays in free air, e.g. the aft chamfer of a body, is
                # an ordinary expansion corner and gets its fan like an upward one.)
                self._set_cut(last, B, S, s_n)
                self.notes.append(f"corner at ({B[0]:.1f},{B[1]:.1f}): flow separates onto a "
                                  "leeward slope (wake not modeled)")
                break
            Delta = PI - psi
            if abs(Delta) > 1e-6:
                if Delta > 0:
                    if S.M <= 1.0 + 1e-6:
                        break
                    if Delta > theta_max(S.M, g)[0]:
                        f["local_bow"] = dict(B=B.copy(), S=S, Delta=Delta, n_prev=n.copy(),
                                              L=float(np.hypot(*(other - B))), top=other.copy(),
                                              last=last)
                        break
                    beta = solve_beta(S.M, Delta, g)
                    S2, _ = oblique_state(S, beta, Delta, s_n, g)
                    plan = dict(origin=B.copy(), ang=S.phi + s_n * beta, S1=S, S2=S2, beta=beta,
                                theta=Delta, s=s_n, kind="oblique", gen=0, cut=None, prefix="S",
                                parent=last)
                    plans.append(plan)
                    last, S = plan, S2
                else:
                    self._set_cut(last, B, S, s_n)
                    had_fan, tot = True, 0.0
                    S, capped = self._fan(B, S, s_n, -Delta, phi_out=math.atan2(c[1], c[0]))
                    last = None
                    if capped:
                        self.notes.append(f"corner at ({B[0]:.1f},{B[1]:.1f}): expansion exceeds "
                                          "the Prandtl-Meyer limit, flow separates")
                        break
            edge, B, t = ei, other.copy(), c
            n = self._adv_normal(c, r)
            s_n = 1 if cross(t, n) > 0 else -1
            S = State(S.M, math.atan2(c[1], c[0]), S.p, S.rho, S.T, S.p0)
            if (edge, K(B)) in visited:
                break
            visited.add((edge, K(B)))
            if S.M <= 1.0 + 1e-6:
                break

    def _fan(self, B, S, sigma, mag, phi_out=None):
        g = self.g
        if phi_out is None:
            phi_out = S.phi - sigma * mag
        dphi = wrap(phi_out - S.phi)
        nu1 = pm(S.M, g)
        room = max(0.0, min(0.995 * nu_max(g), pm(25.0, g)) - nu1)
        capped = mag > room
        if capped:
            mag = room
            dphi = math.copysign(mag, dphi if abs(dphi) > 1e-12 else -sigma)
        if mag < 1e-6:
            return State(S.M, S.phi + dphi, S.p, S.rho, S.T, S.p0), capped
        phi_out = S.phi + dphi
        M2 = inv_pm(nu1 + mag, g)
        k_p = (1 + (g - 1) / 2 * S.M ** 2) / (1 + (g - 1) / 2 * M2 ** 2)
        S2 = State(M2, phi_out, S.p * k_p ** (g / (g - 1)), S.rho * k_p ** (1 / (g - 1)),
                   S.T * k_p, S.p0)
        lines = []
        for k in np.linspace(0, 1, 5):
            Mk = inv_pm(nu1 + k * mag, g)
            ang = S.phi + dphi * k + sigma * math.asin(1 / Mk)
            d = dirv(ang)
            h = self.first_hit(B, d)
            L = min(FAN_LEN, h[0] if h else FAN_LEN)
            lines.append((B.copy(), B + d * L))
        ks = np.linspace(0, 1, 61)
        Ms = np.array([inv_pm(nu1 + k * mag, g) for k in ks])
        phis = S.phi + dphi * ks
        angs = phis + sigma * np.arcsin(1.0 / Ms)
        fan = Fan(self._nid("F"), B.copy(), lines, S, S2, mag, sigma, phi_out, capped,
                  float(angs[0]), angs - angs[0], ks, Ms, phis)
        self.fans.append(fan)
        return S2, capped

    def _trace(self, o, d, cut):
        hit = self.first_hit(o, d)
        tw = hit[0] if hit else 1e9
        tv = exit_dist(o, d)
        tc = 1e9
        if cut is not None:
            r = ray_seg(o, d, cut[0], cut[0] + cut[1])
            if r and r[0] > 1e-6 and r[1] > 1e-6:
                tc = r[0]
        if tc < min(tw, tv):
            return tc, "cut", None, min(tw, tv)
        if tw <= tv:
            return tw, "wall", hit, tw
        return tv, "view", None, tv

    def _launch(self, plan):
        w = Wave(self._nid(plan.get("prefix", "S")), plan["kind"], plan["S1"], plan["S2"],
                 plan["beta"], plan["theta"], plan["s"], plan["gen"], plan["ang"])
        w.cut = plan.get("cut")
        plan["wave"] = w
        o, d = plan["origin"], dirv(plan["ang"])
        tend, kind, hit, ttail = self._trace(o, d, w.cut)

        if plan["kind"] == "oblique" and plan["gen"] == 0 and plan.get("parent") is not None:
            m = self._find_merge(plan, o, d, tend)
            if m is not None:
                tm, cw, over = m
                X = o + d * tm
                w.pts = [o.copy(), X.copy()]
                w.end = "merge"
                self.waves.append(w)
                if over:
                    self._overturned(w, cw, X)
                else:
                    self._merge(w, cw, X)
                return w

        end = o + d * tend
        w.pts = [o.copy(), end]
        w.end = kind
        if kind == "cut":
            w.tail = [end.copy(), o + d * ttail]
        self.waves.append(w)
        if kind == "wall":
            if plan["gen"] < MAX_GEN:
                self.pending.append((w, hit))
            else:
                self.notes.append("reflection limit reached")
        return w

    def _local_bow(self, f):
        g, M0 = self.g, self.M
        lb = f["local_bow"]
        S, B, Delta = lb["S"], lb["B"], lb["Delta"]
        Ml = S.M
        thm_l = theta_max(Ml, g)[0]
        h = max(lb["L"] * math.sin(Delta), 0.3)
        fr = min(1.0, max(0.0, (Delta - thm_l) / (PI / 2 - thm_l)))
        stand = min(h * 0.386 * math.exp(4.67 / Ml ** 2) * (0.15 + 0.85 * fr), 3.0)
        Rc = min(max(h * 1.386 * math.exp(1.8 / max(Ml - 1, 0.05) ** 0.75), 0.3), 50.0)
        R = S.phi - PI
        cR, sR = math.cos(R), math.sin(R)

        def rot(v):
            return np.array([cR * v[0] - sR * v[1], sR * v[0] + cR * v[1]])

        npl = lb["n_prev"]
        sg = 1 if (-sR * npl[0] + cR * npl[1]) >= 0 else -1
        mu = math.asin(1 / Ml)
        yt = Rc * math.sqrt(Ml * Ml - 1)
        ys = np.linspace(0, min(yt, 14.0), 80)
        pts = [B + rot(np.array([stand - y * y / (2 * Rc), sg * y])) for y in ys]
        if yt < 14.0:
            dt = rot(np.array([-math.cos(mu), sg * math.sin(mu)]))
            pts.append(pts[-1] + dt * exit_dist(pts[-1], dt))
        out = [pts[0]]
        for k in range(1, len(pts)):
            a, b = pts[k - 1], pts[k]
            dd = b - a
            L = float(np.hypot(*dd))
            if L < 1e-12:
                continue
            hit = self.first_hit(a, dd / L)
            if hit and hit[0] <= L:
                out.append(a + dd / L * hit[0])
                break
            out.append(b)
        pts = out

        cands, c = [], (lb["last"].get("wave") if lb["last"] else None)
        while c is not None:
            cands.append(c)
            c = c.child
        found = None
        for k in range(len(pts) - 1):
            for c in cands:
                for j in range(len(c.pts) - 1):
                    r = seg_intersect(pts[k], pts[k + 1], c.pts[j], c.pts[j + 1])
                    if r and (found is None or r[0] < found[1]):
                        found = (k, r[0], c)
            if found:
                break
        X = None
        if found:
            k, t, cw = found
            X = pts[k] + t * (pts[k + 1] - pts[k])
            pts = pts[:k + 1] + [X.copy()]
            cw.pts = [cw.pts[0], X.copy()]
            cw.end = "merge"
            cw.tail = []
            self.pending = [(a, hh) for a, hh in self.pending if a is not cw]

        r = shock_ratios(Ml, PI / 2, 0.0, g)
        S2 = State(r["M2"], S.phi, S.p * r["p"], S.rho * r["rho"], S.T * r["T"], S.p0 * r["p0"])
        w = Wave(self._nid("B"), "bow", S, S2, PI / 2, 0.0, 0, 0)
        w.pts, w.end = pts, ("merge" if X is not None else "mach line")
        self.waves.append(w)
        self.notes.append("detached shock ahead of steep corner (local Mach "
                          f"{Ml:.2f}, turn {deg(Delta):.0f} deg > {deg(thm_l):.0f}); schematic")

        if X is None:
            return
        h_eff = max(abs(lb["top"][1] - f["lead"][1]), h)
        self._outer_bow(X, h_eff, 1 if X[1] >= f["lead"][1] else -1)

    def _find_merge(self, plan, o, d, tend):
        parent = plan["parent"].get("wave")
        cands, c = [], parent
        while c is not None:
            cands.append(c)
            c = c.child
        best = None
        for c in cands:
            if c.s != plan["s"]:
                continue
            for k in range(len(c.pts) - 1):
                r = ray_seg(o, d, c.pts[k], c.pts[k + 1])
                if r is None:
                    continue
                t, u = r
                if 1e-6 < t < tend - 1e-7 and -1e-9 <= u <= 1 + 1e-9:
                    if best is None or t < best[0]:
                        best = (t, c)
        if best is None:
            return None
        cw = best[1]
        th_tot = cw.theta + plan["theta"]
        return best[0], cw, th_tot > theta_max(cw.S1.M, self.g)[0] * 0.999

    def _overturned(self, w, cw, X):
        cw.pts = [cw.pts[0], X.copy()]
        cw.end = "merge"
        cw.tail = []
        self.pending = [(a, h) for a, h in self.pending if a is not cw]
        h_eff = max(abs(X[1] - cw.pts[0][1]), 0.3)
        self._outer_bow(X, h_eff, -cw.s)
        self.notes.append(f"{cw.id}+{w.id}: combined turn {deg(cw.theta + w.theta):.1f} deg > "
                          f"theta_max {deg(theta_max(cw.S1.M, self.g)[0]):.1f}: "
                          "shocks meet and continue as a detached curved shock")

    def _outer_bow(self, X, h_eff, sgo):
        g, M0 = self.g, self.M
        th_o, b_o = theta_max(M0, g)
        Rc0 = min(max(h_eff * 1.386 * math.exp(1.8 / max(M0 - 1, 0.05) ** 0.75), 0.3), 50.0)
        mu0 = math.asin(1 / M0)
        u0 = Rc0 / math.tan(b_o)
        u_t = Rc0 * math.sqrt(M0 * M0 - 1)
        us = np.linspace(u0, max(min(u_t, u0 + 14.0), u0 + 1e-3), 70)
        op = [np.array([X[0] - (u * u - u0 * u0) / (2 * Rc0), X[1] + sgo * (u - u0)]) for u in us]
        if u_t < u0 + 14.0:
            dt = np.array([-math.cos(mu0), sgo * math.sin(mu0)])
            op.append(op[-1] + dt * exit_dist(op[-1], dt))
        out = [op[0]]
        for k in range(1, len(op)):
            a, b = op[k - 1], op[k]
            dd = b - a
            L = float(np.hypot(*dd))
            if L < 1e-12:
                continue
            hit = self.first_hit(a, dd / L)
            if hit and hit[0] <= L:
                out.append(a + dd / L * hit[0])
                break
            out.append(b)
        So, _ = oblique_state(self.S0, b_o, th_o, -sgo, g)
        wo = Wave(self._nid("B"), "bow", self.S0, So, b_o, th_o, -sgo, 0)
        wo.pts, wo.end = out, "mach line"
        self.waves.append(wo)

    def _merge(self, w, cw, X):
        g = self.g
        s = w.s
        th_tot = cw.theta + w.theta
        beta_c = solve_beta(cw.S1.M, th_tot, g)
        Sc, _ = oblique_state(cw.S1, beta_c, th_tot, s, g)
        cw.pts = [cw.pts[0], X.copy()]
        cw.end = "merge"
        cw.tail = []
        self.pending = [(a, h) for a, h in self.pending if a is not cw]
        mp = dict(origin=X.copy(), ang=cw.S1.phi + s * beta_c, S1=cw.S1, S2=Sc, beta=beta_c,
                  theta=th_tot, s=s, kind="merged", gen=0, cut=cw.cut, prefix="C", parent=None)
        mw = self._launch(mp)
        cw.child = mw
        d4 = dirv(Sc.phi)
        h = self.first_hit(X, d4)
        self.slips.append((X.copy(), X + d4 * min(3.0, h[0] if h else 3.0)))
        self.notes.append(f"{cw.id} and {w.id} merge into {mw.id} (+ slip line)")

    def _reflect(self, w, H, hit, gen):
        g = self.g
        _, idx, _ = hit
        a, b = self.walls[idx]
        te = unit(b - a)
        d = dirv(w.ang)
        n_f = np.array([-te[1], te[0]])
        if d @ n_f > 0:
            n_f = -n_f
        S2 = w.S2
        if S2.M <= 1.05:
            self.notes.append(f"{w.id}: flow behind it is (near-)sonic, reflection not computed")
            w.end = "wall (subsonic)"
            return
        v2 = dirv(S2.phi)
        tw = te if te @ v2 >= 0 else -te
        dsg = wrap(math.atan2(tw[1], tw[0]) - S2.phi)
        mag = abs(dsg)
        if mag < 1e-5:
            w.end = "wall (grazing)"
            return
        sr = 1 if dsg > 0 else -1
        if (v2 @ n_f) >= 0:
            sigma = 1 if cross(v2, n_f) > 0 else -1
            self._fan(H, S2, sigma, mag, phi_out=S2.phi + dsg)
            self.notes.append(f"{w.id} meets the wall with flow turning away: expansion fan")
            return
        if mag <= theta_max(S2.M, g)[0] * 0.9999:
            beta = solve_beta(S2.M, mag, g)
            S3, _ = oblique_state(S2, beta, mag, sr, g)
            self._launch(dict(origin=H.copy(), ang=S2.phi + sr * beta, S1=S2, S2=S3, beta=beta,
                              theta=mag, s=sr, kind="regular", gen=gen + 1, cut=None, prefix="R"))
        else:
            self._mach(w, H, d, n_f, idx, sr, gen)

    def _mach(self, w, H, d, n_f, idx, sr, gen):
        g = self.g
        S1, S2 = w.S1, w.S2
        th_i = w.theta
        thm2 = theta_max(S2.M, g)[0]
        thm1 = theta_max(S1.M, g)[0]
        p_inc = S2.p / S1.p

        def p_stem(dm):
            dm = min(abs(dm), thm1 * 0.9999)
            return p_ratio(S1.M, solve_beta(S1.M, dm, g, strong=True), g)

        def f(thr):
            return p_stem(th_i - thr) - p_inc * p_ratio(S2.M, solve_beta(S2.M, thr, g), g)

        lo, hi = 1e-6, thm2 * 0.9999
        if f(lo) * f(hi) < 0:
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if f(lo) * f(mid) <= 0:
                    hi = mid
                else:
                    lo = mid
            thr = 0.5 * (lo + hi)
        else:
            thr = 0.9 * thm2
            self.notes.append(f"{w.id}: three-shock solution not found, schematic Mach reflection")
        dm = th_i - thr
        s_stem = w.s if dm >= 0 else -w.s
        beta_st = solve_beta(S1.M, min(abs(dm), thm1 * 0.9999), g, strong=True)
        S3, _ = oblique_state(S1, beta_st, abs(dm), s_stem, g)
        beta_r = solve_beta(S2.M, thr, g)
        S4, _ = oblique_state(S2, beta_r, thr, sr, g)
        Lw = float(np.hypot(*(H - w.pts[0])))
        T = H - d * max(0.12, min(0.18 * Lw, 0.8 * Lw))
        a, b = self.walls[idx]
        seg = b - a
        uu = min(1.0, max(0.0, float((T - a) @ seg) / float(seg @ seg)))
        F = a + uu * seg
        w.pts[-1] = T
        w.end = "triple point"
        stem = Wave(self._nid("M"), "stem", S1, S3, beta_st, abs(dm), s_stem, gen + 1,
                    math.atan2(*(T - F)[::-1]))
        stem.pts = [F.copy(), T.copy()]
        stem.end = "wall"
        self.waves.append(stem)
        d4 = dirv(S4.phi)
        h = self.first_hit(T, d4)
        self.slips.append((T.copy(), T + d4 * min(3.0, h[0] if h else 3.0)))
        self._launch(dict(origin=T.copy(), ang=S2.phi + sr * beta_r, S1=S2, S2=S4, beta=beta_r,
                          theta=thr, s=sr, kind="refl", gen=gen + 1, cut=None, prefix="R"))
        self.notes.append(f"{w.id}: Mach reflection (stem height is schematic)")

    def _bow(self, faces):
        M, g = self.M, self.g
        thm = theta_max(M, g)[0]
        lead = faces[0]["lead"]
        h, dmax, both = 0.0, 0.0, False
        for f in faces:
            h = max(h, f["L"] * math.sin(f["delta"]) * (0.5 if f["vertical"] else 1.0))
            dmax = max(dmax, f["delta"], f.get("detach", 0.0))
            h = max(h, f.get("rise", 0.0))
            both = both or f["vertical"]
        h = max(h, 0.3)
        fr = min(1.0, max(0.0, (dmax - thm) / (PI / 2 - thm)))
        stand = min(h * 0.386 * math.exp(4.67 / M ** 2) * (0.15 + 0.85 * fr), 5.0)
        Rc = min(max(h * 1.386 * math.exp(1.8 / max(M - 1, 0.05) ** 0.75), 0.3), 50.0)
        V0 = np.array([lead[0] + stand, lead[1]])
        mu = math.asin(1 / M)
        yt = Rc * math.sqrt(M * M - 1)
        sides = {1, -1} if both else {-f["s_n"] for f in faces}
        halves = {}
        for sg in sides:
            ys = np.linspace(0, min(yt, 14.0), 70)
            pts = [np.array([V0[0] - y * y / (2 * Rc), V0[1] + sg * y]) for y in ys]
            if yt < 14.0:
                dtail = np.array([-math.cos(mu), sg * math.sin(mu)])
                pts.append(pts[-1] + dtail * exit_dist(pts[-1], dtail))
            out = [pts[0]]
            for k in range(1, len(pts)):
                a, b = pts[k - 1], pts[k]
                dd = b - a
                L = float(np.hypot(*dd))
                if L < 1e-12:
                    continue
                hit = self.first_hit(a, dd / L)
                if hit and hit[0] <= L:
                    out.append(a + dd / L * hit[0])
                    break
                out.append(b)
            halves[sg] = out
        pts = []
        if -1 in halves:
            pts += halves[-1][::-1]
        if 1 in halves:
            pts += halves[1] if not pts else halves[1][1:]
        r = shock_ratios(M, PI / 2, 0.0, g)
        S2 = State(r["M2"], PI, r["p"], r["rho"], r["T"], r["p0"])
        w = Wave(self._nid("B"), "bow", self.S0, S2, PI / 2, 0.0, 0, 0)
        w.pts, w.end = pts, "mach line"
        self.waves.append(w)
        self.bow = dict(standoff=stand, Rc=Rc, vertex=V0)
        self.notes.append("bow shock: standoff/curvature are empirical estimates")

    def _find_crossings(self):
        """Crossings of shocks that are not resolved by the tracer. Weak waves (< 2 deg turn, the
        near-Mach waves of curved surfaces) pass through each other almost undisturbed, so their
        crossings are not marked."""
        segs = []
        for wi, w in enumerate(self.waves):
            if w.kind != "bow" and w.theta < math.radians(2.0):
                continue
            for k in range(len(w.pts) - 1):
                segs.append((wi, w.pts[k], w.pts[k + 1]))
        for i in range(len(segs)):
            for j in range(i + 1, len(segs)):
                if segs[i][0] == segs[j][0]:
                    continue
                r = seg_intersect(segs[i][1], segs[i][2], segs[j][1], segs[j][2], tol=-1e-4)
                if r:
                    t, _ = r
                    self.crossings.append(segs[i][1] + t * (segs[i][2] - segs[i][1]))

    def _pieces(self):
        out = []
        for w in self.waves:
            u1 = dirv(w.S1.phi)
            for k in range(len(w.pts) - 1):
                a, b = w.pts[k], w.pts[k + 1]
                e = b - a
                L = float(np.hypot(*e))
                if L < 1e-9:
                    continue
                l = e / L
                nd = np.array([-l[1], l[0]])
                if nd @ u1 < 0:
                    nd = -nd
                out.append((a, b, l, nd, None))
        return out

    def _tail_pieces(self):
        """The dotted tails of cut-off shocks, as shock pieces for the pressure map only.
        Each carries the list of fans born at the point where the shock was cut (a streamline
        that crosses the tail is then inside that fan's wedge, see streamlines)."""
        out = []
        for w in self.waves:
            if len(w.tail) != 2:
                continue
            a, b = np.array(w.tail[0], float), np.array(w.tail[1], float)
            e = b - a
            L = float(np.hypot(*e))
            if L < 1e-9:
                continue
            l = e / L
            nd = np.array([-l[1], l[0]])
            if nd @ dirv(w.S1.phi) < 0:
                nd = -nd
            fans = []
            if w.cut is not None:
                fans = [fn for fn in self.fans
                        if float(np.hypot(*(fn.origin - np.array(w.cut[0], float)))) < 1e-6]
            out.append((a, b, l, nd, fans))
        return out

    def streamlines(self, ys, with_p=False):
        """Streamlines from the inflow edge. with_p=True also returns, per line, the pressure
        (p/p_inf) along each segment (NaN where the flow is subsonic): (pts, sub, seg_p)."""
        g = self.g
        pieces = self._pieces()
        if with_p:                                   # pressure map: dotted shock tails count too
            pieces = pieces + self._tail_pieces()
        result = []
        for y0 in ys:
            p, phi, M = np.array([XMAX, y0]), PI, self.M
            fan, sub = None, None
            pts = [p.copy()]
            pr, segp, fan_ref = 1.0, [], None
            for _ in range(900):
                d = dirv(phi)
                tbest, kind, info = exit_dist(p, d), "view", None
                if fan is not None and STEP < tbest:
                    tbest, kind = STEP, "step"
                hit = self.first_hit(p, d)
                if hit and hit[0] < tbest:
                    tbest, kind = hit[0], "wall"
                if M > 1.0 + 1e-6:
                    for a, b, l, nd, tf in pieces:
                        if d @ nd <= 1e-9:
                            continue
                        r = ray_seg(p, d, a, b)
                        if r and 1e-7 < r[0] < tbest and -1e-9 <= r[1] <= 1 + 1e-9:
                            tbest, kind, info = r[0], "shock", (l, tf)
                    if fan is None:
                        for fn in self.fans:
                            if abs(wrap(phi - fn.S1.phi)) > 0.05 or abs(M - fn.S1.M) > 0.05 * fn.S1.M:
                                continue
                            r = ray_seg(p, d, fn.origin, fn.origin + fn.lead_dir() * 200.0)
                            if r and 1e-7 < r[0] < tbest and r[1] > 0:
                                q = p + d * r[0]
                                if fn.state_at(q + d * 1e-4) is not None and \
                                        fn.state_at(q - d * 1e-4) is None:
                                    tbest, kind, info = r[0], "fan", fn
                p = p + d * tbest
                pts.append(p.copy())
                segp.append(pr if M > 1.0 + 1e-6 else float("nan"))
                if kind in ("view", "wall"):
                    break
                if kind == "shock":
                    l, tail_fans = info
                    l = l if d @ l >= 0 else -l
                    cr = cross(d, l)
                    beta = math.atan2(abs(cr), float(d @ l))
                    if M * math.sin(beta) > 1.0 + 1e-9:
                        th = float(theta_from_beta(M, beta, g))
                        pr *= p_ratio(M, beta, g)
                        phi += (1 if cr > 0 else -1) * th
                        M = shock_ratios(M, beta, th, g)["M2"]
                    fan = None
                    # Crossing the dotted tail of a shock that an expansion fan cut off: the
                    # point is already inside that fan's wedge, so the fan acts on it from here.
                    for fn in (tail_fans or []):
                        q = p + dirv(phi) * 1e-4
                        st = fn.state_at(q)
                        if st is not None and M > 1.0 + 1e-6:
                            fan, fan_ref = fn, (M, pr)
                            M, phi = st
                            pr = fan_ref[1] * self._iso_p(fan_ref[0], M)
                            break
                elif kind == "fan":
                    fan = info
                    fan_ref = (M, pr)
                    st = fan.state_at(p + d * 1e-4)
                    if st:
                        M, phi = st
                        pr = fan_ref[1] * self._iso_p(fan_ref[0], M)
                elif kind == "step":
                    st = fan.state_at(p)
                    if st is None:
                        M, phi, fan = fan.S2.M, fan.phi_out, None
                    else:
                        M, phi = st
                    if fan_ref is not None:
                        pr = fan_ref[1] * self._iso_p(fan_ref[0], M)
                if M <= 1.0 + 1e-6 and sub is None:
                    sub = len(pts) - 1
            result.append((np.array(pts), sub, np.array(segp)) if with_p
                           else (np.array(pts), sub))
        return result

    def _iso_p(self, M0, M1):
        """Isentropic pressure ratio p(M1)/p(M0) (used inside expansion fans)."""
        g = self.g
        return ((1 + (g - 1) / 2 * M0 ** 2) / (1 + (g - 1) / 2 * M1 ** 2)) ** (g / (g - 1))

    def report(self):
        M, g = self.M, self.g
        if not self.main:
            return ["Main flow: OFF (still ambient air)", "",
                    "Shocks are not traced without a main flow.",
                    "Source-line jets are computed by the flow",
                    "solver: View > Map (Mach / Schlieren / ...)."]
        if M <= 1.0 + 1e-9:
            return ["Subsonic freestream (M <= 1):", "no shock waves form in the wave model.",
                    "View > Map runs the flow solver (subsonic too)."]
        thm = theta_max(M, g)[0]
        L = [f"M = {M:.2f}   gamma = {g:.2f}",
             f"Mach angle = {deg(math.asin(1 / M)):.1f} deg   theta_max = {deg(thm):.1f} deg", ""]
        if not self.waves and not self.fans:
            L.append("No shocks yet. Draw a surface that")
            L.append("turns the flow (flow comes from the right).")
            return L
        L.append("ID  type   M1    M2   beta theta  p2/p1  p/pinf  p0/p0inf")
        names = {"oblique": "obl", "merged": "mrg", "bow": "bow", "regular": "reg",
                 "refl": "refl", "stem": "stem"}
        rows = len(self.waves) + len(self.fans)
        if rows > 22:                       # curved (spline) surfaces make many weak waves
            weak = [w for w in self.waves if w.kind in ("oblique", "regular") and w.theta < math.radians(2.0)]
            L.append(f"({rows} waves; {len(weak)} weak shocks and the fans from curved")
            L.append(" surfaces are summarised below)")
            shown = [w for w in self.waves if w not in weak][:14]
        else:
            weak, shown = [], self.waves
        for w in shown:
            L.append(f"{w.id:<4}{names[w.kind]:<5}{w.S1.M:5.2f}{w.S2.M:6.2f}{deg(w.beta):6.1f}"
                     f"{deg(w.theta):6.1f}{w.S2.p / w.S1.p:7.2f}{w.S2.p:7.2f}{w.S2.p0:8.3f}")
        fans = self.fans if not weak else [f for f in self.fans if f.turn >= math.radians(2.0)][:6]
        for fn in fans:
            L.append(f"{fn.id:<4}{'fan':<5}{fn.S1.M:5.2f}{fn.S2.M:6.2f}{'-':>6}"
                     f"{deg(fn.turn):6.1f}{fn.S2.p / fn.S1.p:7.2f}{fn.S2.p:7.2f}{fn.S2.p0:8.3f}")
        if weak or len(fans) < len(self.fans):
            tw = sum(w.theta for w in weak)
            tf = sum(f.turn for f in self.fans if f not in fans)
            L.append(f"+ {len(weak)} weak shocks (total turn {deg(tw):.1f} deg),")
            L.append(f"  {len(self.fans) - len(fans)} small fans (total turn {deg(tf):.1f} deg)")
        if self.bow:
            L.append("")
            L.append(f"bow standoff ~ {self.bow['standoff']:.2f} (est.)")
        if self.crossings:
            L.append("")
            L.append(f"{len(self.crossings)} unresolved shock crossing(s) marked X")
            L.append("(refraction not solved)")
            if len(self.crossings) > 40:
                L.append("Many interacting shocks: the wave model is only")
                L.append("a sketch here - View > Map solves them properly.")
        if self.notes:
            L.append("")
            for n in list(dict.fromkeys(self.notes))[:8]:
                L.append("- " + n)
        return L


# =============================================================================
# Interactive app
# =============================================================================
class App:
    def __init__(self):
        set_app_id()
        plt.rcParams["toolbar"] = "None"
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = ["Microsoft Sans Serif", "Arial", "DejaVu Sans"]
        self.fig = plt.figure(figsize=(17, 7.7))
        try:
            self.fig.canvas.manager.set_window_title("TBG")
            set_window_icon(self.fig.canvas.manager.window)
            self.fig.canvas.mpl_disconnect(self.fig.canvas.manager.key_press_handler_id)
        except Exception:
            pass
        self.ax = self.fig.add_axes([0.012, 0.15, 0.725, 0.80])
        self.ax.set_anchor("NW")
        pan = self.pan_ax = self.fig.add_axes([0.745, 0.15, 0.25, 0.80])
        pan.axis("off")
        self.panel = pan.text(0.0, 1.0, "", va="top", family="monospace", fontsize=7.6)

        # only the Mach slider and the Randomise button live on the main window
        self.sM = Slider(self.fig.add_axes([0.07, 0.078, 0.46, 0.030]), "Mach", 0.5, 10.0, valinit=2.5)
        self.sM.on_changed(lambda _: self.recompute())

        # settings edited through  Edit > Setup...
        self.gamma = 1.4
        self.nlines = 15
        self.m_lo, self.m_hi = 2.0, 3.0
        self.random_spacing = False
        self._rand_fracs = None

        self.flutter = False
        self._busy = False
        self.pmap_on = False              # View > Map (tbg_pmap.py): a map is shown
        self.map_kind = "pressure"        # which map: pressure / mach / density / schlieren / vorticity
        self.eddy_on = False              # View > Eddies (tbg_eddy.py, schematic wake overlay)
        self.shock_on = True              # View > Shock lines (wave-model shocks, fans, slip lines)
        self.shock_var = None
        self.eddy_var = None
        self.pmap_var = None              # View > Map radio buttons ("off" or a kind)
        self.main_on = True               # Setup: main (free-stream) flow on / off
        self.main_var = None
        self.visc_on = False              # Setup: viscosity (flow solver only)
        self.reynolds = 2000.0            # Reynolds number per unit length
        self.cfd_cells = 300              # flow solver cells across the view
        self._pm = None
        self._pm_timer = self.fig.canvas.new_timer(interval=300)
        self._pm_timer.single_shot = True
        self._pm_timer.add_callback(self._pm_fine)
        self._pm_job = None               # background flow solver (tbg_pmap.CfdJob)
        self._cfd_cache = None            # (settings signature, solver, grid) of the last solution
        self._pm_seen = 0
        self._pm_status = ""
        self._pm_poll = self.fig.canvas.new_timer(interval=250)
        self._pm_poll.add_callback(self._pm_poll_cb)
        self.timer = self.fig.canvas.new_timer(interval=70)
        self.timer.add_callback(self.tick)

        self.scene = Scene()              # geometry + actuators come from the Draw window
        self.act_vals = {}                # actuator id -> slider value
        self.act_sliders, self.act_axes = [], []
        self.history = []                 # earlier scenes (for Edit > Undo)
        self.draw_win = None
        self.plane_on = False
        self.plane_var = None
        self.overlay = {}                 # Draw window: overlay image settings (kept between opens)
        self.setup_win = None
        self.walls = self.build()
        self.flut_btn = self.round_button([0.58, 0.064, 0.14, 0.056], "Randomise: OFF",
                                          self.toggle_flutter)
        self.fig.text(0.012, 0.012, "File > Draw... to sketch the geometry and actuators   |   "
                      "Ctrl+N new   Ctrl+O open   Ctrl+S save   |   U undo   P plane   M map on/off   E eddies   L shock lines   |   "
                      "Made with <3 by Shreeram Narayanan 2026",
                      fontsize=8, va="bottom")
        self.fig.text(0.995, 0.012, "Shreeram Narayanan", fontsize=8, va="bottom", ha="right",
                      color="#555555")
        self.fig.text(0.988, 0.012, "Shreeram Narayanan", fontsize=8, va="bottom", ha="right",
                      color="#555555")

        c = self.fig.canvas
        c.mpl_connect("key_press_event", self.on_key)
        c.mpl_connect("close_event", lambda e: self.timer.stop())
        c.mpl_connect("resize_event", lambda e: self.redraw() if hasattr(self, "flow") else None)
        self.build_menu()
        self.after_scene_change()

    # ---- scene (geometry and actuators come from the Draw window) ----------------------
    def build(self):
        walls = base_walls() if self.plane_on else []
        for p, q in scene_walls(self.scene, self.act_vals):
            walls = add_wall(walls, p, q)
        return walls

    def push(self):
        self.history.append(self.scene.copy())

    def undo(self):
        if self.history:
            self.scene = self.history.pop()
            self.after_scene_change()

    def clear(self):
        self.push()
        self.scene = Scene()
        self.after_scene_change()

    def open_draw(self):
        if self.win is None:
            return
        if self.draw_win is not None and self.draw_win.alive():
            self.draw_win.lift()
            return
        from tbg_draw import DrawWindow
        self.draw_win = DrawWindow(self.win, self.scene.copy(), self.apply_drawing,
                                   plane_getter=lambda: self.plane_on,
                                   view=(XMIN, XMAX, YMIN, YMAX), overlay=self.overlay)

    def apply_drawing(self, scene):
        """Called by the Draw window's Finish button."""
        self.push()
        had_src = bool(self.scene.sources)
        self.scene = scene
        if scene.sources and not had_src and not self.pmap_on:
            self.set_map("mach", recompute=False)      # source jets are only seen in a solver map
        self.after_scene_change()

    def after_scene_change(self, recompute=True):
        """New / opened / drawn / undone scene: reset actuator positions and rebuild sliders."""
        self.act_vals = {a.id: default_value(self.scene, a) for a in self.scene.acts}
        self.rebuild_act_sliders()
        if recompute:
            self.recompute()

    def rebuild_act_sliders(self):
        """One slider per actuator, stacked above the Mach slider. The plot shrinks to make room."""
        for s in self.act_sliders:
            try:
                s.disconnect_events()
            except Exception:
                pass
        for ax in self.act_axes:
            try:
                self.fig.delaxes(ax)
            except Exception:
                pass
        self.act_sliders, self.act_axes = [], []
        acts = self.scene.acts
        self.ax.set_position([0.012, 0.15, 0.725, 0.80])
        self.pan_ax.set_position([0.745, 0.15, 0.25, 0.80])
        for k, act in enumerate(acts):
            ax = self.fig.add_axes([0.80, 0.30, 0.15, 0.026])     # real position: place_act_sliders()
            lo, hi = act_range(self.scene, act)
            fmt = "%.2f" if act.kind == "linear" else "%.0f\u00b0"
            s = Slider(ax, act.name, lo, hi, valinit=self.act_vals[act.id], valfmt=fmt,
                       color=ACT_COLORS[k % len(ACT_COLORS)])
            s.on_changed(lambda v, aid=act.id: self.on_act(aid, v))
            self.act_axes.append(ax)
            self.act_sliders.append(s)
        self.place_act_sliders()

    def place_act_sliders(self):
        """Stack the actuator sliders in the right-hand column, a little below the end of the
        report text (M1, M2, ...). Follows the text as it grows or shrinks."""
        if not self.act_axes:
            return
        top = 0.62                                            # fallback if the text can't be measured
        try:
            rend = self.fig.canvas.get_renderer()
            bb = self.panel.get_window_extent(rend).transformed(self.fig.transFigure.inverted())
            top = bb.y0 - 0.06                                # gap under the report
        except Exception:
            pass
        pitch, h = 0.045, 0.026
        top = min(top, 0.93)
        n = len(self.act_axes)
        top = max(top, 0.15 + h + (n - 1) * pitch)            # never run into the Mach slider area
        for k, ax in enumerate(self.act_axes):
            ax.set_position([0.80, top - h - k * pitch, 0.15, h])

    def on_act(self, aid, v):
        self.act_vals[aid] = float(v)
        self.recompute()

    def _draw_actuators(self, ax):
        """Small, unobtrusive markers so the sliders can be matched to the picture."""
        for k, act in enumerate(self.scene.acts):
            col = ACT_COLORS[k % len(ACT_COLORS)]
            if act.kind == "linear":
                a, b = np.array(act.a, float), np.array(act.b, float)
                d = (b - a) / float(np.hypot(*(b - a)))
                pos = a + d * self.act_vals.get(act.id, 0.0)
                ax.plot([a[0], b[0]], [a[1], b[1]], color=col, lw=1.2, ls=(0, (4, 3)),
                        alpha=0.8, zorder=4.5)
                ax.plot(pos[0], pos[1], "o", ms=6, mfc=col, mec="white", zorder=4.6)
            else:
                ax.plot(act.a[0], act.a[1], "o", ms=8, mfc="white", mec=col, mew=2, zorder=4.6)

    def round_button(self, rect, label, fn, fc="#d9d9d9", hover="#f2f2f2", ec="#8c8c8c"):
        """Button drawn as a rounded rectangle (axes in inches so the corners stay circular)."""
        ax = self.fig.add_axes(rect)
        b = Button(ax, label, color=fc, hovercolor=hover)
        b.on_clicked(lambda _e: fn())
        w_in = rect[2] * self.fig.get_figwidth()
        h_in = rect[3] * self.fig.get_figheight()
        ax.set_xlim(0, w_in)
        ax.set_ylim(0, h_in)
        ax.axis("off")
        r = min(0.12, h_in / 2 - 0.01)
        pad = 0.02
        patch = FancyBboxPatch((pad, pad), w_in - 2 * pad, h_in - 2 * pad,
                               boxstyle=f"round,pad=0,rounding_size={r}",
                               fc=fc, ec=ec, lw=1.0, zorder=0, transform=ax.transData)
        ax.add_patch(patch)
        state = {"over": False}

        def motion(ev):
            over = ev.inaxes is ax
            if over != state["over"]:
                state["over"] = over
                patch.set_facecolor(hover if over else fc)
                self.fig.canvas.draw_idle()

        self.fig.canvas.mpl_connect("motion_notify_event", motion)
        return b

    # ---- text boxes / randomised inflow ---------------------------------------------
    def _line_fracs(self):
        """Positions of the airflow lines as fractions 0..1 of the available height."""
        n = self.nlines
        if not self.random_spacing:
            return np.linspace(0.0, 1.0, n)
        if self._rand_fracs is None or len(self._rand_fracs) != n:
            self._rand_fracs = np.sort(np.random.random(n))     # natural clumps and gaps
        return self._rand_fracs

    def _set_mach_silent(self, M):
        self.sM.eventson = False
        self.sM.set_val(M)
        self.sM.eventson = True

    def toggle_flutter(self):
        self.flutter = not self.flutter
        self.flut_btn.label.set_text("Randomise: ON" if self.flutter else "Randomise: OFF")
        if self.flutter:
            self.base_M = self.sM.val
            self.M_now = min(max(self.base_M, self.m_lo), self.m_hi)
            self.timer.start()
        else:
            self.timer.stop()
            self._set_mach_silent(self.base_M)
            self.recompute()
        self.fig.canvas.draw_idle()

    def tick(self):
        """Jitter the freestream Mach number inside [m_lo, m_hi] (smoothed random draws)."""
        if not self.flutter or self._busy:
            return
        self._busy = True
        try:
            target = random.uniform(self.m_lo, self.m_hi)
            self.M_now += 0.55 * (target - self.M_now)
            self._set_mach_silent(self.M_now)
            self.recompute()
        finally:
            self._busy = False

    # ---- plane ---------------------------------------------------------------------
    def set_plane(self, on):
        self.plane_on = bool(on)
        if self.plane_var is not None:
            self.plane_var.set(self.plane_on)
        self.recompute()

    def toggle_plane(self):
        self.set_plane(not self.plane_on)

    def set_main(self, on):
        self.main_on = bool(on)
        if self.main_var is not None:
            self.main_var.set(self.main_on)
        self.recompute()

    def set_shock(self, on):
        """View > Shock lines: show / hide the wave model's shocks, fans, slip lines and crossing
        marks (display only - the solution and the data panel are unchanged)."""
        self.shock_on = bool(on)
        if self.shock_var is not None:
            self.shock_var.set(self.shock_on)
        self.redraw()

    def set_eddy(self, on):
        self.eddy_on = bool(on)
        if self.eddy_var is not None:
            self.eddy_var.set(self.eddy_on)
        self.redraw()                         # overlay only: the flow solution is unchanged

    # ---- menu bar (File / Edit) ------------------------------------------------------
    def build_menu(self):
        try:
            import tkinter as tk
            self.tk = tk
            self.win = self.fig.canvas.manager.window
            win = self.win
            bar = tk.Menu(win, tearoff=0)
            fm = tk.Menu(bar, tearoff=0)
            fm.add_command(label="New", accelerator="Ctrl+N", command=self.clear)
            fm.add_command(label="Open...", accelerator="Ctrl+O", command=self.open_file)
            fm.add_command(label="Save...", accelerator="Ctrl+S", command=self.save_file)
            fm.add_command(label="Draw...", command=self.open_draw)
            fm.add_separator()
            fm.add_command(label="Exit", command=self.exit_app)
            bar.add_cascade(label="File", menu=fm)
            em = tk.Menu(bar, tearoff=0)
            em.add_command(label="Undo", accelerator="U", command=self.undo)
            self.plane_var = tk.BooleanVar(master=win, value=self.plane_on)   # shown in Setup
            em.add_separator()
            em.add_command(label="Setup...", command=self.open_setup)
            bar.add_cascade(label="Edit", menu=em)
            vm = tk.Menu(bar, tearoff=0)
            self.eddy_var = tk.BooleanVar(master=win, value=self.eddy_on)
            self.shock_var = tk.BooleanVar(master=win, value=self.shock_on)
            vm.add_checkbutton(label="Shock lines", accelerator="L", variable=self.shock_var,
                               command=lambda: self.set_shock(self.shock_var.get()))
            vm.add_checkbutton(label="Eddies", accelerator="E", variable=self.eddy_var,
                               command=lambda: self.set_eddy(self.eddy_var.get()))
            mm = tk.Menu(vm, tearoff=0)
            import tbg_pmap
            self.pmap_var = tk.StringVar(master=win, value=self.map_kind if self.pmap_on else "off")
            mm.add_radiobutton(label="Off", value="off", variable=self.pmap_var,
                               command=lambda: self.set_map(None))
            mm.add_separator()
            for kind, (label, _t) in tbg_pmap.MAPS.items():
                mm.add_radiobutton(label=label, value=kind, variable=self.pmap_var,
                                   accelerator="M" if kind == "pressure" else "",
                                   command=lambda k=kind: self.set_map(k))
            vm.add_cascade(label="Map", menu=mm)
            self.main_var = tk.BooleanVar(master=win, value=self.main_on)      # shown in Setup
            bar.add_cascade(label="View", menu=vm)
            am = tk.Menu(bar, tearoff=0)
            am.add_command(label="Version", command=self.show_version)
            am.add_command(label="Help", command=self.show_help)
            bar.add_cascade(label="About", menu=am)
            win.config(menu=bar)
            self.menubar = bar
        except Exception:
            self.win = None                       # not a Tk window: no menu bar

    def exit_app(self):
        self.timer.stop()
        self._pm_stop_job()
        plt.close(self.fig)

    # ---- .tbg files ---------------------------------------------------------------------
    def write_file(self, path):
        save_scene(path, self.scene, self.plane_on, {"main": self.main_on})

    def load_file(self, path):
        scene, plane, data = load_scene(path)
        self.push()
        self.scene = scene
        self.main_on = bool(data.get("main", True))
        if self.main_var is not None:
            self.main_var.set(self.main_on)
        if scene.sources and not self.pmap_on:
            self.set_map("mach", recompute=False)
        self.after_scene_change(recompute=False)
        self.set_plane(plane)                     # also redraws

    def save_file(self):
        if self.win is None:
            return
        from tkinter import filedialog, messagebox
        path = filedialog.asksaveasfilename(parent=self.win, title="Save geometry",
                                            defaultextension=".tbg",
                                            filetypes=[("TBG geometry", "*.tbg")])
        if not path:
            return
        try:
            self.write_file(path)
        except Exception as e:
            messagebox.showerror("Save failed", str(e), parent=self.win)

    def open_file(self):
        if self.win is None:
            return
        from tkinter import filedialog, messagebox
        path = filedialog.askopenfilename(parent=self.win, title="Open geometry",
                                          filetypes=[("TBG geometry", "*.tbg"), ("All files", "*.*")])
        if not path:
            return
        try:
            self.load_file(path)
        except Exception as e:
            messagebox.showerror("Open failed", f"Could not read this file:\n{e}", parent=self.win)

    # ---- About menu ----------------------------------------------------------------------
    def _popup(self, attr, title, width_hint=None):
        """Create (or raise) a small child window stored in self.<attr>."""
        old = getattr(self, attr, None)
        if old is not None and old.winfo_exists():
            old.lift()
            old.focus_force()
            return None
        top = self.tk.Toplevel(self.win)
        setattr(self, attr, top)
        top.title(title)
        top.transient(self.win)
        top.geometry(f"+{self.win.winfo_rootx() + 110}+{self.win.winfo_rooty() + 70}")
        top.bind("<Escape>", lambda _e: top.destroy())
        return top

    def show_version(self):
        if self.win is None:
            return
        top = self._popup("version_win", "Version")
        if top is None:
            return
        tk = self.tk
        import webbrowser
        top.resizable(False, False)
        f = ("Microsoft Sans Serif", 10)
        box = tk.Frame(top, padx=34, pady=26)
        box.pack()
        tk.Label(box, text=APP_NAME, font=("Microsoft Sans Serif", 15, "bold")).pack(pady=(0, 10))
        tk.Label(box, text=f"Version: {APP_VERSION}", font=f).pack()
        tk.Label(box, text=f"Latest Update on: {APP_DATE}", font=f).pack(pady=(2, 0))
        tk.Label(box, text=APP_AUTHOR, font=f).pack(pady=(14, 2))
        link = tk.Label(box, text=APP_URL, font=("Microsoft Sans Serif", 10, "underline"),
                        fg="#0b57d0", cursor="hand2")
        link.pack()
        link.bind("<Button-1>", lambda _e: webbrowser.open(APP_URL))
        tk.Button(box, text="OK", width=10, command=top.destroy).pack(pady=(18, 0))

    HELP_TEXT = [
        ("h", "Drawing a geometry"),
        ("p", "Choose File > Draw... to open the drawing window. Pick a tool from the Draw menu "
              "(Normal line, Spline, Source line, Linear actuator, Rotary actuator). For a line, click to place the start "
              "point, then either click the end point, or type a Length, press Tab, type an Angle and "
              "press Enter. The angle is measured counter-clockwise from the +x axis and an arc shows "
              "it. The chain continues from each end point; right-click or Esc ends it. Line ends "
              "snap like a magnet to existing vertices and lines, and clicking back on the first vertex "
              "closes a body (a closed body is solid and shocks never form inside it). You can draw "
              "anywhere: there is no grid snapping. Mouse wheel zooms, middle-drag pans, Fit resets."),
        ("h", "Line thickness"),
        ("p", "The Thickness box sets the thickness of the lines you draw next. 0 is an ideal thin wall. "
              "Anything above 0 becomes a solid plate with flat ends, so a plate facing the flow has a "
              "blunt leading edge. Two equally thick lines that meet are joined with a clean mitred "
              "corner. Right-click a line to change its thickness or delete it."),
        ("h", "Actuators (moving parts)"),
        ("p", "1) Draw your geometry.  2) Choose Linear actuator and draw its track like a line (for "
              "example type 20, Tab, 0). Or choose Rotary actuator and click where the hinge should "
              "be.  3) Right-click the actuator, choose Pair geometry..., and drag a box around the "
              "lines it should move (left to right: only lines fully inside; right to left: any line the "
              "box touches).  4) Click Finish.  The simulation window then shows one slider per "
              "actuator. A linear slider runs between the two ends of the track; the paired geometry "
              "slides parallel to the track. A rotary slider turns the paired geometry about the "
              "actuator point (set its range with right-click > Set angle range...). Each slider starts "
              "where you drew the geometry."),
        ("h", "How the flow works"),
        ("p", "Air arrives from the right and moves left at the Mach number set by the slider. "
              "Surfaces that face the flow turn it and create shocks; corners further along a "
              "surface create more shocks (compression) or expansion fans (the surface turns away). "
              "Shocks that hit a wall reflect, and two shocks of the same family that meet merge "
              "into one stronger shock."),
        ("h", "Reading the picture"),
        ("p", "Dark/white lines are shocks (the dark half faces the oncoming flow). Gradient wedges "
              "are expansion fans (darkest at the corner). Grey lines are streamlines. A gold X "
              "marks two shocks of opposite families crossing; their interaction is not solved. "
              "Dotted streamlines are in subsonic flow and are schematic. Dashed grey lines are slip lines. "
              "Coloured dashed lines and dots mark the actuators and match the slider colours."),
        ("h", "The data panel (right)"),
        ("p", "Each shock is listed with M1 and M2 (Mach before and after), beta (shock angle), "
              "theta (flow turning), the pressure ratio across it, pressure relative to freestream "
              "and the total-pressure ratio p0/p0inf (1.0 means no loss). Types: obl = oblique shock, "
              "mrg = merged shock, bow = detached curved shock, reg = regular reflection, "
              "refl and stem = Mach reflection (reflected shock and Mach stem), fan = expansion fan. "
              "Notes at the bottom explain special cases the solver met."),
        ("h", "Menus"),
        ("p", "File > New clears the geometry (Undo brings it back). Open and Save use .tbg files, "
              "which keep your lines and splines (with thickness), source lines, actuators, pairings and whether "
              "the plane and the main flow are on "
              "(older .tbg files still open). Draw... opens the drawing window. Exit closes TBG. "
              "Edit > Undo steps back through New / Open / Finish, and Setup opens the settings window. "
              "View > Shock lines (L) shows or hides the traced shocks, fans and slip lines (handy on top of a "
              "map), View > Eddies switches the wake overlay, View > Map chooses a solver map."),
        ("h", "Setup window"),
        ("p", "Plane: switches the horizontal ground wall on or off (applies at once; P does the same). "
              "Main flow: switches the free stream off for jets into still air (applies at once). "
              "Viscosity, Reynolds number and Solver cells set up the flow solver behind the maps. "
              "Gamma value: ratio of specific heats (1.4 for air). Flow lines: how many streamlines "
              "are drawn. Randomise min / max: the Mach range used by the Randomise button. "
              "Spacing: Equal gives evenly spaced streamlines, Random clusters them; pressing "
              "Apply with Random chosen rolls a new pattern. Press Apply (or Enter) to use the values."),
        ("h", "Randomise"),
        ("p", "The Randomise button makes the Mach number jitter quickly inside the min-max range, "
              "so the shocks flutter. Turn it off and your slider value comes back. Many streamlines "
              "or a complicated geometry slow the flicker, so lower the flow-line count if it stutters."),
        ("h", "Keyboard"),
        ("p", "Simulation window: Ctrl+N new, Ctrl+O open, Ctrl+S save, U undo, C clear, P plane, M map on/off, E eddies, L shock lines.  "
              "Drawing window: Esc ends the chain, the spline or point editing, or cancels pairing, Ctrl+Z "
              "undoes, Tab moves between Length and Angle, Enter places the line (or finishes a spline)."),
        ("h", "Tips"),
        ("p", "Switch the plane off if the ground wall adds reflections you do not want. A flat, blunt "
              "front face always gives a detached bow shock, because no attached shock can turn the "
              "flow 90 degrees. For an attached shock use a pointed or shallow-angle leading edge "
              "(below the maximum turn angle shown in the panel, theta_max). If a surface looks ignored, "
              "check that it faces the oncoming flow and is not hidden behind other geometry."),
        ("h", "Eddies (wake overlay)"),
        ("p", "Eddies are a viscous effect: the boundary layer separates at a sharp aft corner or base and "
              "the free shear layer rolls up. They are not caused by the speed being supersonic, and the "
              "inviscid solver cannot compute them. View > Eddies (E) draws them schematically: behind every "
              "leeward face steeper than 25 deg a recirculation bubble (shaded) with counter-rotating eddies, "
              "the shear layers (dashed), a recompression shock at the wake neck (dashed dark line, M > 1) "
              "and a decaying vortex street. A base on the ground is treated as a backward-facing step. "
              "Thin-line trailing edges shed a small street. With Randomise on, the eddies drift downstream. "
              "Sizes come from empirical recirculation lengths (W1, W2... in the panel), not from the solver. "
              "To see computed eddies, switch Viscosity on in Setup and use View > Map > Vorticity Map."),
        ("h", "Splines (curved geometry)"),
        ("p", "Draw > Spline: click the points the curve should pass through; it is drawn live. "
              "Press Enter (with the Length / Angle boxes empty), right-click or Esc to finish. "
              "Clicking the first point again closes the curve into a solid body (an airfoil, a nose "
              "cone). Typed Length / Angle + Enter adds a point exactly. Right-click a spline for "
              "Edit points (drag the points, Esc when done), Set thickness or Delete. The solver "
              "sees a spline as many short straight pieces, so a curved ramp gives a chain of weak "
              "shocks that coalesce and a curved shoulder a spread-out expansion, as in reality."),
        ("h", "Overlay image (retracing a design)"),
        ("p", "In the drawing window press Image... and Load a picture (png, jpg, bmp...). Set its "
              "Width, X and Y (world units) and Opacity, or use Move (drag it with the left button) "
              "and Scale by 2 points (click two points on the picture whose real distance you know, "
              "then type that distance). The picture is only a tracing aid; it is not part of the "
              "geometry and is not saved in .tbg files."),
        ("h", "Source lines (rocket nozzles, jets)"),
        ("p", "Draw > Source line, drawn like a line. Flow comes out of it normal to the line; green "
              "arrows show which side (right-click > Flip direction). Right-click > Set flow... sets "
              "the Mach number at the line (1.0 for a nozzle throat), p0/p_amb (chamber or total "
              "pressure over ambient pressure, the nozzle pressure ratio) and T0/T_amb (hot gas: 3 to "
              "10). Workflow for a nozzle: draw the bell walls with splines, put a source line across "
              "the throat (Mach 1), switch the main flow off in Edit > Setup for a static test in still "
              "air, then View > Map > Mach Map or Schlieren. Over-expanded (p_exit < p_amb) jets show "
              "oblique shocks from the lip and diamonds; strongly under-expanded jets show a Mach disk; "
              "very low pressure ratios separate inside the bell. The data panel gives the exit-side "
              "static pressure p/pa and the mass flow. Source lines are handled by the flow solver only; "
              "the wave model ignores them."),
        ("h", "Maps and the flow solver"),
        ("p", "View > Map picks one map: Pressure (p/p_inf), Mach (dashed = sonic line), Density, "
              "Schlieren (dark = strong density gradient: shocks, shear layers, diamonds) or Vorticity "
              "(red = counter-clockwise, blue = clockwise; eddies and shear layers). M switches the "
              "map on and off. The pressure map shows an instant preview while you drag; the solver "
              "picture follows when things settle and refreshes while it converges. The solver is a "
              "2-D Euler code; with Viscosity on (Setup) it becomes laminar Navier-Stokes with no-slip "
              "walls, so separation bubbles, wakes and eddies appear in the maps. Lower Reynolds "
              "numbers (500 - 5000 per unit length) are resolved on the default grid; higher ones "
              "need more Solver cells (slower)."),
        ("h", "Limits you should know about"),
        ("p", "The wave model (the lines) is a fast tracing of steady, inviscid, perfect-gas flow, "
              "meant for understanding shock patterns. Bow-shock standoff and curvature are empirical "
              "estimates, Mach-stem height is schematic, crossings of opposite-family shocks are "
              "marked but not solved there (the solver maps do solve them), and subsonic regions are "
              "not traced. The flow solver is planar 2-D (an axisymmetric rocket nozzle is shown as a "
              "2-D slot nozzle: diamond spacing is similar, Mach-disk sizes differ), steady (jets and "
              "wakes that are really unsteady show a snapshot that may keep moving), and with viscosity "
              "laminar (no turbulence model). Thin lines are two solver cells thick. Actuators move the "
              "geometry quasi-statically: each slider position is a separate steady solution."),
    ]

    def show_help(self):
        if self.win is None:
            return
        top = self._popup("help_win", "Help")
        if top is None:
            return
        tk = self.tk
        top.geometry("640x560")
        txt = tk.Text(top, wrap="word", padx=18, pady=14, bd=0, bg="#fafafa",
                      font=("Microsoft Sans Serif", 10), spacing3=4)
        sb = tk.Scrollbar(top, command=txt.yview)
        txt.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        txt.pack(side="left", fill="both", expand=True)
        txt.tag_configure("title", font=("Microsoft Sans Serif", 15, "bold"), spacing3=8)
        txt.tag_configure("h", font=("Microsoft Sans Serif", 11, "bold"), foreground="#000000",
                          spacing1=12, spacing3=3)
        txt.insert("end", "TBG help\n", "title")
        for kind, body in self.HELP_TEXT:
            txt.insert("end", body + "\n", "h" if kind == "h" else ())
        txt.configure(state="disabled")

    # ---- Edit > Setup window ------------------------------------------------------------
    def open_setup(self):
        if self.win is None:
            return
        tk = self.tk
        from tkinter import ttk
        if self.setup_win is not None and self.setup_win.winfo_exists():
            self.setup_win.lift()
            self.setup_win.focus_force()
            return
        top = tk.Toplevel(self.win)
        self.setup_win = top
        top.title("Setup")
        top.resizable(False, False)
        top.transient(self.win)
        ttk.Style(top).configure(".", font=("Microsoft Sans Serif", 9))

        v_g = tk.StringVar(master=top, value=f"{self.gamma:g}")
        v_n = tk.StringVar(master=top, value=str(self.nlines))
        v_lo = tk.StringVar(master=top, value=f"{self.m_lo:g}")
        v_hi = tk.StringVar(master=top, value=f"{self.m_hi:g}")
        v_sp = tk.StringVar(master=top, value="Random" if self.random_spacing else "Equal")
        msg = tk.StringVar(master=top, value="")

        frm = ttk.Frame(top, padding=16)
        frm.grid()
        rows = [("Gamma value", v_g), ("Flow lines", v_n),
                ("Randomise min (Mach)", v_lo), ("Randomise max (Mach)", v_hi)]
        first = None
        for r, (lab, var) in enumerate(rows):
            ttk.Label(frm, text=lab).grid(row=r, column=0, sticky="w", pady=5, padx=(0, 18))
            e = ttk.Entry(frm, textvariable=var, width=10)
            e.grid(row=r, column=1, sticky="w")
            first = first or e
        ttk.Label(frm, text="Spacing").grid(row=4, column=0, sticky="w", pady=5)
        ttk.Combobox(frm, textvariable=v_sp, values=["Equal", "Random"], state="readonly",
                     width=9).grid(row=4, column=1, sticky="w")
        ttk.Label(frm, text="Plane (ground wall)").grid(row=5, column=0, sticky="w", pady=5)
        if self.plane_var is not None:                # applies at once, like the P key
            ttk.Checkbutton(frm, variable=self.plane_var,
                            command=lambda: self.set_plane(self.plane_var.get())).grid(
                row=5, column=1, sticky="w")
        ttk.Label(frm, text="Main flow (free stream)").grid(row=6, column=0, sticky="w", pady=5)
        if self.main_var is not None:                 # applies at once
            ttk.Checkbutton(frm, variable=self.main_var,
                            command=lambda: self.set_main(self.main_var.get())).grid(
                row=6, column=1, sticky="w")
        ttk.Separator(frm).grid(row=7, column=0, columnspan=2, sticky="ew", pady=(8, 4))
        ttk.Label(frm, text="Flow solver (maps)", foreground="#555").grid(row=8, column=0, sticky="w")
        v_vi = tk.BooleanVar(master=top, value=self.visc_on)
        v_re = tk.StringVar(master=top, value=f"{self.reynolds:g}")
        v_ce = tk.StringVar(master=top, value=str(self.cfd_cells))
        ttk.Label(frm, text="Viscosity (eddies)").grid(row=9, column=0, sticky="w", pady=5)
        ttk.Checkbutton(frm, variable=v_vi).grid(row=9, column=1, sticky="w")
        ttk.Label(frm, text="Reynolds no. per unit length").grid(row=10, column=0, sticky="w", pady=5)
        ttk.Entry(frm, textvariable=v_re, width=10).grid(row=10, column=1, sticky="w")
        ttk.Label(frm, text="Solver cells across").grid(row=11, column=0, sticky="w", pady=5)
        ttk.Entry(frm, textvariable=v_ce, width=10).grid(row=11, column=1, sticky="w")
        tk.Label(frm, textvariable=msg, fg="#b00020",
                 font=("Microsoft Sans Serif", 9)).grid(row=12, column=0, columnspan=2,
                                                        sticky="w", pady=(8, 0))

        def apply(_ev=None):
            try:
                g = float(v_g.get())
                n = int(float(v_n.get()))
                lo, hi = float(v_lo.get()), float(v_hi.get())
                re_ = float(v_re.get())
                ce = int(float(v_ce.get()))
            except ValueError:
                msg.set("Please enter numbers in every box.")
                return
            if not 1.05 <= g <= 2.0:
                msg.set("Gamma must be between 1.05 and 2.0.")
                return
            if not 2 <= n <= 300:
                msg.set("Flow lines must be between 2 and 300.")
                return
            if not 50.0 <= re_ <= 1e7:
                msg.set("Reynolds number must be between 50 and 1e7.")
                return
            if not 80 <= ce <= 1200:
                msg.set("Solver cells must be between 80 and 1200.")
                return
            lo, hi = max(1.05, min(lo, hi)), min(10.0, max(lo, hi))
            msg.set("")
            self.visc_on, self.reynolds, self.cfd_cells = bool(v_vi.get()), re_, ce
            v_re.set(f"{re_:g}")
            v_ce.set(str(ce))
            self.gamma, self.nlines, self.m_lo, self.m_hi = g, n, lo, hi
            self.random_spacing = (v_sp.get() == "Random")
            self._rand_fracs = None               # Apply with Random gives a fresh pattern
            v_g.set(f"{g:g}")
            v_n.set(str(n))
            v_lo.set(f"{lo:g}")
            v_hi.set(f"{hi:g}")
            self.recompute()

        def close():
            top.destroy()
            self.setup_win = None

        bar = ttk.Frame(frm)
        bar.grid(row=13, column=0, columnspan=2, sticky="e", pady=(14, 0))
        ttk.Button(bar, text="Apply", command=apply).grid(row=0, column=0, padx=(0, 8))
        ttk.Button(bar, text="Close", command=close).grid(row=0, column=1)
        top.bind("<Return>", apply)
        top.bind("<Escape>", lambda _e: close())
        top.protocol("WM_DELETE_WINDOW", close)
        top.update_idletasks()
        top.geometry(f"+{self.win.winfo_rootx() + 90}+{self.win.winfo_rooty() + 90}")
        self._setup_apply = apply                 # (kept for testing)
        self._setup_vars = (v_g, v_n, v_lo, v_hi, v_sp, v_vi, v_re, v_ce)
        if first is not None:
            first.focus_set()

    def on_key(self, ev):
        if ev.key == "ctrl+n":
            self.clear()
        elif ev.key == "ctrl+o":
            self.open_file()
        elif ev.key == "ctrl+s":
            self.save_file()
        elif ev.key == "u":
            self.undo()
        elif ev.key == "c":
            self.clear()
        elif ev.key == "p":
            self.toggle_plane()
        elif ev.key == "m":
            self.set_pmap(not self.pmap_on)
        elif ev.key == "l":
            self.set_shock(not self.shock_on)
        elif ev.key == "e":
            self.set_eddy(not self.eddy_on)

    # ---- compute and draw -------------------------------------------------------------
    def recompute(self):
        self.walls = self.build()
        self.flow = Flow(self.walls, self.sM.val, self.gamma, self.plane_on, self.main_on)
        self.sources = scene_sources(self.scene, self.act_vals)
        self._pm_stop_job()
        if self.pmap_on:
            self._pm_update(fine=False)               # quick preview now ...
            self._pm_timer.stop()
            if not self.flutter:
                self._pm_timer.start()                # ... full resolution once things settle
        self.redraw()

    # ---- pressure map (separate module: tbg_pmap.py) -----------------------------------
    def set_pmap(self, on):
        """M key: last chosen map on / off."""
        self.set_map(self.map_kind if on else None)

    def set_map(self, kind, recompute=True):
        """View > Map: kind = None (off) or one of tbg_pmap.MAPS."""
        self.pmap_on = kind is not None
        if kind is not None:
            self.map_kind = kind
        if self.pmap_var is not None:
            self.pmap_var.set(kind if kind is not None else "off")
        self._pm = None
        self._pm_timer.stop()
        self._pm_stop_job()
        if recompute:
            self.recompute()

    def _preview_ok(self):
        """The instant streamline preview exists only for the pressure map of a supersonic main
        flow without source lines; everything else waits for the flow solver."""
        return (self.map_kind == "pressure" and self.main_on and not self.sources
                and self.flow.M > 1.0 + 1e-9)

    def _pm_update(self, fine):
        if not self._preview_ok():
            self._pm = None
            return
        try:
            import tbg_pmap
            ext = (XMIN, XMAX, YMIN, YMAX)
            nx, ny = tbg_pmap.resolution(self.ax, fine, ext)
            self._pm = tbg_pmap.compute(self.flow, ext, nx, ny, plane=self.plane_on)
            if self._pm is not None:
                self._pm["fine"] = bool(fine)
        except Exception as e:                           # never let the map break the solver
            self._pm = None
            self.flow.notes.append(f"pressure map failed: {e}")

    def _pm_fine(self):
        """Things have settled: solve the whole flow field in a background thread. The quick
        streamline preview stays on screen until the first solver picture is ready."""
        if not (self.pmap_on and not self.flutter and not self._busy):
            return
        if not self.sources and not (self.main_on and self.flow.M > 0.2):
            self._pm, self._pm_status = None, ("nothing to solve: main flow off and no source lines"
                                               if not self.main_on else "Mach too low for the solver")
            self.redraw()
            return
        sig = self._cfd_sig()
        if self._cfd_cache is not None and self._cfd_cache[0] == sig:      # same flow, other map
            try:
                import tbg_pmap
                ext = (XMIN, XMAX, YMIN, YMAX)
                nx, ny = tbg_pmap.resolution(self.ax, True, ext)
                _s, sol, geo, status = self._cfd_cache
                self._pm = tbg_pmap.to_pm(sol.field(self.map_kind), geo, ext, nx, ny, self.map_kind)
                self._pm.update(fine=True, cfd=True)
                self._pm_status = status
                self.redraw()
                return
            except Exception:
                self._cfd_cache = None
        try:
            import tbg_pmap
            ext = (XMIN, XMAX, YMIN, YMAX)
            nx, ny = tbg_pmap.resolution(self.ax, True, ext)
            self._pm_stop_job()
            job = tbg_pmap.CfdJob(self.flow, ext, nx, ny, self.plane_on, kind=self.map_kind,
                                  main=self.main_on, sources=self.sources,
                                  visc=self.reynolds if self.visc_on else None,
                                  cells=self.cfd_cells)
            job.sig = sig
            self._pm_job, self._pm_seen = job, 0
            self._pm_status = "solving flow field..."
            job.start()
            self._pm_poll.start()
        except Exception as e:                           # fall back to the streamline map
            self._pm_job = None
            self.flow.notes.append(f"flow solver unavailable: {e}")
            self._pm_update(fine=True)
        self.redraw()

    def _cfd_sig(self):
        """Everything the solver result depends on (a different map kind reuses the solution)."""
        r = lambda v: round(float(v), 6)
        return (tuple((K(a), K(b)) for a, b in self.walls), r(self.flow.M), r(self.gamma),
                self.plane_on, self.main_on,
                tuple((K(sc["a"]), K(sc["b"]), r(sc["n"][0]), r(sc["n"][1]), r(sc["M"]), r(sc["npr"]),
                       r(sc["t0r"])) for sc in self.sources),
                self.reynolds if self.visc_on else None, self.cfd_cells)

    def _pm_stop_job(self):
        job, self._pm_job = self._pm_job, None
        if job is not None:
            job.cancel()
        self._pm_poll.stop()
        self._pm_status = ""

    def _pm_poll_cb(self):
        job = self._pm_job
        if job is None:
            self._pm_poll.stop()
            return
        pm, done, status, gen = job.snapshot()
        changed = done
        self._pm_status = "" if done and pm is not None else status
        if pm is not None and gen != self._pm_seen and self.pmap_on:
            self._pm_seen, self._pm, changed = gen, pm, True
        if done:
            self._pm_poll.stop()
            if pm is None:                               # the solver failed: streamline map instead
                self.flow.notes.append(status)
                self._pm_update(fine=True)
            else:
                self._pm_status = status
                if job.sol is not None:
                    self._cfd_cache = (getattr(job, "sig", None), job.sol, job.geo, status)
        if changed and self.pmap_on:
            self.redraw()

    def _draw_fan(self, ax, fn):
        """Gradient triangle: C_FAN_NEAR at the corner -> C_FAN_FAR at the far end."""
        pts = [fn.origin]
        for k in np.linspace(0.0, 1.0, 25):
            d = dirv(fn.a0 + float(np.interp(k, fn.ks, fn.rel)))
            h = self.flow.first_hit(fn.origin, d)
            L = min(FAN_LEN, h[0] if h else FAN_LEN)
            pts.append(fn.origin + d * L)
        pts = np.array(pts)
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        if x1 - x0 < 1e-6 or y1 - y0 < 1e-6:
            return
        nn = 200 if fn.turn > math.radians(4.0) else 60
        X, Y = np.meshgrid(np.linspace(x0, x1, nn), np.linspace(y0, y1, nn))
        r = np.clip(np.hypot(X - fn.origin[0], Y - fn.origin[1]) / FAN_LEN, 0, 1)
        im = ax.imshow(r, cmap=FAN_CMAP, vmin=0, vmax=1, extent=(x0, x1, y0, y1),
                       origin="lower", interpolation="bilinear", aspect="auto", zorder=3.5)
        patch = Polygon(pts, closed=True, fc="none", ec="none", transform=ax.transData)
        ax.add_patch(patch)
        im.set_clip_path(patch)

    def _draw_fan_rays(self, ax, fn):
        """Fan outline only (first / last characteristic) - used when the pressure map is on."""
        for k in (0, -1):
            a, b = fn.lines[k]
            ax.plot([a[0], b[0]], [a[1], b[1]], color="white", lw=1.0, ls=(0, (4, 3)),
                    alpha=0.85, zorder=4)

    def _upp(self):
        """Data units per point on screen (for sub-line offsets)."""
        self.ax.apply_aspect()
        wpx = self.ax.get_window_extent().width
        return (self.fig.dpi / 72.0) / (wpx / (XMAX - XMIN))

    def _white_half(self, ax, w, lw, upp):
        """Light half of a shock line: half width, shifted to the downstream side by lw/4
        (dark half faces the oncoming flow, white half is at the back)."""
        P = np.array(w.pts, float)
        if len(P) < 2:
            return
        u1 = dirv(w.S1.phi)
        segn = []
        for k in range(len(P) - 1):
            e = P[k + 1] - P[k]
            L = float(np.hypot(*e))
            if L < 1e-12:
                segn.append(np.zeros(2))
                continue
            n = np.array([-e[1], e[0]]) / L
            if n @ u1 < 0:
                n = -n                        # downstream side
            segn.append(n)
        vn = []
        for k in range(len(P)):
            v = np.zeros(2)
            if k > 0:
                v += segn[k - 1]
            if k < len(segn):
                v += segn[k]
            nv = float(np.hypot(*v))
            vn.append(v / nv if nv > 1e-12 else v)
        Q = P + np.array(vn) * (lw / 4.0) * upp
        ax.plot(Q[:, 0], Q[:, 1], color=C_SHOCK_HI, lw=lw / 2.0, zorder=6.1,
                solid_capstyle="butt")

    def _draw_sources(self, ax):
        """Source lines: green line, outflow arrows, label J1, J2 ..."""
        for k, sc in enumerate(self.sources):
            a, b, n = sc["a"], sc["b"], sc["n"]
            ax.plot([a[0], b[0]], [a[1], b[1]], color=SRC_COLOR, lw=3.2, zorder=5.2,
                    solid_capstyle="butt")
            m = max(2, int(sc["L"] / 0.35) + 1)
            for t in np.linspace(0.15, 0.85, m):
                o = a + t * (b - a)
                ax.annotate("", xy=o + n * 0.32, xytext=o, zorder=5.3,
                            arrowprops=dict(arrowstyle="-|>", color=SRC_COLOR, lw=1.4,
                                            mutation_scale=9))
            c = 0.5 * (a + b) - n * 0.22
            ax.text(c[0], c[1], f"J{k + 1}", color=SRC_COLOR, fontsize=8, ha="center",
                    va="center", zorder=7, fontweight="bold")

    def _source_report(self):
        g = self.gamma
        L = ["Source lines (flow solver only)",
             "ID  M    p0/pa  T0/Ta  p/pa   mdot   V/a_amb"]
        for k, sc in enumerate(self.sources):
            M = max(sc["M"], 1e-6)
            f = 1.0 + 0.5 * (g - 1.0) * M * M
            ps = sc["npr"] / f ** (g / (g - 1.0))
            ts = sc["t0r"] / f
            V = M * math.sqrt(ts)
            mdot = ps / ts * V * sc["L"]                    # rho_amb * a_amb * (unit depth)
            L.append(f"J{k + 1:<2}{sc['M']:5.2f}{sc['npr']:7.2f}{sc['t0r']:6.2f}{ps:7.2f}"
                     f"{mdot:7.2f}{V:7.2f}")
        L.append("(p/pa < 1 at the exit: over-expanded, > 1: under-")
        L.append(" expanded; mdot in rho_amb a_amb per unit depth)")
        return L

    def redraw(self):
        ax, fl = self.ax, self.flow
        ax.clear()
        ax.set_facecolor(C_BG)
        ax.set_xlim(XMIN, XMAX)
        ax.set_ylim(YMIN, YMAX)
        ax.set_aspect("equal", adjustable="box", anchor="NW")
        ax.grid(alpha=0.15)
        ax.set_title(f"Freestream Mach {fl.M:.2f}" if self.main_on else
                     "Main flow off  (still ambient air)", fontsize=11)

        if self.plane_on:
            ax.fill_between([XMIN, XMAX], [YMIN, YMIN], [0, 0], color=C_SOLID, zorder=2)
        for poly in bounded_faces(self.walls):
            ax.fill([p[0] for p in poly], [p[1] for p in poly], color=C_SOLID, zorder=2)
        for a, b in self.walls:
            ax.plot([a[0], b[0]], [a[1], b[1]], color="black", lw=3, solid_capstyle="round", zorder=5)
        self._draw_actuators(ax)
        if self.pmap_on and self._pm is not None:
            import tbg_pmap
            tbg_pmap.draw(ax, self._pm)

        self._draw_sources(ax)
        if self.main_on and fl.M > 1.0 + 1e-9:
            if self.plane_on:
                ys = 0.12 + self._line_fracs() * (YMAX - 0.25 - 0.12)
            else:
                ys = (YMIN + 0.15) + self._line_fracs() * (YMAX - 0.25 - YMIN - 0.15)
            for y0, (ln, sub) in zip(ys, fl.streamlines(ys)):
                if sub is None:
                    ax.plot(ln[:, 0], ln[:, 1], color=C_STREAM, lw=0.9, zorder=3)
                else:
                    ax.plot(ln[:sub + 1, 0], ln[:sub + 1, 1], color=C_STREAM, lw=0.9, zorder=3)
                    ax.plot(ln[sub:, 0], ln[sub:, 1], color=C_STREAM, lw=0.9, ls=":", zorder=3)
                ax.annotate("", xy=(XMAX - 1.0, y0), xytext=(XMAX - 0.45, y0),
                            arrowprops=dict(arrowstyle="-|>", color=C_STREAM, lw=1), zorder=3)
        waves = fl.waves if self.shock_on else []       # View > Shock lines
        fans = fl.fans if self.shock_on else []
        for fn in fans:
            if self.pmap_on and self._pm is not None:
                self._draw_fan_rays(ax, fn)           # the map already colours the fan region
            else:
                self._draw_fan(ax, fn)
        for a, b in (fl.slips if self.shock_on else []):
            ax.plot([a[0], b[0]], [a[1], b[1]], color=C_SLIP, ls="--", lw=1.3, zorder=4)
        eddy_rep = []
        if self.eddy_on and self.main_on:
            try:
                import tbg_eddy
                eddy_rep = tbg_eddy.draw(ax, fl, (XMIN, XMAX, YMIN, YMAX), plane=self.plane_on,
                                         animate=self.flutter)
            except Exception as e:                    # never let the overlay break the app
                eddy_rep = [f"eddy overlay failed: {e}"]
        upp = self._upp()
        nlab = 0
        for w in waves:
            xs, ys = [p[0] for p in w.pts], [p[1] for p in w.pts]
            lw = 3.4 if w.kind == "stem" else 2.6
            ax.plot(xs, ys, color=C_SHOCK, lw=lw, zorder=6, solid_capstyle="round")
            self._white_half(ax, w, lw, upp)
            if w.tail:
                ax.plot([w.tail[0][0], w.tail[1][0]], [w.tail[0][1], w.tail[1][1]],
                        color=C_SHOCK, ls=":", lw=1.6, alpha=0.7, zorder=6)
            if (w.theta < math.radians(2.0) and w.kind in ("oblique", "regular")) or nlab >= 40:
                continue                              # weak waves of curved surfaces: no label
            nlab += 1
            k = len(w.pts) // 2
            mid = 0.5 * (w.pts[k - 1] + w.pts[k]) if len(w.pts) > 1 else w.pts[0]
            ax.text(mid[0] + 0.1, mid[1] + 0.12, w.id, color=C_SHOCK, fontsize=8, zorder=7,
                    bbox=dict(boxstyle="round,pad=0.12", fc=C_BG, ec="none", alpha=0.75))
        for fn in fans:
            if fn.turn >= math.radians(2.0):
                ax.text(fn.origin[0], fn.origin[1] + 0.25, fn.id, color=C_FAN_NEAR, fontsize=8,
                        zorder=7)
        for p in (fl.crossings[:80] if self.shock_on else []):
            ax.plot(p[0], p[1], marker="X", ms=9, mfc="gold", mec="black", zorder=8)

        # imshow (fans) can change the axes settings: restore them
        ax.set_xlim(XMIN, XMAX)
        ax.set_ylim(YMIN, YMAX)
        ax.set_aspect("equal", adjustable="box", anchor="NW")

        rep = fl.report()
        if self.sources:
            rep += [""] + self._source_report()
        if eddy_rep:
            rep += [""] + eddy_rep
        if self.pmap_on and self._pm is not None:
            pm = self._pm
            if pm.get("cfd"):
                import tbg_pmap
                name = tbg_pmap.MAPS[pm.get("kind", "pressure")][0]
                rep += ["", f"{name} {pm['nx']}x{pm['ny']}  (flow solver"
                        + (f", viscous Re {self.reynolds:g}/unit)" if self.visc_on else ", inviscid)"),
                        f"range {pm['pmin']:.3g} .. {pm['pmax']:.3g}"]
            else:
                rep += ["", f"pressure map {pm['nx']}x{pm['ny']}{'' if pm.get('fine') else ' (preview)'}",
                        f"p/pinf {pm['pmin']:.2f} .. {pm['pmax']:.2f}   hatched = not solved"]
            if self._pm_status:
                rep.append(self._pm_status)
        elif self.pmap_on and self._pm_status:
            rep += ["", self._pm_status]
        self.panel.set_text("\n".join(rep))
        self.place_act_sliders()
        self.fig.canvas.draw_idle()


def run():
    App()
    plt.show()


if __name__ == "__main__":
    run()
