import json
import math
import os
import sys
from dataclasses import dataclass, field

import numpy as np

# ---- app identity -----------------------------------------------------------------
ICON_FILE = "icon.ico"      # looked up in the assets folder (.ico on Windows; .png also works)
APP_ID = "TBG.App.1"
APP_NAME = "Theta Beta Gamma \u03b1-build"
APP_VERSION = "0.89"
APP_DATE = "28-10-2026"
APP_AUTHOR = "Shreeram Narayanan 2026"
APP_URL = "https://github.com/ShreeramN"
SPLASH_FILE = "splash.png"  # optional splash image in the assets folder (png/gif)
SPLASH_MS = 3000            # how long the splash stays up
SPLASH_TITLE = "TBG"        # shown if there is no splash image

# one colour per actuator (same order in the drawing window and on the sliders)
SRC_COLOR = "#0ca678"       # source lines (drawing window and simulation window)
SPLINE_STEP = 0.1           # spline sampling: one straight piece per ~this many world units

ACT_COLORS = ["#e8590c", "#1c7ed6", "#2f9e44", "#ae3ec9", "#f08c00", "#0c8599",
              "#c2255c", "#5c7cfa"]


def asset_path(name):
    """Find a file in the assets folder. Looks in ../assets (source/ + assets/ layout),
    ./assets, next to the script, and inside a PyInstaller bundle. Returns None if missing."""
    here = os.path.dirname(os.path.abspath(__file__))
    dirs = [os.path.join(here, "..", "assets"), os.path.join(here, "assets"), here]
    bundle = getattr(sys, "_MEIPASS", None)             # set when running as a built .exe
    if bundle:
        dirs = [os.path.join(bundle, "assets"), bundle] + dirs
    for d in dirs:
        q = os.path.normpath(os.path.join(d, name))
        if os.path.exists(q):
            return q
    return None


def set_app_id():
    """Windows: give the process its own identity so the taskbar shows our icon, not Python's.
    Must run BEFORE the first window is created."""
    if sys.platform.startswith("win"):
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)
        except Exception:
            pass


def set_window_icon(win):
    """Give a Tk window (root or Toplevel) the TBG icon, if assets/icon.ico exists."""
    path = asset_path(ICON_FILE)
    if not path:
        return
    try:
        if hasattr(win, "iconbitmap"):                       # Tk
            if path.lower().endswith(".ico") and sys.platform.startswith("win"):
                win.iconbitmap(path)
            else:
                import tkinter
                win._tbg_icon = tkinter.PhotoImage(file=path)
                win.iconphoto(True, win._tbg_icon)
        elif hasattr(win, "setWindowIcon"):                  # Qt backends
            from matplotlib.backends.qt_compat import QtGui
            win.setWindowIcon(QtGui.QIcon(path))
    except Exception:
        pass


# =============================================================================
# Scene model
# =============================================================================
@dataclass
class Line:
    id: int
    p: tuple
    q: tuple
    t: float = 0.0                 # thickness (world units). 0 = ideal thin wall


@dataclass
class Actuator:
    id: int
    kind: str                      # "linear" | "rotary"
    a: tuple                       # linear: first end   | rotary: centre of rotation
    b: tuple = None                # linear: second end
    paired: list = field(default_factory=list)      # ids of the Lines this actuator moves
    lo: float = -90.0              # rotary: slider range (degrees)
    hi: float = 90.0
    name: str = ""


@dataclass
class Spline:
    id: int
    pts: list                      # control points; the curve passes through every one
    t: float = 0.0                 # thickness, as for a Line
    closed: bool = False


@dataclass
class Source:
    """A source line: flow comes out of it, normal to the line (rocket engine throat etc.)."""
    id: int
    p: tuple
    q: tuple
    M: float = 1.0                 # Mach number at the line (1 = sonic throat)
    npr: float = 10.0              # total pressure / ambient static pressure
    t0r: float = 1.0               # total temperature / ambient temperature
    flip: bool = False             # False: flow leaves to the right of p -> q


class Scene:
    def __init__(self):
        self.lines = []
        self.splines = []
        self.sources = []
        self.acts = []
        self.next_id = 1

    # ---- editing helpers
    def _new_id(self):
        i = self.next_id
        self.next_id += 1
        return i

    def add_line(self, p, q, t=0.0):
        ln = Line(self._new_id(), (float(p[0]), float(p[1])), (float(q[0]), float(q[1])), float(t))
        self.lines.append(ln)
        return ln

    def add_spline(self, pts, t=0.0, closed=False):
        sp = Spline(self._new_id(), [(float(x), float(y)) for x, y in pts], float(t), bool(closed))
        self.splines.append(sp)
        return sp

    def add_source(self, p, q):
        sc = Source(self._new_id(), (float(p[0]), float(p[1])), (float(q[0]), float(q[1])))
        self.sources.append(sc)
        return sc

    def add_actuator(self, kind, a, b=None):
        act = Actuator(self._new_id(), kind, (float(a[0]), float(a[1])),
                       None if b is None else (float(b[0]), float(b[1])))
        self.acts.append(act)
        self.renumber()
        return act

    def renumber(self):
        n = {"linear": 0, "rotary": 0}
        for a in self.acts:
            n[a.kind] += 1
            a.name = f"{'Linear' if a.kind == 'linear' else 'Rotary'} {n[a.kind]}"

    def line(self, i):
        for ln in self.lines:
            if ln.id == i:
                return ln
        return None

    def item(self, i):
        """Any piece of geometry (line, spline or source line) by id."""
        for ob in self.lines + self.splines + self.sources:
            if ob.id == i:
                return ob
        return None

    def delete_item(self, i):
        self.lines = [ob for ob in self.lines if ob.id != i]
        self.splines = [ob for ob in self.splines if ob.id != i]
        self.sources = [ob for ob in self.sources if ob.id != i]
        for a in self.acts:
            a.paired = [k for k in a.paired if k != i]

    def delete_line(self, i):
        self.delete_item(i)

    def delete_actuator(self, i):
        self.acts = [a for a in self.acts if a.id != i]
        self.renumber()

    def is_empty(self):
        return not (self.lines or self.splines or self.sources or self.acts)

    # ---- (de)serialisation
    def to_dict(self):
        return {
            "lines": [{"id": l.id, "p": list(l.p), "q": list(l.q), "t": l.t} for l in self.lines],
            "splines": [{"id": c.id, "pts": [list(p) for p in c.pts], "t": c.t, "closed": c.closed}
                        for c in self.splines],
            "sources": [{"id": c.id, "p": list(c.p), "q": list(c.q), "M": c.M, "npr": c.npr,
                         "t0r": c.t0r, "flip": c.flip} for c in self.sources],
            "actuators": [{"id": a.id, "kind": a.kind, "name": a.name, "a": list(a.a),
                           "b": None if a.b is None else list(a.b), "paired": list(a.paired),
                           "lo": a.lo, "hi": a.hi} for a in self.acts],
            "next_id": self.next_id,
        }

    @classmethod
    def from_dict(cls, d):
        s = cls()
        for l in d.get("lines", []):
            s.lines.append(Line(int(l["id"]), tuple(map(float, l["p"])), tuple(map(float, l["q"])),
                                float(l.get("t", 0.0))))
        for c in d.get("splines", []):
            s.splines.append(Spline(int(c["id"]), [tuple(map(float, p)) for p in c["pts"]],
                                    float(c.get("t", 0.0)), bool(c.get("closed", False))))
        for c in d.get("sources", []):
            s.sources.append(Source(int(c["id"]), tuple(map(float, c["p"])), tuple(map(float, c["q"])),
                                    float(c.get("M", 1.0)), float(c.get("npr", 10.0)),
                                    float(c.get("t0r", 1.0)), bool(c.get("flip", False))))
        for a in d.get("actuators", []):
            s.acts.append(Actuator(int(a["id"]), a["kind"], tuple(map(float, a["a"])),
                                   None if a.get("b") is None else tuple(map(float, a["b"])),
                                   [int(k) for k in a.get("paired", [])],
                                   float(a.get("lo", -90.0)), float(a.get("hi", 90.0)),
                                   a.get("name", "")))
        ids = [o.id for o in s.lines + s.splines + s.sources] + [a.id for a in s.acts]
        s.next_id = max(int(d.get("next_id", 1)), (max(ids) + 1) if ids else 1)
        s.renumber()
        return s

    def copy(self):
        return Scene.from_dict(self.to_dict())


def save_scene(path, scene, plane, extra=None):
    """extra: further settings to keep with the geometry (e.g. {"main": False})."""
    data = {"format": "tbg", "version": 3, "plane": bool(plane)}
    data.update(extra or {})
    data.update(scene.to_dict())
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)


def load_scene(path):
    """Returns (Scene, plane_on, raw file dict). Also reads the old v1 / v2 files."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    plane = bool(data.get("plane", True))
    if "segments" in data and "lines" not in data:          # v1 file
        s = Scene()
        for a, b in data["segments"]:
            s.add_line(a, b, 0.0)
        return s, plane, data
    return Scene.from_dict(data), plane, data


# =============================================================================
# Splines
# =============================================================================
def spline_curve(pts, closed=False, step=SPLINE_STEP):
    """Centripetal Catmull-Rom spline through the control points, sampled into straight pieces
    (array of points; closed curves end on their first point). It passes exactly through every
    control point, never overshoots into loops, and two control points give a straight line."""
    P = []
    for p in pts:
        p = np.array(p, float)
        if not P or float(np.hypot(*(p - P[-1]))) > 1e-9:
            P.append(p)
    if closed and len(P) > 2 and float(np.hypot(*(P[0] - P[-1]))) < 1e-9:
        P.pop()
    n = len(P)
    if n < 2:
        return np.array(P)
    if n == 2 or (closed and n < 3):
        return np.array(P)
    G = ([P[-1]] + P + [P[0], P[1]]) if closed else ([2 * P[0] - P[1]] + P + [2 * P[-1] - P[-2]])
    out = []
    for i in range(n if closed else n - 1):
        p0, p1, p2, p3 = G[i], G[i + 1], G[i + 2], G[i + 3]
        t0 = 0.0
        t1 = t0 + max(float(np.hypot(*(p1 - p0))), 1e-9) ** 0.5
        t2 = t1 + max(float(np.hypot(*(p2 - p1))), 1e-9) ** 0.5
        t3 = t2 + max(float(np.hypot(*(p3 - p2))), 1e-9) ** 0.5
        k = int(min(32, max(4, math.ceil(float(np.hypot(*(p2 - p1))) / step))))
        out.append(p1.copy())                               # exact control point
        for j in range(1, k):
            t = t1 + (t2 - t1) * j / k
            A1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
            A2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
            A3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
            B1 = (t2 - t) / (t2 - t0) * A1 + (t - t0) / (t2 - t0) * A2
            B2 = (t3 - t) / (t3 - t1) * A2 + (t - t1) / (t3 - t1) * A3
            out.append((t2 - t) / (t2 - t1) * B1 + (t - t1) / (t2 - t1) * B2)
    out.append(P[0].copy() if closed else P[-1].copy())
    return np.array(out)


# =============================================================================
# Actuator motion
# =============================================================================
def _item_points(ob):
    if isinstance(ob, Spline):
        return [np.array(p, float) for p in ob.pts]
    return [np.array(ob.p, float), np.array(ob.q, float)]


def _centroid(scene, ids):
    pts = []
    for i in ids:
        ob = scene.item(i)
        if ob is not None:
            pts += _item_points(ob)
    return np.mean(pts, axis=0) if pts else None


def linear_length(act):
    return max(float(np.hypot(*(np.array(act.b, float) - np.array(act.a, float)))), 1e-6)


def act_range(scene, act):
    """(min, max) of the slider. Linear: distance along the line from end A. Rotary: degrees."""
    if act.kind == "linear":
        return 0.0, linear_length(act)
    lo, hi = float(act.lo), float(act.hi)
    if hi - lo < 1e-6:
        hi = lo + 1e-6
    return lo, hi


def default_value(scene, act):
    """Slider value that reproduces the geometry exactly as it was drawn."""
    if act.kind == "linear":
        a, b = np.array(act.a, float), np.array(act.b, float)
        L = linear_length(act)
        d = (b - a) / L
        c = _centroid(scene, act.paired)
        if c is None:
            return 0.5 * L
        return float(np.clip((c - a) @ d, 0.0, L))
    lo, hi = act_range(scene, act)
    return float(min(max(0.0, lo), hi))


def moved_points(scene, vals):
    """Points of every line / spline / source line at the current actuator positions:
    {id: [np arrays]}. Linear actuator: the paired geometry's centroid rides along the actuator
    line (slider = distance from end A). Rotary actuator: paired geometry rotates about the
    actuator point by the slider angle (counter-clockwise positive)."""
    cur = {ob.id: _item_points(ob) for ob in scene.lines + scene.splines + scene.sources}
    for act in scene.acts:
        ids = [i for i in act.paired if i in cur]
        if not ids:
            continue
        v = vals.get(act.id, default_value(scene, act))
        if act.kind == "linear":
            a, b = np.array(act.a, float), np.array(act.b, float)
            d = (b - a) / linear_length(act)
            disp = (v - default_value(scene, act)) * d
            for i in ids:
                cur[i] = [p + disp for p in cur[i]]
        else:
            th = math.radians(v)
            c, s = math.cos(th), math.sin(th)
            R = np.array([[c, -s], [s, c]])
            piv = np.array(act.a, float)
            for i in ids:
                cur[i] = [piv + R @ (p - piv) for p in cur[i]]
    return cur


def moved_lines(scene, vals):
    """Every straight piece at the current actuator positions: list of (p, q, thickness).
    Splines are sampled into short straight pieces (the solver works with straight walls; a
    curved surface becomes many small turns, i.e. a chain of weak waves, as in reality)."""
    cur = moved_points(scene, vals)
    out = [(cur[ln.id][0], cur[ln.id][1], ln.t) for ln in scene.lines]
    for sp in scene.splines:
        C = spline_curve(cur[sp.id], sp.closed)
        out += [(C[k], C[k + 1], sp.t) for k in range(len(C) - 1)]
    return out


def scene_sources(scene, vals=None):
    """Source lines at the current actuator positions:
    [{"id", "a", "b", "n" (unit outflow normal), "L", "M", "npr", "t0r"}]."""
    cur = moved_points(scene, vals or {})
    out = []
    for sc in scene.sources:
        a, b = cur[sc.id]
        e = b - a
        L = float(np.hypot(*e))
        if L < 1e-9:
            continue
        n = np.array([e[1], -e[0]]) / L                       # right-hand side of a -> b
        if sc.flip:
            n = -n
        out.append(dict(id=sc.id, a=a, b=b, n=n, L=L, M=sc.M, npr=sc.npr, t0r=sc.t0r))
    return out


# =============================================================================
# Scene -> wall segments for the solver
# =============================================================================
def _k(p):
    return (round(float(p[0]), 6), round(float(p[1]), 6))


def _thick_outlines(items):
    """Thick lines -> closed rectangular bodies (4 wall segments each). Where exactly two
    equally thick lines meet at a point the corner is mitred, so a bent plate is one clean
    body; otherwise the ends are flat (butt)."""
    n = len(items)
    d, nn, h = [], [], []
    for p, q, t in items:
        v = q - p
        dd = v / float(np.hypot(*v))
        d.append(dd)
        nn.append(np.array([-dd[1], dd[0]]))
        h.append(t / 2.0)
    corner = {}
    for i, (p, q, t) in enumerate(items):
        corner[(i, 0)] = {1: p + h[i] * nn[i], -1: p - h[i] * nn[i]}
        corner[(i, 1)] = {1: q + h[i] * nn[i], -1: q - h[i] * nn[i]}
    ends = {}
    for i, (p, q, t) in enumerate(items):
        ends.setdefault(_k(p), []).append((i, 0))
        ends.setdefault(_k(q), []).append((i, 1))
    for lst in ends.values():
        if len(lst) != 2:
            continue
        (i, ei), (j, ej) = lst
        if i == j or abs(h[i] - h[j]) > 1e-9:
            continue
        V = items[i][ei]
        ai = d[i] if ei == 0 else -d[i]               # arm directions pointing away from V
        aj = d[j] if ej == 0 else -d[j]
        cr = ai[0] * aj[1] - ai[1] * aj[0]
        if abs(cr) < 1e-6:
            if float(ai @ aj) < 0:                    # straight continuation: share corners
                for s in (1, -1):
                    s2 = 1 if float(nn[j] @ nn[i]) * s > 0 else -1
                    corner[(j, ej)][s2] = corner[(i, ei)][s]
            continue
        new_i, new_j, ok = {}, {}, True
        for s in (1, -1):
            inner_i = s * float(nn[i] @ aj) > 0
            s2 = 1
            for cand in (1, -1):
                if (cand * float(nn[j] @ ai) > 0) == inner_i:
                    s2 = cand
                    break
            P1 = V + s * h[i] * nn[i]
            P2 = V + s2 * h[j] * nn[j]
            u = ((P2[0] - P1[0]) * aj[1] - (P2[1] - P1[1]) * aj[0]) / cr
            X = P1 + u * ai
            if float(np.hypot(*(X - V))) > 4.0 * h[i] + 1e-9:    # very sharp bend: no mitre
                ok = False
                break
            new_i[s] = X
            new_j[s2] = X
        if ok:
            corner[(i, ei)].update(new_i)
            corner[(j, ej)].update(new_j)
    walls = []
    for i in range(n):
        A, B = corner[(i, 0)][1], corner[(i, 1)][1]
        C, D = corner[(i, 1)][-1], corner[(i, 0)][-1]
        for u, v in ((A, B), (B, C), (C, D), (D, A)):
            if float(np.hypot(*(v - u))) > 1e-9:
                walls.append((u.copy(), v.copy()))
    return walls


def scene_walls(scene, vals=None):
    """Plain wall segments [(np.array p, np.array q), ...] for the solver, at the actuator
    positions in `vals` ({actuator id: slider value}). Thin lines stay single walls; lines with
    thickness become closed rectangular bodies."""
    vals = vals or {}
    walls, thick = [], []
    for p, q, t in moved_lines(scene, vals):
        if float(np.hypot(*(q - p))) < 1e-9:
            continue
        if t > 1e-9:
            thick.append((p, q, t))
        else:
            walls.append((p.copy(), q.copy()))
    return walls + _thick_outlines(thick)
