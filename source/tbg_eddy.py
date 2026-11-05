"""
TBG - eddies / wake overlay  (View > Eddies in the simulation window)
=====================================================================
Eddies are a VISCOUS effect: they come from boundary-layer separation and the roll-up of
free shear layers, not from the flow being supersonic. The wave-tracing solver (and the Euler
pressure map) are inviscid, so they cannot produce them. This module draws a SCHEMATIC wake
on top of the solved picture, the same way the bow-shock standoff is an empirical estimate:

  * base wakes      leeward faces of closed bodies that are steeper than SEP_DEG separate.
                    Shear layers leave the separation corners and converge at the wake neck,
                    enclosing a recirculation bubble with a pair of counter-rotating eddies.
                    A recompression shock stands at the neck (M > 1), and a decaying vortex
                    street follows downstream.
  * ground steps    a base sitting on the ground (plane on) is a backward-facing step:
                    one shear layer reattaches on the ground, with a main eddy and a small
                    counter-rotating corner eddy, then a row of boundary-layer eddies.
  * trailing edges  free downstream ends of thin lines shed a thin vortex street.

Recirculation length:  free wake  L ~ h (1 + 1/M^2),   ground step  L ~ h (2.5 + 3.5/M^2)
(h = base height). Nothing here feeds back into the solver: streamlines, shocks and the
pressure map are unchanged.

Only drawn with the main flow on. For eddies that are actually computed, switch Viscosity on in
Edit > Setup and use View > Map > Vorticity Map.

Used by tbg_sim.py. Needs only numpy and matplotlib.
"""

import math
import time

import numpy as np
from matplotlib.patches import Polygon

C_BUBBLE = "#6b6b6b"      # recirculation region fill
C_EDDY = "#2b2b2b"        # eddy spirals
C_SHEAR = "#3d3d3d"       # free shear layers
C_RECOMP = "#242424"      # recompression shocks

SEP_DEG = 25.0            # leeward faces steeper than this (deg from the flow axis) separate
SEP_SLOPE = math.sin(math.radians(SEP_DEG))
THIN_H = 0.18             # effective wake height behind a thin trailing edge
MAX_EDDIES = 14           # per wake
CONVECT = 0.9             # eddy drift speed (world units / s) when animated (Randomise on)


# ---- small geometry helpers -------------------------------------------------------
def _area(P):
    return 0.5 * sum(P[i][0] * P[(i + 1) % len(P)][1] - P[(i + 1) % len(P)][0] * P[i][1]
                     for i in range(len(P)))


def _bezier(p0, p1, p2, n=40):
    t = np.linspace(0.0, 1.0, n)[:, None]
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2


def _wiggle(P, amp, waves=5, phase=0.0):
    """Kelvin-Helmholtz-like ripple on a shear layer, growing along it."""
    d = np.gradient(P, axis=0)
    L = np.hypot(d[:, 0], d[:, 1])[:, None]
    L[L < 1e-12] = 1.0
    nrm = np.stack([-d[:, 1], d[:, 0]], axis=1) / L
    s = np.linspace(0.0, 1.0, len(P))[:, None]
    return P + nrm * amp * s * np.sin(2 * math.pi * waves * s + phase)


def _clip(flow, P):
    """Cut a polyline at the first wall it runs into."""
    out = [P[0]]
    for k in range(1, len(P)):
        a, b = P[k - 1], P[k]
        dd = b - a
        L = float(np.hypot(*dd))
        if L < 1e-12:
            continue
        hit = flow.first_hit(a, dd / L, 1e-6)
        if hit and hit[0] <= L:
            out.append(a + dd / L * hit[0])
            break
        out.append(b)
    return np.array(out)


def _ray(flow, o, d, ext):
    """End point of a ray from o along d: first wall, or just outside the view."""
    xmin, xmax, ymin, ymax = ext
    ts = []
    if abs(d[0]) > 1e-12:
        ts.append(((xmax + 1.0 if d[0] > 0 else xmin - 1.0) - o[0]) / d[0])
    if abs(d[1]) > 1e-12:
        ts.append(((ymax + 1.0 if d[1] > 0 else ymin - 1.0) - o[1]) / d[1])
    t = min([x for x in ts if x > 0], default=0.0)
    h = flow.first_hit(o, d)
    if h and h[0] < t:
        t = h[0]
    return o + d * t


# ---- where the flow separates -------------------------------------------------------
def _runs(flow, poly):
    """Contiguous leeward (separating) edge runs of one closed body, top vertex first."""
    P = [np.array(p, float) for p in poly]
    if _area(P) < 0:
        P = P[::-1]                                       # CCW: outward normal = (dy, -dx)
    n = len(P)
    lee = []
    for i in range(n):
        a, b = P[i], P[(i + 1) % n]
        e = b - a
        L = float(np.hypot(*e))
        if L < 1e-9:
            lee.append(False)
            continue
        nrm = np.array([e[1], -e[0]]) / L
        lee.append(nrm[0] < -SEP_SLOPE and not flow._solid((a + b) / 2 + nrm * 1e-3))
    if not any(lee) or all(lee):
        return []
    runs = []
    for i in range(n):
        if lee[i] and not lee[i - 1]:
            pts, k = [P[i]], i
            while lee[k % n]:
                pts.append(P[(k + 1) % n])
                k += 1
            if pts[0][1] < pts[-1][1]:
                pts = pts[::-1]
            runs.append(pts)
    return runs


def _bases(flow, plane, ext):
    xmin, xmax, ymin, ymax = ext
    out = []
    for poly in flow.polys:
        for run in _runs(flow, poly):
            top = max(run, key=lambda p: p[1])
            bot = min(run, key=lambda p: p[1])
            h = float(top[1] - bot[1])
            x0 = min(float(p[0]) for p in run)
            if h < 0.05 or x0 < xmin or x0 > xmax:
                continue
            out.append(dict(top=top, bot=bot, h=h, x0=x0, run=run, thin=False,
                            ground=bool(plane and bot[1] < 1e-3)))
    for i, (a, b) in enumerate(flow.walls):                # free trailing edges of thin lines
        for end, other, d in ((a, b, flow.deg[i][0]), (b, a, flow.deg[i][1])):
            if d != 1 or end[0] >= other[0] - 1e-9:
                continue
            if not (xmin <= end[0] <= xmax and ymin <= end[1] <= ymax):
                continue
            off = np.array([0.0, THIN_H / 2])
            out.append(dict(top=end + off, bot=end - off, h=THIN_H, x0=float(end[0]),
                            run=None, thin=True, ground=False))
    return out


def _recirc_len(M, h, ground):
    m2 = max(M, 1.0) ** 2
    return h * (2.5 + 3.5 / m2) if ground else h * (1.0 + 1.0 / m2)


# ---- drawing --------------------------------------------------------------------------
def _spiral(ax, c, r, ccw, alpha, z=4.2):
    """Eddy marker: an outward spiral with an arrowhead showing the sense of rotation."""
    th = np.linspace(0.0, 2.4 * math.pi, 60)
    rr = r * (0.12 + 0.88 * th / th[-1])
    s = 1.0 if ccw else -1.0
    x, y = c[0] + rr * np.cos(s * th), c[1] + rr * np.sin(s * th)
    ax.plot(x, y, color=C_EDDY, lw=1.1, alpha=alpha, zorder=z, solid_capstyle="round")
    ax.annotate("", xy=(x[-1], y[-1]), xytext=(x[-4], y[-4]), zorder=z,
                arrowprops=dict(arrowstyle="-|>", color=C_EDDY, lw=1.1, alpha=alpha,
                                mutation_scale=8))


def _street(ax, flow, start, h, b0, rows, ext, t):
    """Decaying vortex street from `start` downstream (-x). rows=2: Karman street
    (upper CCW, lower CW); rows=1: single row of CCW eddies (on the ground)."""
    xmin = ext[0]
    step = 1.78 * b0 if rows == 2 else 1.6 * b0
    shift = (t * CONVECT) % (2 * step)
    n = 0
    for k in range(-2, 60):
        x = start[0] - step * (k + 0.5) - shift
        if x > start[0] - 0.3 * step:
            continue
        if x < xmin - 1.0 or n >= MAX_EDDIES:
            break
        s = start[0] - x
        grow = min(2.2, 1.0 + 0.10 * s / h)
        b = b0 * grow
        if rows == 2:
            upper = (k % 2 == 0)
            c = np.array([x, start[1] + (0.5 if upper else -0.5) * b])
        else:
            upper = True
            c = np.array([x, start[1] + 0.5 * b])
        if flow._solid(c):
            continue
        _spiral(ax, c, 0.38 * b, upper, max(0.15, 0.85 * math.exp(-s / (8.0 * h))))
        n += 1
    return n


def draw(ax, flow, extent, plane=True, animate=False):
    """Draw the schematic wakes into ax. Returns report lines for the data panel."""
    t = time.time() if animate else 0.0
    M = flow.M
    bases = _bases(flow, plane, extent)
    if not bases:
        return []
    lines = ["Wakes / eddies (schematic: viscous, not solved)"]
    n_te, n_te_eddies, wid = 0, 0, 0
    beta = min(1.25 * math.asin(1.0 / M), 1.2) if M > 1.0 + 1e-9 else None

    for bs in bases:
        h = bs["h"]
        if bs["thin"]:
            n_te += 1
            n_te_eddies += _street(ax, flow, np.array([bs["x0"], (bs["top"][1] + bs["bot"][1]) / 2]),
                                   h, 0.5 * h, 2, extent, t)
            continue
        wid += 1
        top, bot, x0, run = bs["top"], bs["bot"], bs["x0"], bs["run"]
        Lr = _recirc_len(M, h, bs["ground"])
        amp, ph = 0.05 * h, 6.0 * t

        if bs["ground"]:
            R = np.array([x0 - Lr, 0.0])
            sh = _clip(flow, _wiggle(_bezier(top, np.array([top[0] - 0.5 * (top[0] - R[0]), top[1]]), R),
                                     amp, phase=ph))
            ax.add_patch(Polygon(np.vstack([sh] + run[::-1]), closed=True, fc=C_BUBBLE,
                                 ec="none", alpha=0.22, zorder=2.6))
            ax.plot(sh[:, 0], sh[:, 1], color=C_SHEAR, lw=1.2, ls=(0, (5, 2)), zorder=4.1)
            _spiral(ax, np.array([x0 - 0.45 * Lr, 0.42 * h]), min(0.3 * h, 0.2 * Lr), True, 0.9)
            _spiral(ax, np.array([x0 - 0.15 * h, 0.15 * h]), 0.08 * h, False, 0.8)
            ne = _street(ax, flow, R, h, 0.45 * h, 1, extent, t)
            neck, cen = R, np.array([x0 - 0.45 * Lr, 0.42 * h + min(0.3 * h, 0.2 * Lr) + 0.15])
            rc = [(R, np.array([-math.cos(beta), math.sin(beta)]))] if beta else []
        else:
            yc = 0.5 * (top[1] + bot[1])
            N = np.array([x0 - Lr, yc])
            sh_t = _clip(flow, _wiggle(_bezier(top, np.array([top[0] - 0.55 * (top[0] - N[0]), top[1]]), N),
                                       amp, phase=ph))
            sh_b = _clip(flow, _wiggle(_bezier(bot, np.array([bot[0] - 0.55 * (bot[0] - N[0]), bot[1]]), N),
                                       amp, phase=ph + 1.0))
            ax.add_patch(Polygon(np.vstack([sh_t, sh_b[::-1]] + run[::-1]), closed=True,
                                 fc=C_BUBBLE, ec="none", alpha=0.22, zorder=2.6))
            for sh in (sh_t, sh_b):
                ax.plot(sh[:, 0], sh[:, 1], color=C_SHEAR, lw=1.2, ls=(0, (5, 2)), zorder=4.1)
            r = min(0.18 * h, 0.25 * Lr)
            _spiral(ax, np.array([x0 - 0.45 * Lr, yc + 0.22 * h]), r, True, 0.9)
            _spiral(ax, np.array([x0 - 0.45 * Lr, yc - 0.22 * h]), r, False, 0.9)
            ne = _street(ax, flow, N, h, 0.25 * h, 2, extent, t)
            neck, cen = N, np.array([x0 - 0.45 * Lr, yc])
            rc = [(N + np.array([0.0, s * 0.1 * h]), np.array([-math.cos(beta), s * math.sin(beta)]))
                  for s in (1, -1)] if beta else []

        for o, d in rc:                                    # recompression shock(s) at the neck
            e = _ray(flow, o, d, extent)
            ax.plot([o[0], e[0]], [o[1], e[1]], color=C_RECOMP, lw=1.8, ls=(0, (6, 3)),
                    alpha=0.75, zorder=5.5)
        ax.text(cen[0], cen[1], f"W{wid}", color=C_EDDY, fontsize=8, ha="center", va="center",
                zorder=7)
        kind = "step" if bs["ground"] else "base"
        lines.append(f"W{wid}  {kind} h={h:.2f}  recirc L~{Lr:.2f}  eddies {ne + 2}"
                     + ("  +RC shock" if rc else ""))
    if n_te:
        lines.append(f"{n_te} trailing-edge wake(s), {n_te_eddies} eddies")
    return lines
