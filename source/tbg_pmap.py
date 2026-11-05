"""
TBG - pressure map
==================
Turns the solved flow into a pressure image (p / p_freestream), one value per pixel.

How: a dense bundle of streamlines is marched through the solved wave pattern with
Flow.streamlines(..., with_p=True) (pressure is updated at every shock and fan crossing).
Each streamline segment is painted onto a pixel grid, small gaps between neighbouring
streamlines (e.g. behind expansions) are filled from their neighbours, and walls act as
barriers so pressure never leaks through a surface.

Not coloured (hatched): regions no streamline reaches (wakes / separated flow, shadowed
pockets) and subsonic regions, where the solver stops tracking waves. Beyond the dotted tail
of a shock (where an expansion fan has cut it off) the solver has no shock, so the map shows
the undisturbed pressure there.

Used by tbg_sim.py (View > Map). Needs only numpy and matplotlib.

Maps (View > Map):  Pressure  p/p_inf     Mach       local Mach number (sonic line dashed)
                    Density   rho/rho_inf Schlieren  numerical schlieren exp(-k |grad rho|)
                    Vorticity dv/dx - du/dy (eddies; needs viscosity in Setup to be physical)
Only the pressure map has the instant streamline preview; the others come from the solver.

Two back-ends:
  compute()    the streamline-marching map described above. Instant, used as the quick preview
               while a slider is being dragged.
  CfdJob       the real thing: a 2-D Euler solver (tbg_cfd.py) run in a background thread once
               things settle. It solves the WHOLE field, including detached bow shocks, subsonic
               regions, ducts / inlets, reflections and wakes, so nothing is left hatched, and
               the map is smooth. The picture refreshes while it converges.
"""

import math
import threading
import time

import numpy as np
import matplotlib
from matplotlib.colors import TwoSlopeNorm

PM_MIN_NX = 1024            # fine map: at least this many pixels across ...
PM_MAX_NX = 2048            # ... at most this many, otherwise follows the canvas width
PREVIEW_NX = 320            # quick preview while a slider is being dragged
OVERSAMPLE = 1.6            # streamlines per pixel row of the grid
REFINE_GAP = 1.2            # streamlines further apart than this (pixels) at the exit get extra neighbours
REFINE_MAX = 14             # at most this many sub-divisions of one gap
FILL_PASSES = 10            # how far (pixels) gaps may be filled
CMAP = "coolwarm"          # diverging, neutral at the freestream value

# kind: (menu label, colour-bar title)
MAPS = {"pressure": ("Pressure Map", "p / p\u221e"),
        "mach": ("Mach Map", "Mach  (dashed: M = 1)"),
        "density": ("Density Map", "\u03c1 / \u03c1\u221e"),
        "schlieren": ("Schlieren", None),
        "vorticity": ("Vorticity Map", "\u03c9  (a\u221e / unit length)")}
SCHLIEREN_K = 12.0


def resolution(ax, fine, extent):
    """(nx, ny) for the grid. Fine = follows the canvas (>= 1024 across); preview = coarse."""
    xmin, xmax, ymin, ymax = extent
    if fine:
        try:
            ax.apply_aspect()
            w = float(ax.get_window_extent().width)
        except Exception:
            w = PM_MIN_NX
        nx = int(min(PM_MAX_NX, max(PM_MIN_NX, round(w))))
    else:
        nx = PREVIEW_NX
    ny = max(2, int(round(nx * (ymax - ymin) / (xmax - xmin))))
    return nx, ny


def _paint(grid, A, B, V, extent, nx, ny, order="last"):
    """Draw segments A->B (world coords) with values V into grid[iy, ix] (row 0 = bottom).

    order = "last" (default): the last sample drawn on a pixel wins.
    order = "strong" / "weak": where several samples land on one pixel, the one furthest from /
    closest to the freestream value (|log p|) wins."""
    if len(A) == 0:
        return
    xmin, xmax, ymin, ymax = extent
    sx, sy = nx / (xmax - xmin), ny / (ymax - ymin)
    d = B - A
    Lpx = np.hypot(d[:, 0] * sx, d[:, 1] * sy)
    cnt = np.clip(np.ceil(Lpx * 2.0).astype(np.int64) + 2, 2, 5000)
    idx = np.repeat(np.arange(len(A)), cnt)
    start = np.cumsum(cnt) - cnt
    within = np.arange(cnt.sum()) - np.repeat(start, cnt)
    t = within / (cnt[idx] - 1)
    P = A[idx] + t[:, None] * d[idx]
    ix = np.floor((P[:, 0] - xmin) * sx).astype(np.int64)
    iy = np.floor((P[:, 1] - ymin) * sy).astype(np.int64)
    ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    ix, iy, val = ix[ok], iy[ok], V[idx[ok]]
    if order != "last":
        key = np.abs(np.log(np.maximum(val, 1e-9)))
        o = np.argsort(key if order == "strong" else -key, kind="stable")
        ix, iy, val = ix[o], iy[o], val[o]                  # the winner is written last
    grid[iy, ix] = val


def _fill(grid, barrier, passes):
    """Fill empty pixels with the mean of their filled neighbours; never into barrier pixels."""
    g = grid.copy()
    for _ in range(passes):
        empty = np.isnan(g) & ~barrier
        if not empty.any():
            break
        pad = np.pad(g, 1, constant_values=np.nan)
        s = np.zeros_like(g)
        c = np.zeros_like(g)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nb = pad[1 + dy:1 + dy + g.shape[0], 1 + dx:1 + dx + g.shape[1]]
                v = ~np.isnan(nb)
                s[v] += nb[v]
                c[v] += 1
        fill = empty & (c > 0)
        if not fill.any():
            break
        g[fill] = s[fill] / c[fill]
    return g


def _shift(a, dy, dx, fill):
    """a shifted by (dy, dx) with `fill` coming in at the edges (out[y, x] = a[y+dy, x+dx])."""
    out = np.full_like(a, fill)
    h, w = a.shape
    ys, yd = (slice(dy, h), slice(0, h - dy)) if dy >= 0 else (slice(0, h + dy), slice(-dy, h))
    xs, xd = (slice(dx, w), slice(0, w - dx)) if dx >= 0 else (slice(0, w + dx), slice(-dx, w))
    out[yd, xd] = a[ys, xs]
    return out


def _soften(grid, thr=0.25, passes=2):
    """Anti-alias the hard edges of the map. A pixel whose 3x3 neighbourhood spans more than
    `thr` in log(p) (a region boundary: stair-stepped by the pixel grid) is replaced by the
    weighted mean of its finite neighbours, in log space. Smooth areas are left untouched, and
    unsolved (NaN) pixels are neither filled nor used."""
    fin = np.isfinite(grid)
    if not fin.any():
        return grid
    lg = np.where(fin, np.log(np.maximum(grid, 1e-9)), np.nan)
    for _ in range(passes):
        s = np.zeros_like(lg)
        c = np.zeros_like(lg)
        mx = np.full_like(lg, -np.inf)
        mn = np.full_like(lg, np.inf)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                v = _shift(lg, dy, dx, np.nan)
                f = np.isfinite(v)
                w = 2.0 if (dy == 0 and dx == 0) else 1.0
                s[f] += w * v[f]
                c[f] += w
                mx = np.where(f, np.maximum(mx, v), mx)
                mn = np.where(f, np.minimum(mn, v), mn)
        edge = fin & ((mx - mn) > thr) & (c > 0)
        lg = np.where(edge, s / np.maximum(c, 1e-9), lg)
    return np.where(fin, np.exp(lg), np.nan)


def _resolve_overlaps(gs, gw, win=3, dens=0.30, grow=2, reach=2):
    """Where the solver lets streamlines cross (flow turned up by an expansion fan meeting the
    flow that passes over the top of a body), both sets of streamlines claim the same pixels.
    Drawing order would then cut the stronger one off along an arbitrary streamline, and since
    streamlines spread out inside an expansion, 'strongest wins' alone leaves a speckle.

    gs / gw are the grids with the strongest / weakest sample winning each pixel. Pixels where
    they differ are conflicts. A conflict that is a thin line is just a shock or fan edge and is
    left alone; a conflict covering an AREA is an overlap, and there every pixel takes the
    strongest value within `reach` pixels, which closes the gaps between the stronger lines."""
    both = np.isfinite(gs) & np.isfinite(gw)
    with np.errstate(invalid="ignore", divide="ignore"):
        diff = np.abs(np.log(np.maximum(gs, 1e-9)) - np.log(np.maximum(gw, 1e-9)))
    conf = both & (diff > 0.25)      # fan marching steps differ by ~0.05-0.1, real overlaps by >1
    if not conf.any():
        return gs
    c = conf.astype(np.float32)
    acc = np.zeros_like(c)
    n = 0
    for dy in range(-win, win + 1):
        for dx in range(-win, win + 1):
            acc += _shift(c, dy, dx, 0.0)
            n += 1
    zone = (acc / n) > dens
    if not zone.any():
        return gs
    z0 = zone.copy()
    for dy in range(-grow, grow + 1):
        for dx in range(-grow, grow + 1):
            zone |= _shift(z0, dy, dx, False)
    out = gs.copy()
    best = np.where(np.isfinite(gs), np.abs(np.log(np.maximum(gs, 1e-9))), -1.0)
    for dy in range(-reach, reach + 1):
        for dx in range(-reach, reach + 1):
            v = _shift(gs, dy, dx, np.nan)
            with np.errstate(invalid="ignore", divide="ignore"):
                k = np.where(np.isfinite(v), np.abs(np.log(np.maximum(v, 1e-9))), -1.0)
            better = zone & (k > best)
            out[better] = v[better]
            best = np.where(better, k, best)
    return out


def _refine(flow, ys, lines, extent, ny):
    """Streamlines spread out inside an expansion (the lower the pressure, the wider), so a
    uniform set of inflow seeds leaves holes there. Wherever two neighbouring streamlines end
    far apart on the left edge, seed more lines between them at the inflow and trace those too.
    Returns (extra_ys, extra_lines)."""
    xmin, _xmax, ymin, ymax = extent
    sy = ny / (ymax - ymin)
    tol = 1e-6 + 0.5 * (ymax - ymin) / ny
    ends = []
    for pts, _s, _p in lines:
        e = pts[-1]
        ends.append(float(e[1]) if len(pts) > 1 and e[0] <= xmin + tol else None)
    new = []
    for i in range(len(ys) - 1):
        a, b = ends[i], ends[i + 1]
        if a is None or b is None:
            continue
        k = int(min(REFINE_MAX, math.ceil(abs(b - a) * sy / REFINE_GAP)))
        if k > 1:
            new.extend(np.linspace(ys[i], ys[i + 1], k + 1)[1:-1])
    if not new:
        return [], []
    new = np.array(new)
    return new, flow.streamlines(new, with_p=True)


def compute(flow, extent, nx, ny, plane=True):
    """Pressure image of the solved Flow. Returns a dict, or None if there is nothing to show."""
    if flow.M <= 1.0 + 1e-9:
        return None
    xmin, xmax, ymin, ymax = extent
    dy = (ymax - ymin) / ny
    n = int(max(64, ny * OVERSAMPLE))
    y_lo = (0.0 if plane else ymin) + 0.25 * dy
    ys = np.linspace(y_lo, ymax - 0.25 * dy, n)
    lines = flow.streamlines(ys, with_p=True)
    _ey, extra = _refine(flow, ys, lines, extent, ny)
    lines = list(lines) + list(extra)

    A, B, V = [], [], []
    for pts, _sub, sp in lines:
        if len(pts) < 2 or len(sp) != len(pts) - 1:
            continue
        A.append(pts[:-1])
        B.append(pts[1:])
        V.append(sp)
    grid = np.full((ny, nx), np.nan)
    if A:
        A, B, V = np.vstack(A), np.vstack(B), np.concatenate(V)
        keep = np.isfinite(V)
        _paint(grid, A[keep], B[keep], V[keep], extent, nx, ny, order="strong")
        weak = np.full((ny, nx), np.nan)
        _paint(weak, A[keep], B[keep], V[keep], extent, nx, ny, order="weak")
        grid = _resolve_overlaps(grid, weak)

    # walls (and the ground) as 3-pixel-wide barriers for the gap filling
    bar = np.zeros((ny, nx))
    wa = [np.array(a, float) for a, _b in flow.walls]
    wb = [np.array(b, float) for _a, b in flow.walls]
    if wa:
        _paint(bar, np.array(wa), np.array(wb), np.ones(len(wa)), extent, nx, ny)
    b0 = np.pad(bar > 0, 1)
    barrier = np.zeros((ny, nx), bool)
    for oy in range(3):
        for ox in range(3):
            barrier |= b0[oy:oy + ny, ox:ox + nx]
    grid[barrier] = np.nan
    grid = _fill(grid, barrier, FILL_PASSES)
    grid = _soften(grid)
    grid[barrier] = np.nan

    good = grid[np.isfinite(grid)]
    if good.size == 0:
        return None
    return dict(grid=grid, extent=extent, nx=nx, ny=ny,
                pmin=float(good.min()), pmax=float(good.max()),
                coverage=float(good.size) / grid.size)


def _ticks(vmin, vmax):
    """Colour-bar ticks: the extremes, the freestream value 1.0, and a midpoint on each side."""
    t = {vmin, vmax}
    if vmin < 0.999:
        t.add(math.sqrt(vmin))
    if vmax > 1.001:
        t.add(math.sqrt(vmax))
    if vmin < 1.0 < vmax:
        t.add(1.0)
    return sorted(t)


def draw(ax, pm, zorder=1.6):
    """Draw a map dict (see compute / to_pm) into ax, styled for its kind."""
    kind = pm.get("kind", "pressure") if pm is not None else "pressure"
    if kind in ("pressure", "density"):
        return _draw_log(ax, pm, zorder)
    return _draw_lin(ax, pm, kind, zorder)


def _colorbar(ax, im, title, ticks=None, labels=None):
    try:
        cax = ax.inset_axes([0.02, 0.90, 0.24, 0.022])
        cb = ax.figure.colorbar(im, cax=cax, orientation="horizontal")
        if ticks is not None:
            cb.set_ticks(ticks)
            cb.set_ticklabels(labels)
        cb.ax.tick_params(labelsize=7, length=2, pad=1)
        cb.ax.set_title(title, fontsize=8, pad=2)
        cb.minorticks_off()
    except Exception:
        pass


def _draw_lin(ax, pm, kind, zorder):
    """Mach, schlieren and vorticity maps (linear colour scales)."""
    if pm is None:
        return None
    grid = pm["grid"]
    if not np.isfinite(grid).any():
        return None
    if kind == "schlieren":
        ref = max(float(np.nanpercentile(grid, 99.5)), 1e-9)
        img = np.exp(-SCHLIEREN_K * np.nan_to_num(grid, nan=0.0) / ref)
        img = np.ma.masked_where(~np.isfinite(grid), img)
        cm = matplotlib.colormaps["gray"].copy()
        cm.set_bad((0, 0, 0, 0))
        return ax.imshow(img, extent=pm["extent"], origin="lower", cmap=cm, vmin=0.0, vmax=1.0,
                         interpolation="bilinear", aspect="auto", zorder=zorder)
    g = np.ma.masked_invalid(grid)
    if kind == "vorticity":
        lim = max(float(np.nanpercentile(np.abs(grid), 99.0)), 1e-6)
        cm = matplotlib.colormaps["RdBu_r"].copy()
        vmin, vmax = -lim, lim
    else:                                                       # mach
        cm = matplotlib.colormaps["turbo"].copy()            # no black: walls stay visible
        vmin, vmax = 0.0, max(float(np.nanpercentile(grid, 99.9)), 1.05)
    cm.set_bad((0, 0, 0, 0))
    im = ax.imshow(g, extent=pm["extent"], origin="lower", cmap=cm, vmin=vmin, vmax=vmax,
                   interpolation="bilinear", aspect="auto", zorder=zorder)
    if kind == "mach" and vmax > 1.0 and np.nanmin(grid) < 1.0:
        try:                                                    # sonic line
            ny, nx = grid.shape
            x0, x1, y0, y1 = pm["extent"]
            X = np.linspace(x0, x1, nx)
            Y = np.linspace(y0, y1, ny)
            ax.contour(X, Y, g, levels=[1.0], colors="black", linewidths=0.8,
                       linestyles="--", zorder=zorder + 0.1)
        except Exception:
            pass
    _colorbar(ax, im, MAPS[kind][1])
    return im


def _draw_log(ax, pm, zorder=1.6):
    """Draw the pressure image into ax.

    Colours: a soft diverging map centred on the freestream value (p/p_inf = 1 is a neutral
    grey, compression warm, expansion cool), on a log scale so a 0.13x drop and a 3x rise
    get comparable weight. Fluid pixels that were NOT solved (wakes, subsonic regions,
    pockets no streamline reaches) are hatched instead of left as plain background.
    """
    if pm is None:
        return None
    grid = pm["grid"]
    g = np.ma.masked_invalid(grid)
    if g.count() == 0:
        return None
    ny, nx = grid.shape
    vmin, vmax = pm["pmin"], pm["pmax"]
    lg = np.ma.log10(np.ma.masked_less_equal(g, 1e-9))
    lo = min(math.log10(max(vmin, 1e-9)), -0.02)
    hi = max(math.log10(max(vmax, 1e-9)), 0.02)
    norm = TwoSlopeNorm(vcenter=0.0, vmin=lo, vmax=hi)
    cm = matplotlib.colormaps[CMAP].copy()
    cm.set_bad((0, 0, 0, 0))
    im = ax.imshow(lg, extent=pm["extent"], origin="lower", cmap=cm, norm=norm,
                   interpolation="nearest", aspect="auto", zorder=zorder)

    # hatch over the unsolved pixels (the solid bodies are painted over it by the caller)
    nanmask = np.isnan(grid)
    if pm.get("hatch", True) and nanmask.any():
        iy, ix = np.indices(grid.shape)
        stripe = nanmask & (((ix + iy) % 12) < 2)
        rgba = np.zeros((ny, nx, 4), dtype=np.uint8)
        rgba[stripe] = (70, 70, 70, 120)
        ax.imshow(rgba, extent=pm["extent"], origin="lower", interpolation="nearest",
                  aspect="auto", zorder=zorder + 0.05)
    ticks = _ticks(vmin, vmax)
    _colorbar(ax, im, MAPS[pm.get("kind", "pressure")][1],
              [math.log10(max(t, 1e-9)) for t in ticks], [f"{t:.2f}" for t in ticks])
    return im


# =============================================================================
# CFD back-end
# =============================================================================
CFD_NX = 300                 # cells across the view for the final grid (coarser grids come first)
CFD_TIME = 30.0              # seconds the solver may use at most
CFD_ITERS = 400              # iterations on the final grid


def _upsample(a, w, nxd, nyd, extent, gx0, gdx, gy0, gdy):
    """Bilinear resampling of a (ny, nx) field with weights w (0 = no data) to nxd x nyd pixels."""
    xmin, xmax, ymin, ymax = extent
    ny, nx = a.shape
    xs = xmin + (np.arange(nxd) + 0.5) * (xmax - xmin) / nxd
    ys = ymin + (np.arange(nyd) + 0.5) * (ymax - ymin) / nyd
    fx = np.clip((xs - gx0) / gdx - 0.5, 0.0, nx - 1.000001)
    fy = np.clip((ys - gy0) / gdy - 0.5, 0.0, ny - 1.000001)
    i0, j0 = np.floor(fx).astype(int), np.floor(fy).astype(int)
    tx, ty = fx - i0, fy - j0

    def interp(f):
        r = f[:, i0] * (1 - tx) + f[:, i0 + 1] * tx            # (ny, nxd)
        return r[j0, :] * (1 - ty)[:, None] + r[j0 + 1, :] * ty[:, None]

    num = interp(a * w)
    den = interp(w)
    return num, den


def to_pm(p, geo, extent, nxd, nyd, kind="pressure"):
    """CFD field grid (NaN in solids) -> map dict (nyd x nxd pixels). Pressure and density are
    interpolated in log space, the other fields linearly."""
    fluid = np.isfinite(p)
    if not fluid.any():
        return None
    good = p[fluid]
    logk = kind in ("pressure", "density")
    lg = np.where(fluid, np.log(np.maximum(p, 1e-6)) if logk else p, np.nan)
    lg = _fill(lg, np.zeros(lg.shape, bool), 3)          # run a little way into the solids, so
    w = np.isfinite(lg).astype(float)                    # the picture reaches right up to the walls
    num, den = _upsample(np.where(np.isfinite(lg), lg, 0.0), w, nxd, nyd, extent,
                         geo.xc[0], geo.dx, geo.yc[0], geo.dy)
    ok = den > 0.5
    grid = np.full((nyd, nxd), np.nan)
    grid[ok] = np.exp(num[ok] / den[ok]) if logk else num[ok] / den[ok]
    lo, hi = np.percentile(good, [0.02, 99.98])
    if logk:
        lo, hi = min(lo, 1.0), max(hi, 1.0)
    return dict(grid=grid, extent=extent, nx=nxd, ny=nyd, pmin=float(lo), pmax=float(hi),
                coverage=float(ok.mean()), hatch=False, kind=kind)


class CfdJob(threading.Thread):
    """Runs the Euler solver in the background. The GUI polls .snapshot() from its own thread.

        job = CfdJob(flow, extent, nx, ny, plane);  job.start()
        pm, done, status = job.snapshot()      # pm: map dict or None, status: short text
        job.cancel()                           # when the geometry changes
    """

    def __init__(self, flow, extent, nx, ny, plane=True, kind="pressure", main=True,
                 sources=(), visc=None, cells=None):
        """sources: scene_sources() dicts; visc: Reynolds number per unit length or None;
        cells: solver cells across the view (default CFD_NX)."""
        super().__init__(daemon=True)
        import tbg_cfd
        g = float(flow.g)
        self.kind, self.main, self.visc = kind, bool(main), visc
        self.sources = [dict(a=np.array(sc["a"], float), b=np.array(sc["b"], float),
                             n=np.array(sc["n"], float),
                             W=tbg_cfd.source_state(sc["M"], sc["npr"], sc["t0r"], sc["n"], g))
                        for sc in sources]
        self.walls = [(np.array(a, float), np.array(b, float)) for a, b in flow.walls]
        self.polys = [[tuple(p) for p in poly] for poly in flow.polys]
        self.ground = bool(flow.ground)
        self.M, self.g = float(flow.M), float(flow.g)
        self.extent, self.nxd, self.nyd = extent, nx, ny
        gx = int(cells or CFD_NX)
        gy = max(24, int(round(gx * (extent[3] - extent[2]) / (extent[1] - extent[0]))))
        self.gnx, self.gny = gx, gy
        self._cancel = False
        self._pm = None
        self._status = "starting"
        self._done = False
        self._gen = 0                                    # bumps whenever a new picture is ready
        self.info = {}
        self.sol = self.geo = None                       # the finished solution

    def cancel(self):
        self._cancel = True

    def snapshot(self):
        return self._pm, self._done, self._status, self._gen

    def _publish(self, info, sol, geo, final=False):
        try:
            pm = to_pm(sol.field(self.kind), geo, self.extent, self.nxd, self.nyd, self.kind)
        except Exception:
            pm = None
        if pm is None or self._cancel:
            return
        pm["fine"] = True
        pm["cfd"] = True
        pm["cfd_grid"] = (info.get("nx"), info.get("ny"))
        pm["cfd_it"] = info.get("it")
        pm["cfd_final"] = final
        self.info = info
        self._pm = pm
        self._gen += 1
        if not final:
            self._status = f"solving flow field...  grid {info['level']}/{info['levels']}  " \
                           f"{info['it']} it  {info['time']:.0f}s"

    def run(self):
        try:
            import tbg_cfd
            src = bool(self.sources)
            sol, info, geo = tbg_cfd.solve(
                self.walls, self.polys, self.ground, self.M, self.g, self.extent,
                self.gnx, self.gny, max_iter=CFD_ITERS * (5 if src else 1) * (2 if self.visc else 1),
                time_limit=CFD_TIME * (2.0 if src else 1.0) * (1.5 if self.visc else 1.0),
                cancel=lambda: self._cancel, progress=self._publish,
                main=self.main, sources=self.sources, visc=self.visc)
            if sol is not None and not self._cancel:
                self.sol, self.geo = sol, geo                    # kept for other map kinds
                if info.get("diverged"):
                    self._status = "flow solver diverged - try a lower source pressure or Mach"
                else:
                    self._publish(info, sol, geo, final=True)
                    self._status = (f"flow field solved  {info['nx']}x{info['ny']} cells, "
                                    f"{info['time']:.0f}s" + ("" if info.get("converged") else
                                    " (time limit, still settling)"))
        except Exception as e:                           # never let the map break the app
            self._status = f"flow solver failed: {e}"
        self._done = True
