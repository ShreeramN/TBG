import math
import time
import numpy as np

NG = 2                       # padding layers around the grid
DT = np.float32              # solver precision (float32 is ~1.7x faster and plenty for a picture)
CFL = 0.8
PRANDTL = 0.72
OUTFLOW_EXTRAP = 0.3         # still-air boundaries: outflow faster than this x a is extrapolated
MU_EXP = 0.76                # mu ~ T^0.76 (power-law fit to Sutherland for air)
FIRST_ORDER_ITERS = 150      # robust start-up
SHOCK_THR = 0.25             # relative pressure jump that switches HLLC -> HLL


def _key(p):
    return (round(float(p[0]), 6), round(float(p[1]), 6))


def source_state(M, npr, t0r, n, g):
    """Primitive state (rho, u, v, p) of a source line in solver units (ambient: rho = 1, a = 1,
    p = 1/g). M: Mach number at the line, npr: total pressure / ambient static pressure,
    t0r: total temperature / ambient temperature, n: unit vector of the outflow direction."""
    k = 1.0 + 0.5 * (g - 1.0) * M * M
    ps = npr / k ** (g / (g - 1.0))                  # static p / p_amb
    ts = t0r / k                                     # static T / T_amb
    rho = ps / ts
    V = M * math.sqrt(ts)
    return np.array([rho, V * n[0], V * n[1], ps / g], dtype=float)


# =============================================================================
# geometry -> solid mask, ghost cells and their image points
# =============================================================================
def _dist_seg(px, py, a, b):
    ex, ey = b[0] - a[0], b[1] - a[1]
    L2 = ex * ex + ey * ey
    if L2 < 1e-18:
        qx = np.full_like(px, a[0])
        qy = np.full_like(py, a[1])
    else:
        u = np.clip(((px - a[0]) * ex + (py - a[1]) * ey) / L2, 0.0, 1.0)
        qx, qy = a[0] + u * ex, a[1] + u * ey
    return np.hypot(px - qx, py - qy), qx, qy


class Geometry:
    """Solid mask, ghost-cell list and image-point interpolation for one scene."""

    def __init__(self, walls, polys, ground, extent, nx, ny, sources=()):
        xmin, xmax, ymin, ymax = extent
        self.extent = extent
        self.nx, self.ny = nx, ny
        self.dx, self.dy = (xmax - xmin) / nx, (ymax - ymin) / ny
        dx, dy = self.dx, self.dy
        self.xc = xmin + (np.arange(nx) + 0.5) * dx
        self.yc = ymin + (np.arange(ny) + 0.5) * dy
        X, Y = np.meshgrid(self.xc, self.yc)
        h = max(dx, dy)
        delta = 1.05 * h                              # half thickness of a free thin line

        inbody = np.zeros((ny, nx), bool)
        body_edges = set()
        for poly in polys:
            n = len(poly)
            for k in range(n):
                body_edges.add(frozenset((_key(poly[k]), _key(poly[(k + 1) % n]))))
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            i0 = max(0, int(math.floor((min(xs) - xmin) / dx)) - 1)
            i1 = min(nx, int(math.ceil((max(xs) - xmin) / dx)) + 2)
            j0 = max(0, int(math.floor((min(ys) - ymin) / dy)) - 1)
            j1 = min(ny, int(math.ceil((max(ys) - ymin) / dy)) + 2)
            if i1 <= i0 or j1 <= j0:
                continue
            bx, by = X[j0:j1, i0:i1], Y[j0:j1, i0:i1]
            ins = np.zeros(bx.shape, bool)
            for k in range(n):
                x1, y1 = poly[k]
                x2, y2 = poly[(k + 1) % n]
                if y1 == y2:
                    continue
                cond = (y1 > by) != (y2 > by)
                xint = (x2 - x1) * (by - y1) / (y2 - y1) + x1
                ins ^= cond & (bx < xint)
            inbody[j0:j1, i0:i1] |= ins
        if ground:
            inbody |= Y < 0.0

        body_segs, thin_segs = [], []
        for a, b in walls:
            a, b = np.asarray(a, float), np.asarray(b, float)
            if frozenset((_key(a), _key(b))) in body_edges:
                body_segs.append((a, b))
            elif ground and abs(a[1]) < 1e-9 and abs(b[1]) < 1e-9:
                continue                               # part of the ground wall
            else:
                thin_segs.append((a, b))
        if ground:
            body_segs.append((np.array([xmin - 50.0, 0.0]), np.array([xmax + 50.0, 0.0])))
        src_segs = [(np.asarray(sc["a"], float), np.asarray(sc["b"], float)) for sc in sources]

        solid = inbody.copy()
        for a, b in thin_segs + src_segs:
            m = delta + 2 * h
            i0 = max(0, int(math.floor((min(a[0], b[0]) - m - xmin) / dx)))
            i1 = min(nx, int(math.ceil((max(a[0], b[0]) + m - xmin) / dx)) + 1)
            j0 = max(0, int(math.floor((min(a[1], b[1]) - m - ymin) / dy)))
            j1 = min(ny, int(math.ceil((max(a[1], b[1]) + m - ymin) / dy)) + 1)
            if i1 <= i0 or j1 <= j0:
                continue
            d, _qx, _qy = _dist_seg(X[j0:j1, i0:i1], Y[j0:j1, i0:i1], a, b)
            solid[j0:j1, i0:i1] |= d < delta
        self.solid = solid

        fl = ~solid
        nbf = np.zeros_like(solid)
        nbf[:, 1:] |= fl[:, :-1]
        nbf[:, :-1] |= fl[:, 1:]
        nbf[1:, :] |= fl[:-1, :]
        nbf[:-1, :] |= fl[1:, :]
        ghost = solid & nbf
        gj, gi = np.nonzero(ghost)
        self.gj, self.gi = gj, gi
        nG = len(gj)

        # ---- image points: reflect each ghost cell through the nearest wall surface
        gx, gy = X[gj, gi], Y[gj, gi]
        ib = inbody[gj, gi]
        best = np.full(nG, np.inf)
        nxv, nyv = np.zeros(nG), np.zeros(nG)
        which = np.full(nG, -1)                        # index of the source line, -1 = wall
        side = np.zeros(nG)                            # (ghost - wall point) . source normal

        def consider(a, b, thin, src=-1, sn=None):
            nonlocal best, nxv, nyv, which, side
            d, qx, qy = _dist_seg(gx, gy, a, b)
            if thin:
                depth = delta - d
                ok = (d < delta) & (depth < best)
                sx, sy = gx - qx, gy - qy
            else:
                depth = d
                ok = ib & (d < best)
                sx, sy = qx - gx, qy - gy
            if not ok.any():
                return
            dd = np.hypot(sx, sy)
            tiny = dd < 1e-9
            ex, ey = b[0] - a[0], b[1] - a[1]
            L = max(math.hypot(ex, ey), 1e-12)
            ux = np.where(tiny, -ey / L, sx / np.maximum(dd, 1e-300))
            uy = np.where(tiny, ex / L, sy / np.maximum(dd, 1e-300))
            best = np.where(ok, depth, best)
            nxv = np.where(ok, ux, nxv)
            nyv = np.where(ok, uy, nyv)
            which = np.where(ok, src, which)
            if sn is not None:
                side = np.where(ok, sx * sn[0] + sy * sn[1], side)

        for a, b in body_segs:
            consider(a, b, False)
        for a, b in thin_segs:
            consider(a, b, True)
        for k, (a, b) in enumerate(src_segs):
            consider(a, b, True, k, np.asarray(sources[k]["n"], float))
        best = np.where(np.isfinite(best), best, 0.0)
        self.nxv, self.nyv = nxv, nyv           # (cast to DT below)

        ix, iy = gx + 2.0 * best * nxv, gy + 2.0 * best * nyv
        fx = np.clip((ix - xmin) / dx - 0.5, 0.0, nx - 1.000001)
        fy = np.clip((iy - ymin) / dy - 0.5, 0.0, ny - 1.000001)
        i0, j0 = np.floor(fx).astype(int), np.floor(fy).astype(int)
        tx, ty = fx - i0, fy - j0
        J = np.stack([j0, j0, j0 + 1, j0 + 1], axis=1)
        I = np.stack([i0, i0 + 1, i0, i0 + 1], axis=1)
        w = np.stack([(1 - tx) * (1 - ty), tx * (1 - ty), (1 - tx) * ty, tx * ty], axis=1)
        w = w * fl[J, I]
        ws = w.sum(axis=1)
        for k in np.nonzero(ws < 1e-12)[0]:             # image stencil is all solid: nearest fluid
            for dj, di in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                jj, ii = gj[k] + dj, gi[k] + di
                if 0 <= jj < ny and 0 <= ii < nx and fl[jj, ii]:
                    J[k, :], I[k, :] = jj, ii
                    w[k, :] = (1.0, 0.0, 0.0, 0.0)
                    break
            else:
                w[k, :] = (1.0, 0.0, 0.0, 0.0)
        w = w / np.maximum(w.sum(axis=1, keepdims=True), 1e-12)
        Nx = nx + 2 * NG
        self.gidx = (J + NG) * Nx + (I + NG)            # flat indices into the padded array
        self.gw = w.astype(DT)
        self.gflat = (gj + NG) * Nx + (gi + NG)
        # source ghosts: on the outflow side of a source line they hold the source state
        sel = (which >= 0) & (side > 0.0)
        self.src_flat = self.gflat[sel]
        self.src_W = (np.stack([np.asarray(sources[k]["W"], float) for k in which[sel]], axis=1)
                      .astype(DT) if sel.any() else np.zeros((4, 0), DT))
        self.n_src_cells = int(sel.sum())
        valid = np.zeros((ny + 2 * NG, nx + 2 * NG), bool)     # fluid, ghost or outer padding
        valid[:, :] = True
        valid[NG:-NG, NG:-NG] = fl
        valid.reshape(-1)[self.gflat] = True
        self.valid_pad = valid
        self.nxv, self.nyv = self.nxv.astype(DT), self.nyv.astype(DT)
        # padded solid mask (domain padding is never solid)
        sp = np.zeros((ny + 2 * NG, nx + 2 * NG), bool)
        sp[NG:-NG, NG:-NG] = solid
        self.solid_pad = sp
        self.fluid = fl
        self.n_fluid = int(fl.sum())


# =============================================================================
# numerics
# =============================================================================
def _limit(a, b):
    """van Leer slope limiter: (|ab| + ab) / (a + b)."""
    ab = a * b
    den = a + b
    den[den == 0.0] = 1.0
    return (np.abs(ab) + ab) / den


def _flux_hllc(L, R, g):
    rl, ul, vl, pl = L
    rr, ur, vr, pr = R
    al, ar = np.sqrt(g * pl / rl), np.sqrt(g * pr / rr)
    sl = np.minimum(ul - al, ur - ar)
    sr = np.maximum(ul + al, ur + ar)
    ql, qr = rl * (sl - ul), rr * (sr - ur)
    ss = (pr - pl + rl * ul * (sl - ul) - rr * ur * (sr - ur)) / (ql - qr)
    left = ss >= 0.0
    # pick the side the contact lies on, then one star flux:  F = F_k + S_k (U*_k - U_k)
    rk = np.where(left, rl, rr)
    uk = np.where(left, ul, ur)
    vk = np.where(left, vl, vr)
    pk = np.where(left, pl, pr)
    sk = np.where(left, sl, sr)
    act = np.where(left, sl < 0.0, sr > 0.0)          # False = supersonic, plain upwind flux
    Ek = pk / (g - 1.0) + 0.5 * rk * (uk * uk + vk * vk)
    qk = rk * (sk - uk)
    ck = qk / (sk - ss)
    # (act is False -> sk is zeroed so the correction vanishes)
    sk = np.where(act, sk, 0.0)
    ck = np.where(act, ck, rk)
    ssn = np.where(act, ss, uk)
    pk_over_q = np.where(act, pk / np.where(qk == 0.0, 1.0, qk), 0.0)
    rku = rk * uk
    F0 = rku + sk * (ck - rk)
    F1 = rku * uk + pk + sk * (ck * ssn - rku)
    F2 = rku * vk + sk * (ck * vk - rk * vk)
    F3 = (Ek + pk) * uk + sk * (ck * (Ek / rk + (ssn - uk) * (ssn + pk_over_q)) - Ek)
    return np.stack([F0, F1, F2, F3])


def _flux_hll(L, R, g):
    rl, ul, vl, pl = L
    rr, ur, vr, pr = R
    al, ar = np.sqrt(g * pl / rl), np.sqrt(g * pr / rr)
    sl = np.minimum(ul - al, ur - ar)
    sr = np.maximum(ul + al, ur + ar)
    El = pl / (g - 1.0) + 0.5 * rl * (ul * ul + vl * vl)
    Er = pr / (g - 1.0) + 0.5 * rr * (ur * ur + vr * vr)
    UL = (rl, rl * ul, rl * vl, El)
    UR = (rr, rr * ur, rr * vr, Er)
    FL = (rl * ul, rl * ul * ul + pl, rl * ul * vl, (El + pl) * ul)
    FR = (rr * ur, rr * ur * ur + pr, rr * ur * vr, (Er + pr) * ur)
    den = np.where(sr - sl == 0.0, 1.0, sr - sl)
    out = []
    for k in range(4):
        fh = (sr * FL[k] - sl * FR[k] + sl * sr * (UR[k] - UL[k])) / den
        out.append(np.where(sl >= 0.0, FL[k], np.where(sr <= 0.0, FR[k], fh)))
    return np.stack(out)


def _sweep(Wr, solid, flagc, g, order):
    """Flux difference along the last axis.
    Wr: (4, R, N) primitives (rho, u_n, u_t, p) incl. 2 padding cells at both ends.
    solid: (R, N) bool.  flagc: (R, N) bool (HLL zones).
    Returns d(mass, mom_n, mom_t, E)/dt * dx  for the N-4 interior cells, shape (4, R, N-4)."""
    c = Wr[:, :, 1:-2], Wr[:, :, 2:-1]
    if order == 2:
        a = Wr[:, :, 1:-1] - Wr[:, :, :-2]
        b = Wr[:, :, 2:] - Wr[:, :, 1:-1]
        s = _limit(a, b)
        WL = Wr[:, :, 1:-2] + 0.5 * s[:, :, :-1]
        WR = Wr[:, :, 2:-1] - 0.5 * s[:, :, 1:]
        bad = ((WL[0] <= 1e-6) | (WL[3] <= 1e-6) | (WR[0] <= 1e-6) | (WR[3] <= 1e-6))
        if bad.any():
            WL[:, bad] = c[0][:, bad]
            WR[:, bad] = c[1][:, bad]
    else:
        WL, WR = c[0].copy(), c[1].copy()
    sl, sr = solid[:, 1:-2], solid[:, 2:-1]
    m = sr & ~sl                       # wall on the right of the face: right state = ghost cell
    if m.any():
        WR[:, m] = c[1][:, m]
    m = sl & ~sr
    if m.any():
        WL[:, m] = c[0][:, m]
    F = _flux_hllc(WL, WR, g)
    fl = flagc[:, 1:-2] | flagc[:, 2:-1]
    fl &= ~(sl | sr)
    if fl.any():
        F[:, fl] = _flux_hll(WL[:, fl], WR[:, fl], g)
    return F[:, :, :-1] - F[:, :, 1:]


class Solver:
    def __init__(self, geo, M, g, first_order=FIRST_ORDER_ITERS, main=True, visc=None):
        """main: freestream on (else quiescent ambient). visc: None (Euler) or Reynolds number
        per unit length, based on the freestream speed (main flow on) or the ambient speed of
        sound (main flow off)."""
        self.geo, self.M, self.g = geo, float(M), float(g)
        self.first_order = first_order
        self.main = bool(main)
        self.mu0 = 0.0 if not visc else max(self.M if self.main else 1.0, 0.2) / float(visc)
        ny, nx = geo.ny, geo.nx
        self.pinf = 1.0 / self.g
        self.Winf = np.array([1.0, -self.M if self.main else 0.0, 0.0, self.pinf], dtype=DT)
        self.U = np.empty((4, ny, nx), dtype=DT)
        self._set_cons(self.U, np.broadcast_to(self.Winf[:, None, None], (4, ny, nx)))
        self.Ny, self.Nx = ny + 2 * NG, nx + 2 * NG
        self.sp_x = geo.solid_pad[NG:-NG, :]                 # (ny, Nx)
        self.sp_y = np.ascontiguousarray(geo.solid_pad[:, NG:-NG].T)   # (nx, Ny)
        self.fluid = geo.fluid
        self.it = 0
        self.res = 1.0
        self.res_max = 0.0

    def init_from(self, c):
        """Start from a converged coarser solution (bilinear, fluid cells only)."""
        g, cg = self.geo, c.geo
        Wc = c._prim(c.U)
        fx = np.clip((g.xc - cg.extent[0]) / cg.dx - 0.5, 0.0, cg.nx - 1.000001)
        fy = np.clip((g.yc - cg.extent[2]) / cg.dy - 0.5, 0.0, cg.ny - 1.000001)
        i0, j0 = np.floor(fx).astype(int), np.floor(fy).astype(int)
        tx, ty = (fx - i0)[None, :], (fy - j0)[:, None]
        acc = np.zeros((4, g.ny, g.nx))
        ws = np.zeros((g.ny, g.nx))
        for dj, di, wt in ((0, 0, (1 - ty) * (1 - tx)), (0, 1, (1 - ty) * tx),
                           (1, 0, ty * (1 - tx)), (1, 1, ty * tx)):
            J, I = (j0 + dj)[:, None], (i0 + di)[None, :]
            w = wt * c.fluid[J, I]
            acc += w * Wc[:, J, I]
            ws += w
        ok = ws > 1e-6
        Wn = np.where(ok, acc / np.maximum(ws, 1e-6), self.Winf[:, None, None])
        self._set_cons(self.U, Wn.astype(DT))

    def _set_cons(self, U, W):
        g = self.g
        U[0] = W[0]
        U[1] = W[0] * W[1]
        U[2] = W[0] * W[2]
        U[3] = W[3] / (g - 1.0) + 0.5 * W[0] * (W[1] ** 2 + W[2] ** 2)

    def _prim(self, U):
        g = self.g
        r = np.maximum(U[0], 1e-8)
        u, v = U[1] / r, U[2] / r
        p = np.maximum((g - 1.0) * (U[3] - 0.5 * r * (u * u + v * v)), 1e-8)
        return np.stack([r, u, v, p]).astype(DT, copy=False)

    def _padded(self, Wi):
        """Interior primitives -> padded array with boundary conditions and wall ghost cells."""
        geo, g = self.geo, self.g
        Ny, Nx = self.Ny, self.Nx
        W = np.empty((4, Ny, Nx), dtype=DT)
        W[:] = self.Winf[:, None, None]
        W[:, NG:-NG, NG:-NG] = Wi
        if self.main:
            # top / bottom: zero gradient
            W[:, :NG, NG:-NG] = Wi[:, :1, :]
            W[:, -NG:, NG:-NG] = Wi[:, -1:, :]
            # left: zero gradient, ambient pressure if the flow leaves subsonically
            W[:, NG:-NG, :NG] = self._pressure_bc(W[:, NG:-NG, NG:NG + 1], 1, -1)
            # right: freestream inflow (already set)
        else:
            # quiescent ambient: pressure boundaries all round
            W[:, NG:-NG, :NG] = self._pressure_bc(W[:, NG:-NG, NG:NG + 1], 1, -1)
            W[:, NG:-NG, -NG:] = self._pressure_bc(W[:, NG:-NG, -NG - 1:-NG], 1, 1)
            W[:, :NG, NG:-NG] = self._pressure_bc(W[:, NG:NG + 1, NG:-NG], 2, -1)
            W[:, -NG:, NG:-NG] = self._pressure_bc(W[:, -NG - 1:-NG, NG:-NG], 2, 1)
        # wall ghost cells: mirror image of the flow at the wall
        Wf = W.reshape(4, -1)
        vals = Wf[:, geo.gidx]                                # (4, nG, 4)
        w = geo.gw
        r = (vals[0] * w).sum(axis=1)
        u = (vals[1] * w).sum(axis=1)
        v = (vals[2] * w).sum(axis=1)
        p = (vals[3] * w).sum(axis=1)
        Wf[0, geo.gflat] = r
        Wf[3, geo.gflat] = p
        if self.mu0 > 0.0:                                    # no-slip: velocity reversed
            Wf[1, geo.gflat] = -u
            Wf[2, geo.gflat] = -v
        else:                                                 # slip: normal component reversed
            un = u * geo.nxv + v * geo.nyv
            Wf[1, geo.gflat] = u - 2.0 * un * geo.nxv
            Wf[2, geo.gflat] = v - 2.0 * un * geo.nyv
        if geo.n_src_cells:                                   # source lines inject their state
            Wf[:, geo.src_flat] = geo.src_W
        return W

    def _pressure_bc(self, e, axis, sgn):
        """Ghost state beyond an outer edge. e: (4, ...) edge cells, axis 1 = x (u), 2 = y (v),
        sgn = +1 if the outward normal points along +axis. Supersonic outflow: extrapolate;
        subsonic outflow: ambient pressure; inflow: ambient density and pressure."""
        e = e.copy()
        g = self.g
        un = sgn * e[axis]                                    # outward normal velocity
        a = np.sqrt(g * e[3] / e[0])
        if self.main:
            e[3] = np.where(un < a, self.pinf, e[3])
            return e
        # quiescent ambient: characteristic (Riemann-invariant) far field, so waves leave
        # instead of reflecting. R+ from inside, R- from the ambient (u = 0, a = 1).
        tang = 3 - axis                                       # the other velocity component
        Rp = un + 2.0 * a / (g - 1.0)
        Rm = -2.0 / (g - 1.0)
        ub = 0.5 * (Rp + Rm)
        ab = np.maximum(0.25 * (g - 1.0) * (Rp - Rm), 1e-3)
        out = ub >= 0.0
        s_in = e[3] / e[0] ** g                               # entropy: inside or ambient
        s_b = np.where(out, s_in, self.pinf)
        rb = (ab * ab / (g * s_b)) ** (1.0 / (g - 1.0))
        pb = s_b * rb ** g
        # gas that is clearly leaving (jet, plume) is extrapolated: a pressure condition there
        # feeds back upstream and terminates a jet core early in a box this small
        sup = (un >= a) | (un > OUTFLOW_EXTRAP * a)
        e[0] = np.where(sup, e[0], rb)
        e[3] = np.where(sup, e[3], pb)
        e[axis] = np.where(sup, e[axis], sgn * ub)
        e[tang] = np.where(sup | out, e[tang], 0.0)
        return e

    def _grad_c(self, f, valid, axis, h):
        """Central difference along axis (0 = y, 1 = x) on the padded grid, one-sided next to
        invalid (deep solid) cells, zero if both neighbours are invalid."""
        fp = np.roll(f, -1, axis)
        fm = np.roll(f, 1, axis)
        vp = np.roll(valid, -1, axis)
        vm = np.roll(valid, 1, axis)
        return np.where(vp & vm, (fp - fm) / (2 * h),
                        np.where(vp, (fp - f) / h, np.where(vm, (f - fm) / h, 0.0)))

    def _viscous(self, W):
        """Viscous flux divergence (4, ny, nx) for the interior cells (laminar, Stokes)."""
        geo, g = self.geo, self.g
        dx, dy = geo.dx, geo.dy
        V = geo.valid_pad
        u, v = W[1], W[2]
        T = W[3] / W[0]                                       # p / rho  (a^2 = g T)
        mu = self.mu0 * (g * T) ** MU_EXP
        kk = mu * g / ((g - 1.0) * PRANDTL)
        uy, vy, ux, vx = (self._grad_c(u, V, 0, dy), self._grad_c(v, V, 0, dy),
                          self._grad_c(u, V, 1, dx), self._grad_c(v, V, 1, dx))
        R = np.zeros((4, geo.ny, geo.nx), dtype=DT)
        r = slice(NG, -NG)
        # x faces (between columns c and c+1), rows interior
        a, b = slice(NG - 1, -NG), slice(NG, -NG + 1)
        ok = (V[r, a] & V[r, b]).astype(DT)
        muf = 0.5 * (mu[r, a] + mu[r, b]) * ok
        kf = 0.5 * (kk[r, a] + kk[r, b]) * ok
        dudx = (u[r, b] - u[r, a]) / dx
        dvdx = (v[r, b] - v[r, a]) / dx
        dTdx = (T[r, b] - T[r, a]) / dx
        dudy = 0.5 * (uy[r, a] + uy[r, b])
        dvdy = 0.5 * (vy[r, a] + vy[r, b])
        uf, vf = 0.5 * (u[r, a] + u[r, b]), 0.5 * (v[r, a] + v[r, b])
        txx = muf * (4.0 / 3.0 * dudx - 2.0 / 3.0 * dvdy)
        txy = muf * (dudy + dvdx)
        q = uf * txx + vf * txy + kf * dTdx
        for k, F in ((1, txx), (2, txy), (3, q)):
            R[k] += (F[:, 1:] - F[:, :-1]) / dx
        # y faces (between rows), columns interior
        ok = (V[a, r] & V[b, r]).astype(DT)
        muf = 0.5 * (mu[a, r] + mu[b, r]) * ok
        kf = 0.5 * (kk[a, r] + kk[b, r]) * ok
        dudy = (u[b, r] - u[a, r]) / dy
        dvdy = (v[b, r] - v[a, r]) / dy
        dTdy = (T[b, r] - T[a, r]) / dy
        dudx = 0.5 * (ux[a, r] + ux[b, r])
        dvdx = 0.5 * (vx[a, r] + vx[b, r])
        uf, vf = 0.5 * (u[a, r] + u[b, r]), 0.5 * (v[a, r] + v[b, r])
        tyy = muf * (4.0 / 3.0 * dvdy - 2.0 / 3.0 * dudx)
        txy = muf * (dudy + dvdx)
        q = uf * txy + vf * tyy + kf * dTdy
        for k, F in ((1, txy), (2, tyy), (3, q)):
            R[k] += (F[1:, :] - F[:-1, :]) / dy
        return R

    def _shock_cells(self, W):
        """Cells next to a strong pressure jump (fluid-fluid faces only)."""
        sp = self.geo.solid_pad
        p = W[3]
        flag = np.zeros_like(sp)
        jx = np.abs(p[:, 1:] - p[:, :-1]) / np.minimum(p[:, 1:], p[:, :-1]) > SHOCK_THR
        jx &= ~(sp[:, 1:] | sp[:, :-1])
        flag[:, 1:] |= jx
        flag[:, :-1] |= jx
        jy = np.abs(p[1:, :] - p[:-1, :]) / np.minimum(p[1:, :], p[:-1, :]) > SHOCK_THR
        jy &= ~(sp[1:, :] | sp[:-1, :])
        flag[1:, :] |= jy
        flag[:-1, :] |= jy
        return flag

    def _rhs(self, Wi, order):
        """dU/dt for the interior cells (4, ny, nx)."""
        geo, g = self.geo, self.g
        W = self._padded(Wi)
        flag = self._shock_cells(W) if order == 2 else np.zeros_like(geo.solid_pad)
        dUx = _sweep(W[:, NG:-NG, :], self.sp_x, flag[NG:-NG, :], g, order)
        Wy = np.ascontiguousarray(W[[0, 2, 1, 3]][:, :, NG:-NG].transpose(0, 2, 1))
        dUy = _sweep(Wy, self.sp_y, np.ascontiguousarray(flag[:, NG:-NG].T), g, order)
        R = np.empty_like(dUx)
        R[0] = dUx[0] / geo.dx + dUy[0].T / geo.dy
        R[1] = dUx[1] / geo.dx + dUy[2].T / geo.dy
        R[2] = dUx[2] / geo.dx + dUy[1].T / geo.dy
        R[3] = dUx[3] / geo.dx + dUy[3].T / geo.dy
        if self.mu0 > 0.0:
            R += self._viscous(W)
        R[:, ~self.fluid] = 0.0
        return R

    def step(self, order=2):
        geo, g = self.geo, self.g
        U = self.U
        W0 = self._prim(U)
        a = np.sqrt(g * W0[3] / W0[0])
        lam = (np.abs(W0[1]) + a) / geo.dx + (np.abs(W0[2]) + a) / geo.dy
        if self.mu0 > 0.0:                                    # viscous time-step limit
            nu = max(4.0 / 3.0, g / PRANDTL) * self.mu0 * (g * W0[3] / W0[0]) ** MU_EXP / W0[0]
            lam = lam + 2.0 * nu * (1.0 / geo.dx ** 2 + 1.0 / geo.dy ** 2)
        dt = CFL / lam
        R0 = self._rhs(W0, order)
        U1 = U + dt * R0
        W1 = self._prim(U1)
        R1 = self._rhs(W1, order)
        Un = 0.5 * (U + U1 + dt * R1)
        Un[:, ~self.fluid] = U[:, ~self.fluid]
        self.res = float(np.sqrt(np.mean((R0[0][self.fluid] * geo.dx / max(self.M, 1.0)) ** 2)))
        if not np.all(np.isfinite(Un[0])):
            self.res = float("nan")
        self.res_max = max(self.res_max, self.res) if self.it > self.first_order else self.res
        self.U = Un
        self.it += 1

    def pressure(self):
        """p / p_inf on the grid, NaN inside solids."""
        p = self._prim(self.U)[3] * self.g
        return np.where(self.fluid, p, np.nan)

    def field(self, kind):
        """A field on the grid (NaN inside solids):
        pressure  p / p_inf          mach       local Mach number
        density   rho / rho_inf      vorticity  dv/dx - du/dy  (in a_inf / unit length)
        schlieren |grad rho|  (the display turns it into a schlieren picture)"""
        g = self.g
        W = self._prim(self.U).astype(float)
        r, u, v, p = W
        if kind == "pressure":
            f = p * g
        elif kind == "mach":
            f = np.hypot(u, v) / np.sqrt(g * p / r)
        elif kind == "density":
            f = r
        else:
            fl = self.fluid
            if kind == "vorticity":
                uu, vv = np.where(fl, u, np.nan), np.where(fl, v, np.nan)
                dvdx = np.gradient(vv, self.geo.dx, axis=1)
                dudy = np.gradient(uu, self.geo.dy, axis=0)
                f = dvdx - dudy
            else:                                             # schlieren
                rr = np.where(fl, r, np.nan)
                f = np.hypot(np.gradient(rr, self.geo.dx, axis=1),
                             np.gradient(rr, self.geo.dy, axis=0))
            bad = ~np.isfinite(f) & fl                        # cells next to a wall: one-sided
            if bad.any():
                f[bad] = 0.0
        return np.where(self.fluid, f, np.nan)

    def converged(self, tol=2e-4):
        return self.it > self.first_order + 100 and self.res < tol * max(self.res_max, 1e-12)


def solve(walls, polys, ground, M, g, extent, nx, ny, max_iter=1500, time_limit=30.0,
          cancel=None, progress=None, every=1.5, levels=(4, 2, 1), main=True, sources=(),
          visc=None):
    """Run to steady state on a sequence of grids (coarse -> fine; the last one is nx x ny),
    stopping when converged or when the iteration / time budget is used up.
    sources: [{"a", "b", "n", "W"}] source lines (W from source_state). visc: Reynolds number
    per unit length, or None for inviscid flow.
    progress(info, solver, geometry) is called about every `every` seconds.
    Returns (solver, info, geometry); solver is None if cancelled."""
    t0 = time.time()
    if sources:              # internal (nozzle) flows: a coarse grid must still resolve the throat
        h = (extent[1] - extent[0]) / nx  # (>= 6 cells across the shortest source line), else
        Lmin = min(float(np.hypot(*(np.asarray(sc["b"]) - np.asarray(sc["a"])))) for sc in sources)
        levels = tuple(f for f in levels if f == 1 or Lmin / (h * f) >= 6.0) or (1,)  # it blocks
    info = dict(it=0, res=1.0, converged=False, nx=nx, ny=ny, time=0.0, level=0, levels=len(levels))
    prev = None
    sol = geo = None
    tlast = t0
    for lv, fac in enumerate(levels):
        last = lv == len(levels) - 1
        lnx = nx if last else max(48, int(round(nx / fac)))
        lny = ny if last else max(24, int(round(ny / fac)))
        geo = Geometry(walls, polys, ground, extent, lnx, lny, sources)
        sol = Solver(geo, M, g, first_order=0 if prev is not None else FIRST_ORDER_ITERS,
                     main=main, visc=visc)
        if prev is not None:
            sol.init_from(prev)
        budget = max_iter if last else (1500 if lv == 0 else 500)
        tcap = t0 + (time_limit * (0.18 if lv == 0 else 0.42) if not last else time_limit)
        for it in range(budget):
            if cancel is not None and cancel():
                return None, info, geo
            sol.step(1 if it < sol.first_order else 2)
            if not np.isfinite(sol.res):
                info["diverged"] = True
                break
            now = time.time()
            info.update(it=sol.it, res=sol.res / max(sol.res_max, 1e-12), time=now - t0,
                        nx=lnx, ny=lny, level=lv + 1)
            if progress is not None and now - tlast > every:
                tlast = now
                progress(dict(info), sol, geo)
            if sol.converged(2e-3 if not last else 2e-4):
                info["converged"] = last
                break
            if now > tcap:
                break
        prev = sol
    info.update(time=time.time() - t0)
    return sol, info, geo
