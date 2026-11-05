"""
TBG - drawing window  (File > Draw... in the simulation window)

  Tools (drop-down)   Normal line | Spline | Source line | Linear actuator | Rotary actuator
  Thickness           thickness of the lines you draw (0 = ideal thin wall; > 0 = a solid plate)

  Drawing a line      click to place the start point, then EITHER click the end point
                      OR type a length, press Tab, type an angle, press Enter.
                      The angle is measured counter-clockwise from the +x axis (an arc shows it).
                      The chain continues from the end point. Esc / right-click ends the chain.
  Magnet              line ends snap to existing vertices, then to existing lines.
  Linear actuator     draw its track like a line (click or Length/Tab/Angle).
  Rotary actuator     click to place its centre (snaps to vertices).
  Spline              click the points the curve passes through; Enter / right-click / Esc
                      finishes, clicking the first point closes it. Right-click > Edit points:
                      drag points, click the curve to insert one, right-click a point to delete it.
  Source line         drawn like a line; flow leaves it on the arrow side. Right-click >
                      Set flow... (Mach, p0/p_amb, T0/T_amb), Flip direction.
  Image...            overlay picture for retracing (load, move, scale by two points, opacity).
  Pairing             right-click an actuator > Pair geometry..., then drag a box:
                      left -> right = geometry fully inside the box,
                      right -> left = anything the box touches.
  View                mouse wheel zooms, middle-button drag pans, Fit resets.
  Finish              sends the sketch to the simulation window.

Pure tkinter + numpy (no matplotlib); Pillow (installed with matplotlib) for the overlay image.
"""

import math
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk

import numpy as np

from tbg_common import ACT_COLORS, SRC_COLOR, Scene, Spline, Source, set_window_icon, spline_curve

TOOL_LINE, TOOL_LIN, TOOL_ROT = "Normal line", "Linear actuator", "Rotary actuator"
TOOL_SPL, TOOL_SRC = "Spline", "Source line"
TOOLS = [TOOL_LINE, TOOL_SPL, TOOL_SRC, TOOL_LIN, TOOL_ROT]
CANVAS_BG = "#fbfbfb"
SNAP_VERTEX_PX = 12
SNAP_EDGE_PX = 8
HIT_PX = 9
ALIGN_PX = 8                 # centre-guide pull distance
GUIDE_MID = "#d6336c"        # guide from a line's centre (midpoint)
GUIDE_PT = "#7b8794"         # guide from an ordinary vertex / the origin / the start point
FONT = ("Microsoft Sans Serif", 9)


def _seg_point(p, a, b):
    """nearest point of segment a-b to p (all np arrays) and its parameter"""
    e = b - a
    L2 = float(e @ e)
    u = 0.0 if L2 < 1e-18 else min(1.0, max(0.0, float((p - a) @ e) / L2))
    return a + u * e, u


def _seg_hits_rect(a, b, x0, y0, x1, y1):
    """Liang-Barsky: does segment a-b touch the rectangle?"""
    dx, dy = b[0] - a[0], b[1] - a[1]
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, a[0] - x0), (dx, x1 - a[0]), (-dy, a[1] - y0), (dy, y1 - a[1])):
        if abs(p) < 1e-15:
            if q < 0:
                return False
        else:
            r = q / p
            if p < 0:
                if r > t1:
                    return False
                t0 = max(t0, r)
            else:
                if r < t0:
                    return False
                t1 = min(t1, r)
    return True


class DrawWindow:
    def __init__(self, master, scene, on_finish, plane_getter=lambda: True,
                 view=(-7.0, 8.0, -1.2, 6.2), overlay=None):
        self.master = master
        self.scene = scene                       # an editable COPY
        self.on_finish = on_finish
        self.plane_getter = plane_getter
        self.view = view
        self.undo_stack = []
        self.modified = False

        self.start = None                        # world point where the current segment starts
        self.typed = {"len": False, "ang": False}
        self.cursor = None                       # snapped cursor (world)
        self.cursor_kind = None                  # "vertex" | "edge" | None
        self.guides = []                         # [(ref point, is_midpoint)] aligned with the cursor
        self.pair_act = None                     # actuator being paired
        self.rb = None                           # rubber band [x0, y0, x1, y1] in canvas px
        self.sc, self.ox, self.oy = 40.0, 0.0, 0.0
        self._fitted = False
        self._pan = None
        self.spl_pts = []                        # spline being drawn (control points)
        self.edit_sp = None                      # spline whose points are being edited
        self.drag_idx = None                     # index of the dragged control point
        self.ov = overlay if overlay is not None else {}     # overlay image settings
        self.ov_mode = None                      # None | "move" | "scale"
        self.ov_drag = None
        self.ov_pts = []
        self._ov_tk = None
        self._ov_cache = None
        self.ov_win = None

        self.top = tk.Toplevel(master)
        self.top.title("TBG - Draw")
        self.top.geometry("1220x800")
        set_window_icon(self.top)
        self.tool = tk.StringVar(master=self.top, value=TOOL_LINE)
        self.thick = tk.StringVar(master=self.top, value="0")
        self.len_var = tk.StringVar(master=self.top, value="")
        self.ang_var = tk.StringVar(master=self.top, value="")
        self.status = tk.StringVar(master=self.top, value="")
        self.coords = tk.StringVar(master=self.top, value="")
        self.guides_on = tk.BooleanVar(master=self.top, value=True)
        self._build_ui()
        self.set_hint()
        self.top.protocol("WM_DELETE_WINDOW", self.request_close)

    # ================================================================== UI
    def _build_ui(self):
        top = self.top
        style = ttk.Style(top)
        style.configure(".", font=FONT)
        bar = ttk.Frame(top, padding=(10, 8))
        bar.pack(side="top", fill="x")
        ttk.Label(bar, text="Draw").pack(side="left")
        cb = ttk.Combobox(bar, textvariable=self.tool, values=TOOLS, state="readonly", width=16)
        cb.pack(side="left", padx=(6, 18))
        cb.bind("<<ComboboxSelected>>", lambda _e: self.on_tool_change())
        ttk.Label(bar, text="Thickness").pack(side="left")
        ttk.Spinbox(bar, from_=0.0, to=10.0, increment=0.05, textvariable=self.thick,
                    width=6).pack(side="left", padx=(6, 2))
        ttk.Label(bar, text="(0 = thin wall)", foreground="#777").pack(side="left", padx=(0, 22))
        ttk.Label(bar, text="Length").pack(side="left")
        self.len_e = ttk.Entry(bar, textvariable=self.len_var, width=9)
        self.len_e.pack(side="left", padx=(6, 14))
        ttk.Label(bar, text="Angle").pack(side="left")
        self.ang_e = ttk.Entry(bar, textvariable=self.ang_var, width=9)
        self.ang_e.pack(side="left", padx=(6, 0))
        ttk.Label(bar, text="\u00b0", foreground="#777").pack(side="left", padx=(1, 0))
        ttk.Checkbutton(bar, text="Center guides", variable=self.guides_on,
                        command=self._guides_toggled).pack(side="left", padx=(22, 0))
        ttk.Button(bar, text="Finish", command=self.finish).pack(side="right")
        ttk.Button(bar, text="Fit", command=self.fit_view).pack(side="right", padx=(0, 8))
        ttk.Button(bar, text="Undo", command=self.undo).pack(side="right", padx=(0, 8))
        ttk.Button(bar, text="Image...", width=9, command=self.open_image_panel).pack(side="right", padx=(0, 8))
        self.len_e.state(["disabled"])
        self.ang_e.state(["disabled"])
        for which, ent in (("len", self.len_e), ("ang", self.ang_e)):
            ent.bind("<KeyPress>", lambda e, w=which: self._typing(w, e))
            ent.bind("<Tab>", lambda e, w=which: self._tab_from(w))
            ent.bind("<Return>", lambda e: self._enter_key())
            ent.bind("<KP_Enter>", lambda e: self._enter_key())

        foot = ttk.Frame(top, padding=(10, 4))
        foot.pack(side="bottom", fill="x")
        ttk.Label(foot, textvariable=self.status).pack(side="left")
        ttk.Label(foot, textvariable=self.coords, foreground="#555").pack(side="right")

        self.canvas = tk.Canvas(top, bg=CANVAS_BG, highlightthickness=0, cursor="crosshair")
        self.canvas.pack(side="top", fill="both", expand=True)
        c = self.canvas
        c.bind("<Configure>", self._on_configure)
        c.bind("<Motion>", self.on_motion)
        c.bind("<Button-1>", self.on_press)
        c.bind("<B1-Motion>", self.on_drag)
        c.bind("<ButtonRelease-1>", self.on_release)
        c.bind("<Button-3>", self.on_right)
        c.bind("<Button-2>", self._pan_start)
        c.bind("<B2-Motion>", self._pan_move)
        c.bind("<MouseWheel>", self._wheel)
        c.bind("<Button-4>", lambda e: self._zoom(e.x, e.y, 1.15))
        c.bind("<Button-5>", lambda e: self._zoom(e.x, e.y, 1 / 1.15))
        top.bind("<Escape>", lambda _e: self.on_escape())
        top.bind("<Control-z>", lambda _e: self.undo())

    # ================================================================== helpers
    def alive(self):
        try:
            return bool(self.top.winfo_exists())
        except Exception:
            return False

    def lift(self):
        self.top.lift()
        self.top.focus_force()

    def thickness(self):
        try:
            return max(0.0, float(self.thick.get()))
        except ValueError:
            return 0.0

    def w2c(self, x, y):
        return self.ox + self.sc * x, self.oy - self.sc * y

    def c2w(self, cx, cy):
        return (cx - self.ox) / self.sc, (self.oy - cy) / self.sc

    def act_color(self, act):
        return ACT_COLORS[self.scene.acts.index(act) % len(ACT_COLORS)]

    def push_undo(self):
        self.undo_stack.append(self.scene.to_dict())
        self.modified = True

    def undo(self):
        if self.spl_pts:                                     # spline in progress: drop a point
            self.spl_pts.pop()
            self.start = self.spl_pts[-1] if self.spl_pts else None
            if self.start is None:
                self._leave_entry_mode()
            else:
                self._enter_entry_mode()
            self.set_hint()
            self.redraw_overlay()
            return
        if not self.undo_stack:
            return
        self.scene = Scene.from_dict(self.undo_stack.pop())
        self._cancel_ops()
        self.redraw()

    def _cancel_ops(self):
        self.start = None
        self.pair_act = None
        self.rb = None
        self.spl_pts = []
        self.edit_sp = None
        self.drag_idx = None
        self.ov_mode = None
        self.ov_pts = []
        self._leave_entry_mode()
        self.set_hint()

    def set_hint(self):
        tool = self.tool.get()
        if self.ov_mode == "move":
            self.status.set("Image: drag it with the left button.  Esc when done.")
            return
        if self.ov_mode == "scale":
            self.status.set(f"Image scale: click point {len(self.ov_pts) + 1} of 2 on the picture "
                            "(two points whose real distance you know).  Esc cancels.")
            return
        if self.edit_sp is not None:
            self.status.set("Editing spline: drag a point, click the curve to insert a point, "
                            "right-click a point to delete it.  Esc when done.")
            return
        if tool == TOOL_SPL and self.spl_pts:
            self.status.set(f"Spline: {len(self.spl_pts)} point(s). Click the next point (or type length "
                            "/ Tab / angle, Enter). Enter with empty boxes, right-click or Esc finishes; "
                            "click the first point to close it.")
            return
        if self.pair_act is not None:
            self.status.set(f"Pairing {self.pair_act.name}: drag a box around the geometry to move "
                            "(left\u2192right = fully inside, right\u2192left = touching).  Esc cancels.")
        elif self.start is not None:
            self.status.set("Click the end point  \u2014  or type a length, press Tab, type an angle "
                            "A (counter-clockwise from +x), press Enter.   \u03b8 = flow deflection "
                            "angle (inclination to the flow axis).   Esc ends the chain.")
        elif tool == TOOL_ROT:
            self.status.set("Click to place the rotary actuator (its centre of rotation).")
        elif tool == TOOL_LIN:
            self.status.set("Click to start the linear actuator track, then click or type "
                            "length / Tab / angle.")
        elif tool == TOOL_SPL:
            self.status.set("Spline: click the points the curve should pass through.")
        elif tool == TOOL_SRC:
            self.status.set("Source line: click its start, then click or type length / Tab / angle. "
                            "Flow leaves on the right-hand side (arrows); right-click it to set the flow.")
        else:
            self.status.set("Click to start a line.   Right-click an actuator or line for options.   "
                            "Wheel = zoom, middle-drag = pan.   Dotted guides run from line centres "
                            "(pink) and vertices (grey).")

    def on_tool_change(self):
        self._cancel_ops()
        self.canvas.focus_set()
        self.redraw()

    # ================================================================== view
    def _on_configure(self, _e):
        if not self._fitted and self.canvas.winfo_width() > 50:
            self.fit_view()
            self._fitted = True
        else:
            self.redraw()

    def fit_view(self):
        x0, x1, y0, y1 = self.view
        for ln in self.scene.lines:
            for p in (ln.p, ln.q):
                x0, x1 = min(x0, p[0]), max(x1, p[0])
                y0, y1 = min(y0, p[1]), max(y1, p[1])
        for a in self.scene.acts:
            for p in (a.a, a.b):
                if p is not None:
                    x0, x1 = min(x0, p[0]), max(x1, p[0])
                    y0, y1 = min(y0, p[1]), max(y1, p[1])
        for ob in self.scene.splines + self.scene.sources:
            for p in self._pts(ob):
                x0, x1 = min(x0, p[0]), max(x1, p[0])
                y0, y1 = min(y0, p[1]), max(y1, p[1])
        w = max(self.canvas.winfo_width(), 300)
        h = max(self.canvas.winfo_height(), 200)
        self.sc = min((w - 90) / (x1 - x0), (h - 90) / (y1 - y0))
        self.ox = w / 2 - self.sc * (x0 + x1) / 2
        self.oy = h / 2 + self.sc * (y0 + y1) / 2
        self.redraw()

    def _zoom(self, cx, cy, f):
        wx, wy = self.c2w(cx, cy)
        self.sc = min(5000.0, max(2.0, self.sc * f))
        self.ox = cx - self.sc * wx
        self.oy = cy + self.sc * wy
        self.cursor, self.cursor_kind = self.snap(cx, cy)
        self.redraw()

    def _wheel(self, ev):
        self._zoom(ev.x, ev.y, 1.15 if ev.delta > 0 else 1 / 1.15)

    def _pan_start(self, ev):
        self._pan = (ev.x, ev.y)

    def _pan_move(self, ev):
        if self._pan is None:
            return
        self.ox += ev.x - self._pan[0]
        self.oy += ev.y - self._pan[1]
        self._pan = (ev.x, ev.y)
        self.redraw()

    # ================================================================== snapping
    @staticmethod
    def _pts(ob):
        return list(ob.pts) if isinstance(ob, Spline) else [ob.p, ob.q]

    def _curve(self, sp):
        return spline_curve(sp.pts, sp.closed)

    def _segs(self):
        """Every straight piece of geometry (world): lines, sampled splines, source lines."""
        out = [(np.array(ln.p, float), np.array(ln.q, float)) for ln in self.scene.lines]
        for sp in self.scene.splines:
            C = self._curve(sp)
            out += [(C[k], C[k + 1]) for k in range(len(C) - 1)]
        out += [(np.array(sc.p, float), np.array(sc.q, float)) for sc in self.scene.sources]
        return out

    def _vertices(self):
        pts = []
        for ln in self.scene.lines:
            pts += [ln.p, ln.q]
        for sp in self.scene.splines:
            pts += [p for k, p in enumerate(sp.pts)
                    if not (sp is self.edit_sp and k == self.drag_idx)]
        for sc in self.scene.sources:
            pts += [sc.p, sc.q]
        pts += list(self.spl_pts)
        for a in self.scene.acts:
            pts.append(a.a)
            if a.b is not None:
                pts.append(a.b)
        return pts

    def _guides_toggled(self):
        self.guides = []
        self.redraw()

    def _guide_refs(self):
        """Points the centre guides run from: line midpoints (the 'centres'), the origin, every
        vertex, and the start of the line being drawn. Returns [(point, is_midpoint)]."""
        refs = [((0.0, 0.0), False)]
        for ln in self.scene.lines + self.scene.sources:
            refs.append((((ln.p[0] + ln.q[0]) / 2.0, (ln.p[1] + ln.q[1]) / 2.0), True))
            refs += [(ln.p, False), (ln.q, False)]
        for sp in self.scene.splines:
            refs += [(p, False) for p in sp.pts]
        refs += [(p, False) for p in self.spl_pts[:-1]]
        for a in self.scene.acts:
            if a.b is not None:
                refs.append((((a.a[0] + a.b[0]) / 2.0, (a.a[1] + a.b[1]) / 2.0), True))
                refs += [(a.a, False), (a.b, False)]
            else:
                refs.append((a.a, False))
        if self.start is not None:
            refs.append((self.start, False))
        return refs

    def snap(self, cx, cy):
        """Snapping + centre guides. Returns (world point, kind), kind in vertex / edge / None.
        Guide alignment only moves a free cursor; it is reported in self.guides, not in kind."""
        pt, kind = self._snap_geometry(cx, cy)
        self.guides = []
        if not self.guides_on.get():
            return pt, kind
        refs = self._guide_refs()
        x, y = pt
        if kind is None:                                      # pull onto the nearest alignment
            bx = by = None
            for (rx, ry), mid in refs:
                dx = abs(rx - x) * self.sc
                dy = abs(ry - y) * self.sc
                if dx < ALIGN_PX and (bx is None or dx - (2 if mid else 0) < bx[0]):
                    bx = (dx - (2 if mid else 0), rx)
                if dy < ALIGN_PX and (by is None or dy - (2 if mid else 0) < by[0]):
                    by = (dy - (2 if mid else 0), ry)
            if bx is not None:
                x = bx[1]
            if by is not None:
                y = by[1]
            pt = (x, y)
        for (rx, ry), mid in refs:                            # every reference now in line
            if math.hypot(rx - x, ry - y) * self.sc < 1.0:
                continue
            if abs(rx - x) * self.sc < 0.5 or abs(ry - y) * self.sc < 0.5:
                self.guides.append(((rx, ry), mid))
        return pt, kind

    def _snap_geometry(self, cx, cy):
        wx, wy = self.c2w(cx, cy)
        best, bd = None, SNAP_VERTEX_PX
        for pt in self._vertices():
            px, py = self.w2c(*pt)
            d = math.hypot(px - cx, py - cy)
            if d < bd:
                best, bd = pt, d
        if best is not None:
            return (float(best[0]), float(best[1])), "vertex"
        be, bd = None, SNAP_EDGE_PX
        P = np.array([wx, wy])
        for a, b in self._segs():
            q, _u = _seg_point(P, a, b)
            qx, qy = self.w2c(q[0], q[1])
            d = math.hypot(qx - cx, qy - cy)
            if d < bd:
                be, bd = (float(q[0]), float(q[1])), d
        if be is not None:
            return be, "edge"
        return (wx, wy), None

    # ================================================================== typed input
    def _enter_entry_mode(self):
        self.typed = {"len": False, "ang": False}
        self.len_e.state(["!disabled"])
        self.ang_e.state(["!disabled"])
        self.len_var.set("")
        self.ang_var.set("")
        self._sync_entries()
        self.len_e.focus_set()

    def _leave_entry_mode(self):
        self.typed = {"len": False, "ang": False}
        self.len_var.set("")
        self.ang_var.set("")
        self.len_e.state(["disabled"])
        self.ang_e.state(["disabled"])
        try:
            self.canvas.focus_set()
        except Exception:
            pass

    def _typing(self, which, ev):
        if ev.keysym in ("Tab", "ISO_Left_Tab", "Return", "KP_Enter", "Escape", "Shift_L", "Shift_R",
                         "Control_L", "Control_R", "Alt_L", "Alt_R", "Left", "Right", "Home", "End"):
            return None
        edits = ev.keysym in ("BackSpace", "Delete") or (ev.char and ev.char in "0123456789.-+")
        if not edits:
            return "break"                                   # no letters in numeric boxes
        ent = self.len_e if which == "len" else self.ang_e
        if not self.typed[which]:
            ent.delete(0, "end")                             # first key replaces the live value
            self.typed[which] = True
        self.top.after_idle(self._after_typed, which)
        return None

    def _after_typed(self, which):
        var = self.len_var if which == "len" else self.ang_var
        if var.get().strip() == "":
            self.typed[which] = False
        self._sync_entries()
        self.redraw_overlay()

    def _tab_from(self, which):
        (self.ang_e if which == "len" else self.len_e).focus_set()
        return "break"

    def _enter_key(self):
        if self.tool.get() == TOOL_SPL and self.spl_pts and not (self.typed["len"] or self.typed["ang"]):
            self.finish_spline()                             # Enter with empty boxes finishes
        else:
            self.commit(from_click=False)
        return "break"

    def _val(self, which):
        if not self.typed[which]:
            return None
        try:
            return float((self.len_var if which == "len" else self.ang_var).get())
        except ValueError:
            return None

    def _compute_end(self):
        """End point from the start, the (snapped) mouse and whichever values were typed.
        Returns (end, length, angle_deg, followed_snap)."""
        sx, sy = self.start
        cx, cy = self.cursor if self.cursor is not None else self.start
        dx, dy = cx - sx, cy - sy
        L = math.hypot(dx, dy)
        A = math.degrees(math.atan2(dy, dx)) % 360.0
        lv, av = self._val("len"), self._val("ang")
        if lv is not None:
            if lv < 0:
                lv, A = -lv, (A + 180.0) % 360.0
            L = lv
        if av is not None:
            A = av % 360.0
        E = (sx + L * math.cos(math.radians(A)), sy + L * math.sin(math.radians(A)))
        followed = lv is None and av is None and self.cursor_kind is not None
        return E, L, A, followed

    def _sync_entries(self):
        if self.start is None:
            return
        _E, L, A, _f = self._compute_end()
        if not self.typed["len"]:
            self.len_var.set(f"{L:.3f}")
        if not self.typed["ang"]:
            self.ang_var.set(f"{A:.2f}")

    # ================================================================== mouse
    def on_motion(self, ev):
        self.cursor, self.cursor_kind = self.snap(ev.x, ev.y)
        wx, wy = self.cursor
        self.coords.set(f"x = {wx:.3f}   y = {wy:.3f}" + ("   [snap]" if self.cursor_kind else ""))
        if self.start is not None:
            self._sync_entries()
        self.redraw_overlay()

    def on_press(self, ev):
        if self.pair_act is not None:
            self.rb = [ev.x, ev.y, ev.x, ev.y]
            return
        if self.ov_mode is not None:
            self._image_press(ev)
            return
        if self.edit_sp is not None:
            self._edit_press(ev)
            return
        self.cursor, self.cursor_kind = self.snap(ev.x, ev.y)
        tool = self.tool.get()
        if tool == TOOL_SPL:
            pt = (float(self.cursor[0]), float(self.cursor[1]))
            if len(self.spl_pts) >= 3:
                fx, fy = self.w2c(*self.spl_pts[0])
                if math.hypot(fx - ev.x, fy - ev.y) < SNAP_VERTEX_PX:
                    self.finish_spline(closed=True)
                    return
            if self.spl_pts and math.hypot(pt[0] - self.spl_pts[-1][0],
                                           pt[1] - self.spl_pts[-1][1]) < 1e-9:
                return
            self.spl_pts.append(pt)
            self.start = pt
            self._enter_entry_mode()
            self.set_hint()
            self.redraw_overlay()
            return
        if tool == TOOL_ROT:
            self.push_undo()
            self.scene.add_actuator("rotary", self.cursor)
            self.tool.set(TOOL_LINE)
            self.set_hint()
            self.redraw()
            return
        if self.start is None:
            self.start = (float(self.cursor[0]), float(self.cursor[1]))
            self._enter_entry_mode()
            self.set_hint()
            self.redraw_overlay()
            return
        self.commit(from_click=True)

    def on_drag(self, ev):
        if self.rb is not None:
            self.rb[2], self.rb[3] = ev.x, ev.y
            self.redraw_overlay()
        elif self.ov_drag is not None:
            wx, wy = self.c2w(ev.x, ev.y)
            x0, y0, ax_, ay_ = self.ov_drag
            self.ov["x"], self.ov["y"] = x0 + wx - ax_, y0 + wy - ay_
            self.redraw()
        elif self.edit_sp is not None and self.drag_idx is not None:
            self.cursor, self.cursor_kind = self.snap(ev.x, ev.y)
            pts = list(self.edit_sp.pts)
            pts[self.drag_idx] = (float(self.cursor[0]), float(self.cursor[1]))
            self.edit_sp.pts = pts
            self.redraw()

    def on_release(self, ev):
        if self.ov_drag is not None:
            self.ov_drag = None
            self._sync_image_panel()
            return
        if self.drag_idx is not None:
            self.drag_idx = None
            self.redraw()
            return
        if self.rb is None:
            return
        self.rb[2], self.rb[3] = ev.x, ev.y
        self._finish_pairing()

    def on_escape(self):
        if self.ov_mode is not None:
            self.ov_mode, self.ov_pts, self.ov_drag = None, [], None
            self.set_hint()
            self.redraw()
        elif self.edit_sp is not None:
            self.edit_sp, self.drag_idx = None, None
            self.set_hint()
            self.redraw()
        elif self.spl_pts:
            self.finish_spline()
        elif self.pair_act is not None or self.rb is not None:
            self.pair_act, self.rb = None, None
            self.set_hint()
            self.redraw()
        elif self.start is not None:
            self.end_chain()

    def end_chain(self):
        if self.spl_pts:
            self.finish_spline()
            return
        self.start = None
        self._leave_entry_mode()
        if self.tool.get() != TOOL_LINE:
            self.tool.set(TOOL_LINE)
        self.set_hint()
        self.redraw_overlay()

    # ================================================================== committing
    def commit(self, from_click):
        if self.start is None:
            return
        E, L, A, followed = self._compute_end()
        if L < 1e-6:
            return
        tool = self.tool.get()
        if tool == TOOL_SPL:                                 # typed length / angle: a new point
            self.spl_pts.append((float(E[0]), float(E[1])))
            self.start = self.spl_pts[-1]
            self._enter_entry_mode()
            self.set_hint()
            self.redraw_overlay()
            return
        self.push_undo()
        if tool == TOOL_LIN:
            self.scene.add_actuator("linear", self.start, E)
            self.start = None
            self.tool.set(TOOL_LINE)
            self._leave_entry_mode()
        elif tool == TOOL_SRC:
            self.scene.add_source(self.start, E)
            self.start = None
            self.tool.set(TOOL_LINE)
            self._leave_entry_mode()
        else:
            self.scene.add_line(self.start, E, self.thickness())
            if from_click and followed:                      # clicked onto existing geometry
                self.start = None                            # (closing a body) -> chain ends
                self._leave_entry_mode()
            else:
                self.start = E
                self._enter_entry_mode()
        self.set_hint()
        self.redraw()

    # ================================================================== right-click menu
    def hit_test(self, cx, cy):
        best, bd = None, 1e9
        P = np.array([cx, cy])
        for a in self.scene.acts:
            if a.kind == "linear":
                A = np.array(self.w2c(*a.a))
                B = np.array(self.w2c(*a.b))
                q, _ = _seg_point(P, A, B)
                d = float(np.hypot(*(q - P)))
                lim = HIT_PX
            else:
                d = math.hypot(self.w2c(*a.a)[0] - cx, self.w2c(*a.a)[1] - cy)
                lim = HIT_PX + 4
            if d < lim and d < bd:
                best, bd = ("act", a), d
        if best is not None:
            return best
        for sc in self.scene.sources:
            A = np.array(self.w2c(*sc.p))
            B = np.array(self.w2c(*sc.q))
            q, _ = _seg_point(P, A, B)
            d = float(np.hypot(*(q - P)))
            if d < HIT_PX and d < bd:
                best, bd = ("source", sc), d
        if best is not None:
            return best
        for ln in self.scene.lines:
            A = np.array(self.w2c(*ln.p))
            B = np.array(self.w2c(*ln.q))
            q, _ = _seg_point(P, A, B)
            d = float(np.hypot(*(q - P)))
            if d < max(HIT_PX, ln.t * self.sc / 2) and d < bd:
                best, bd = ("line", ln), d
        for sp in self.scene.splines:
            C = self._curve(sp)
            for k in range(len(C) - 1):
                A = np.array(self.w2c(*C[k]))
                B = np.array(self.w2c(*C[k + 1]))
                q, _ = _seg_point(P, A, B)
                d = float(np.hypot(*(q - P)))
                if d < max(HIT_PX, sp.t * self.sc / 2) and d < bd:
                    best, bd = ("spline", sp), d
        return best

    def on_right(self, ev):
        if self.ov_mode is not None:
            self.on_escape()
            return
        if self.edit_sp is not None:
            k = self._near_point(self.edit_sp, ev.x, ev.y)
            if k is not None and len(self.edit_sp.pts) > 2:
                self.push_undo()
                self.edit_sp.pts = [p for i, p in enumerate(self.edit_sp.pts) if i != k]
                self.redraw()
            else:
                self.on_escape()
            return
        if self.spl_pts:
            self.finish_spline()
            return
        if self.pair_act is not None:
            self.on_escape()
            return
        if self.start is not None:
            self.end_chain()
            return
        hit = self.hit_test(ev.x, ev.y)
        if hit is None:
            return
        kind, obj = hit
        m = tk.Menu(self.top, tearoff=0)
        if kind == "act":
            m.add_command(label="Pair geometry...", command=lambda: self.begin_pair(obj))
            m.add_command(label="Clear pairing", command=lambda: self.clear_pairing(obj))
            if obj.kind == "rotary":
                m.add_command(label="Set angle range...", command=lambda: self.set_range(obj))
            m.add_separator()
            m.add_command(label="Delete actuator", command=lambda: self.delete_act(obj))
        elif kind == "source":
            m.add_command(label="Set flow...", command=lambda: self.set_source_flow(obj))
            m.add_command(label="Flip direction", command=lambda: self.flip_source(obj))
            m.add_separator()
            m.add_command(label="Delete source line", command=lambda: self.delete_line(obj))
        elif kind == "spline":
            m.add_command(label="Edit points", command=lambda: self.begin_edit(obj))
            m.add_command(label="Set thickness...", command=lambda: self.set_line_thickness(obj))
            m.add_separator()
            m.add_command(label="Delete spline", command=lambda: self.delete_line(obj))
        else:
            m.add_command(label="Set thickness...", command=lambda: self.set_line_thickness(obj))
            m.add_separator()
            m.add_command(label="Delete line", command=lambda: self.delete_line(obj))
        try:
            m.tk_popup(ev.x_root, ev.y_root)
        finally:
            m.grab_release()

    def begin_pair(self, act):
        self._cancel_ops()
        self.pair_act = act
        self.set_hint()
        self.redraw()

    def _finish_pairing(self):
        act = self.pair_act
        x0, y0, x1, y1 = self.rb
        crossing = x1 < x0                                   # dragged right -> left
        wxa, wya = self.c2w(min(x0, x1), max(y0, y1))
        wxb, wyb = self.c2w(max(x0, x1), min(y0, y1))
        rx0, rx1 = min(wxa, wxb), max(wxa, wxb)
        ry0, ry1 = min(wya, wyb), max(wya, wyb)
        self.rb = None
        picked = []
        for ob in self.scene.lines + self.scene.splines + self.scene.sources:
            P = self._curve(ob) if isinstance(ob, Spline) else np.array([ob.p, ob.q], float)
            if crossing:
                ok = any(_seg_hits_rect(P[k], P[k + 1], rx0, ry0, rx1, ry1)
                         for k in range(len(P) - 1))
            else:
                ok = all(rx0 <= v[0] <= rx1 and ry0 <= v[1] <= ry1 for v in P)
            if ok:
                picked.append(ob.id)
        self.pair_act = None
        if not picked:
            self.status.set("Nothing inside that box - right-click the actuator and try again.")
        else:
            self.push_undo()
            act.paired = sorted(set(act.paired) | set(picked))
            self.status.set(f"{act.name}: {len(act.paired)} item(s) paired.")
        self.redraw()

    def clear_pairing(self, act):
        if act.paired:
            self.push_undo()
            act.paired = []
        self.redraw()

    def set_range(self, act):
        lo = simpledialog.askfloat("Angle range", "Minimum angle (degrees):", parent=self.top,
                                   initialvalue=act.lo, minvalue=-360, maxvalue=360)
        if lo is None:
            return
        hi = simpledialog.askfloat("Angle range", "Maximum angle (degrees):", parent=self.top,
                                   initialvalue=act.hi, minvalue=-360, maxvalue=360)
        if hi is None:
            return
        if hi <= lo:
            messagebox.showerror("Angle range", "Maximum must be larger than minimum.", parent=self.top)
            return
        self.push_undo()
        act.lo, act.hi = float(lo), float(hi)

    def delete_act(self, act):
        self.push_undo()
        self.scene.delete_actuator(act.id)
        self.redraw()

    def delete_line(self, ln):
        self.push_undo()
        self.scene.delete_line(ln.id)
        self.redraw()

    # ---- splines
    def finish_spline(self, closed=False):
        pts = list(self.spl_pts)
        self.spl_pts = []
        self.start = None
        self._leave_entry_mode()
        if len(pts) >= 2 and not (closed and len(pts) < 3):
            self.push_undo()
            self.scene.add_spline(pts, self.thickness(), closed)
        self.set_hint()
        self.redraw()

    def begin_edit(self, sp):
        self._cancel_ops()
        self.edit_sp = sp
        self.set_hint()
        self.redraw()

    def _near_point(self, sp, cx, cy):
        best, bd = None, SNAP_VERTEX_PX
        for k, p in enumerate(sp.pts):
            px, py = self.w2c(*p)
            d = math.hypot(px - cx, py - cy)
            if d < bd:
                best, bd = k, d
        return best

    def _edit_press(self, ev):
        sp = self.edit_sp
        k = self._near_point(sp, ev.x, ev.y)
        if k is not None:                                    # grab a point
            self.push_undo()
            self.drag_idx = k
            return
        P = np.array(self.c2w(ev.x, ev.y))                   # click on the curve: insert a point
        C = self._curve(sp)
        d = min(float(np.hypot(*(_seg_point(P, C[i], C[i + 1])[0] - P))) for i in range(len(C) - 1))
        if d * self.sc > HIT_PX:
            return
        pts = [np.array(p, float) for p in sp.pts]
        n = len(pts)
        spans = range(n if sp.closed else n - 1)
        best = min(spans, key=lambda i: float(np.hypot(*(_seg_point(P, pts[i], pts[(i + 1) % n])[0] - P))))
        self.push_undo()
        new = list(sp.pts)
        new.insert(best + 1, (float(P[0]), float(P[1])))
        sp.pts = new
        self.drag_idx = best + 1
        self.redraw()

    # ---- source lines
    def set_source_flow(self, sc):
        top = tk.Toplevel(self.top)
        top.title("Source flow")
        top.transient(self.top)
        top.resizable(False, False)
        frm = ttk.Frame(top, padding=14)
        frm.grid()
        vs = [tk.StringVar(master=top, value=f"{v:g}") for v in (sc.M, sc.npr, sc.t0r)]
        labels = ["Mach number at the line (1 = throat)", "p0 / p_ambient  (nozzle pressure ratio)",
                  "T0 / T_ambient  (1 = cold, 3-10 = hot)"]
        for r, (lab, v) in enumerate(zip(labels, vs)):
            ttk.Label(frm, text=lab).grid(row=r, column=0, sticky="w", pady=4, padx=(0, 14))
            ttk.Entry(frm, textvariable=v, width=9).grid(row=r, column=1, sticky="w")
        msg = tk.StringVar(master=top, value="")
        tk.Label(frm, textvariable=msg, fg="#b00020", font=FONT).grid(row=3, column=0, columnspan=2,
                                                                      sticky="w")

        def ok(_e=None):
            try:
                M, npr, t0r = (float(v.get()) for v in vs)
            except ValueError:
                msg.set("Please enter numbers.")
                return
            if not (0.2 <= M <= 6.0 and 1.0 < npr <= 2000.0 and 0.2 <= t0r <= 20.0):
                msg.set("Mach 0.2-6, p0/p_amb 1-2000, T0/T_amb 0.2-20.")
                return
            k = 1 + 0.2 * M * M
            if npr / k ** 3.5 < 0.05:
                msg.set("Static pressure at the line < 0.05 p_amb: raise p0/p_amb.")
                return
            self.push_undo()
            sc.M, sc.npr, sc.t0r = M, npr, t0r
            top.destroy()
            self.redraw()

        bar = ttk.Frame(frm)
        bar.grid(row=4, column=0, columnspan=2, sticky="e", pady=(10, 0))
        ttk.Button(bar, text="OK", command=ok).grid(row=0, column=0, padx=(0, 6))
        ttk.Button(bar, text="Cancel", command=top.destroy).grid(row=0, column=1)
        top.bind("<Return>", ok)
        top.bind("<Escape>", lambda _e: top.destroy())
        top.geometry(f"+{self.top.winfo_rootx() + 160}+{self.top.winfo_rooty() + 120}")

    def flip_source(self, sc):
        self.push_undo()
        sc.flip = not sc.flip
        self.redraw()

    # ---- overlay image
    def open_image_panel(self):
        if self.ov_win is not None and self.ov_win.winfo_exists():
            self.ov_win.lift()
            return
        top = self.ov_win = tk.Toplevel(self.top)
        top.title("Overlay image")
        top.transient(self.top)
        top.resizable(False, False)
        frm = ttk.Frame(top, padding=12)
        frm.grid()
        self._ov_vars = {k: tk.StringVar(master=top) for k in ("w", "x", "y")}
        self._ov_show = tk.BooleanVar(master=top, value=self.ov.get("show", True))
        self._ov_alpha = tk.DoubleVar(master=top, value=self.ov.get("alpha", 0.5))
        ttk.Button(frm, text="Load...", command=self._image_load).grid(row=0, column=0, sticky="w")
        ttk.Button(frm, text="Remove", command=self._image_remove).grid(row=0, column=1, sticky="w")
        ttk.Checkbutton(frm, text="Show", variable=self._ov_show,
                        command=self._image_apply).grid(row=0, column=2, sticky="w", padx=(10, 0))
        for r, (k, lab) in enumerate((("w", "Width"), ("x", "X (left)"), ("y", "Y (bottom)")), start=1):
            ttk.Label(frm, text=lab).grid(row=r, column=0, sticky="w", pady=3)
            e = ttk.Entry(frm, textvariable=self._ov_vars[k], width=10)
            e.grid(row=r, column=1, sticky="w")
            e.bind("<Return>", lambda _e: self._image_apply())
        ttk.Label(frm, text="Opacity").grid(row=4, column=0, sticky="w", pady=3)
        ttk.Scale(frm, from_=0.1, to=1.0, variable=self._ov_alpha,
                  command=lambda _v: self._image_apply()).grid(row=4, column=1, columnspan=2, sticky="ew")
        ttk.Button(frm, text="Move (drag)", command=lambda: self._image_mode("move")).grid(
            row=5, column=0, sticky="w", pady=(8, 0))
        ttk.Button(frm, text="Scale by 2 points", command=lambda: self._image_mode("scale")).grid(
            row=5, column=1, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Button(frm, text="Fit to view", command=self._image_fit).grid(row=6, column=0, sticky="w",
                                                                         pady=(4, 0))
        ttk.Button(frm, text="Apply", command=self._image_apply).grid(row=6, column=1, sticky="w",
                                                                     pady=(4, 0))
        self._ov_msg = tk.StringVar(master=top, value="")
        ttk.Label(frm, textvariable=self._ov_msg, foreground="#777").grid(row=7, column=0, columnspan=3,
                                                                         sticky="w", pady=(6, 0))
        top.geometry(f"+{self.top.winfo_rootx() + 60}+{self.top.winfo_rooty() + 90}")
        self._sync_image_panel()

    def _sync_image_panel(self):
        if self.ov_win is None or not self.ov_win.winfo_exists():
            return
        for k in ("w", "x", "y"):
            self._ov_vars[k].set(f"{self.ov[k]:.3f}" if k in self.ov else "")
        self._ov_msg.set(self.ov.get("path", "no image loaded").replace("\\", "/").split("/")[-1])

    def _image_load(self):
        from tkinter import filedialog
        path = filedialog.askopenfilename(parent=self.ov_win, title="Overlay image",
                                          filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.gif *.tif *.tiff"),
                                                     ("All files", "*.*")])
        if not path:
            return
        try:
            from PIL import Image
            img = Image.open(path).convert("RGB")
        except Exception as e:
            messagebox.showerror("Overlay image", f"Could not open the image:\n{e}", parent=self.ov_win)
            return
        self.ov.update(path=path, img=img, show=True)
        self._ov_show.set(True)
        self._image_fit()

    def _image_fit(self):
        if "img" not in self.ov:
            return
        x0, x1, y0, y1 = self.view
        iw, ih = self.ov["img"].size
        w = min(0.9 * (x1 - x0), 0.9 * (y1 - y0) * iw / ih)
        self.ov.update(w=w, x=(x0 + x1) / 2 - w / 2, y=(y0 + y1) / 2 - w * ih / iw / 2)
        self._sync_image_panel()
        self.redraw()

    def _image_remove(self):
        for k in ("img", "path", "w", "x", "y"):
            self.ov.pop(k, None)
        self._ov_cache = self._ov_tk = None
        self._sync_image_panel()
        self.redraw()

    def _image_apply(self):
        if self.ov_win is not None and self.ov_win.winfo_exists():
            self.ov["show"] = bool(self._ov_show.get())
            self.ov["alpha"] = float(self._ov_alpha.get())
            if "img" in self.ov:
                try:
                    w = float(self._ov_vars["w"].get())
                    x = float(self._ov_vars["x"].get())
                    y = float(self._ov_vars["y"].get())
                    if w > 1e-6:
                        self.ov.update(w=w, x=x, y=y)
                except ValueError:
                    pass
        self.redraw()

    def _image_mode(self, mode):
        if "img" not in self.ov:
            return
        self._cancel_ops()
        self.ov_mode, self.ov_pts = mode, []
        self.set_hint()
        self.canvas.focus_set()
        self.redraw_overlay()

    def _image_press(self, ev):
        wx, wy = self.c2w(ev.x, ev.y)
        if self.ov_mode == "move":
            self.ov_drag = (self.ov["x"], self.ov["y"], wx, wy)
            return
        self.ov_pts.append((wx, wy))                         # scale by two points
        self.set_hint()
        self.redraw_overlay()
        if len(self.ov_pts) < 2:
            return
        (ax_, ay_), (bx_, by_) = self.ov_pts
        d = math.hypot(bx_ - ax_, by_ - ay_)
        self.ov_mode, self.ov_pts = None, []
        if d > 1e-9:
            true = simpledialog.askfloat("Scale image", "Real distance between the two points "
                                         "(world units):", parent=self.top, initialvalue=round(d, 4),
                                         minvalue=1e-6)
            if true:
                f = true / d
                self.ov["w"] *= f
                self.ov["x"] = ax_ + (self.ov["x"] - ax_) * f
                self.ov["y"] = ay_ + (self.ov["y"] - ay_) * f
        self._sync_image_panel()
        self.set_hint()
        self.redraw()

    def _draw_image(self):
        """Overlay picture: crop to the visible part, resize, blend with the canvas colour."""
        ov = self.ov
        if not ov.get("show", True) or "img" not in ov or "w" not in ov:
            return
        c = self.canvas
        W, H = c.winfo_width(), c.winfo_height()
        img = ov["img"]
        iw, ih = img.size
        x0, y0, w = ov["x"], ov["y"], ov["w"]
        h = w * ih / iw
        vx0, vy0 = self.c2w(0, H)
        vx1, vy1 = self.c2w(W, 0)
        ax0, ax1 = max(x0, vx0), min(x0 + w, vx1)
        ay0, ay1 = max(y0, vy0), min(y0 + h, vy1)
        if ax1 <= ax0 or ay1 <= ay0:
            return
        box = (int((ax0 - x0) / w * iw), int((y0 + h - ay1) / h * ih),
               int(math.ceil((ax1 - x0) / w * iw)), int(math.ceil((y0 + h - ay0) / h * ih)))
        box = (max(0, box[0]), max(0, box[1]), min(iw, max(box[2], box[0] + 1)), min(ih, max(box[3], box[1] + 1)))
        # world extent of the integer crop box, so the picture does not drift while zooming
        cx0, cx1 = x0 + box[0] / iw * w, x0 + box[2] / iw * w
        cy1, cy0 = y0 + h - box[1] / ih * h, y0 + h - box[3] / ih * h
        px = (int(round((cx1 - cx0) * self.sc)), int(round((cy1 - cy0) * self.sc)))
        if px[0] < 2 or px[1] < 2 or px[0] > 8000 or px[1] > 8000:
            return
        alpha = float(ov.get("alpha", 0.5))
        key = (id(img), box, px, round(alpha, 3))
        try:
            if self._ov_cache != key:
                from PIL import Image, ImageTk
                im = img.crop(box).resize(px, Image.BILINEAR)
                bg = Image.new("RGB", px, CANVAS_BG)
                self._ov_tk = ImageTk.PhotoImage(Image.blend(bg, im, alpha))
                self._ov_cache = key
            sx, sy = self.w2c(cx0, cy1)
            c.create_image(sx, sy, image=self._ov_tk, anchor="nw", tags="static")
        except Exception:
            self._ov_cache = None

    def set_line_thickness(self, ln):
        t = simpledialog.askfloat("Thickness", "Line thickness (0 = thin wall):", parent=self.top,
                                  initialvalue=ln.t, minvalue=0.0, maxvalue=10.0)
        if t is None:
            return
        self.push_undo()
        ln.t = float(t)
        self.redraw()

    # ================================================================== finish / close
    def finish(self):
        self.scene.renumber()
        self.on_finish(self.scene.copy())
        self.modified = False
        self.close()

    def request_close(self):
        if self.modified:
            ans = messagebox.askyesnocancel("TBG - Draw",
                                            "Send your sketch to the simulation before closing?\n\n"
                                            "Yes = finish   No = discard changes   Cancel = keep drawing",
                                            parent=self.top)
            if ans is None:
                return
            if ans:
                self.finish()
                return
        self.close()

    def close(self):
        try:
            self.top.destroy()
        except Exception:
            pass

    # ================================================================== drawing
    def redraw(self):
        self.redraw_static()
        self.redraw_overlay()

    def _grid_step(self):
        for s in (0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500):
            if s * self.sc >= 45:
                return s
        return 1000

    def redraw_static(self):
        c = self.canvas
        c.delete("static")
        W, H = c.winfo_width(), c.winfo_height()
        self._draw_image()
        # grid (display only - nothing snaps to it)
        step = self._grid_step()
        xa, ya = self.c2w(0, H)
        xb, yb = self.c2w(W, 0)
        k = math.floor(xa / step)
        while k * step <= xb:
            x = k * step
            cx, _ = self.w2c(x, 0)
            c.create_line(cx, 0, cx, H, fill="#ececec", tags="static")
            c.create_text(cx + 3, H - 10, text=f"{x:g}", anchor="w", fill="#aaa", font=("Arial", 7),
                          tags="static")
            k += 1
        k = math.floor(ya / step)
        while k * step <= yb:
            y = k * step
            _, cy = self.w2c(0, y)
            c.create_line(0, cy, W, cy, fill="#ececec", tags="static")
            c.create_text(4, cy - 8, text=f"{y:g}", anchor="w", fill="#aaa", font=("Arial", 7),
                          tags="static")
            k += 1
        # ground plane (read from the simulation window)
        if self.plane_getter():
            _, gy = self.w2c(0, 0)
            c.create_rectangle(0, gy, W, max(H, gy + 1), fill="#d6d6d6", outline="", tags="static")
            c.create_line(0, gy, W, gy, fill="#222", width=2, tags="static")
            c.create_text(8, gy + 12, text="ground plane (Edit > Setup in the simulation window)",
                          anchor="w", fill="#777", font=FONT, tags="static")
        # simulation view window
        x0, x1, y0, y1 = self.view
        ax_, ay_ = self.w2c(x0, y1)
        bx_, by_ = self.w2c(x1, y0)
        c.create_rectangle(ax_, ay_, bx_, by_, outline="#5b8def", dash=(6, 4), tags="static")
        c.create_text(ax_ + 6, ay_ + 10, text="simulation view", anchor="w", fill="#5b8def",
                      font=FONT, tags="static")
        # origin
        ox_, oy_ = self.w2c(0, 0)
        c.create_line(ox_ - 6, oy_, ox_ + 6, oy_, fill="#999", tags="static")
        c.create_line(ox_, oy_ - 6, ox_, oy_ + 6, fill="#999", tags="static")

        # pairing halos, then lines
        halo = {}
        for a in self.scene.acts:
            col = self.act_color(a)
            for i in a.paired:
                halo.setdefault(i, []).append(col)
        for ob in self.scene.lines + self.scene.splines + self.scene.sources:
            P = self._curve(ob) if isinstance(ob, Spline) else [ob.p, ob.q]
            flat = [v for p in P for v in self.w2c(*p)]
            width = max(2.5, getattr(ob, "t", 0.0) * self.sc)
            for n, col in enumerate(halo.get(ob.id, [])):
                c.create_line(*flat, width=width + 10 + 6 * n, fill=col,
                              stipple="gray50", capstyle="round", joinstyle="round", tags="static")
        for ln in self.scene.lines:
            x0_, y0_ = self.w2c(*ln.p)
            x1_, y1_ = self.w2c(*ln.q)
            if ln.t > 1e-9:
                c.create_line(x0_, y0_, x1_, y1_, width=max(2.0, ln.t * self.sc), fill="#2b2b2b",
                              capstyle="butt", tags="static")
                c.create_line(x0_, y0_, x1_, y1_, width=1, fill="#8a8a8a", tags="static")
            else:
                c.create_line(x0_, y0_, x1_, y1_, width=2.5, fill="black", capstyle="round",
                              tags="static")
            for (vx, vy) in ((x0_, y0_), (x1_, y1_)):
                c.create_rectangle(vx - 2.5, vy - 2.5, vx + 2.5, vy + 2.5, fill="#fff",
                                   outline="#555", tags="static")
        for sp in self.scene.splines:
            flat = [v for p in self._curve(sp) for v in self.w2c(*p)]
            if len(flat) >= 4:
                if sp.t > 1e-9:
                    c.create_line(*flat, width=max(2.0, sp.t * self.sc), fill="#2b2b2b",
                                  capstyle="butt", joinstyle="round", tags="static")
                    c.create_line(*flat, width=1, fill="#8a8a8a", tags="static")
                else:
                    c.create_line(*flat, width=2.5, fill="black", capstyle="round",
                                  joinstyle="round", tags="static")
            editing = sp is self.edit_sp
            for k, p in enumerate(sp.pts):
                vx, vy = self.w2c(*p)
                r = 5 if editing else 3
                c.create_polygon(vx, vy - r, vx + r, vy, vx, vy + r, vx - r, vy,
                                 fill="#ffd43b" if editing else "#fff",
                                 outline="#e67700" if editing else "#555", tags="static")
        for sc in self.scene.sources:
            A, B = np.array(sc.p, float), np.array(sc.q, float)
            e = B - A
            L = float(np.hypot(*e))
            if L < 1e-12:
                continue
            n = np.array([e[1], -e[0]]) / L * (-1 if sc.flip else 1)
            c.create_line(*self.w2c(*A), *self.w2c(*B), width=4, fill=SRC_COLOR, capstyle="butt",
                          tags="static")
            m = max(2, int(L * self.sc / 28) + 1)
            for t in np.linspace(0.12, 0.88, m):
                o = A + t * e
                c.create_line(*self.w2c(*o), *self.w2c(*(o + n * 22 / self.sc)), fill=SRC_COLOR,
                              width=2, arrow="last", arrowshape=(8, 10, 4), tags="static")
            mx, my = self.w2c(*(0.5 * (A + B) - n * 8 / self.sc))
            anchor = ("e" if n[0] > 0.5 else "w" if n[0] < -0.5 else "n" if n[1] > 0 else "s")
            c.create_text(mx, my, anchor=anchor, text=f"M {sc.M:g}  p0/pa {sc.npr:g}" + (f"  T0/Ta {sc.t0r:g}" if abs(sc.t0r - 1) > 1e-9 else ""),
                          fill=SRC_COLOR, font=("Microsoft Sans Serif", 9, "bold"), tags="static")

        # actuators
        for a in self.scene.acts:
            col = self.act_color(a)
            if a.kind == "linear":
                A, B = self.w2c(*a.a), self.w2c(*a.b)
                c.create_line(*A, *B, width=3, fill=col, dash=(9, 5), arrow="both",
                              arrowshape=(11, 13, 4), tags="static")
                mx, my = (A[0] + B[0]) / 2, (A[1] + B[1]) / 2
                L = math.hypot(a.b[0] - a.a[0], a.b[1] - a.a[1])
                c.create_text(mx, my - 14, text=f"{a.name}  (length {L:.2f})", fill=col,
                              font=("Microsoft Sans Serif", 9, "bold"), tags="static")
                ref = (mx, my)
            else:
                cx_, cy_ = self.w2c(*a.a)
                c.create_oval(cx_ - 9, cy_ - 9, cx_ + 9, cy_ + 9, width=2.5, outline=col,
                              fill="#fff", tags="static")
                c.create_line(cx_ - 9, cy_, cx_ + 9, cy_, fill=col, width=1.5, tags="static")
                c.create_line(cx_, cy_ - 9, cx_, cy_ + 9, fill=col, width=1.5, tags="static")
                c.create_text(cx_ + 14, cy_ - 26, text=f"{a.name}  ({a.lo:g}\u00b0 to {a.hi:g}\u00b0)",
                              anchor="w", fill=col, font=("Microsoft Sans Serif", 9, "bold"),
                              tags="static")
                ref = (cx_, cy_)
            for i in a.paired:                               # connectors to the paired geometry
                ob = self.scene.item(i)
                if ob is None:
                    continue
                P = self._pts(ob)
                mx2, my2 = self.w2c(sum(p[0] for p in P) / len(P), sum(p[1] for p in P) / len(P))
                c.create_line(ref[0], ref[1], mx2, my2, fill=col, dash=(2, 4), tags="static")

    def redraw_overlay(self):
        c = self.canvas
        c.delete("ov")
        # snap marker
        if self.cursor is not None and self.cursor_kind and self.rb is None:
            px, py = self.w2c(*self.cursor)
            if self.cursor_kind == "vertex":
                c.create_rectangle(px - 7, py - 7, px + 7, py + 7, outline="#16a34a", width=2,
                                   tags="ov")
            else:
                c.create_polygon(px, py - 8, px + 8, py, px, py + 8, px - 8, py, outline="#16a34a",
                                 fill="", width=2, tags="ov")
        # centre guides: dotted line from the reference point to the cursor
        if self.cursor is not None and self.guides and self.rb is None and self.guides_on.get():
            px, py = self.w2c(*self.cursor)
            for (rx, ry), mid in self.guides:
                gx, gy = self.w2c(rx, ry)
                col = GUIDE_MID if mid else GUIDE_PT
                c.create_line(gx, gy, px, py, fill=col, dash=(2, 4), width=1.5, tags="ov")
                r = 4 if mid else 3
                c.create_oval(gx - r, gy - r, gx + r, gy + r, outline=col, width=1.5, fill="#fff",
                              tags="ov")
        # image scale points
        for (wx, wy) in self.ov_pts:
            px, py = self.w2c(wx, wy)
            c.create_oval(px - 5, py - 5, px + 5, py + 5, outline="#d6336c", width=2, tags="ov")
        # live line / angle arc
        if self.start is not None:
            E, L, A, _f = self._compute_end()
            sx, sy = self.w2c(*self.start)
            ex, ey = self.w2c(*E)
            tool = self.tool.get()
            if tool == TOOL_LIN:
                c.create_line(sx, sy, ex, ey, width=3, fill="#e8590c", dash=(9, 5), tags="ov")
            elif tool == TOOL_SRC:
                c.create_line(sx, sy, ex, ey, width=4, fill=SRC_COLOR, tags="ov")
            elif tool == TOOL_SPL:
                pts = list(self.spl_pts) + ([E] if L > 1e-9 else [])
                C = spline_curve(pts) if len(pts) >= 2 else np.array(pts)
                flat = [v for p in C for v in self.w2c(*p)]
                if len(flat) >= 4:
                    t = self.thickness()
                    if t > 1e-9:
                        c.create_line(*flat, width=max(2.0, t * self.sc), fill="#7aa7e6",
                                      capstyle="butt", joinstyle="round", tags="ov")
                    c.create_line(*flat, width=2, fill="#1c63d5", tags="ov")
                for p in self.spl_pts:
                    vx, vy = self.w2c(*p)
                    c.create_polygon(vx, vy - 4, vx + 4, vy, vx, vy + 4, vx - 4, vy, fill="#fff",
                                     outline="#1c63d5", tags="ov")
            else:
                t = self.thickness()
                if t > 1e-9:
                    c.create_line(sx, sy, ex, ey, width=max(2.0, t * self.sc), fill="#7aa7e6",
                                  capstyle="butt", tags="ov")
                c.create_line(sx, sy, ex, ey, width=2, fill="#1c63d5", tags="ov")
            # reference ray (+x) and the arc the angle is measured on
            ref_len = max(70, min(160, math.hypot(ex - sx, ey - sy)))
            c.create_line(sx, sy, sx + ref_len, sy, fill="#999", dash=(4, 3), tags="ov")
            r = min(80, max(30, 0.35 * math.hypot(ex - sx, ey - sy)))
            if L > 1e-9:
                c.create_arc(sx - r, sy - r, sx + r, sy + r, start=0, extent=A, style="arc",
                             outline="#e67700", width=2, tags="ov")
                mid = math.radians(A / 2)
                lx, ly = sx + (r + 28) * math.cos(mid), sy - (r + 28) * math.sin(mid)
                c.create_text(lx, ly, text=f"A = {A:.1f}\u00b0", fill="#e67700",
                              font=("Microsoft Sans Serif", 10, "bold"), tags="ov")
                if tool not in (TOOL_LIN, TOOL_SRC):
                    # theta = flow deflection angle the solver uses: the line's inclination to the
                    # horizontal (flow runs along -x), folded into 0..90 deg
                    th = math.degrees(math.asin(min(1.0, abs(math.sin(math.radians(A))))))
                    c.create_text(lx, ly + 16, text=f"\u03b8 = {th:.1f}\u00b0", fill="#0b7285",
                                  font=("Microsoft Sans Serif", 10, "bold"), tags="ov")
            mx, my = (sx + ex) / 2, (sy + ey) / 2
            c.create_text(mx, my - 14, text=f"{L:.3f}", fill="#1c63d5",
                          font=("Microsoft Sans Serif", 10, "bold"), tags="ov")
            c.create_oval(sx - 4, sy - 4, sx + 4, sy + 4, fill="#16a34a", outline="", tags="ov")
        # rubber band
        if self.rb is not None:
            x0, y0, x1, y1 = self.rb
            col = "#16a34a" if x1 < x0 else "#1c63d5"
            c.create_rectangle(x0, y0, x1, y1, outline=col, dash=(5, 3) if x1 < x0 else (),
                               width=2, tags="ov")
