import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # find the sibling modules

from tbg_common import (ICON_FILE, SPLASH_FILE, SPLASH_MS, SPLASH_TITLE, asset_path,
                        set_app_id)


class Splash:
    """Borderless, centred splash window. It is painted immediately, stays up while the heavy
    modules load, and finish() keeps it up until SPLASH_MS have passed in total. If tkinter or a
    display is unavailable it silently does nothing."""

    def __init__(self, ms=SPLASH_MS):
        self.t0 = time.time()
        self.ms = ms
        self.root = None
        try:
            import tkinter as tk
            root = tk.Tk()
            root.overrideredirect(True)
            root.attributes("-topmost", True)
            img = None
            path = asset_path(SPLASH_FILE)
            if path:
                try:
                    img = tk.PhotoImage(file=path)
                except Exception:
                    img = None
            if img is not None:
                tk.Label(root, image=img, bd=0).pack()
                root._img = img
            else:                                           # text fallback
                root.configure(bg="#0d0f13")
                tk.Label(root, text=SPLASH_TITLE, fg="#d7dde6", bg="#0d0f13",
                         font=("Microsoft Sans Serif", 40, "bold"), padx=90, pady=50).pack()
                tk.Label(root, text="loading...", fg="#6f7887", bg="#0d0f13",
                         font=("Microsoft Sans Serif", 10), pady=10).pack()
            icon = asset_path(ICON_FILE)
            if icon and sys.platform.startswith("win"):
                try:
                    root.iconbitmap(icon)
                except Exception:
                    pass
            root.update_idletasks()
            w, h = root.winfo_reqwidth(), root.winfo_reqheight()
            x = (root.winfo_screenwidth() - w) // 2
            y = (root.winfo_screenheight() - h) // 2
            root.geometry(f"{w}x{h}+{x}+{y}")
            root.update()
            self.root = root
        except Exception:
            self.root = None

    def pump(self):
        if self.root is not None:
            try:
                self.root.update()
            except Exception:
                self.root = None

    def finish(self):
        """Wait until the splash has been visible for the full time, then close it."""
        if self.root is None:
            return
        while (time.time() - self.t0) * 1000.0 < self.ms:
            self.pump()
            if self.root is None:
                return
            time.sleep(0.02)
        try:
            self.root.destroy()
        except Exception:
            pass
        self.root = None


def main():
    set_app_id()                  # before any window exists, so the taskbar uses our icon
    splash = Splash()
    import tbg_sim                # heavy imports (numpy / matplotlib) happen behind the splash
    splash.pump()
    splash.finish()
    tbg_sim.run()


if __name__ == "__main__":
    main()
