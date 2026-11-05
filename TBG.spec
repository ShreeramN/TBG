# -*- mode: python ; coding: utf-8 -*-
# TBG build file. Put it in TBG_alphabuild (next to source\ and assets\) and run:
#     pyinstaller --clean --noconfirm TBG.spec
# Result: dist\TBG\TBG.exe  (ship the whole dist\TBG folder)
import os

ROOT = os.path.abspath(SPECPATH)                 # the folder this file is in
SRC = os.path.join(ROOT, "source")
ASSETS = os.path.join(ROOT, "assets")
ICON = os.path.join(ASSETS, "icon.ico")

a = Analysis(
    [os.path.join(SRC, "tbg.py")],
    pathex=[SRC],
    binaries=[],
    datas=[(ASSETS, "assets")] if os.path.isdir(ASSETS) else [],
    hiddenimports=[                              # modules TBG loads only when a menu is used
        "tbg_common", "tbg_sim", "tbg_draw", "tbg_pmap", "tbg_cfd", "tbg_eddy",
        "PIL.ImageTk", "PIL._tkinter_finder", "matplotlib.backends.backend_tkagg",
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=["PyQt5", "PyQt6", "PySide2", "PySide6", "IPython", "pandas", "scipy"],  # not used: smaller build
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="TBG",
    debug=False,
    strip=False,
    upx=False,
    console=False,                               # no console window (set True to see errors)
    icon=ICON if os.path.exists(ICON) else None,
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="TBG")
