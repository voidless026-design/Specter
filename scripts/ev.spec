# PyInstaller spec for E.V.'s standalone executable (Windows .exe or Fedora
# binary). PyInstaller does not cross-compile, so build it ON the target OS:
#   Windows: scripts\build-windows.ps1   (produces dist\ev\ev.exe)
#   Fedora:  scripts/build-bundle-fedora.sh (produces dist/ev/ev)
# Or directly, from a venv with this project + pyinstaller installed:
#   pyinstaller scripts/ev.spec
#
# This is a --onedir build (a folder, not a single file): it starts faster
# and is far easier to debug than --onefile when a hidden import turns out
# missing. To hand someone a single file, zip the dist/ev folder.
#
# Honesty note: this spec has NOT been build-verified on a real Windows or
# Fedora host - it was written in a Linux container with no GUI/audio host
# available, from known PyInstaller gotchas with this dependency set:
#   - uvicorn and edge_tts pull implementation modules via dynamic imports
#     that PyInstaller's static analysis can miss.
#   - vosk and sounddevice each ship a native shared library (libvosk /
#     bundled PortAudio) alongside their Python code, only bundled if asked.
#   - the GUI's index.html is data, not code, and must be added explicitly.
# The collect_* calls and datas below cover these. If the built binary fails
# at runtime with ModuleNotFoundError or a missing-file error, add the named
# module to hiddenimports or the file to datas - see the README Troubleshooting.

import os

from PyInstaller.utils.hooks import collect_dynamic_libs, collect_submodules

REPO_ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))

hiddenimports = (
    collect_submodules("uvicorn")
    + collect_submodules("edge_tts")
    + collect_submodules("anthropic")
)
binaries = collect_dynamic_libs("vosk") + collect_dynamic_libs("sounddevice")

# Bundle the GUI so `ev gui` / the daemon can serve it from inside the binary.
datas = [(os.path.join(REPO_ROOT, "ev_assistant", "gui", "index.html"), "ev_assistant/gui")]

a = Analysis(
    [os.path.join(REPO_ROOT, "ev_assistant", "__main__.py")],
    pathex=[REPO_ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="ev", console=True)
coll = COLLECT(exe, a.binaries, a.datas, name="ev")
