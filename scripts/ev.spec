# PyInstaller spec for E.V.'s Windows executable.
#
# Must be BUILT ON WINDOWS - PyInstaller does not cross-compile. Run (via
# scripts/build-windows.ps1, or directly from an activated venv that has
# this project + pyinstaller installed):
#   pyinstaller scripts\ev.spec
# The result is dist\ev\ev.exe (a --onedir build: slightly more files than
# a single .exe, but starts faster and is far easier to debug than --onefile
# when a hidden import turns out to be missing).
#
# Honesty note: this spec has NOT been build-verified on a real Windows
# machine - it was written in a Linux container with no Windows host
# available, from known PyInstaller gotchas with this dependency set:
#   - uvicorn picks its event loop / HTTP protocol implementation via
#     dynamic imports, which PyInstaller's static analysis can miss.
#   - pyttsx3 loads its platform driver (sapi5 on Windows) via
#     importlib.import_module(f"pyttsx3.drivers.{name}") - a string-built
#     import PyInstaller cannot see statically at all.
#   - vosk and sounddevice each ship a native shared library
#     (libvosk.dll / a bundled PortAudio DLL) alongside their Python code,
#     which PyInstaller only bundles if told to explicitly.
# collect_submodules / collect_dynamic_libs below cover all three. If
# ev.exe still fails at runtime with ModuleNotFoundError or a DLL load
# error, that means one more hidden import or binary - see the README's
# Troubleshooting section for how to add it.

import os

from PyInstaller.utils.hooks import collect_dynamic_libs, collect_submodules

REPO_ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))

hiddenimports = collect_submodules("uvicorn") + collect_submodules("pyttsx3.drivers")
binaries = collect_dynamic_libs("vosk") + collect_dynamic_libs("sounddevice")

a = Analysis(
    [os.path.join(REPO_ROOT, "ev_assistant", "__main__.py")],
    pathex=[REPO_ROOT],
    binaries=binaries,
    datas=[],
    hiddenimports=hiddenimports,
    hookspath=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ev",
    console=True,
)

coll = COLLECT(exe, a.binaries, a.datas, name="ev")
