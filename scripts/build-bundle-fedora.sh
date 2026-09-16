#!/usr/bin/env bash
# Builds a standalone E.V. binary bundle on Fedora - a self-contained folder
# (dist/ev/) you can copy to another Fedora PC that doesn't have Python set
# up. PyInstaller can't cross-compile, so run this ON a Fedora machine.
#
#   bash scripts/build-bundle-fedora.sh
#
# The other PC still needs the system audio libraries (portaudio, espeak-ng)
# and, for the best voice, internet for edge-tts. Copy your exported brain
# with `ev export` / `ev import` to bring your config and memory along.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

echo "==> Ensuring build tools (portaudio/espeak needed at runtime too)..."
sudo dnf install -y python3 python3-pip portaudio espeak-ng

echo "==> Creating build virtualenv..."
python3 -m venv .venv-build
.venv-build/bin/pip install --upgrade pip -q
.venv-build/bin/pip install . pyinstaller -q

echo "==> Running PyInstaller..."
.venv-build/bin/pyinstaller scripts/ev.spec --noconfirm --distpath dist --workpath build

cat <<EOF

Built: $REPO_DIR/dist/ev/ev

To move E.V. to another Fedora PC:
  1. On this PC:   ev export ev-brain.tar.gz
  2. Copy the whole dist/ev/ folder AND ev-brain.tar.gz to the other PC.
  3. On the other PC: sudo dnf install portaudio espeak-ng
  4. Run: ./ev/ev import ev-brain.tar.gz   then   ./ev/ev daemon

To point a laptop at this PC's brain instead of bundling, set remote_host /
remote_port under [control_api] in the laptop's config (see the README's
"Remote brain" section).
EOF
