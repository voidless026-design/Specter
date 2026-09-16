#!/usr/bin/env bash
# Installs E.V. as a systemd --user service on Fedora, so she starts when
# you log in and keeps listening for "Hey E.V." in the background.
#
# Run as your normal user (not root) from anywhere:
#   bash scripts/install-fedora.sh
set -euo pipefail

if [ "$(id -u)" -eq 0 ]; then
  echo "Run this as your normal user, not root - it installs a systemd --user service" >&2
  echo "and needs your desktop audio session, which root doesn't have." >&2
  exit 1
fi

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INSTALL_DIR="$HOME/.local/share/ev-assistant"
CONFIG_DIR="$HOME/.config/ev-assistant"
UNIT_DIR="$HOME/.config/systemd/user"

echo "==> Installing system packages (sudo password may be requested)..."
# portaudio + espeak-ng + alsa-utils: microphone capture and offline voice.
# ffmpeg-free: ffplay, to play the Australian neural voice (edge-tts mp3).
# playerctl: media control (play/pause/next) via voice.
# xdg-utils: xdg-open, so E.V. can open websites.
sudo dnf install -y python3 python3-pip portaudio espeak-ng alsa-utils \
  ffmpeg-free playerctl xdg-utils

echo "==> Creating virtualenv at $INSTALL_DIR"
mkdir -p "$INSTALL_DIR"
python3 -m venv "$INSTALL_DIR/venv"
"$INSTALL_DIR/venv/bin/pip" install --upgrade pip -q
"$INSTALL_DIR/venv/bin/pip" install "$REPO_DIR" -q

echo "==> Writing default config and generating the control token"
# `ev init` writes both config.toml and the env file (with a fresh
# EV_CONTROL_TOKEN and a placeholder API key) under $CONFIG_DIR.
"$INSTALL_DIR/venv/bin/ev" init
chmod 600 "$CONFIG_DIR/env" 2>/dev/null || true

# E.V.'s default brain is a local Ollama model - no API key, no cost, works
# offline. Offer to install it so she can converse out of the box.
if ! command -v ollama >/dev/null 2>&1; then
  echo ""
  read -r -p "Install Ollama so E.V. has a free local brain (no API key)? [Y/n] " reply
  if [[ ! "$reply" =~ ^[Nn]$ ]]; then
    echo "==> Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh || echo "  (Ollama install failed - install it yourself from https://ollama.com)"
  fi
fi
if command -v ollama >/dev/null 2>&1; then
  echo "==> Pulling a tool-capable local model (llama3.1, ~4.7GB - Ctrl-C to skip)..."
  ollama pull llama3.1 || echo "  (skip/failed - run 'ollama pull llama3.1' later)"
fi

echo "==> Installing systemd --user unit"
mkdir -p "$UNIT_DIR"
sed \
  -e "s#__VENV_BIN__#$INSTALL_DIR/venv/bin#g" \
  -e "s#__INSTALL_DIR__#$INSTALL_DIR#g" \
  -e "s#__CONFIG_DIR__#$CONFIG_DIR#g" \
  "$REPO_DIR/scripts/ev-assistant.service" > "$UNIT_DIR/ev-assistant.service"

systemctl --user daemon-reload
systemctl --user enable ev-assistant.service

cat <<EOF

Setup done. One manual step left: edit $CONFIG_DIR/env and set ANTHROPIC_API_KEY
(get one at https://console.anthropic.com/settings/keys), then:

  systemctl --user start ev-assistant
  systemctl --user status ev-assistant
  journalctl --user -u ev-assistant -f      # watch the logs / first-run model download

Say "Hey E.V." (or "Yo E.V.", or just "E.V.") to talk to her once she's
running. From any shell on this machine (SSH included):

  $INSTALL_DIR/venv/bin/ev ask "what's the weather like"
  $INSTALL_DIR/venv/bin/ev gui            # open the cyberpunk console
  $INSTALL_DIR/venv/bin/ev mic-test       # check your microphone + wake word

Add $INSTALL_DIR/venv/bin to your PATH to just run \`ev\` directly:

  echo 'export PATH="$INSTALL_DIR/venv/bin:\$PATH"' >> ~/.bashrc && source ~/.bashrc

To change personality, voice, permissions, wake names, or feeds, edit
$CONFIG_DIR/config.toml (or use the SETTINGS tab in \`ev gui\`), then restart:

  systemctl --user restart ev-assistant
EOF
