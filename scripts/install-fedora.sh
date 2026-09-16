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
sudo dnf install -y python3 python3-pip portaudio espeak-ng alsa-utils

echo "==> Creating virtualenv at $INSTALL_DIR"
mkdir -p "$INSTALL_DIR"
python3 -m venv "$INSTALL_DIR/venv"
"$INSTALL_DIR/venv/bin/pip" install --upgrade pip -q
"$INSTALL_DIR/venv/bin/pip" install "$REPO_DIR" -q

echo "==> Writing default config"
"$INSTALL_DIR/venv/bin/ev" init

mkdir -p "$CONFIG_DIR"
if [ ! -f "$CONFIG_DIR/env" ]; then
  cp "$REPO_DIR/.env.example" "$CONFIG_DIR/env"
  TOKEN="$(python3 -c 'import secrets; print(secrets.token_hex(32))')"
  sed -i "s/change-me-to-a-random-value/$TOKEN/" "$CONFIG_DIR/env"
  echo "    Wrote $CONFIG_DIR/env with a freshly generated EV_CONTROL_TOKEN."
  echo "    You still need to edit it and set your real ANTHROPIC_API_KEY."
else
  echo "    $CONFIG_DIR/env already exists - leaving it as-is."
fi
chmod 600 "$CONFIG_DIR/env"

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

Say "Hey E.V." to talk to her once she's running, or from any SSH session on
this machine:

  $INSTALL_DIR/venv/bin/ev ask "what's the weather like"

Add $INSTALL_DIR/venv/bin to your PATH to just run \`ev\` directly. To review
or change personality, wake phrases, or news feeds, edit:

  $CONFIG_DIR/config.toml
EOF
