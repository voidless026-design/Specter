#!/usr/bin/env bash
# Stops and removes the E.V. systemd --user service, its virtualenv, and
# (optionally) its config and stored memory. Run as your normal user.
set -euo pipefail

# Install dir doubles as E.V.'s XDG data dir (platformdirs default for the
# "ev-assistant" app name), so it holds the venv, the sqlite memory, and the
# downloaded speech model all together.
INSTALL_DIR="$HOME/.local/share/ev-assistant"
CONFIG_DIR="$HOME/.config/ev-assistant"
UNIT_DIR="$HOME/.config/systemd/user"

systemctl --user stop ev-assistant.service 2>/dev/null || true
systemctl --user disable ev-assistant.service 2>/dev/null || true
rm -f "$UNIT_DIR/ev-assistant.service"
systemctl --user daemon-reload

read -r -p "Delete $INSTALL_DIR (venv, memory, downloaded speech model) and $CONFIG_DIR (config, API key)? [y/N] " reply
if [[ "$reply" =~ ^[Yy]$ ]]; then
  rm -rf "$INSTALL_DIR" "$CONFIG_DIR"
  echo "Removed everything."
else
  echo "Service uninstalled. Left $INSTALL_DIR and $CONFIG_DIR in place."
fi
