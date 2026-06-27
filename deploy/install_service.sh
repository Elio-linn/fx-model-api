#!/usr/bin/env bash
# Install & start the FX Model API systemd service.
# Usage: sudo bash deploy/install_service.sh
set -euo pipefail

SRC="$(cd "$(dirname "$0")" && pwd)/fx-model-api.service"
DEST="/etc/systemd/system/fx-model-api.service"

echo "Installing $SRC -> $DEST"
install -m 644 "$SRC" "$DEST"

systemctl daemon-reload
systemctl enable fx-model-api
systemctl restart fx-model-api

echo
systemctl --no-pager status fx-model-api || true
echo
echo "Follow logs with:  journalctl -u fx-model-api -f"
