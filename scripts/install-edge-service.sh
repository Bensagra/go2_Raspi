#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICE_NAME="${GO2_SERVICE_NAME:-go2-edge}"
SERVICE_USER="${GO2_SERVICE_USER:-${SUDO_USER:-$(id -un)}}"
SERVICE_FILE="/etc/systemd/system/$SERVICE_NAME.service"

if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
  echo "Este paso modifica systemd. Ejecútalo así:"
  echo "  sudo $PROJECT_ROOT/go2 service-install"
  exit 1
fi

if [[ ! -x "$PROJECT_ROOT/go2" ]]; then
  echo "No encuentro el lanzador: $PROJECT_ROOT/go2" >&2
  exit 1
fi

temporary_file="$(mktemp)"
trap 'rm -f "$temporary_file"' EXIT

cat >"$temporary_file" <<EOF
[Unit]
Description=Go2 Edge Gateway
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$SERVICE_USER
WorkingDirectory=$PROJECT_ROOT
Environment=GO2_CONFIG_FILE=$PROJECT_ROOT/config/.env
ExecStart=$PROJECT_ROOT/go2 edge
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

install -m 0644 "$temporary_file" "$SERVICE_FILE"
systemctl daemon-reload
systemctl enable --now "$SERVICE_NAME.service"

echo "Servicio instalado: $SERVICE_NAME"
echo "Estado: sudo systemctl status $SERVICE_NAME"
echo "Logs:   journalctl -u $SERVICE_NAME -f"
