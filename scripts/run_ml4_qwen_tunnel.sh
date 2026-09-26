#!/usr/bin/env bash
set -euo pipefail

# Makes ML4 image workers available to the bot on local-only ports.
# The key is dedicated to this tunnel and has no interactive shell use.
readonly ML4_HOST="${ML4_HOST:-gater.frccsc.ru}"
readonly ML4_PORT="${ML4_PORT:-9191}"
readonly ML4_USER="${ML4_USER:-ggorbov}"
readonly TUNNEL_KEY="${ML4_TUNNEL_KEY:-$HOME/.ssh/item_cards_ml4}"

while true; do
  ssh -N \
    -i "$TUNNEL_KEY" \
    -p "$ML4_PORT" \
    -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=3 \
    -o StrictHostKeyChecking=yes \
    -L 127.0.0.1:8766:127.0.0.1:8766 \
    -L 127.0.0.1:8767:127.0.0.1:8767 \
    "$ML4_USER@$ML4_HOST" || true
  sleep 5
done
