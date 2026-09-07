#!/usr/bin/env bash
set -euo pipefail

# Install the EGL loader needed for MuJoCo offscreen rendering.
# Run as root, or use: sudo bash scripts/setup_runpold.sh
apt-get update
apt-get install -y --no-install-recommends libegl1
