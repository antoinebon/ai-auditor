#!/usr/bin/env bash
set -euo pipefail

# Install Docker Engine on Ubuntu using the official repository

if command -v docker &>/dev/null; then
  echo "Docker is already installed: $(docker --version)"
  exit 0
fi

echo "==> Cleaning up any stale Docker repo entries..."
sudo rm -f /etc/apt/sources.list.d/docker.list /etc/apt/sources.list.d/docker.sources

echo "==> Installing prerequisites..."
sudo apt-get update
sudo apt-get install -y ca-certificates curl

echo "==> Adding Docker GPG key..."
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

echo "==> Adding Docker repository..."
# Ubuntu 25.04 (plucky) may not have a Docker repo yet; fall back to noble (24.04 LTS)
CODENAME=$(. /etc/os-release && echo "${VERSION_CODENAME}")
if ! curl -fsSL "https://download.docker.com/linux/ubuntu/dists/${CODENAME}/Release" &>/dev/null; then
  echo "    No Docker repo for '${CODENAME}', falling back to 'noble'..."
  CODENAME="noble"
fi
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu ${CODENAME} stable" |
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

echo "==> Installing Docker Engine..."
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

echo "==> Adding $USER to the docker group..."
sudo usermod -aG docker "$USER"

echo "==> Verifying installation..."
sudo docker run --rm hello-world

echo ""
echo "Done! Log out and back in (or run 'newgrp docker') for group changes to take effect."
