#!/usr/bin/env bash
# Clone every repository in repos.lock with full history and check out the pinned SHA.
set -euo pipefail
mkdir -p repos
while read -r name url sha; do
  [ -d "repos/$name/.git" ] || git clone -q "$url" "repos/$name"
  git -C "repos/$name" checkout -q "$sha"
  echo "$name pinned at $sha"
done < repos.lock
