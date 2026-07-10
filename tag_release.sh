#!/usr/bin/env bash
# Usage: ./tag_release.sh [major|minor|patch]
set -euo pipefail

BUMP="${1:-patch}"

# Get latest tag or start from v0.0.0
LATEST=$(git tag --sort=-v:refname | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' | head -1 || echo "v0.0.0")
VERSION="${LATEST#v}"

IFS='.' read -r MAJOR MINOR PATCH <<< "$VERSION"

case "$BUMP" in
  major) MAJOR=$((MAJOR + 1)); MINOR=0; PATCH=0 ;;
  minor) MINOR=$((MINOR + 1)); PATCH=0 ;;
  patch) PATCH=$((PATCH + 1)) ;;
  *) echo "Usage: $0 [major|minor|patch]"; exit 1 ;;
esac

NEW_TAG="v${MAJOR}.${MINOR}.${PATCH}"
echo "Tagging $NEW_TAG"
git tag "$NEW_TAG"
git push origin "$NEW_TAG"
