#!/usr/bin/env bash
#
# Build the bonus deliverable bundle for TA verification.
#
# Stages every file the TA needs into dist/bonus_bundle/, then zips it.
# Excluded by design:
#   * data/ and out/ — TA regenerates via download + run scripts
#   * report/, PPT/, testdata/ — separate deliverables, not for grading bonus
#   * .git, .venv, __pycache__ — build noise
#   * the team-internal README.md — replaced by BONUS_README.md (TA-focused)
#
# FoundationStereo source is vendored with its LICENSE preserved (NVIDIA's
# non-commercial license permits redistribution for academic use as long as
# the LICENSE travels with the source).
#
# Pretrained model weights (~750 MB-3.3 GB) are NOT included — the TA fetches
# them via bonus/download_fs_weights.py, which the bundle's README documents.
#
# Usage:
#   bash scripts/package_bonus.sh                              # default name
#   BUNDLE_NAME=bonus_bundle_v1.0 bash scripts/package_bonus.sh
#   FOUNDATION_STEREO_COMMIT=<sha> bash scripts/package_bonus.sh
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIST="$REPO_ROOT/dist"
BUNDLE_NAME="${BUNDLE_NAME:-bonus_bundle}"
STAGE="$DIST/$BUNDLE_NAME"
ZIP_OUT="$DIST/${BUNDLE_NAME}.zip"
FS_COMMIT="${FOUNDATION_STEREO_COMMIT:-}"

echo "[stage] repo root: $REPO_ROOT"
echo "[stage] bundle target: $STAGE"

rm -rf "$STAGE" "$ZIP_OUT"
mkdir -p "$STAGE"

# ---------------------------------------------------------------------------
# 1. Source code the bonus pipeline depends on
# ---------------------------------------------------------------------------
# The team's final classical implementation is v8. We ship it as the
# bundle's canonical computeDisp.py — the TA only ever sees one classical
# matcher (v8) and one SOTA deep matcher (FoundationStereo). v6 from the
# team's repo is deliberately NOT shipped to keep the deliverable focused.
echo "[copy] python source — computeDisp.py := v8, FoundationStereo for deep"
cp "$REPO_ROOT/computeDisp_v8.py"  "$STAGE/computeDisp.py"
cp -r "$REPO_ROOT/bonus"           "$STAGE/"
cp -r "$REPO_ROOT/experiments"     "$STAGE/"

# Drop the v6 entry from MATCHER_KINDS and remap v8 to the canonical
# `computeDisp` module name the bundle ships. Done as a staging patch so the
# team's source tree stays untouched and the dispatch tables stay simple.
echo "[patch] simplify MATCHER_KINDS to {v8 (classical), foundation_stereo (deep)}"
for f in "$STAGE/bonus/run_bonus_tartanair.py" "$STAGE/experiments/eval_tartanair.py"; do
  sed -i '/^[[:space:]]*"v6":[[:space:]]*("classical"/d' "$f"
  sed -i 's/("classical", "computeDisp_v8")/("classical", "computeDisp")/' "$f"
done

# ---------------------------------------------------------------------------
# 2. Build / dependency manifests (uv preferred, pip fallback)
# ---------------------------------------------------------------------------
echo "[copy] env manifests"
cp "$REPO_ROOT/pyproject.toml"     "$STAGE/"
[ -f "$REPO_ROOT/uv.lock" ]        && cp "$REPO_ROOT/uv.lock"     "$STAGE/"
[ -f "$REPO_ROOT/requirement.txt" ] && cp "$REPO_ROOT/requirement.txt" "$STAGE/"

# ---------------------------------------------------------------------------
# 3. TA-facing README (renamed to README.md in the bundle for prominence)
# ---------------------------------------------------------------------------
if [ -f "$REPO_ROOT/BONUS_README.md" ]; then
  echo "[copy] BONUS_README.md -> bundle README.md"
  cp "$REPO_ROOT/BONUS_README.md" "$STAGE/README.md"
else
  echo "[warn] BONUS_README.md not found in repo root; bundle will lack README"
fi

# ---------------------------------------------------------------------------
# 4. Vendor FoundationStereo source (LICENSE preserved)
# ---------------------------------------------------------------------------
mkdir -p "$STAGE/third_party"
if [ -d "$REPO_ROOT/third_party/FoundationStereo" ]; then
  echo "[copy] vendored third_party/FoundationStereo"
  cp -r "$REPO_ROOT/third_party/FoundationStereo" "$STAGE/third_party/"
else
  echo "[clone] NVlabs/FoundationStereo into bundle"
  git clone --depth 1 https://github.com/NVlabs/FoundationStereo.git \
            "$STAGE/third_party/FoundationStereo"
  if [ -n "$FS_COMMIT" ]; then
    git -C "$STAGE/third_party/FoundationStereo" fetch --depth 1 origin "$FS_COMMIT"
    git -C "$STAGE/third_party/FoundationStereo" checkout "$FS_COMMIT"
  fi
fi

# Verify the LICENSE is in place — non-negotiable per NVIDIA's terms.
if [ ! -f "$STAGE/third_party/FoundationStereo/LICENSE" ]; then
  echo "[error] FoundationStereo LICENSE missing — refusing to ship"
  exit 1
fi

# ---------------------------------------------------------------------------
# 5. Cleanup: strip build noise
# ---------------------------------------------------------------------------
echo "[clean] strip .git, __pycache__, *.pyc"
find "$STAGE" -name ".git" -type d -exec rm -rf {} + 2>/dev/null || true
find "$STAGE" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find "$STAGE" -name "*.pyc" -delete 2>/dev/null || true
find "$STAGE" -name ".DS_Store" -delete 2>/dev/null || true

# ---------------------------------------------------------------------------
# 6. Provenance stamp (helps trace which commit the bundle was built from)
# ---------------------------------------------------------------------------
{
  echo "Build date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "Repo HEAD:  $(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo 'unknown')"
  echo "Repo branch:$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')"
  if [ -d "$STAGE/third_party/FoundationStereo/.git" ]; then
    echo "FS HEAD:    $(git -C "$STAGE/third_party/FoundationStereo" rev-parse HEAD 2>/dev/null || echo 'unknown')"
  fi
} > "$STAGE/PROVENANCE.txt"

# ---------------------------------------------------------------------------
# 7. Zip it up
# ---------------------------------------------------------------------------
echo "[zip] creating $ZIP_OUT"
( cd "$DIST" && zip -qr "${BUNDLE_NAME}.zip" "$BUNDLE_NAME" )

SIZE=$(du -h "$ZIP_OUT" | cut -f1)
FILES=$(unzip -l "$ZIP_OUT" | tail -1 | awk '{print $2}')
echo ""
echo "[done] $ZIP_OUT  ($SIZE, $FILES files)"
echo "[hint] verify with: unzip -l $ZIP_OUT | head -20"
