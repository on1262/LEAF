#!/usr/bin/env bash
# Fetch the three third-party code bases at their pinned upstream commits and
# apply the patches in patches/.  Nothing is committed: the modifications stay
# as working-tree changes, so `git -C third_party/SDPO diff --binary` reproduces
# patches/SDPO-7c457fc.patch.
#
#   third_party/SDPO           verl fork (SDPO)      7c457fc1b1f636ae794eb0362ba37d4743b06fbc  + patches/SDPO-7c457fc.patch
#   third_party/LiveCodeBench  LiveCodeBench          28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24  + patches/LiveCodeBench-28fef95.patch
#   third_party/EvalPlus       EvalPlus               26d6d00bb1fd0fa37f39c99d5290da67891d1c5e  (unmodified)
#
# Usage: bash setup.sh
# Environment:
#   THIRD_PARTY_DIR  target directory (default: <repo>/third_party)
#   CLONE_SOURCE     optional base URL or directory that holds SDPO, LiveCodeBench
#                    and EvalPlus repositories, used instead of the GitHub URLs
#                    (e.g. a local mirror).
#
# The script is idempotent: an existing checkout at the pinned commit is kept,
# and a patch that is already applied is skipped.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
THIRD_PARTY_DIR="${THIRD_PARTY_DIR:-$REPO/third_party}"
PATCH_DIR="$REPO/patches"

# name | upstream URL | pinned commit | patch file (empty = none)
REPOS=(
  "SDPO|https://github.com/lasgroup/SDPO.git|7c457fc1b1f636ae794eb0362ba37d4743b06fbc|SDPO-7c457fc.patch"
  "LiveCodeBench|https://github.com/LiveCodeBench/LiveCodeBench.git|28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24|LiveCodeBench-28fef95.patch"
  "EvalPlus|https://github.com/evalplus/evalplus.git|26d6d00bb1fd0fa37f39c99d5290da67891d1c5e|"
)

log() { printf '[setup] %s\n' "$*"; }
die() { printf '[setup] ERROR: %s\n' "$*" >&2; exit 1; }

command -v git >/dev/null || die "git is required"

fetch_repo() {  # name url commit
  local name="$1" url="$2" commit="$3" dir="$THIRD_PARTY_DIR/$1"
  if [[ -n "${CLONE_SOURCE:-}" ]]; then
    url="${CLONE_SOURCE%/}/$name"
  fi
  if [[ ! -e "$dir" ]]; then
    log "$name: cloning $url"
    git clone --quiet "$url" "$dir"
  elif [[ ! -d "$dir/.git" ]]; then
    die "$dir exists but is not a git checkout; move it away and re-run"
  fi
  if ! git -C "$dir" cat-file -e "${commit}^{commit}" 2>/dev/null; then
    log "$name: fetching $commit"
    git -C "$dir" fetch --quiet origin "$commit" || die "$name: cannot fetch $commit"
  fi
  local head
  head="$(git -C "$dir" rev-parse --verify --quiet HEAD || true)"
  if [[ "$head" != "$commit" ]]; then
    if [[ -n "$head" && -n "$(git -C "$dir" status --porcelain --untracked-files=no)" ]]; then
      die "$name: checkout at $head has local changes; cannot switch to $commit"
    fi
    log "$name: checking out $commit"
    git -C "$dir" -c advice.detachedHead=false checkout --quiet --detach "$commit"
  else
    log "$name: already at $commit"
  fi
}

apply_patch() {  # name patch-file
  local name="$1" patch="$PATCH_DIR/$2" dir="$THIRD_PARTY_DIR/$1"
  [[ -f "$patch" ]] || die "missing $patch"
  if git -C "$dir" apply --reverse --check "$patch" 2>/dev/null; then
    log "$name: $(basename "$patch") already applied"
  elif git -C "$dir" apply --check "$patch"; then
    git -C "$dir" apply "$patch"
    # Record the files the patch creates as intent-to-add (no commit) so that
    # `git diff` lists them together with the modified files.
    local created
    created="$(git -C "$dir" apply --summary "$patch" | awk '$1 == "create" && $2 == "mode" {print $4}')"
    if [[ -n "$created" ]]; then
      # shellcheck disable=SC2086
      git -C "$dir" add --intent-to-add -- $created
    fi
    log "$name: applied $(basename "$patch")"
  else
    die "$name: $(basename "$patch") neither applies nor is already applied"
  fi
}

mkdir -p "$THIRD_PARTY_DIR"
for entry in "${REPOS[@]}"; do
  IFS='|' read -r name url commit patch <<< "$entry"
  fetch_repo "$name" "$url" "$commit"
  if [[ -n "$patch" ]]; then
    apply_patch "$name" "$patch"
  elif [[ -n "$(git -C "$THIRD_PARTY_DIR/$name" status --porcelain --untracked-files=no)" ]]; then
    log "$name: WARNING: working tree differs from $commit (expected unmodified)"
  fi
done

log "done. Put the repository root first on PYTHONPATH, e.g."
log "  export PYTHONPATH=$REPO:$THIRD_PARTY_DIR/SDPO:$THIRD_PARTY_DIR/LiveCodeBench"
log "(run/common.sh exports exactly this; EvalPlus is added to sys.path by leaf/verifiers/mbppplus.py)."
