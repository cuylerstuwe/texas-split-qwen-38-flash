#!/bin/bash
# Moves beamivalice/homebrew-tap's Formula/sushi.rb to a published release with this machine's `gh` login, so no
# cross-repo token has to live in a repo secret. The tap's hourly bump.yml schedule is the fallback.
#
#   scripts/bump-tap.sh            # the latest published release
#   scripts/bump-tap.sh v1.2.0
set -euo pipefail
TAP=beamivalice/homebrew-tap
REPO=beamivalice/sushi
TAG="${1:-$(gh release view -R "$REPO" --json tagName --jq .tagName)}"

before=$(gh run list -R "$TAP" --workflow bump.yml --limit 1 --json databaseId --jq '.[0].databaseId // 0')
gh workflow run bump.yml -R "$TAP"
for _ in $(seq 1 30); do
    id=$(gh run list -R "$TAP" --workflow bump.yml --limit 1 --json databaseId --jq '.[0].databaseId // 0')
    [ "$id" != "$before" ] && break
    sleep 4
done
[ "$id" != "$before" ] || { echo "the tap's bump run never started" >&2; exit 1; }
gh run watch "$id" -R "$TAP" --exit-status > /dev/null

url=$(gh api "repos/$TAP/contents/Formula/sushi.rb" --jq .content | base64 -d | sed -n 's/^  url "\(.*\)"$/\1/p')
case "$url" in
    */download/"$TAG"/*) echo "Formula/sushi.rb names $TAG" ;;
    *) echo "Formula/sushi.rb still points at $url, not $TAG" >&2; exit 1 ;;
esac
