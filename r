#!/bin/sh
set -euo pipefail
set -x

REMOTE=tiny10
RDIR=src/tinygrad
DIFF="$(mktemp)"

cd "$(dirname $0)"

git commit --allow-empty --no-gpg-sign -m 'r checkpoint'
git add -A
git diff --cached > $DIFF
git reset --soft 'HEAD~1'
if [ -s "$DIFF" ]; then
  cat "$DIFF" | ssh "$REMOTE" "cd $RDIR && git apply"
fi
rm $DIFF
ssh -L 127.0.0.1:8000:127.0.0.1:8000 -t "$REMOTE" "cd $RDIR && touch ~/.cache/tinygrad/cache.db.hack && rm ~/.cache/tinygrad/cache.db* && source .venv-3.11/bin/activate && $@"
