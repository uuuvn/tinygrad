#!/bin/sh
set -euo pipefail

# WCP="$(jj log --no-graph -r @ -T 'parents.map(|c| "-d " ++ c.commit_id().short()).join(" ")')"

if [ -n "$(jj diff --summary)" ]; then
  jj squash --into private_nix_direnv .envrc
  jj squash --into private_scripts push.sh fun*.py
fi

if [ -n "$(jj diff --summary)" ]; then
  jj squash --into lora
fi

for remote in "$@"; do
  jj git push --config git.sign-on-push=false --remote "$remote" --allow-empty-description --allow-private --all --deleted
  ssh "$remote" "cd src/tinygrad && jj rebase -r @ -d private_nix_direnv -d private_scripts -d lora"
done
