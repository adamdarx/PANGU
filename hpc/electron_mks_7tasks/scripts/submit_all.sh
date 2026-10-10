#!/usr/bin/env bash
set -euo pipefail
scripts=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
for name in sane_a0 sane_a05 sane_a09 sane_a09375 mad_am09375 mad_a0 mad_ap09375; do
  bash "$scripts/submit.sh" "$name"
done
