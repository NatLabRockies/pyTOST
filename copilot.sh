#!/usr/bin/env bash
set -euo pipefail

exec ~/src/open-llms/scripts/llm-copilot.sh \
  debug \
  -- \
  --allow-all-tools \
  "$@"
