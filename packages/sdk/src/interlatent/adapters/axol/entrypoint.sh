#!/usr/bin/env bash
set -euo pipefail

# Smoke-check that the SDK imported cleanly, then hand the user an interactive
# shell. Any args passed to the container run first; bash starts when done.
python -c "import interlatent; print('interlatent imported OK')"

if [ "$#" -gt 0 ]; then
    "$@"
fi

exec bash
