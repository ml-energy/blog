#!/usr/bin/env bash

# MacOS brew cairo fix
if [[ -d /opt/homebrew/lib ]]; then
    export DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib
fi

uv run --with-requirements requirements.txt mkdocs serve -a localhost:7778 -w docs -w mkdocs.yml
