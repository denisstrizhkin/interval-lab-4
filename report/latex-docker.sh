#!/usr/bin/env sh
podman run --rm -it -v "$PWD:/work:Z" -w /work texlive "$@"
