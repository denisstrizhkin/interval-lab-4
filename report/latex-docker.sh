#!/usr/bin/env sh
if ! podman image exists texlive; then
    echo "Image 'texlive' not found. Building from https://github.com/denisstrizhkin/texlive-container.git ..."
    podman build -t texlive https://github.com/denisstrizhkin/texlive-container.git
fi

podman run --rm -it -v "$PWD:/work:Z" -w /work texlive "$@"
