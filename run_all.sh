#!/usr/bin/env sh
set -e

echo "1. Generating data with Rust..."
cargo run --release

echo "2. Generating plots with Gnuplot..."
mkdir -p report/images
gnuplot plot_graphs.gp

echo "3. Compiling LaTeX report..."
# Build texlive image if not present
if ! podman image exists texlive; then
    echo "Image 'texlive' not found. Building from https://github.com/denisstrizhkin/texlive-container.git ..."
    podman build -t texlive https://github.com/denisstrizhkin/texlive-container.git
fi

# Run the compilation
podman run --rm -v "$(pwd):/work:Z" -w /work/report texlive pdflatex report.tex

echo "All done! Report generated at report/report.pdf"
