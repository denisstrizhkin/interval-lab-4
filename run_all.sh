#!/usr/bin/env sh
set -e

echo "1. Generating data with Rust..."
cargo run --release

echo "2. Generating plots with Gnuplot..."
mkdir -p report/images
gnuplot plot_graphs.gp

echo "3. Compiling LaTeX report..."
# Use the local docker script if available, or call podman directly
if [ -f "report/latex-docker.sh" ]; then
    # The script uses $PWD, so we run it from the root but target the report folder
    podman run --rm -v "$(pwd):/work:Z" -w /work/report texlive pdflatex report.tex
else
    echo "Warning: report/latex-docker.sh not found, skipping PDF compilation."
fi

echo "All done! Report generated at report/report.pdf"
