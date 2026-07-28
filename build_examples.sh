#!/usr/bin/env bash
# Build and run each example's self_test. Output goes straight to stdout.
# Stops on the first error. If an example has no training data, it says which
# README to read for the download, and skips it.
set -e

cd "$(dirname "$0")"

for dir in examples/*/; do
    [ -d "${dir}targets/self_test" ] || continue
    name="$(basename "$dir")"

    if [ -z "$(find "${dir}training_data" -type f ! -name .gitkeep 2>/dev/null)" ]; then
        echo ">>> $name: no training data -- read ${dir}README.md to install it. Skipping."
        continue
    fi

    echo ">>> $name: cleaning old build (so training actually re-runs, not a stale cache)"
    make -C "$dir" clean
    echo ">>> $name: compiling, training from scratch, exporting, building client, running it"
    make -C "$dir" self_test
    echo ">>> $name: PASS"
done

echo ">>> done"
