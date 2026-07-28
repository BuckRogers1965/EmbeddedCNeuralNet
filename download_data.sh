#!/usr/bin/env bash
#
# download_data.sh -- download the REAL datasets for the examples into their
# training_data/ directories. These files are gitignored (large, freely
# available), so a fresh checkout has none until you run this.
#
# Usage:
#   ./download_data.sh                 # all datasets
#   ./download_data.sh iris            # just iris (tiny, quick)
#   ./download_data.sh mnist fashion   # a subset
#
# Idempotent: files already present are skipped. After it finishes, run
# ./build_examples.sh to train + verify each example.

set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# --- pick a downloader ------------------------------------------------------
if command -v wget >/dev/null 2>&1; then
    fetch() { wget -O "$2" "$1"; }             # $1=url  $2=out
elif command -v curl >/dev/null 2>&1; then
    fetch() { curl -fL --progress-bar -o "$2" "$1"; }
else
    echo "ERROR: need 'wget' or 'curl' installed." >&2
    exit 1
fi
if ! command -v gunzip >/dev/null 2>&1; then
    echo "ERROR: need 'gunzip' installed (for the MNIST/Fashion .gz files)." >&2
    exit 1
fi

failed=""

# --- iris: one CSV file -----------------------------------------------------
get_iris() {
    local dir="examples/iris/training_data"
    local out="$dir/iris.data"
    echo "=== iris ==="
    if [ -s "$out" ]; then echo "  already present -- skipping"; return 0; fi
    if fetch "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data" "$out" \
        && [ -s "$out" ]; then
        echo "  OK: $out ($(wc -l < "$out") rows)"
        return 0
    fi
    echo "  FAILED. Source/help: examples/iris/README.md" >&2
    rm -f "$out"; failed="$failed iris"; return 1
}

# --- idx datasets: MNIST and Fashion-MNIST share the format and filenames ---
# Each file is tried against every mirror in order until one succeeds, so a
# single stale URL does not break the download. All mirror URLs below were
# checked to return HTTP 200 (curl -I) before being committed here.
get_idx() {                     # $1=label  $2=dir  $3.. = mirror base urls
    local label="$1" dir="$2"; shift 2
    local rc=0
    echo "=== $label ==="
    for f in train-images-idx3-ubyte train-labels-idx1-ubyte \
             t10k-images-idx3-ubyte  t10k-labels-idx1-ubyte; do
        if [ -s "$dir/$f" ]; then echo "  $f present -- skipping"; continue; fi
        local got=0 base
        for base in "$@"; do
            echo "  downloading $f.gz from $base ..."
            if fetch "$base/$f.gz" "$dir/$f.gz" && [ -s "$dir/$f.gz" ] && gunzip -f "$dir/$f.gz"; then
                echo "  OK: $dir/$f"; got=1; break
            fi
            rm -f "$dir/$f.gz"
        done
        [ "$got" -eq 1 ] || { echo "  FAILED (all mirrors) for $f" >&2; rc=1; }
    done
    [ $rc -eq 0 ] || { echo "  $label incomplete. Help: examples/$label*/README.md" >&2; failed="$failed $label"; }
    return $rc
}

get_mnist()   { get_idx "mnist"         "examples/mnist/training_data" \
                    "https://storage.googleapis.com/cvdf-datasets/mnist" \
                    "https://ossci-datasets.s3.amazonaws.com/mnist"; }
get_fashion() { get_idx "fashion-mnist" "examples/fashion-mnist/training_data" \
                    "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion"; }

# --- select which datasets --------------------------------------------------
run_one() {
    case "$1" in
        iris)                 get_iris ;;
        mnist)                get_mnist ;;
        fashion|fashion-mnist) get_fashion ;;
        *) echo "unknown dataset: $1 (expected: iris, mnist, fashion-mnist)" >&2; failed="$failed $1" ;;
    esac
}

if [ "$#" -eq 0 ]; then
    get_iris; get_mnist; get_fashion
else
    for d in "$@"; do run_one "$d"; done
fi

echo
if [ -n "$failed" ]; then
    echo "Some datasets failed:$failed"
    echo "URLs can go stale -- check the named READMEs for the current source, then re-run."
    exit 1
fi
echo "All requested datasets are in place. Next: ./build_examples.sh"
