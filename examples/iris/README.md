# Iris — flower species classification (CSV)

The classic 150-sample dataset: 4 numeric features (sepal/petal length & width)
and 3 species. It's here to show the library works on plain tabular CSV data,
not just images — only the loader changes. Net is 4 → 16 (leaky ReLU) → 3
(softmax), trained with the **Adam** optimizer (mini-batch of 16). The 150 rows
are shuffled once and split 120 train / 30 test; expect ~90–100% test accuracy
(it varies run to run — the net is time-seeded and the test set is only 30 rows).

## Get the data

One file, `iris.data`, goes in `training_data/` (gitignored). From this
directory:

```bash
wget https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data \
     -O training_data/iris.data
```

Each line is `sepal_len,sepal_wid,petal_len,petal_wid,Iris-<species>`, e.g.:

```
5.1,3.5,1.4,0.2,Iris-setosa
```

The loader expects exactly 150 such rows. The UCI file sometimes has a trailing
blank line — that's fine, it's ignored.

## Build, deploy, and test the whole path

The point of the example is the **client**: `make` trains the net, exports it to
`iris_model.h`, builds the embedded client (`client.c`, compiled only against the
pure-C `client/nn_infer.c` — no BLAS, no training code — just like firmware),
and runs it to confirm the deployed model reproduces the training library's
predictions.

```bash
make          # train -> export -> build client -> test the whole path
```

Expected tail:

```
client reproduced 30/30 training-library predictions on held-out data.
  PASS: end-to-end path verified (train -> export -> client inference).
```

Requires BLAS for the training step (`sudo apt-get install libcblas-base-dev`);
the client build needs neither BLAS nor SIMD. Run from this directory (the
`training_data/` paths are relative). `make clean` removes the binaries and the
generated `iris_model.h` / `iris_samples.bin`. The trained net is time-seeded, so
accuracy varies run to run (~90–100%); the client always reproduces it exactly.
