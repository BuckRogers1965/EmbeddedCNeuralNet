# MNIST — handwritten digit classification

Classic 28×28 grayscale digits (0–9), 60,000 training + 10,000 test images.
The trainer builds a 784 → 128 → 64 → 10 dense net (ReLU hidden layers, softmax
output) and reports test accuracy.

## Get the data

The four IDX files go in `training_data/` (gitignored). From this directory:

```bash
cd training_data
for f in train-images-idx3-ubyte train-labels-idx1-ubyte \
         t10k-images-idx3-ubyte  t10k-labels-idx1-ubyte; do
    wget https://storage.googleapis.com/cvdf-mirror/mnist/$f.gz
    gunzip $f.gz
done
cd ..
```

After this `training_data/` should contain the four uncompressed files:

```
training_data/
├── train-images-idx3-ubyte
├── train-labels-idx1-ubyte
├── t10k-images-idx3-ubyte
└── t10k-labels-idx1-ubyte
```

(The original source is http://yann.lecun.com/exdb/mnist/ if the mirror is down.
Note the trainer reads the raw `idx-ubyte` files, not the `.gz`.)

## Build, deploy, and test the whole path

The point of the example is the **client**: `make` trains the net, exports it to
`mnist_model.h`, builds the embedded client (`client.c`, compiled only against
the pure-C `client/nn_infer.c` — no BLAS, no training code — just like firmware),
and runs it to confirm the deployed model reproduces the training library's
predictions on 1000 held-out test images.

```bash
make          # train -> export -> build client -> test the whole path
```

Expected tail:

```
client reproduced 1000/1000 training-library predictions on held-out data.
  PASS: end-to-end path verified (train -> export -> client inference).
```

Requires BLAS for the training step (`sudo apt-get install libcblas-base-dev`);
the client build needs neither BLAS nor SIMD. Run from this directory (the
`training_data/` paths are relative). **Training takes a few minutes.** `make
clean` removes the binaries and generated `mnist_model.h` / `mnist_samples.bin`.
The MNIST net is ~109k params (~427 KB as float) — it fits ESP32/Pico/Pi flash,
but not an Arduino Uno.
