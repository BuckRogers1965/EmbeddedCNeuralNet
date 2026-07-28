# Fashion-MNIST — clothing image classification

A drop-in-harder replacement for MNIST: same 28×28 grayscale format, same
60,000 / 10,000 split, same 10 classes count — but the images are articles of
clothing (T-shirt, Trouser, Pullover, …), which are tougher to separate than
digits. Good for comparing the same 784 → 128 → 64 → 10 net on a harder task.

## Get the data

Fashion-MNIST ships the **same filenames and IDX format** as MNIST. Put the four
files in `training_data/` (gitignored). From this directory:

```bash
cd training_data
base=https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion
for f in train-images-idx3-ubyte train-labels-idx1-ubyte \
         t10k-images-idx3-ubyte  t10k-labels-idx1-ubyte; do
    wget $base/$f.gz
    gunzip $f.gz
done
cd ..
```

Resulting layout:

```
training_data/
├── train-images-idx3-ubyte
├── train-labels-idx1-ubyte
├── t10k-images-idx3-ubyte
└── t10k-labels-idx1-ubyte
```

Project page: https://github.com/zalandoresearch/fashion-mnist

## Build, deploy, and test the whole path

The point of the example is the **client**: `make` trains the net, exports it to
`fashion_mnist_model.h`, builds the embedded client (`client.c`, compiled only
against the pure-C `client/nn_infer.c` — no BLAS, no training code — just like
firmware), and runs it to confirm the deployed model reproduces the training
library's predictions on 1000 held-out test images.

```bash
make          # train -> export -> build client -> test the whole path
```

Expected tail:

```
client reproduced 1000/1000 training-library predictions on held-out data.
  PASS: end-to-end path verified (train -> export -> client inference).
```

Requires BLAS for the training step (`sudo apt-get install libcblas-base-dev`);
the client build needs neither BLAS nor SIMD. Run from this directory. **Training
takes a few minutes.** `make clean` removes the binaries and generated
`fashion_mnist_model.h` / `fashion_mnist_samples.bin`.
