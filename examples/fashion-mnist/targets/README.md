# Deployment targets

Each subdirectory here is one **deployment target** for this example's trained
model: the code and directions for running the exported network somewhere. The
trainer (`../fashion_mnist.c`) produces `fashion_mnist_model.h` (weights) and
`fashion_mnist_samples.bin` (held-out samples); a target consumes those.

| Target          | Where it runs        | What it does                                  |
|-----------------|----------------------|-----------------------------------------------|
| `self_test/`    | your desktop         | builds the pure-C client and verifies the path |
| `pico_arduino/` | Pico / ESP32         | flashable sketch that classifies on-device    |

Run the desktop verification from the example root:

```bash
make self_test
```

or `cd` into a target and follow its `README.md`.

## Add your own target

1. `mkdir targets/<your_target>/` and drop in your code plus a `README.md` with
   build/flash directions.
2. Compile against `../../../../client/nn_infer.c` and the generated
   `../../fashion_mnist_model.h` (define `NN_MODEL_HEADER`, or copy it in as
   `nn_model.h`).
3. Feed inputs scaled **identically to training** (each pixel divided by `512.0`).
