# Deployment targets

Each subdirectory here is one **deployment target** for this example's trained
model: the code and the directions for running the exported network somewhere.
The training side (`../iris.c`) produces `iris_model.h` (weights) and
`iris_samples.bin` (held-out samples); a target consumes those.

| Target          | Where it runs        | What it does                                  |
|-----------------|----------------------|-----------------------------------------------|
| `self_test/`    | your desktop         | builds the pure-C client and verifies the path |
| `pico_arduino/` | Pico / Arduino / ESP32 | flashable sketch that classifies on-device   |

Run one from the example root:

```bash
make self_test          # desktop verification (from ../)
```

or `cd` into a target and follow its `README.md`.

## Add your own target

1. `mkdir targets/<your_target>/` and drop in your code plus a `README.md` with
   build/flash directions.
2. Consume the model the same way the others do: compile against
   `../../../../client/nn_infer.c` and the generated `../../iris_model.h`
   (define `NN_MODEL_HEADER` to point at it, or copy it in as `nn_model.h`).
3. Feed inputs scaled **identically to training** (the Iris loader divides each
   feature by `8.0`).

That's the whole contract — a target is just "here is how this model runs on X."
