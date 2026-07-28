# Examples

Each subdirectory is a dataset. Training just produces a model; the interesting
part is the **deployment targets**. One trainer exports the model, and each
target under `targets/` is a self-contained way to *run* that model — with its
own code and directions. Anyone can add a new target dir.

```
examples/<name>/
├── <name>.c            # trainer: load data, train, export <name>_model.h + <name>_samples.bin
├── Makefile            # orchestrates training; `make self_test`
├── README.md           # dataset-specific download + run steps
├── training_data/      # you download the dataset here (gitignored)
└── targets/
    ├── self_test/      # desktop: builds the pure-C client and verifies the whole path
    │   ├── client.c
    │   ├── Makefile
    │   └── README.md
    └── pico_arduino/   # microcontroller: an Arduino/Pico sketch + directions
        ├── <name>_infer.ino
        ├── Makefile
        └── README.md
```

Every target consumes the same two exported files (`<name>_model.h` weights,
`<name>_samples.bin` held-out samples) and links against the one pure-C client in
`../../../../client/`. `training_data/` contents and all generated
`*_model.h` / `*_samples.bin` / binaries / copied sketch files are **not**
committed — each README explains how to download the data.

## Available examples

| Directory        | Data type      | Task                              | Optimizer |
|------------------|----------------|-----------------------------------|-----------|
| `mnist/`         | 28×28 images   | Handwritten digit classification  | SGD       |
| `fashion-mnist/` | 28×28 images   | Clothing classification (harder)  | SGD       |
| `iris/`          | CSV / tabular  | Flower species classification     | Adam      |

## Run one

```bash
cd examples/<name>
# download the dataset into training_data/ first -- see this dir's README.md
make self_test          # train -> export -> build the pure-C client -> verify the path
```

A passing run ends with:

```
client reproduced N/N training-library predictions on held-out data.
  PASS: end-to-end path verified (train -> export -> client inference).
```

To target hardware instead, `cd targets/pico_arduino && make prepare` assembles a
flashable Arduino sketch folder (see that target's README).

The training step needs BLAS (`libcblas-base-dev`); the **client/target builds
need neither BLAS nor SIMD** — that is exactly what you cross-compile for an MCU.

## Common pattern

Every trainer builds `double **` arrays (one pointer per sample), one-hot encodes
labels, then uses the same library API: `create_neural_net` → `add_layer` × N →
`train` → `test` → `export_inference_header` + a dump of held-out samples. The
only real per-example difference is the data loader and class names. Every
`self_test/client.c` is the same shape: read the dumped samples, run
`nn_classify`, compare to the reference predictions. On real hardware (the
`pico_arduino` target) you drop the file I/O and feed live sensor data instead.

## Adding a new example, or a new target

**New dataset:** copy the closest example directory (trainer, `Makefile`,
`targets/`), then adjust the loader, the model/sample filenames, and the class
names. Have the trainer call `export_inference_header(net, "<name>_model.h")` and
dump held-out samples + predictions to `<name>_samples.bin`. Generated files are
already covered by the root `.gitignore`; add the trainer's binary name there.

**New target** (e.g. STM32, an HTTP server, a different MCU): `mkdir
examples/<name>/targets/<your_target>/`, drop in your code and a `README.md`, and
consume the model like the others — compile against `../../../../client/nn_infer.c`
and the generated `../../<name>_model.h`, feeding inputs scaled identically to
training. See any `targets/README.md` for the contract.
