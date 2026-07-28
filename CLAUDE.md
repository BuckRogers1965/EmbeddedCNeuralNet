# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A from-scratch feed-forward neural network library written in C, meant to be **trained on a desktop and then run (forward pass only) on tiny embedded hardware** (Arduino, ESP32, Pico, Raspberry Pi). The design assumption throughout: training is slow and one-time; inference is cheap, single-sample, and runs millions of times on constrained memory. The library also serves as a readable reference implementation of many neural-net techniques (it is deliberately *not* optimized for speed).

## Build & run

Requires BLAS. On Ubuntu: `sudo apt-get install libcblas-base-dev`.

Each dataset is a directory `examples/<name>/` with the trainer (`<name>.c`), an
orchestration `Makefile`, a gitignored `training_data/`, and a `targets/` dir.
Training just produces a model; each subdir of `targets/` is a self-contained
**deployment target** (its own code + `README.md`) that consumes the exported
model. `make self_test` is the closest thing to a test suite this repo has.

```bash
cd examples/mnist        # or fashion-mnist, iris
# download data into training_data/ first (see the dir's README.md)
make self_test           # train -> export <name>_model.h + <name>_samples.bin
                         #   -> build self_test client (no BLAS) -> run + verify
```

Flow: the trainer does `create_neural_net` → `add_layer` × N → `train` → `test` →
`export_inference_header("<name>_model.h")` + dumps held-out samples with the
library's predictions to `<name>_samples.bin` (both land in the example root).
The `targets/self_test/client.c` (linked only against `client/nn_infer.c` + the
generated header, no BLAS) replays those samples through `nn_classify` and
asserts they match — `PASS`/mismatch. `targets/pico_arduino/` is an Arduino/Pico
sketch; `make prepare` there copies `nn_infer.*` + the model (as `nn_model.h`)
into the sketch folder for the Arduino IDE.

Makefiles are recursive: the `self_test`/`pico_arduino` Makefiles call `make -C`
back to the example root to build the model if missing. **Gotcha:** don't put an
inline `# comment` after a Makefile `VAR = value` — Make keeps the trailing
spaces in the value (this bit a relative path once). Adding a dataset = copy an
example dir; adding a target = new subdir under `targets/`. See
`examples/README.md`. Generated `*_model.h` / `*_samples.bin` / binaries / copied
sketch files are gitignored.

Note: `neural_net.c` includes `<immintrin.h>` and uses SSE2 intrinsics (`add_biases_simd`, `__m128d`), baseline on x86-64, for the *training* build. The edge client (`client/`) has its own pure-C forward pass and shares none of this.

## Architecture

The whole library is two files: `library/neural_net.h` (public API) and `library/neural_net.c` (~1200 lines, everything else). The `examples/*/` drivers train it; the standalone `client/` runs inference (see the "Edge deployment" section).

**Opaque handle.** `struct NeuralNet` and `struct Layer` are defined only in `neural_net.c`. Callers get a `NeuralNet *` and go through accessor functions (`get_*`/`set_*` in the header). Don't reach into the struct from example/driver code — extend the accessor API instead.

**Building a net.** `create_neural_net(input_size, ...)` sets the input width, then each `add_layer(net, output_size, activation)` appends a fully-connected layer. Each new layer's `input_size` is taken from the previous layer's output (or the net's `input_size` for the first layer). Historical bug worth knowing: the first layer once ignored `input_size` and mis-sized itself to the output — if training silently fails to learn, re-check that wiring in `add_layer`.

**Pluggable strategy via function pointers.** The net stores function pointers chosen at configuration time rather than branching per-sample:
- **Activations** (`ActivationFunction` enum): each has a `*_activation`/`*_derivative` pair; `add_layer` binds them onto the layer. Softmax is special-cased (operates over the whole output vector, see `softmax`).
- **Optimizers** (`OptimizationMethod` enum): SGD, momentum, RMSProp, Adam, NAG. Each has a `create_*` (allocates per-weight state into `opt_params`) and a step function matching the `net->chooser` signature; `setup_chooser_params` wires them up.
- **Training is proper mini-batch.** Per batch, `train()` calls `backward_pass` for each sample — which now only **accumulates** gradients into each layer's `weight_gradients`/`bias_gradients` (added in `add_layer`/`load_neural_net`, freed in `free_neural_net`) — then calls `apply_gradients(net, batch_count)` **once** to take a single optimizer step on the mean gradient and zero the accumulators. Weight updates go through `net->chooser` so the selected optimizer actually takes effect (previously `backward_pass` hardcoded a per-sample SSE2 SGD update and ignored `opt_method`). Adam's timestep `t` advances once per batch inside `apply_gradients`.
- **Biases use plain SGD** in `apply_gradients` on purpose: each optimizer's state arrays are sized for weights only (`input_size*output_size` per layer), so routing biases through the chooser would alias weight state — per-bias optimizer state is a TODO.
- Adam gotchas fixed earlier: `OPT_ADAM` allocated an `RMSpropParams` (wrong struct → segfault), and `t` starting at 0 gave a `1-beta^0=0` div-by-zero; fixed in `setup_chooser_params`/`create_adam`. Plain ReLU + Adam can dead-unit on tiny data — the Iris example uses leaky ReLU for that reason.
- **Loss**: `calculate_error` / `calculate_error_derivative` pointers (MSE and cross-entropy implemented).

**Backward pass is in flux.** There are multiple backward-pass implementations in the file (`backward_pass`, `backward_pass_no_jitter`, `backward_pass_revised`, `backward_passbad`). This is intentional — the author is actively expanding backward-pass options. Confirm which one `train` actually calls before assuming behavior.

**Training loop.** `train` shuffles data each epoch (`shuffle_data`), runs forward/backward per batch, adjusts learning rate per epoch via `adj_lr_epoch`, and optionally applies input "jitter" (data augmentation, controllable via `set_use_jitter`/`set_jitter_strength`/`set_jitter_decay_rate`). Progress is reported by test-running the first 1000 shuffled samples at each epoch end — there is no proper training-loss metric yet.

**Persistence.** `save_neural_net`/`load_neural_net` write/read the full net (weights, biases, config) to a file. Note `load_neural_net`'s activation `switch` is stubbed (only `ACTIVATION_SIGMOID` is wired), so a desktop round-trip mis-runs every layer as sigmoid — don't rely on it. `classify(net, input)` is the single-sample inference entry point.

## Edge deployment (`client/`)

The intended shipping path is: **train on desktop, export a C header, compile that header into firmware, run inference on the MCU.** The edge client is fully separate from the training library — no BLAS, no SIMD, no `malloc`, `float` (not `double`) — so it builds for tiny targets.

- `export_inference_header(net, "client/nn_model.h")` (in `neural_net.c`) emits a self-contained header: `float` weights/biases, per-layer sizes + activation codes (codes match the `ActivationFunction` enum), and `NN_INPUT_SIZE`/`NN_OUTPUT_SIZE`/`NN_LAYER_COUNT`/`NN_MAX_WIDTH` `#define`s. Training metadata is intentionally dropped.
- `client/nn_infer.{h,c}` is the standalone forward pass: all 12 activations ported to `float`, two static ping-pong scratch buffers of `NN_MAX_WIDTH` (no heap), `nn_classify()` / `nn_forward()`. It `#include`s the generated `nn_model.h` (override the name with `-DNN_MODEL_HEADER=...`).
- **Verification is two-stage** (see `client/README.md`): `gen_reference.c` (links the training lib, needs BLAS) builds a net, exports the header, and dumps library predictions to `ref.bin`; `client_test.c` (pure C) replays them through the client and asserts every argmax matches. Passing run: `500/500`.
- Client and training now compute the same forward pass (the old `add_biases_simd` stride-by-4 bias-skipping bug is fixed), so their outputs agree.
- `nn_model.h` and `ref.bin` are generated artifacts (gitignored).

## Conventions

- Data is passed as `double **` — arrays of per-sample pointers. Labels are one-hot (`OUTPUT_SIZE`-wide). MNIST pixels are scaled by `/512.0` in the example loader.
- New capabilities (activation, optimizer, loss, regularizer, LR schedule) are added by implementing the function(s) and wiring them into the corresponding enum + selector, keeping the opaque-struct/accessor pattern intact.

## Python

`python/train.2023-07-19.py` is a separate, earlier standalone NumPy MNIST trainer the author wrote by hand (reached ~96%). It is historical reference, not part of the C build.
