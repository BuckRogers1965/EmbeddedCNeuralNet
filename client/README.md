# Edge inference client

Standalone forward-pass ("classify") code for running a trained network on an
embedded target. No training, no BLAS, no SIMD, no `malloc` — pure C99 plus
`<math.h>`. This is the code you flash onto the device.

## Files

- `nn_infer.h` / `nn_infer.c` — the client. Portable, dependency-free.
- `nn_model.h` — **generated** by the training side; holds the weights. Not
  committed (see `.gitignore`).
- `gen_reference.c`, `client_test.c` — desktop-only verification harness.

## How a model gets here

On the training side, after `train(...)`, call:

```c
export_inference_header(net, "client/nn_model.h");
```

This writes a self-contained header with:

- `float` weights and biases (half the size of the training-side `double`s),
- per-layer input/output sizes and activation codes (the codes match the
  `ActivationFunction` enum in `library/neural_net.h`),
- `#define`s for `NN_INPUT_SIZE`, `NN_OUTPUT_SIZE`, `NN_LAYER_COUNT`,
  `NN_MAX_WIDTH`.

The training metadata (epochs, learning rate, optimizer, …) is deliberately left
out — the client doesn't need it.

## Firmware usage

Compile `nn_infer.c` together with your firmware and call:

```c
#include "nn_infer.h"

float input[/* nn_input_size() */];   /* scale inputs exactly as in training */
int   digit = nn_classify(input);     /* argmax class */
/* or nn_forward(input, output) for the full probability vector */
```

RAM cost is fixed at compile time: `2 * NN_MAX_WIDTH` floats of scratch plus the
output buffer. Weights stay in flash via the `const` arrays.

Point the client at a differently-named header with
`-DNN_MODEL_HEADER='"my_model.h"'`.

## Verifying the client (desktop)

Two stages, from the repo root. Stage 1 links the training library and needs
BLAS; stage 2 is pure C, like the real firmware:

```bash
# Stage 1: build a net, export nn_model.h, record library predictions -> ref.bin
gcc library/neural_net.c client/gen_reference.c -I. -lm -lcblas -O2 -o gen_reference
./gen_reference

# Stage 2: replay ref.bin through the client, check every argmax matches
gcc client/nn_infer.c client/client_test.c -Iclient -lm -O2 -Wall -o client_test
./client_test        # expect "500/500 predictions match"
```

## Notes / caveats

- **AVR / Arduino (Uno, Nano):** these Harvard-architecture chips need weights
  read from flash with `PROGMEM` + `pgm_read_float()`. The current header uses
  plain `const`, which already lands in flash on ARM Cortex-M, ESP32, Pico, and
  Raspberry Pi. A `PROGMEM` export variant is a straightforward follow-up.
- **Deliberate divergence from the training forward pass:** the training-side
  `add_biases_simd()` in `neural_net.c` strides by 4 while adding only 2 doubles,
  so it skips biases on some neurons. The client adds *every* bias (correct).
  With trained (non-zero) biases the client's outputs can therefore differ from
  the desktop's own numbers — the client is the correct one. Verification uses
  an untrained net (biases = 0) so this doesn't mask real mismatches.
