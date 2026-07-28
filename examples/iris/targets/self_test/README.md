# Target: self_test (desktop)

Verifies the whole train → export → deploy path on your desktop. It builds the
embedded client (`client.c`, linked only against the pure-C `client/nn_infer.c`
plus the generated `iris_model.h` — **no BLAS, no SIMD, no training code**, just
like firmware) and replays the held-out samples the trainer dumped, asserting
`nn_classify()` reproduces the training library's predictions.

```bash
make run          # builds the model first if needed, then builds + runs the client
```

Expected tail:

```
client reproduced 30/30 training-library predictions on held-out data.
  PASS: end-to-end path verified (train -> export -> client inference).
```

`client.c` takes the sample-dump path as an argument (defaults to
`iris_samples.bin`); the Makefile points it at the copy in the example root. On
real hardware you delete the file I/O and feed live sensor data into
`nn_classify()` — see the `pico_arduino` target for that.
