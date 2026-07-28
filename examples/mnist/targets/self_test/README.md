# Target: self_test (desktop)

Verifies the whole train → export → deploy path on your desktop. It builds the
embedded client (`client.c`, linked only against the pure-C `client/nn_infer.c`
plus the generated `mnist_model.h` — **no BLAS, no SIMD, no training code**) and
replays 1000 held-out test images, asserting `nn_classify()` reproduces the
training library's predictions.

```bash
make run          # builds the model first if needed (training takes a few minutes)
```

Expected tail:

```
client reproduced 1000/1000 training-library predictions on held-out data.
  PASS: end-to-end path verified (train -> export -> client inference).
```

The MNIST model is ~109k params (~427 KB as `float`) — it fits ESP32/Pico/Pi
flash, but not an Arduino Uno. See the `pico_arduino` target for on-device use.
