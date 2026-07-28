# Target: pico_arduino (Raspberry Pi Pico / Arduino)

Runs the trained Iris classifier on a microcontroller using the Arduino
framework. The sketch (`iris_infer.ino`) reads a sample, scales it exactly as
training did, calls `nn_classify()`, and prints the species over Serial.

The Iris model is tiny (~131 params, ~0.5 KB as `float`), so it fits essentially
any board — including an Arduino Uno.

## Steps

1. **Generate the model and assemble the sketch folder.** From this directory:

   ```bash
   make prepare
   ```

   This trains the net if needed (via the example-root Makefile) and copies
   `nn_infer.c`, `nn_infer.h`, and the trained model (as `nn_model.h`) in beside
   the `.ino`. The Arduino IDE compiles every `.c`/`.h` in the sketch folder, so
   the folder is now self-contained.

2. **Open `iris_infer.ino` in the Arduino IDE** (or `arduino-cli`).

3. **Install your board's core** if needed — e.g. *Raspberry Pi Pico/RP2040*
   (arduino-pico), *esp32*, or an AVR board — and select the board + port.

4. **Upload**, then open the Serial Monitor at **115200 baud**. Expected:

   ```
   predicted: Iris-setosa
   ```

## Notes

- `make clean` removes the copied `nn_infer.*` / `nn_model.h` (they are
  generated and gitignored; only `iris_infer.ino` and this README are tracked).
- **Weights live in flash** via the `const` arrays in `nn_model.h` on ARM/Xtensa
  boards (Pico, ESP32) and, for a model this small, on AVR too. Large models on
  AVR (Uno/Nano) would need a `PROGMEM` export — that's on the project roadmap.
- **Match the input scaling to training.** The desktop Iris loader divides each
  raw feature by `8.0`; the sketch does the same. If you change preprocessing in
  the trainer, change it here too. (Baking the scale into the exported header is
  a roadmap item.)
