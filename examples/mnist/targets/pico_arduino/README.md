# Target: pico_arduino (Raspberry Pi Pico / ESP32)

Runs the trained MNIST classifier on a microcontroller using the Arduino
framework. `mnist_infer.ino` fills a 784-pixel buffer (from a camera on a real
device), scales it as training did, calls `nn_classify()`, and prints the digit
over Serial.

**Board requirement:** the model is ~109k params (~427 KB as `float`), so it
needs a board with room in flash — a **Raspberry Pi Pico (2 MB)** or **ESP32
(4 MB)**. It does **not** fit an Arduino Uno/Nano (32 KB). RAM use is tiny.

## Steps

1. From this directory:

   ```bash
   make prepare
   ```

   Trains the net if needed and copies `nn_infer.c`, `nn_infer.h`, and the
   trained model (as `nn_model.h`) in beside the `.ino`.

2. Open `mnist_infer.ino` in the Arduino IDE.

3. Install the board core (*Raspberry Pi Pico/RP2040* via arduino-pico, or
   *esp32*) and select the board + port.

4. Upload, then open the Serial Monitor at **115200 baud**:

   ```
   predicted digit: <n>
   ```

## Notes

- The sketch ships with a blank input so it compiles and demonstrates the call.
  Feed real pixels (`pixels[i] = raw_pixel / 512.0f`) to get a real prediction;
  use the `self_test` target to verify correctness against the training library.
- `make clean` removes the copied `nn_infer.*` / `nn_model.h` (generated,
  gitignored). Only `mnist_infer.ino` and this README are tracked.
- Weights land in flash via the `const` arrays on Pico/ESP32. Match the input
  scaling to training (`/512.0`); baking the scale into the header is a roadmap
  item.
