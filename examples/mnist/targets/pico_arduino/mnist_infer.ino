// MNIST digit classifier for Raspberry Pi Pico / ESP32 (Arduino framework).
//
// Run `make prepare` in this folder first: it copies the pure-C inference client
// (nn_infer.c/.h) and the trained model (as nn_model.h) in beside this sketch,
// so the folder opens directly in the Arduino IDE. Select your board and flash,
// then open the Serial Monitor at 115200 baud.
//
// NOTE ON FIT: this model is ~109k parameters (~427 KB as float). It fits the
// flash of a Raspberry Pi Pico (2 MB) or ESP32 (4 MB), but NOT an Arduino Uno
// (32 KB). RAM use is tiny: 2 * NN_MAX_WIDTH floats of scratch.

#include "nn_infer.h"

// A 28x28 = 784 pixel input. On a real device you fill this from a camera and
// scale EXACTLY as training did -- the desktop loader divides each 0..255 pixel
// by 512.0. Here it is left blank so the sketch compiles and demonstrates the
// call; feed real pixels to get a real digit.
static float pixels[784];

void setup() {
  Serial.begin(115200);
  while (!Serial) { /* wait for USB serial on native-USB boards */ }

  for (int i = 0; i < 784; ++i) pixels[i] = 0.0f;   // replace with camera / raw_pixel / 512.0f

  int digit = nn_classify(pixels);
  Serial.print("predicted digit: ");
  Serial.println(digit);
}

void loop() {
  // A real device would capture a fresh frame here and classify on demand.
}
