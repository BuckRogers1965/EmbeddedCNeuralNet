// Iris species classifier for Raspberry Pi Pico / Arduino.
//
// This is the whole edge program. Run `make prepare` in this folder first: it
// copies the pure-C inference client (nn_infer.c/.h) and the trained model
// (as nn_model.h) in beside this sketch, so the folder opens directly in the
// Arduino IDE. Then select your board and flash. Open the Serial Monitor at
// 115200 baud to see the prediction.
//
// The model is ~131 parameters (~0.5 KB as float), so it fits even an Arduino
// Uno. Inference is integer/float math with no dynamic allocation.

#include "nn_infer.h"

static const char *SPECIES[3] = { "Iris-setosa", "Iris-versicolor", "Iris-virginica" };

// One flower's 4 features: sepal length, sepal width, petal length, petal width.
// Scale EXACTLY as training did -- the desktop loader divides each raw feature
// by 8.0. On real hardware, read these from your sensor and apply the same scale.
static float read_sample(float *x) {
  const float raw[4] = { 5.1f, 3.5f, 1.4f, 0.2f }; // a clear Iris-setosa
  for (int i = 0; i < 4; ++i) x[i] = raw[i] / 8.0f;
  return 0;
}

void setup() {
  Serial.begin(115200);
  while (!Serial) { /* wait for USB serial on native-USB boards */ }

  float x[4];
  read_sample(x);

  int cls = nn_classify(x);
  Serial.print("predicted: ");
  Serial.println(SPECIES[cls]);
}

void loop() {
  // A real device would read a fresh sample here and classify on demand.
}
