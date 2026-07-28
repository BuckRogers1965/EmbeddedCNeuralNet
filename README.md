# EmbeddedCNeuralNet
A Neural Net library designed for use on embedded systems like arduino, esp32, pico and raspberry pi. 

I know, I know, you are going to say, "But there is pytorch, why write your own?" 

For one reason, because it was there and I want to know how it works.  A second reason is that you can't run torch on tiny embedded machines.  

Because this library is aiming to add "classify" functions to these small embedded machines.  I have some ideas for toys, smart thermostats, weather stations, automatic lights and other things that are going to require limited AI to run on very low end hardware.

This is taking advantage of the fact that the training will be ran on desktop machines just for a few hours and then loaded onto embedded hardware to run millions of times one sample at a time. It doesn't even matter if it takes a few seconds to classify something. All you need to classify on the hardware is the forward pass with the activation functions you need to perform that action. And enough memory to hold the neural net, of course. If you choose the right activation functions it can be very efficient and fast. And of course, the smaller the memory, the smaller the neural net has to be.  But I am sure if someone is clever they can make very small networks do amazing things with the proper training. 

I started this effort around the Holidays in 2023, but it wasn't working and I couldn't figure out why.  I poked at it a few times since then without luck. Oh, and I lost 3 months because of the widow maker heart attack and open heart surgery. Two days ago I told myself that I was going to make it work come hell or high water and figured out what was wrong. The input size in the create neural net function was not hooked up to anything. The first layer just set the input size to the same size as the output, instead of to the size of the mnist data set. So it would have never worked with that error. This is how I know AI is not ready to replace actual programmers. 

In the last few days I have made some amazing progress and it is training on the mnist data set, even hitting 90% accuracy a few times.  I know I can do better.  

This is a work in progress. The last two days I got the neural networks training and added in a lot of different optmizers for applying the gradients.  Today I made the library opaque with accessor functions.  I am planning on making major changes in the next few days adding a lot of capability to the backward pass to give people a lot of choices.  

My plan for this net phase is here:  https://mystry-geek.blogspot.com/2024/08/plans-for-making-backward-pass-more.html

This library is also intended to demostrate cutting edge neural network techniques in a strait forward, easy to understand way. It is not optimized for speed, but I want it to be a very complete example of all the techniques for training neural networks. And I want it to be very easy to load in a training set, building a network and train it overnight. 

I am just putting the code out there for anyone to use with the MIT licsense. None of this stuff is hard to do.  If you can use this to train a neuralnet on a different training set, please send me the file to put here as an example to others of how to load and train on that data.  Also put in a feature request if you need a different activation fuction, optimizer, loss function, regularizer, or learning rate function you want added.  Please include example code and links to that technique in the request and give me a few weeks to add it.  Thanks!  

And if anyone can help me come up with a loss function I can use to report training process, I would be eternally grateful.  Right now I am just doing a test run of the first 1000 shuffled results at the end of each epoch and reporting that.


## What's new

A recent push moved this project from "it trains on MNIST" to "it trains
*properly* and can actually be deployed to an edge device":

- **The edge client is real and verified.** `client/` is a standalone,
  dependency-free forward pass (`float`, no BLAS, no SIMD, no `malloc`, static
  buffers) supporting all 12 activation functions. `export_inference_header()`
  bakes a trained net into a self-contained C header, and a two-stage test
  harness confirms the client's predictions match the training library exactly
  (500/500). See the "Deploying to an embedded device" section below — the whole
  chain is demonstrated with the Iris example.
- **Optimizers actually work now.** `train()` used to hardcode plain SGD and
  silently ignore the optimizer you chose. It now does proper **mini-batch
  gradient accumulation** and takes one step per batch through the selected
  method. Adam in particular went from "segfaults and is never applied" to
  converging robustly — the Iris example trains with Adam.
- **Real bugs fixed** (several caught under AddressSanitizer): the SIMD bias-add
  helper skipped half the neurons; the Adam path allocated the wrong struct and
  divided by zero on step 0; the per-epoch progress check read past small
  datasets.
- **Examples are organized around deployment targets.** Each
  `examples/<name>/` trains a model, and each directory under its `targets/` is a
  self-contained way to run that model with its own code and directions:
  `self_test/` (desktop — builds the pure-C client and verifies the whole path)
  and `pico_arduino/` (a flashable Arduino/Pico sketch). Adding a new target is
  just a new folder. `make self_test` runs the end-to-end check. MNIST,
  Fashion-MNIST, and Iris (CSV) so far.

## How to build

Each dataset lives in its own directory under `examples/`, and each keeps its
downloaded data in a `training_data/` subdirectory (which is gitignored). See
`examples/README.md` for the list of examples, and `examples/<name>/README.md`
for the exact download steps for that dataset.

```
EmbeddedCNeuralNet
├── library/
│   ├── neural_net.c
│   └── neural_net.h
├── client/                  # standalone edge-inference client (see client/README.md)
├── examples/
│   ├── README.md
│   ├── mnist/
│   │   ├── mnist.c
│   │   ├── README.md
│   │   └── training_data/   # you download MNIST here (gitignored)
│   ├── fashion-mnist/
│   └── iris/
├── python/
├── LICENSE
└── README.md
```

On my ubuntu machine I had to install BLAS first:

```
sudo apt-get install libcblas-base-dev
```

Then build and run an example **from inside its own directory** so the
`training_data/` paths resolve (using MNIST here):

```
cd examples/mnist
# download the data into training_data/ first -- see this dir's README.md
gcc ../../library/neural_net.c mnist.c -I../../library -lm -lcblas -O2 -o mnist
./mnist
```

## Deploying to an embedded device

Yes — this works end to end today, and **every example already does it**. Each
`examples/<name>/` exports a trained model that its `targets/` consume: `make
self_test` (from the example dir) trains, exports, builds the pure-C client, and
verifies the deployed model matches training; `targets/pico_arduino/` assembles a
flashable Arduino/Pico sketch (see `examples/README.md`). The walkthrough below
spells out those same steps by hand, using the Iris example because its trained
net is tiny (4 → 16 → 3, ~131 parameters, ~0.5 KB as `float`) and fits on
essentially any microcontroller.

### 1. Train on your desktop and export the model

The Iris trainer calls `export_inference_header()` after training. It writes
`iris_model.h`: a self-contained C header of `float` weights plus the network
geometry. Nothing about *training* (optimizer, epochs, learning rate) goes in it.

```
cd examples/iris
# download the data first -- see this dir's README.md
gcc ../../library/neural_net.c iris.c -I../../library -lm -lcblas -O2 -o iris
./iris
# ... trains, tests, and writes iris_model.h
```

### 2. Write the tiny program the device will run

The only device-side dependency is the client in `client/`. Your program reads a
sample, scales it **exactly** the way training did, and calls `nn_classify()`:

```c
// edge_demo.c
#include <stdio.h>
#include "nn_infer.h"
int main(void) {
    const char *species[3] = {"Iris-setosa", "Iris-versicolor", "Iris-virginica"};
    /* On real hardware this comes from a sensor. Scale identically to training
       -- the Iris loader divides each feature by 8.0. */
    float sample[4] = {5.1f/8.0f, 3.5f/8.0f, 1.4f/8.0f, 0.2f/8.0f};
    printf("predicted: %s\n", species[nn_classify(sample)]);
    return 0;
}
```

### 3. Compile client + model + your program into one binary

```
gcc ../../client/nn_infer.c edge_demo.c \
    -I../../client -I. -DNN_MODEL_HEADER='"iris_model.h"' -lm -O2 -o edge_demo
./edge_demo
# -> predicted: Iris-setosa
```

That single binary has the weights baked in and **no external dependencies**
(run `ldd ./edge_demo` — no `libcblas`). RAM use is fixed at compile time:
`2 * NN_MAX_WIDTH` floats of scratch plus the output buffer.

On a real board you swap `gcc` for the vendor toolchain — `arm-none-eabi-gcc`
for Pico/Cortex-M, the ESP-IDF / Arduino toolchain for ESP32 — on those same
three inputs. The output is a `.elf`/`.uf2` you flash. See `client/README.md`.

### Will my model fit?

Weights dominate the flash budget at 4 bytes each:

| Net                       | Params  | Flash (`float`) | Fits...                     |
|---------------------------|---------|-----------------|-----------------------------|
| Iris `4 -> 16 -> 3`       | ~131    | ~0.5 KB         | anything, incl. Arduino Uno |
| MNIST `784 -> 128 -> 64 -> 10` | ~109k | ~437 KB      | ESP32 / Pico / Pi, not Uno  |

The project's whole premise is *small* nets: pick an architecture that fits your
target's flash and leaves room for the scratch buffers, and train it well.

## Roadmap to painless embedded deployment

Working today: desktop training with real optimizers, `float` model export, a
verified pure-C inference client, and a single-binary build (all demonstrated
above). To make deployment turnkey on the smallest hardware:

- [ ] **Ship a real board sketch.** An ESP32/Pico example (Arduino or CMake)
      that reads a sensor and classifies, so there's a flashable reference — not
      just a desktop binary.
- [ ] **AVR / PROGMEM export.** Plain `const` arrays already land in flash on
      ARM/ESP32/Pico/Pi. Arduino Uno/Nano (Harvard AVR) need `PROGMEM` +
      `pgm_read_float()`; add an export variant that emits that.
- [ ] **int8 quantization.** `float` halves `double`; int8 (with per-layer
      scale/zero-point) quarters it again and unlocks no-FPU chips. Needs a
      quantizing exporter and an integer forward path in the client.
- [ ] **Bake in preprocessing.** Today you must hand-match the training-side
      input scaling on the device (a real footgun). Emit the scaling into the
      model header so the client applies it automatically.
- [ ] **Per-bias optimizer state.** Biases currently train with plain SGD even
      under Adam (the optimizer state arrays are weight-sized). Give biases their
      own state for fully-correct Adam/RMSProp/etc.
- [ ] **Fix `load_neural_net`.** Its activation-restore `switch` is stubbed
      (only sigmoid), so the desktop save/load format round-trips incorrectly.
      The edge path doesn't use it, but training-side persistence does.
- [ ] **Report a real training loss** instead of "accuracy on the first 1000
      shuffled rows" per epoch.
