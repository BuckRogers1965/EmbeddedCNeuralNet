// Embedded inference client for Fashion-MNIST, run on the desktop as an
// end-to-end test of the WHOLE deploy path. Links ONLY against the pure-C client
// in client/ (no BLAS, no SIMD, no training code) plus the generated
// fashion_mnist_model.h -- exactly what you would compile into firmware. It
// replays the samples the trainer dumped and confirms nn_classify() reproduces
// the training library's predictions, proving train -> export -> deploy is faithful.
//
// On real hardware you delete the file I/O and feed camera pixels into
// nn_classify() instead; the nn_infer core is byte-for-byte identical.

#include <stdio.h>
#include <stdlib.h>
#include "nn_infer.h"

static const char *CLASS_NAMES[10] = {
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
};

int main(int argc, char **argv) {
    const char *path = (argc > 1) ? argv[1] : "fashion_mnist_samples.bin";
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return 1; }

    int n = 0, isz = 0;
    if (fread(&n, sizeof(int), 1, f) != 1 || fread(&isz, sizeof(int), 1, f) != 1) {
        fprintf(stderr, "fashion_mnist_samples.bin: bad header\n"); return 1;
    }
    if (isz != nn_input_size()) {
        fprintf(stderr, "input size mismatch: samples=%d, model=%d\n", isz, nn_input_size());
        return 1;
    }

    float *x = malloc(isz * sizeof(float));
    int mismatches = 0, shown = 0;
    for (int s = 0; s < n; ++s) {
        int reference = -1;
        if (fread(x, sizeof(float), isz, f) != (size_t)isz ||
            fread(&reference, sizeof(int), 1, f) != 1) {
            fprintf(stderr, "fashion_mnist_samples.bin: truncated at sample %d\n", s); return 1;
        }
        int got = nn_classify(x);
        if (shown < 5) { printf("  client sample %d -> %s\n", s, CLASS_NAMES[got]); ++shown; }
        if (got != reference) ++mismatches;
    }
    fclose(f);
    free(x);

    printf("client reproduced %d/%d training-library predictions on held-out data.\n",
           n - mismatches, n);
    if (mismatches) {
        printf("  %d mismatch(es) -- the deploy path is NOT faithful.\n", mismatches);
        return 1;
    }
    printf("  PASS: end-to-end path verified (train -> export -> client inference).\n");
    return 0;
}
