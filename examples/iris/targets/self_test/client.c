// The embedded inference client, exercised on the desktop as an end-to-end test
// of the WHOLE deploy path. This program links ONLY against the pure-C client
// in client/ (no BLAS, no SIMD, no training code) plus the generated
// iris_model.h -- exactly what you would compile into firmware. It replays the
// held-out samples the trainer dumped and confirms nn_classify() reproduces the
// training library's predictions, proving train -> export -> deploy is faithful.
//
// On real hardware you delete the file I/O and feed live sensor readings into
// nn_classify() instead; the nn_infer core is byte-for-byte identical.

#include <stdio.h>
#include <stdlib.h>
#include "nn_infer.h"

static const char *SPECIES[3] = {"Iris-setosa", "Iris-versicolor", "Iris-virginica"};

int main(int argc, char **argv) {
    const char *path = (argc > 1) ? argv[1] : "iris_samples.bin";
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return 1; }

    int n = 0, isz = 0;
    if (fread(&n, sizeof(int), 1, f) != 1 || fread(&isz, sizeof(int), 1, f) != 1) {
        fprintf(stderr, "iris_samples.bin: bad header\n"); return 1;
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
            fprintf(stderr, "iris_samples.bin: truncated at sample %d\n", s); return 1;
        }
        int got = nn_classify(x);
        if (shown < 3) {  // show a few real predictions the deployed client makes
            printf("  client sample %d -> %s\n", s, SPECIES[got]);
            ++shown;
        }
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
