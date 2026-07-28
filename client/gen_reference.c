/*
 * Verification stage 1 (training side, desktop only).
 *
 * Builds a deterministic net exercising several activation functions, exports
 * it to client/nn_model.h, then records the library's classify() output for a
 * batch of random inputs into client/ref.bin. Stage 2 (client_test.c) replays
 * those inputs through the standalone client and checks the predictions match.
 *
 * Build (from repo root):
 *   gcc library/neural_net.c client/gen_reference.c -I. -lm -lcblas -O2 -o gen_reference
 *   ./gen_reference
 */

#include <stdio.h>
#include <stdlib.h>
#include "library/neural_net.h"

#define INPUT_SIZE 20
#define NUM_SAMPLES 500

int main(void)
{
    srand(12345); /* deterministic weights + inputs */

    NeuralNet *net = create_neural_net(INPUT_SIZE, 1, 1, 0.01, 0.0, OPT_GRADIENT_DESCENT);
    add_layer(net, 16, ACTIVATION_RELU);
    add_layer(net, 14, ACTIVATION_TANH);
    add_layer(net, 12, ACTIVATION_GELU);
    add_layer(net, 10, ACTIVATION_LEAKY_RELU);
    add_layer(net,  8, ACTIVATION_SWISH);
    add_layer(net,  4, ACTIVATION_SOFTMAX);

    export_inference_header(net, "client/nn_model.h");

    FILE *ref = fopen("client/ref.bin", "wb");
    if (!ref) { perror("client/ref.bin"); return 1; }

    int n = NUM_SAMPLES, isz = INPUT_SIZE;
    fwrite(&n, sizeof(int), 1, ref);
    fwrite(&isz, sizeof(int), 1, ref);

    double *in  = malloc(isz * sizeof(double));
    float  *inf = malloc(isz * sizeof(float));
    for (int s = 0; s < n; s++) {
        for (int j = 0; j < isz; j++) {
            double v = ((double)rand() / RAND_MAX) * 2.0 - 1.0; /* [-1, 1] */
            in[j]  = v;
            inf[j] = (float)v;
        }
        int pred = classify(net, in);
        fwrite(inf, sizeof(float), isz, ref);
        fwrite(&pred, sizeof(int), 1, ref);
    }

    fclose(ref);
    free(in);
    free(inf);
    free_neural_net(net);

    printf("Wrote client/ref.bin: %d reference predictions.\n", n);
    return 0;
}
