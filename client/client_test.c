/*
 * Verification stage 2 (edge client, no BLAS / no SIMD / no malloc paths in the
 * library it exercises).
 *
 * Replays client/ref.bin through the standalone client and checks each argmax
 * against the training library's reference prediction.
 *
 * Build (from repo root, AFTER running ./gen_reference so nn_model.h exists):
 *   gcc client/nn_infer.c client/client_test.c -Iclient -lm -O2 -o client_test
 *   ./client_test
 */

#include <stdio.h>
#include <stdlib.h>
#include "nn_infer.h"

int main(void)
{
    FILE *ref = fopen("client/ref.bin", "rb");
    if (!ref) { perror("client/ref.bin (run ./gen_reference first)"); return 1; }

    int n = 0, isz = 0;
    if (fread(&n, sizeof(int), 1, ref) != 1 ||
        fread(&isz, sizeof(int), 1, ref) != 1) {
        fprintf(stderr, "ref.bin: truncated header\n");
        return 1;
    }
    if (isz != nn_input_size()) {
        fprintf(stderr, "input size mismatch: ref.bin=%d, model=%d\n", isz, nn_input_size());
        return 1;
    }

    float *in = malloc(isz * sizeof(float));
    int mismatches = 0;

    for (int s = 0; s < n; s++) {
        int refpred = -1;
        if (fread(in, sizeof(float), isz, ref) != (size_t)isz ||
            fread(&refpred, sizeof(int), 1, ref) != 1) {
            fprintf(stderr, "ref.bin: truncated at sample %d\n", s);
            return 1;
        }
        int got = nn_classify(in);
        if (got != refpred) {
            if (mismatches < 10)
                printf("  mismatch sample %d: client=%d reference=%d\n", s, got, refpred);
            mismatches++;
        }
    }

    fclose(ref);
    free(in);

    printf("%d/%d predictions match the training library (%d mismatches).\n",
           n - mismatches, n, mismatches);
    return mismatches == 0 ? 0 : 1;
}
