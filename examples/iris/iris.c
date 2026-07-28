#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "neural_net.h"

// Iris: 150 samples, 4 numeric features, 3 species. A tiny CSV dataset that
// shows the library is not image-specific -- the only thing that changes per
// example is how you load and normalize the data into double** arrays.

#define INPUT_SIZE   4
#define HIDDEN_SIZE  16
#define OUTPUT_SIZE  3

#define TOTAL_SIZE   150
#define TRAIN_SIZE   120   // first 120 after shuffling
#define TEST_SIZE    30    // remaining 30

// Features are all in roughly [0, 8]; dividing by this keeps inputs in [0, 1].
#define FEATURE_SCALE 8.0

static const char *CLASS_NAMES[OUTPUT_SIZE] = {
    "Iris-setosa", "Iris-versicolor", "Iris-virginica"
};

static int class_index(const char *name) {
    for (int i = 0; i < OUTPUT_SIZE; ++i)
        if (strcmp(name, CLASS_NAMES[i]) == 0)
            return i;
    return -1;
}

// Loads all 150 rows into the caller's arrays (one-hot labels). Returns the
// number of rows actually read.
static int load_iris(const char *path, double **features, double **labels) {
    FILE *f = fopen(path, "r");
    if (!f) {
        perror("Failed to open training_data/iris.data");
        exit(1);
    }

    double a, b, c, d;
    char name[64];
    int n = 0;
    while (n < TOTAL_SIZE &&
           fscanf(f, "%lf,%lf,%lf,%lf,%63s", &a, &b, &c, &d, name) == 5) {
        int label = class_index(name);
        if (label < 0) {
            fprintf(stderr, "Unknown class '%s' on row %d\n", name, n);
            exit(1);
        }
        features[n][0] = a / FEATURE_SCALE;
        features[n][1] = b / FEATURE_SCALE;
        features[n][2] = c / FEATURE_SCALE;
        features[n][3] = d / FEATURE_SCALE;
        for (int k = 0; k < OUTPUT_SIZE; ++k)
            labels[n][k] = (k == label) ? 1.0 : 0.0;
        ++n;
    }

    fclose(f);
    return n;
}

// Fisher-Yates shuffle of the (features, labels) rows in lockstep.
static void shuffle_rows(double **features, double **labels, int n) {
    for (int i = n - 1; i > 0; --i) {
        int j = rand() % (i + 1);
        double *tf = features[i]; features[i] = features[j]; features[j] = tf;
        double *tl = labels[i];   labels[i]   = labels[j];   labels[j]   = tl;
    }
}

int main() {
    srand(time(NULL));

    double **features = malloc(TOTAL_SIZE * sizeof(double *));
    double **labels   = malloc(TOTAL_SIZE * sizeof(double *));
    if (!features || !labels) { perror("alloc"); exit(1); }
    for (int i = 0; i < TOTAL_SIZE; ++i) {
        features[i] = malloc(INPUT_SIZE * sizeof(double));
        labels[i]   = malloc(OUTPUT_SIZE * sizeof(double));
        if (!features[i] || !labels[i]) { perror("alloc row"); exit(1); }
    }

    printf("Loading Iris data set.\n");
    int n = load_iris("training_data/iris.data", features, labels);
    if (n != TOTAL_SIZE) {
        fprintf(stderr, "Expected %d rows, read %d. Check training_data/iris.data.\n",
                TOTAL_SIZE, n);
        exit(1);
    }

    // Shuffle once, then split into fixed train / test partitions.
    shuffle_rows(features, labels, TOTAL_SIZE);
    double **train_features = features;
    double **train_labels   = labels;
    double **test_features  = features + TRAIN_SIZE;
    double **test_labels    = labels   + TRAIN_SIZE;

    printf("Creating neural net (4 -> 16 -> 3, Adam optimizer).\n");
    NeuralNet *net = create_neural_net(INPUT_SIZE, 250, 16, 0.01, 0.0, OPT_ADAM);
    // Leaky ReLU (not plain ReLU) so hidden units don't die under Adam's
    // aggressive early steps on this tiny dataset.
    add_layer(net, HIDDEN_SIZE, ACTIVATION_LEAKY_RELU);
    add_layer(net, OUTPUT_SIZE, ACTIVATION_SOFTMAX);

    train(net, train_features, train_labels, TRAIN_SIZE);

    printf("Testing on held-out %d samples:\n", TEST_SIZE);
    test(net, test_features, test_labels, TEST_SIZE);

    printf("Sample predictions:\n");
    for (int i = 0; i < 5; ++i) {
        int predicted = classify(net, test_features[i]);
        printf("  test[%d] -> %s\n", i, CLASS_NAMES[predicted]);
    }

    // Export the trained net as a C header for the embedded client (client/).
    // This 4 -> 16 -> 3 net is only ~131 parameters (~0.5 KB as float), so it
    // fits on even small microcontrollers. See the "Deploying to an embedded
    // device" section of the top-level README for the full flow.
    export_inference_header(net, "iris_model.h");

    // Dump the held-out test samples (as float) plus this net's prediction for
    // each, so client.c -- built only against the pure-C inference client -- can
    // replay them and confirm the deployed model reproduces these decisions.
    // This is what makes the example test the *whole* path, not just training.
    FILE *sf = fopen("iris_samples.bin", "wb");
    if (sf) {
        int n = TEST_SIZE, isz = INPUT_SIZE;
        fwrite(&n, sizeof(int), 1, sf);
        fwrite(&isz, sizeof(int), 1, sf);
        float buf[INPUT_SIZE];
        for (int i = 0; i < TEST_SIZE; ++i) {
            for (int j = 0; j < INPUT_SIZE; ++j) buf[j] = (float)test_features[i][j];
            int reference = classify(net, test_features[i]);
            fwrite(buf, sizeof(float), isz, sf);
            fwrite(&reference, sizeof(int), 1, sf);
        }
        fclose(sf);
        printf("Wrote iris_samples.bin (%d held-out samples) for the client test.\n", n);
    }

    for (int i = 0; i < TOTAL_SIZE; ++i) {
        free(features[i]);
        free(labels[i]);
    }
    free(features);
    free(labels);
    free_neural_net(net);

    return 0;
}
