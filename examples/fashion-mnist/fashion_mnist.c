#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "neural_net.h"

// Fashion-MNIST uses the exact same IDX file format and dimensions as MNIST:
// 28x28 grayscale, 60,000 train + 10,000 test, 10 classes. Only the images
// differ (clothing instead of digits), so the loader is identical to mnist.c.

#define INPUT_SIZE   784
#define HIDDEN1_SIZE 128
#define HIDDEN2_SIZE  64
#define OUTPUT_SIZE   10

#define TRAINING_SIZE 60000
#define TEST_SIZE     10000

static const char *CLASS_NAMES[OUTPUT_SIZE] = {
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
};

void load_idx_data(const char *image_path, const char *label_path, double **images, double **labels, int size, int input_size, int output_size) {
    FILE *image_file = fopen(image_path, "rb");
    FILE *label_file = fopen(label_path, "rb");

    if (!image_file || !label_file) {
        perror("Failed to open Fashion-MNIST data files");
        exit(1);
    }

    fseek(image_file, 16, SEEK_SET); // Skip the header
    fseek(label_file, 8, SEEK_SET);  // Skip the header

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < input_size; ++j) {
            images[i][j] = fgetc(image_file) / 512.0;
        }

        int label = fgetc(label_file);
        for (int k = 0; k < output_size; ++k) {
            labels[i][k] = (k == label) ? 1.0 : 0.0;
        }
    }

    fclose(image_file);
    fclose(label_file);
}

int main() {
    srand(time(NULL));

    double **train_images = (double **)malloc(TRAINING_SIZE * sizeof(double *));
    double **train_labels = (double **)malloc(TRAINING_SIZE * sizeof(double *));
    double **test_images = (double **)malloc(TEST_SIZE * sizeof(double *));
    double **test_labels = (double **)malloc(TEST_SIZE * sizeof(double *));

    if (!train_images || !train_labels || !test_images || !test_labels) {
        perror("Failed to allocate memory for image and label arrays");
        exit(1);
    }

    for (int i = 0; i < TRAINING_SIZE; ++i) {
        train_images[i] = (double *)malloc(INPUT_SIZE * sizeof(double));
        train_labels[i] = (double *)malloc(OUTPUT_SIZE * sizeof(double));
        if (!train_images[i] || !train_labels[i]) {
            perror("Failed to allocate memory for training images and labels");
            exit(1);
        }
    }

    for (int i = 0; i < TEST_SIZE; ++i) {
        test_images[i] = (double *)malloc(INPUT_SIZE * sizeof(double));
        test_labels[i] = (double *)malloc(OUTPUT_SIZE * sizeof(double));
        if (!test_images[i] || !test_labels[i]) {
            perror("Failed to allocate memory for test images and labels");
            exit(1);
        }
    }

    printf("Loading Fashion-MNIST training data set. \n");
    load_idx_data("training_data/train-images-idx3-ubyte", "training_data/train-labels-idx1-ubyte", train_images, train_labels, TRAINING_SIZE, INPUT_SIZE, OUTPUT_SIZE);
    printf("Loading Fashion-MNIST testing data set. \n");
    load_idx_data("training_data/t10k-images-idx3-ubyte", "training_data/t10k-labels-idx1-ubyte", test_images, test_labels, TEST_SIZE, INPUT_SIZE, OUTPUT_SIZE);

    printf("Creating neural net and loading layers. \n");
    NeuralNet *net = create_neural_net(INPUT_SIZE, 30, 64, 0.002, -0.01, OPT_GRADIENT_DESCENT);

    add_layer(net, HIDDEN1_SIZE, ACTIVATION_RELU);
    add_layer(net, HIDDEN2_SIZE, ACTIVATION_RELU);
    add_layer(net, OUTPUT_SIZE, ACTIVATION_SOFTMAX);

    printf("Starting batch size: %d \n", get_batch_size(net));
    train(net, train_images, train_labels, TRAINING_SIZE);

    set_learning_rate(net, 0.001);
    set_batch_size(net, 32);
    train(net, train_images, train_labels, TRAINING_SIZE);

    set_learning_rate(net, 0.0005);
    set_batch_size(net, 8);
    train(net, train_images, train_labels, TRAINING_SIZE);

    printf("Testing Fashion-MNIST test set of 10,000 samples: \n");
    test(net, test_images, test_labels, TEST_SIZE);

    // Show a few predictions with their human-readable class names.
    printf("Sample predictions:\n");
    for (int i = 0; i < 5; ++i) {
        int predicted = classify(net, test_images[i]);
        printf("  test[%d] -> %s\n", i, CLASS_NAMES[predicted]);
    }

    // Export the trained net for the embedded client, and dump a subset of the
    // test set (as float) with this net's prediction for each, so client.c can
    // replay them and prove the deployed model reproduces these decisions -- the
    // point of the example is testing this whole path. See README.md / Makefile.
    export_inference_header(net, "fashion_mnist_model.h");
    {
        int dump = 1000; // subset; full 10k images would be ~31 MB
        FILE *sf = fopen("fashion_mnist_samples.bin", "wb");
        if (sf) {
            fwrite(&dump, sizeof(int), 1, sf);
            int isz = INPUT_SIZE;
            fwrite(&isz, sizeof(int), 1, sf);
            float buf[INPUT_SIZE];
            for (int i = 0; i < dump; ++i) {
                for (int j = 0; j < INPUT_SIZE; ++j) buf[j] = (float)test_images[i][j];
                int reference = classify(net, test_images[i]);
                fwrite(buf, sizeof(float), isz, sf);
                fwrite(&reference, sizeof(int), 1, sf);
            }
            fclose(sf);
            printf("Wrote fashion_mnist_samples.bin (%d samples) for the client test.\n", dump);
        }
    }

    for (int i = 0; i < TRAINING_SIZE; ++i) {
        free(train_images[i]);
        free(train_labels[i]);
    }
    for (int i = 0; i < TEST_SIZE; ++i) {
        free(test_images[i]);
        free(test_labels[i]);
    }
    free(train_images);
    free(train_labels);
    free(test_images);
    free(test_labels);

    free_neural_net(net);

    return 0;
}
