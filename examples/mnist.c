#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "library/neural_net.h"

#define INPUT_SIZE   784
#define HIDDEN1_SIZE 128
#define HIDDEN2_SIZE  64
#define OUTPUT_SIZE   10

#define TRAINING_SIZE 60000
#define TEST_SIZE     10000

// Load MNIST data
void load_mnist_data(const char* image_path, const char* label_path, double **images, double **labels, int size, int input_size, int output_size) {
    FILE *image_file = fopen(image_path, "rb");
    FILE *label_file = fopen(label_path, "rb");

    if (!image_file || !label_file) {
        perror("Failed to open MNIST data files");
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
    srand(time(NULL)); // Seed random number generator

    // Allocate memory for training and test data
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

    // Load MNIST data into allocated arrays
    printf("Loading MNIST Traning data set. \n");
    load_mnist_data("../mnist_data/train-images-idx3-ubyte", "../mnist_data/train-labels-idx1-ubyte", train_images, train_labels, TRAINING_SIZE, INPUT_SIZE, OUTPUT_SIZE);
    printf("Loading MNIST Testing data set. \n");
    load_mnist_data("../mnist_data/t10k-images-idx3-ubyte", "../mnist_data/t10k-labels-idx1-ubyte", test_images, test_labels, TEST_SIZE, INPUT_SIZE, OUTPUT_SIZE);

    // Create neural network
    printf("Creating neural net and loading layers. \n");
    NeuralNet * net = create_neural_net(INPUT_SIZE, 30, 64, 0.002, -0.01, OPT_GRADIENT_DESCENT);
    
    // Add layers to neural network
    add_layer(net, HIDDEN1_SIZE, ACTIVATION_RELU);
    add_layer(net, HIDDEN2_SIZE, ACTIVATION_RELU);
    add_layer(net, OUTPUT_SIZE, ACTIVATION_SOFTMAX);

    // Train neural network
    printf("Starting batch size: %d \n", get_batch_size(net));
    train(net, train_images, train_labels, TRAINING_SIZE);
    
    set_learning_rate(net, 0.001);
    set_batch_size(net, 32);
    printf("Changing batch size: %d \n", get_batch_size(net));
    train(net, train_images, train_labels, TRAINING_SIZE);

    set_learning_rate(net, 0.0005);
    set_batch_size(net, 8);
    printf("Changing batch size: %d \n", get_batch_size(net));
    train(net, train_images, train_labels, TRAINING_SIZE);


    // Test neural network
    printf("Testing MNIST Test set of 10,000 samples: \n");
    test(net, test_images, test_labels, TEST_SIZE);

    // Free memory for training and test data
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

    // Free neural network
    free_neural_net(net);

    return 0;
}
