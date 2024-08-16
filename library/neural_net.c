#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "neural_net.h"

// Activation function type definitions
typedef double (*activation_function)(double);
typedef double (*activation_function_derivative)(double);

// Move structure definitions here
typedef struct {
    double *weights;
    double *biases;
    double *activations;
    int activation;
    double *errors;
    int input_size;
    int output_size;
    activation_function activate;
    activation_function_derivative activate_derivative;
} Layer;

struct NeuralNet {
    int input_size;
    Layer *layers;
    int num_layers;
    int epochs;
    int current_epoch;
    double learningrate;
    int batchsize;
    double adj_lr_epoch;
    int use_jitter;
    double jitter_strength;
    double jitter_decay_rate;
    OptimizationMethod opt_method;
    void *opt_params;
    void (*chooser)(void *params, double *param_to_update, int index, int another_param, double gradient, double learningrate);
};

// Private functions 
// not accessible outside the library

void chooser(NeuralNet *net, int layer_index, int weight_index, double gradient, int is_bias);
void forward_pass(NeuralNet *net, double *input, double *output);
void backward_pass(NeuralNet *net, double *input, double *expected, double *output);

// -- Activation function implementations
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}
double sigmoid_derivative(double x) {
    double sig = sigmoid(x);
    return sig * (1 - sig);
}
double tanh_activation(double x) {
    return tanh(x);
}
double tanh_derivative(double x) {
    double tanh_x = tanh(x);
    return 1 - tanh_x * tanh_x;
}
double relu(double x) {
    return x > 0 ? x : 0;
}
double relu_derivative(double x) {
    return x > 0 ? 1 : 0;
}
double leaky_relu(double x) {
    return x > 0 ? x : 0.01 * x;
}
double leaky_relu_derivative(double x) {
    return x > 0 ? 1 : 0.01;
}
double elu(double x) {
    return x >= 0 ? x : (exp(x) - 1);
}
double elu_derivative(double x) {
    return x >= 0 ? 1 : (elu(x) + 1);
}
double swish(double x) {
    return x * sigmoid(x);
}
double swish_derivative(double x) {
    double sig = sigmoid(x);
    return sig + x * sig * (1 - sig);
}
double softplus(double x) {
    return log(1 + exp(x));
}
double softplus_derivative(double x) {
    return sigmoid(x);
}
double softsign(double x) {
    return x / (1 + fabs(x));
}
double softsign_derivative(double x) {
    double abs_x = fabs(x);
    return 1 / ((1 + abs_x) * (1 + abs_x));
}
double selu(double x) {
    const double alpha = 1.6732632423543772848170429916717;
    const double scale = 1.0507009873554804934193349852946;
    return scale * (x > 0 ? x : alpha * (exp(x) - 1));
}
double selu_derivative(double x) {
    const double alpha = 1.6732632423543772848170429916717;
    const double scale = 1.0507009873554804934193349852946;
    return scale * (x > 0 ? 1 : alpha * exp(x));
}
double gelu(double x) {
    return 0.5 * x * (1 + tanh(sqrt(2 / M_PI) * (x + 0.044715 * pow(x, 3))));
}
double gelu_derivative(double x) {
    double cdf = 0.5 * (1 + tanh(sqrt(2 / M_PI) * (x + 0.044715 * pow(x, 3))));
    double pdf = exp(-0.5 * x * x) / sqrt(2 * M_PI);
    return cdf + x * pdf;
}
double mish(double x) {
    return x * tanh(log1p(exp(x)));
}
double mish_derivative(double x) {
    double exp_x = exp(x);
    double exp_2x = exp(2 * x);
    double exp_3x = exp(3 * x);
    double omega = 4 * (x + 1) + 4 * exp_2x + exp_3x + exp_x * (4 * x + 6);
    double delta = 2 * exp_x + exp_2x + 2;
    return exp_x * omega / (delta * delta);
}
double hard_sigmoid(double x) {
    if (x < -2.5) return 0;
    if (x > 2.5) return 1;
    return 0.2 * x + 0.5;
}
double hard_sigmoid_derivative(double x) {
    if (x < -2.5 || x > 2.5) return 0;
    return 0.2;
}
void softmax(double *input, double *output, int size) {
    double max = input[0];
    for (int i = 1; i < size; i++) {
        if (input[i] > max) max = input[i];
    }
    
    double sum = 0.0;
    for (int i = 0; i < size; i++) {
        output[i] = exp(input[i] - max);
        sum += output[i];
    }
    
    for (int i = 0; i < size; i++) {
        output[i] /= sum;
    }
}

// -- utility functions -- 
double jitter(NeuralNet *net) {
    if (net->use_jitter) return 0.000000001;
    double base_jitter = ((double)rand() / (double)RAND_MAX) / (net->jitter_strength/2) - 1.0 / net->jitter_strength;
    return base_jitter * (1.0 - (double)net->current_epoch / (double)net->epochs);
}
void shuffle_data(double **images, double **labels, int size) {
    for (int i = size * 0.7; i >= 0; i--) {
        int j = rand() % size;
        int k = rand() % size;
        
        // Swap images
        double *temp_image = images[k];
        images[k] = images[j];
        images[j] = temp_image;
        
        // Swap corresponding labels
        double *temp_label = labels[k];
        labels[k] = labels[j];
        labels[j] = temp_label;
    }
}

void forward_layer(double *input, double *weights, double *biases, int input_size, int output_size, double *output, activation_function activate, ActivationFunction activation) {
    for (int i = 0; i < output_size; ++i) {
        output[i] = 0.0;
        for (int j = 0; j < input_size; ++j) {
            output[i] += input[j] * weights[i * input_size + j];
        }
        output[i] += biases[i];
    }
    
    if (activation == ACTIVATION_SOFTMAX) {
        softmax(output, output, output_size);
    } else if (activate) {
        for (int i = 0; i < output_size; ++i) {
            output[i] = activate(output[i]);
        }
    }
}
void forward_pass(NeuralNet *net, double *input, double *output) {
    double *temp_input = input;

    for (int i = 0; i < net->num_layers; ++i) {
        forward_layer(temp_input, net->layers[i].weights, net->layers[i].biases,
                      net->layers[i].input_size, net->layers[i].output_size, net->layers[i].activations,
                      net->layers[i].activate, net->layers[i].activation);

        // Update input for next layer
        temp_input = net->layers[i].activations;
    }

    // Copy final output to output buffer
    memcpy(output, net->layers[net->num_layers - 1].activations, net->layers[net->num_layers - 1].output_size * sizeof(double));
}

void backward_pass(NeuralNet *net, double *input, double *expected, double *output) {
    int i, j, k;
    Layer *output_layer = &net->layers[net->num_layers - 1];

    // Calculate output layer errors
    if (output_layer->activation == ACTIVATION_SOFTMAX) {
        for (i = 0; i < output_layer->output_size; ++i) {
            output_layer->errors[i] = output[i] - expected[i];
        }
    } else {
        for (i = 0; i < output_layer->output_size; ++i) {
            output_layer->errors[i] = (output[i] - expected[i]) * output_layer->activate_derivative(output[i]);
        }
    }

    // Backpropagate errors and update weights
    for (i = net->num_layers - 1; i > 0; --i) {
        Layer *current_layer = &net->layers[i];
        Layer *prev_layer = &net->layers[i - 1];

        // Calculate gradients and update weights and biases
        for (j = 0; j < current_layer->output_size; ++j) {
            for (k = 0; k < current_layer->input_size; ++k) {
                int weight_index = j * current_layer->input_size + k;
                double gradient = current_layer->errors[j] * prev_layer->activations[k] / net->batchsize;
                //chooser(net, i, weight_index, gradient -jitter(net), 0);
                net->chooser(net->opt_params, current_layer->weights, i, weight_index, gradient - jitter(net), net->learningrate);
            }
            // Update biases
            //chooser(net, i, j, current_layer->errors[j] / net->batchsize, 1);
            net->chooser(net->opt_params, current_layer->biases, i, j, current_layer->errors[j] / net->batchsize - jitter(net), net->learningrate);

        }

        // Calculate errors for previous layer (if not input layer)
        if (i > 1) {
            for (j = 0; j < prev_layer->output_size; ++j) {
                prev_layer->errors[j] = 0.0;
                for (k = 0; k < current_layer->output_size; ++k) {
                    prev_layer->errors[j] += current_layer->errors[k] * 
                        current_layer->weights[k * current_layer->input_size + j];
                }
                prev_layer->errors[j] *= prev_layer->activate_derivative(prev_layer->activations[j]);
            }
        }
    }
}

// -- descent methods -- 

void gradient_descent(void *params, double * block, int layer_index, int index, double gradient, double learningrate) {
    block[index] -= learningrate * gradient;
}

typedef struct {
    double **velocity;
    double momentum;
} MomentumParams;
void create_momentum(NeuralNet *net) {
    MomentumParams *params = (MomentumParams *)malloc(sizeof(MomentumParams));
    params->velocity = (double **)malloc(net->num_layers * sizeof(double *));
    for (int i = 0; i < net->num_layers; i++) {
        params->velocity[i] = (double *)calloc(net->layers[i].input_size * net->layers[i].output_size, sizeof(double));
    }
    params->momentum = 0.9; // Default momentum value
    net->opt_params = params;
}
void momentum(void *vparams, double *param_to_update, int layer_index, int weight_index, double gradient, double learningrate) {
    MomentumParams *params = (MomentumParams *)vparams; 
    
    double *velocity = params->velocity[layer_index];
    
    velocity[weight_index] = params->momentum * velocity[weight_index] + learningrate * gradient;
    param_to_update[weight_index] -= velocity[weight_index];
}

typedef struct {
    double **square_gradients;
    double decay_rate;
    double epsilon;
} RMSpropParams;
void create_rmsprop(NeuralNet *net) {
    RMSpropParams *params = (RMSpropParams *)malloc(sizeof(RMSpropParams));
    params->square_gradients = (double **)malloc(net->num_layers * sizeof(double *));
    for (int i = 0; i < net->num_layers; i++) {
        params->square_gradients[i] = (double *)calloc(net->layers[i].input_size * net->layers[i].output_size, sizeof(double));
    }
    params->decay_rate = 0.9;
    params->epsilon = 1e-8;
    net->opt_params = params;
}
void rmsprop(void *vparams, double *block, int layer_index, int weight_index, double gradient, double learningrate) {
    RMSpropParams *params = (RMSpropParams *)vparams;
    
    double *param_to_update = block;
    double *square_grad = params->square_gradients[layer_index];
    
    square_grad[weight_index] = params->decay_rate * square_grad[weight_index] + 
        (1 - params->decay_rate) * gradient * gradient;
    param_to_update[weight_index] -= (learningrate * gradient) / 
        (sqrt(square_grad[weight_index]) + params->epsilon);
}

typedef struct {
    double **m;
    double **v;
    double beta1;
    double beta2;
    double epsilon;
    int t; // timestep
} AdamParams;
void create_adam(NeuralNet *net) {
    AdamParams *params = (AdamParams *)malloc(sizeof(AdamParams));
    params->m = (double **)malloc(net->num_layers * sizeof(double *));
    params->v = (double **)malloc(net->num_layers * sizeof(double *));
    for (int i = 0; i < net->num_layers; i++) {
        int size = net->layers[i].input_size * net->layers[i].output_size;
        params->m[i] = (double *)calloc(size, sizeof(double));
        params->v[i] = (double *)calloc(size, sizeof(double));
    }
    params->beta1 = 0.9;
    params->beta2 = 0.999;
    params->epsilon = 1e-8;
    params->t = 0;
    net->opt_params = params;
}
void adam(void *vparams, double *block, int layer_index, int weight_index, double gradient, double learningrate) {
    AdamParams *params = (AdamParams *)vparams;
    
    double *param_to_update = block;
    double *m = params->m[layer_index];
    double *v = params->v[layer_index];
    
    m[weight_index] = params->beta1 * m[weight_index] + (1 - params->beta1) * gradient;
    v[weight_index] = params->beta2 * v[weight_index] + (1 - params->beta2) * gradient * gradient;
    
    double m_hat = m[weight_index] / (1 - pow(params->beta1, params->t));
    double v_hat = v[weight_index] / (1 - pow(params->beta2, params->t));
    
    param_to_update[weight_index] -= (learningrate * m_hat) / (sqrt(v_hat) + params->epsilon);
}

typedef struct {
    double **velocity;
    double momentum;
} NAGParams;
void create_nag(NeuralNet *net) {
    NAGParams *params = (NAGParams *)malloc(sizeof(NAGParams));
    params->velocity = (double **)malloc(net->num_layers * sizeof(double *));
    for (int i = 0; i < net->num_layers; i++) {
        params->velocity[i] = (double *)calloc(net->layers[i].input_size * net->layers[i].output_size, sizeof(double));
    }
    params->momentum = 0.9; // Default momentum value
    net->opt_params = params;
}
void nag(void *vparams, double *block, int layer_index, int weight_index, double gradient, double learningrate) {
    NAGParams *params = (NAGParams *)vparams;
    double v_prev = params->velocity[layer_index][weight_index];
    params->velocity[layer_index][weight_index] = params->momentum * v_prev - learningrate * gradient;
    block[weight_index] += -params->momentum * v_prev + (1 + params->momentum) * params->velocity[layer_index][weight_index];
    
}

void setup_chooser_params (NeuralNet *net) {
    switch (net->opt_method) {
        case OPT_GRADIENT_DESCENT:
            net->chooser = gradient_descent;
            break;
        case OPT_MOMENTUM:
            net->chooser = momentum;
            create_momentum(net);
            break;
        case OPT_RMSPROP:
            net->chooser = rmsprop;
            create_rmsprop(net);
            break;
        case OPT_ADAM:
            net->chooser = adam;
            create_rmsprop(net);
            break;
        case OPT_NAG:
            net->chooser = nag;
            create_nag(net);
            break;
        default:
            break;
    }
}

// -- Public functions - visible to code that uses library




// Implement accessor functions
int get_input_size(NeuralNet *net) {
    return net->input_size;
}

int get_num_layers(NeuralNet *net) {
    return net->num_layers;
}

int get_epochs(NeuralNet *net) {
    return net->epochs;
}

void set_epochs(NeuralNet *net, int epochs) {
    net->epochs = epochs;
}

int get_current_epoch(NeuralNet *net) {
    return net->current_epoch;
}

double get_learning_rate(NeuralNet *net) {
    return net->learningrate;
}

void set_learning_rate(NeuralNet *net, double learning_rate) {
    net->learningrate = learning_rate;
}

int get_batch_size(NeuralNet *net) {
    return net->batchsize;
}

void set_batch_size(NeuralNet *net, int batch_size) {
    net->batchsize = batch_size;
}

double get_adj_lr_epoch(NeuralNet *net) {
    return net->adj_lr_epoch;
}

void set_adj_lr_epoch(NeuralNet *net, double adj_lr_epoch) {
    net->adj_lr_epoch = adj_lr_epoch;
}

int get_use_jitter(NeuralNet *net) {
    return net->use_jitter;
}

void set_use_jitter(NeuralNet *net, int use_jitter) {
    net->use_jitter = use_jitter;
}

double get_jitter_strength(NeuralNet *net) {
    return net->jitter_strength;
}

void set_jitter_strength(NeuralNet *net, double jitter_strength) {
    net->jitter_strength = jitter_strength;
}

double get_jitter_decay_rate(NeuralNet *net) {
    return net->jitter_decay_rate;
}

void set_jitter_decay_rate(NeuralNet *net, double jitter_decay_rate) {
    net->jitter_decay_rate = jitter_decay_rate;
}

OptimizationMethod get_optimization_method(NeuralNet *net) {
    return net->opt_method;
}

void set_optimization_method(NeuralNet *net, OptimizationMethod method) {
    net->opt_method = method;
    // Note: You might want to add logic here to reinitialize opt_params based on the new method
}

NeuralNet *create_neural_net(int input_size, int epochs, int batchsize, double learningrate, double adj_lr_epoch, OptimizationMethod opt_method) {
    NeuralNet *net = (NeuralNet *)malloc(sizeof(NeuralNet));
    if (!net) {
        perror("Failed to allocate memory for neural network");
        exit(1);
    }

    net->num_layers = 0;
    net->layers = NULL;
    net->input_size=input_size;

    net->epochs = epochs;
    net->current_epoch = 1;

    net->learningrate = learningrate;
    net->batchsize = batchsize;
    net->adj_lr_epoch = adj_lr_epoch;

    net->opt_method = opt_method;
    net->opt_params = NULL;

    net->use_jitter = 1;
    net->jitter_strength = 800;
    net->jitter_decay_rate = .95;

    return net;
}
void add_layer(NeuralNet *net, int output_size, ActivationFunction activation) {
    net->layers = (Layer *)realloc(net->layers, (net->num_layers + 1) * sizeof(Layer));
    if (!net->layers) {
        perror("Failed to allocate memory for layers");
        exit(1);
    }

    Layer *layer = &net->layers[net->num_layers];
    int input_size = net->num_layers == 0 ? net->input_size : net->layers[net->num_layers - 1].output_size;
    layer->input_size = input_size;
    layer->output_size = output_size;
    layer->weights = (double *)malloc(input_size * output_size * sizeof(double));
    layer->biases = (double *)malloc(output_size * sizeof(double));
    layer->activations = (double *)malloc(output_size * sizeof(double));
    layer->errors = (double *)malloc(output_size * sizeof(double));
    layer->activation=activation;

    if (!layer->weights || !layer->biases || !layer->activations || !layer->errors) {
        perror("Failed to allocate memory for layer parameters");
        exit(1);
    }

    // Initialize weights and biases
    for (int i = 0; i < input_size * output_size; ++i) {
        layer->weights[i] = ((double)rand() / RAND_MAX) - 0.5;
    }

    for (int i = 0; i < output_size; ++i) {
        layer->biases[i] = 0.0;
    }

    switch (activation) {
        case ACTIVATION_SIGMOID:
            layer->activate = sigmoid;
            layer->activate_derivative = sigmoid_derivative;
            break;
        case ACTIVATION_TANH:
            layer->activate = tanh_activation;
            layer->activate_derivative = tanh_derivative;
            break;
        case ACTIVATION_RELU:
            layer->activate = relu;
            layer->activate_derivative = relu_derivative;
            break;
        case ACTIVATION_LEAKY_RELU:
            layer->activate = leaky_relu;
            layer->activate_derivative = leaky_relu_derivative;
            break;
        case ACTIVATION_ELU:
            layer->activate = elu;
            layer->activate_derivative = elu_derivative;
            break;
        case ACTIVATION_SWISH:
            layer->activate = swish;
            layer->activate_derivative = swish_derivative;
            break;
        case ACTIVATION_SOFTPLUS:
            layer->activate = softplus;
            layer->activate_derivative = softplus_derivative;
            break;
        case ACTIVATION_SOFTMAX:
            layer->activate = NULL;
            layer->activate_derivative = NULL;
            break;
        case ACTIVATION_SELU:
            layer->activate = selu;
            layer->activate_derivative = selu_derivative;
            break;
        case ACTIVATION_GELU:
            layer->activate = gelu;
            layer->activate_derivative = gelu_derivative;
            break;
        case ACTIVATION_MISH:
            layer->activate = mish;
            layer->activate_derivative = mish_derivative;
            break;
        case ACTIVATION_HARD_SIGMOID:
            layer->activate = hard_sigmoid;
            layer->activate_derivative = hard_sigmoid_derivative;
            break;
        default:
            fprintf(stderr, "Unknown activation function.\n");
            exit(1);
    }

    net->num_layers++;
}

void train(NeuralNet *net, double **images, double **labels, int trainsize) {
    int epoch, batch, i;
    double *input = (double *)malloc(net->layers[0].input_size * sizeof(double));
    double *output = (double *)malloc(net->layers[net->num_layers - 1].output_size * sizeof(double));

    if (net->opt_params == NULL) setup_chooser_params(net);

        printf("train\n");

    for (net->current_epoch = 1; net->current_epoch <= net->epochs; ++net->current_epoch) {

        shuffle_data(images, labels, trainsize);
        for (batch = 0; batch < trainsize; batch += net->batchsize) {
            // Reset gradients
            for (i = 0; i < net->num_layers; ++i) {
                memset(net->layers[i].errors, 0, net->layers[i].output_size * sizeof(double));
            }

            // Process mini-batch
            for (i = 0; i < net->batchsize && (batch + i) < trainsize; ++i) {
                memcpy(input, images[batch + i], net->layers[0].input_size * sizeof(double));
                forward_pass(net, input, output);
                backward_pass(net, input, labels[batch + i], output);
            }

            // Update optimization-specific parameters if needed
            if (net->opt_method == OPT_ADAM) {
                AdamParams *params = (AdamParams *)net->opt_params;
                params->t++; // Increment timestep
            }
        }

        // Print epoch results
        printf("Epoch: %d completed, ", net->current_epoch );
        test(net, images, labels, 1000);

        // Adjust learning rate or other parameters if needed
        if (net->adj_lr_epoch > 0 && (net->current_epoch) % (int)net->adj_lr_epoch == 0) {
            net->learningrate *= 0.9;
        }
    }

    free(input);
    free(output);
}
void test(NeuralNet *net, double **images, double **labels, int testsize) {
    int correct = 0;
    double *output = (double *)malloc(net->layers[net->num_layers - 1].output_size * sizeof(double));

    for (int i = 0; i < testsize; ++i) {
        forward_pass(net, images[i], output);

        // Determine if the prediction is correct (assuming classification task)
        int predicted = 0;
        int actual = 0;
        for (int j = 1; j < net->layers[net->num_layers - 1].output_size; ++j) {
            if (output[j] > output[predicted]) {
                predicted = j;
            }
            if (labels[i][j] > labels[i][actual]) {
                actual = j;
            }
        }

        if (predicted == actual) {
            correct++;
        }
    }

    printf("Accuracy: %.2f%%\n", (double)correct / testsize * 100.0);

    free(output);
}

void save_neural_net(NeuralNet *net, const char *filename) {
    FILE *file = fopen(filename, "wb");
    if (!file) {
        perror("Failed to open file for writing");
        return;
    }

    // Write basic network information
    fwrite(&net->input_size, sizeof(int), 1, file);
    fwrite(&net->num_layers, sizeof(int), 1, file);
    fwrite(&net->epochs, sizeof(int), 1, file);
    fwrite(&net->learningrate, sizeof(double), 1, file);
    fwrite(&net->batchsize, sizeof(int), 1, file);
    fwrite(&net->adj_lr_epoch, sizeof(double), 1, file);
    fwrite(&net->opt_method, sizeof(OptimizationMethod), 1, file);

    // Write layer information and weights/biases
    for (int i = 0; i < net->num_layers; i++) {
        Layer *layer = &net->layers[i];
        fwrite(&layer->input_size, sizeof(int), 1, file);
        fwrite(&layer->output_size, sizeof(int), 1, file);
        fwrite(&layer->activation, sizeof(ActivationFunction), 1, file);
        
        // Write weights and biases
        fwrite(layer->weights, sizeof(double), layer->input_size * layer->output_size, file);
        fwrite(layer->biases, sizeof(double), layer->output_size, file);
    }

    fclose(file);
}
NeuralNet *load_neural_net(const char *filename) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        perror("Failed to open file for reading");
        return NULL;
    }

    NeuralNet *net = (NeuralNet *)malloc(sizeof(NeuralNet));
    if (!net) {
        perror("Failed to allocate memory for neural network");
        fclose(file);
        return NULL;
    }

    // Read basic network information
    fread(&net->input_size, sizeof(int), 1, file);
    fread(&net->num_layers, sizeof(int), 1, file);
    fread(&net->epochs, sizeof(int), 1, file);
    fread(&net->learningrate, sizeof(double), 1, file);
    fread(&net->batchsize, sizeof(int), 1, file);
    fread(&net->adj_lr_epoch, sizeof(double), 1, file);
    fread(&net->opt_method, sizeof(OptimizationMethod), 1, file);

    // Allocate memory for layers
    net->layers = (Layer *)malloc(net->num_layers * sizeof(Layer));
    if (!net->layers) {
        perror("Failed to allocate memory for layers");
        free(net);
        fclose(file);
        return NULL;
    }

    // Read layer information and weights/biases
    for (int i = 0; i < net->num_layers; i++) {
        Layer *layer = &net->layers[i];
        fread(&layer->input_size, sizeof(int), 1, file);
        fread(&layer->output_size, sizeof(int), 1, file);
        fread(&layer->activation, sizeof(ActivationFunction), 1, file);

        // Allocate memory for weights and biases
        layer->weights = (double *)malloc(layer->input_size * layer->output_size * sizeof(double));
        layer->biases = (double *)malloc(layer->output_size * sizeof(double));
        layer->activations = (double *)malloc(layer->output_size * sizeof(double));
        layer->errors = (double *)malloc(layer->output_size * sizeof(double));

        if (!layer->weights || !layer->biases || !layer->activations || !layer->errors) {
            perror("Failed to allocate memory for layer parameters");
            // Clean up and return NULL (implementation left as an exercise)
            fclose(file);
            return NULL;
        }

        // Read weights and biases
        fread(layer->weights, sizeof(double), layer->input_size * layer->output_size, file);
        fread(layer->biases, sizeof(double), layer->output_size, file);

        // Set activation function pointers
        switch (layer->activation) {
            case ACTIVATION_SIGMOID:
                layer->activate = sigmoid;
                layer->activate_derivative = sigmoid_derivative;
                break;
            // ... (other activation functions)
        }
    }

    fclose(file);
    return net;
}
int classify(NeuralNet *net, double *input) {
    double *output = (double *)malloc(net->layers[net->num_layers - 1].output_size * sizeof(double));
    if (!output) {
        perror("Failed to allocate memory for output");
        return -1;
    }

    forward_pass(net, input, output);

    // Find the index of the maximum output (class with highest probability)
    int max_index = 0;
    for (int i = 1; i < net->layers[net->num_layers - 1].output_size; i++) {
        if (output[i] > output[max_index]) {
            max_index = i;
        }
    }

    free(output);
    return max_index;
}

void free_neural_net(NeuralNet *net) {
    for (int i = 0; i < net->num_layers; ++i) {
        free(net->layers[i].weights);
        free(net->layers[i].biases);
        free(net->layers[i].activations);
        free(net->layers[i].errors);
    }
        // Free optimization parameters
    switch (net->opt_method) {
        case OPT_MOMENTUM:
            {
                MomentumParams *params = (MomentumParams *)net->opt_params;
                for (int i = 0; i < net->num_layers; i++) {
                    free(params->velocity[i]);
                }
                free(params->velocity);
                free(params);
            }
            break;
        case OPT_RMSPROP:
            {
                RMSpropParams *params = (RMSpropParams *)net->opt_params;
                for (int i = 0; i < net->num_layers; i++) {
                    free(params->square_gradients[i]);
                }
                free(params->square_gradients);
                free(params);
            }
            break;
        case OPT_ADAM:
            {
                AdamParams *params = (AdamParams *)net->opt_params;
                for (int i = 0; i < net->num_layers; i++) {
                    free(params->m[i]);
                    free(params->v[i]);
                }
                free(params->m);
                free(params->v);
                free(params);
            }
            break;
        case OPT_NAG:
            {
                NAGParams *params = (NAGParams *)net->opt_params;
                for (int i = 0; i < net->num_layers; i++) {
                    free(params->velocity[i]);
                }
                free(params->velocity);
                free(params);
            }
            break;
        case OPT_GRADIENT_DESCENT:
        default:
            break;
    }
    free(net->layers);
    free(net);
}
