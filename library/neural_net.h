#ifndef NEURAL_NET_H
#define NEURAL_NET_H

struct NeuralNet; // Forward declaration of the structure
typedef struct NeuralNet NeuralNet; // Opaque pointer to the neural network

// Enum definitions (keep these visible to users)
typedef enum {
    OPT_GRADIENT_DESCENT,
    OPT_MOMENTUM,
    OPT_RMSPROP,
    OPT_ADAM,
    OPT_NAG,
    NUM_OPTIMIZATIONS
} OptimizationMethod;

typedef enum {
    ACTIVATION_SIGMOID,
    ACTIVATION_TANH,
    ACTIVATION_RELU,
    ACTIVATION_LEAKY_RELU,
    ACTIVATION_ELU,
    ACTIVATION_SWISH,
    ACTIVATION_SOFTPLUS,
    ACTIVATION_SOFTMAX,
    ACTIVATION_SELU,
    ACTIVATION_GELU,
    ACTIVATION_MISH,
    ACTIVATION_HARD_SIGMOID,
    NUM_ACTIVATIONS
} ActivationFunction;

// Creation and destruction
NeuralNet *create_neural_net(int input_size, int epochs, int batchsize, double learningrate, double adj_lr_epoch, OptimizationMethod opt_method);
void free_neural_net(NeuralNet *net);

// Layer management
void add_layer(NeuralNet *net, int output_size, ActivationFunction activation);

// Training and testing
void train(NeuralNet *net, double **images, double **labels, int trainsize);
void test(NeuralNet *net, double **images, double **labels, int testsize);

// Persistence
void save_neural_net(NeuralNet *net, const char *filename);
NeuralNet *load_neural_net(const char *filename);

// Classification
int classify(NeuralNet *net, double *input);

// Accessor functions
int get_input_size(NeuralNet *net);
int get_num_layers(NeuralNet *net);
int get_epochs(NeuralNet *net);
void set_epochs(NeuralNet *net, int epochs);
int get_current_epoch(NeuralNet *net);
double get_learning_rate(NeuralNet *net);
void set_learning_rate(NeuralNet *net, double learning_rate);
int get_batch_size(NeuralNet *net);
void set_batch_size(NeuralNet *net, int batch_size);
double get_adj_lr_epoch(NeuralNet *net);
void set_adj_lr_epoch(NeuralNet *net, double adj_lr_epoch);
int get_use_jitter(NeuralNet *net);
void set_use_jitter(NeuralNet *net, int use_jitter);
double get_jitter_strength(NeuralNet *net);
void set_jitter_strength(NeuralNet *net, double jitter_strength);
double get_jitter_decay_rate(NeuralNet *net);
void set_jitter_decay_rate(NeuralNet *net, double jitter_decay_rate);
OptimizationMethod get_optimization_method(NeuralNet *net);
void set_optimization_method(NeuralNet *net, OptimizationMethod method);

#endif