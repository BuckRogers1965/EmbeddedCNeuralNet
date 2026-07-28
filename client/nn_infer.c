/*
 * Edge-inference client implementation. See nn_infer.h.
 *
 * Everything here is single-precision and statically sized from the generated
 * model header, so total RAM use is 2 * NN_MAX_WIDTH floats of scratch plus the
 * output buffer. Weights live in flash via the `const` arrays in nn_model.h.
 */

#include <math.h>
#include <string.h>
#include "nn_infer.h"

/* Which generated header holds the weights. Override with
 * -DNN_MODEL_HEADER='"other_model.h"' at compile time. */
#ifndef NN_MODEL_HEADER
#define NN_MODEL_HEADER "nn_model.h"
#endif
#include NN_MODEL_HEADER

#ifndef NN_PI
#define NN_PI 3.14159265358979323846f
#endif

/* Activation codes, mirroring ActivationFunction in library/neural_net.h.
 * The generated header stores these integer codes per layer. */
enum {
    NN_ACT_SIGMOID = 0,
    NN_ACT_TANH,
    NN_ACT_RELU,
    NN_ACT_LEAKY_RELU,
    NN_ACT_ELU,
    NN_ACT_SWISH,
    NN_ACT_SOFTPLUS,
    NN_ACT_SOFTMAX,
    NN_ACT_SELU,
    NN_ACT_GELU,
    NN_ACT_MISH,
    NN_ACT_HARD_SIGMOID
};

static float nn_sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

/* Elementwise activations (everything except softmax, which needs the whole
 * vector). Formulas match the double-precision versions in neural_net.c. */
static float nn_activate(int fn, float x)
{
    switch (fn) {
    case NN_ACT_SIGMOID:
        return nn_sigmoid(x);
    case NN_ACT_TANH:
        return tanhf(x);
    case NN_ACT_RELU:
        return x > 0.0f ? x : 0.0f;
    case NN_ACT_LEAKY_RELU:
        return x > 0.0f ? x : 0.01f * x;
    case NN_ACT_ELU:
        return x >= 0.0f ? x : (expf(x) - 1.0f);
    case NN_ACT_SWISH:
        return x * nn_sigmoid(x);
    case NN_ACT_SOFTPLUS:
        return log1pf(expf(x));
    case NN_ACT_SELU: {
        const float alpha = 1.67326324235437728481f;
        const float scale = 1.05070098735548049342f;
        return scale * (x > 0.0f ? x : alpha * (expf(x) - 1.0f));
    }
    case NN_ACT_GELU:
        return 0.5f * x * (1.0f + tanhf(sqrtf(2.0f / NN_PI) *
                                        (x + 0.044715f * x * x * x)));
    case NN_ACT_MISH:
        return x * tanhf(log1pf(expf(x)));
    case NN_ACT_HARD_SIGMOID:
        if (x < -2.5f) return 0.0f;
        if (x >  2.5f) return 1.0f;
        return 0.2f * x + 0.5f;
    case NN_ACT_SOFTMAX: /* handled by nn_softmax(); identity here */
    default:
        return x;
    }
}

static void nn_softmax(float *v, int n)
{
    float max = v[0];
    for (int i = 1; i < n; i++)
        if (v[i] > max) max = v[i];

    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        v[i] = expf(v[i] - max);
        sum += v[i];
    }
    for (int i = 0; i < n; i++)
        v[i] /= sum;
}

/* Two ping-pong scratch buffers; a layer reads one and writes the other. */
static float nn_buf_a[NN_MAX_WIDTH];
static float nn_buf_b[NN_MAX_WIDTH];

void nn_forward(const float *input, float *output)
{
    const float *in = input;
    float *pool[2] = { nn_buf_a, nn_buf_b };
    int cur = 0;

    for (int l = 0; l < NN_LAYER_COUNT; l++) {
        int in_size  = nn_layer_input[l];
        int out_size = nn_layer_output[l];
        int act      = nn_layer_activation[l];
        const float *w = nn_layer_weights[l];
        const float *b = nn_layer_biases[l];
        float *out = pool[cur];

        /* Dense layer: out[i] = b[i] + sum_j in[j] * w[i*in_size + j] */
        for (int i = 0; i < out_size; i++) {
            const float *wrow = w + (size_t)i * in_size;
            float acc = b[i];
            for (int j = 0; j < in_size; j++)
                acc += in[j] * wrow[j];
            out[i] = acc;
        }

        if (act == NN_ACT_SOFTMAX)
            nn_softmax(out, out_size);
        else
            for (int i = 0; i < out_size; i++)
                out[i] = nn_activate(act, out[i]);

        in = out;      /* next layer reads this layer's output ... */
        cur ^= 1;      /* ... and writes the other buffer (never aliases). */
    }

    memcpy(output, in, NN_OUTPUT_SIZE * sizeof(float));
}

int nn_classify(const float *input)
{
    static float out[NN_OUTPUT_SIZE];
    nn_forward(input, out);

    int best = 0;
    for (int i = 1; i < NN_OUTPUT_SIZE; i++)
        if (out[i] > out[best]) best = i;
    return best;
}

int nn_input_size(void)  { return NN_INPUT_SIZE; }
int nn_output_size(void) { return NN_OUTPUT_SIZE; }
