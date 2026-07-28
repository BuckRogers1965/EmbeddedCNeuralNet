#ifndef NN_INFER_H
#define NN_INFER_H

/*
 * Standalone edge-inference client for EmbeddedCNeuralNet.
 *
 * No training, no BLAS, no SIMD, no malloc. Pure C99 forward pass over a model
 * baked into firmware as a C header (see export_inference_header() on the
 * training side, which emits nn_model.h). nn_infer.c #includes that model
 * header; the rest of the firmware only needs this API.
 *
 * To point the client at a different generated header, define NN_MODEL_HEADER
 * before compiling nn_infer.c, e.g. -DNN_MODEL_HEADER='"my_model.h"'.
 */

#ifdef __cplusplus
extern "C" {
#endif

/* Run the network forward.
 * input  : nn_input_size() floats (pre-scaled the same way as training).
 * output : receives nn_output_size() floats (post-activation of the last layer).
 */
void nn_forward(const float *input, float *output);

/* Forward pass followed by argmax over the output vector. Returns the class
 * index, or -1 if the network has no outputs. */
int nn_classify(const float *input);

/* Model geometry, so callers can size their own buffers without including the
 * generated header. */
int nn_input_size(void);
int nn_output_size(void);

#ifdef __cplusplus
}
#endif

#endif /* NN_INFER_H */
