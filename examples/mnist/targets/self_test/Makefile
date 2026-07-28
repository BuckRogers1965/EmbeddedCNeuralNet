# Deployment target: self_test (desktop).
# Builds the pure-C client (client/nn_infer.c + the generated model header, no
# BLAS, no SIMD) and replays the trainer's held-out samples through it, asserting
# the deployed model reproduces the training library's predictions.
#
#   make run     # (builds the model first if needed)

NAME       = mnist
ROOT       = ../../../..
CLIENT     = $(ROOT)/client/nn_infer.c
CLIENT_INC = -I$(ROOT)/client
# EXDIR = the example root, where the trainer wrote the model header + samples.
EXDIR      = ../..

.PHONY: run clean
$(EXDIR)/$(NAME)_model.h $(EXDIR)/$(NAME)_samples.bin:
	$(MAKE) -C $(EXDIR) model

client: client.c $(CLIENT) $(EXDIR)/$(NAME)_model.h
	gcc $(CLIENT) client.c $(CLIENT_INC) -I$(EXDIR) -DNN_MODEL_HEADER='"$(NAME)_model.h"' -lm -O2 -o client

run: client $(EXDIR)/$(NAME)_samples.bin
	./client $(EXDIR)/$(NAME)_samples.bin

clean:
	rm -f client
