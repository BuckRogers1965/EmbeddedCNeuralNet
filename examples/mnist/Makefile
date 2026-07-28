# Train once, exporting the model header + a held-out sample dump that the
# deployment targets under targets/ consume. Each target (targets/<name>/) has
# its own build + directions; `make self_test` runs the desktop verification one.
# Download the data into training_data/ first (see README.md).
# Note: training MNIST takes a few minutes.

NAME    = mnist
LIB     = ../../library/neural_net.c
LIB_INC = -I../../library

.PHONY: model self_test clean
$(NAME): $(NAME).c $(LIB)
	gcc $(LIB) $(NAME).c $(LIB_INC) -lcblas -lm -O2 -o $(NAME)

$(NAME)_model.h $(NAME)_samples.bin: $(NAME)
	./$(NAME)

model: $(NAME)_model.h              ## train + export model header and sample dump

self_test: model                    ## build + run the desktop client that verifies the path
	$(MAKE) -C targets/self_test run

clean:
	rm -f $(NAME) $(NAME)_model.h $(NAME)_samples.bin
	$(MAKE) -C targets/self_test clean
