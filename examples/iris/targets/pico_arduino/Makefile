# Deployment target: pico_arduino.
# Assembles a self-contained Arduino sketch folder by copying the pure-C client
# and the trained model (as nn_model.h, which nn_infer.c includes by default)
# in beside iris_infer.ino. After `make prepare`, open this folder in the
# Arduino IDE, pick your board, and flash. See README.md.

NAME = iris
ROOT = ../../../..

.PHONY: prepare clean
prepare: nn_infer.c nn_infer.h nn_model.h   ## copy client + trained model into the sketch folder

nn_infer.c: $(ROOT)/client/nn_infer.c
	cp $< $@
nn_infer.h: $(ROOT)/client/nn_infer.h
	cp $< $@

# nn_infer.c includes "nn_model.h" by default, so copy the trained model to that
# name. The model is produced by the example-root Makefile (build it if missing).
nn_model.h: ../../$(NAME)_model.h
	cp $< $@
../../$(NAME)_model.h:
	$(MAKE) -C ../.. model

clean:
	rm -f nn_infer.c nn_infer.h nn_model.h
