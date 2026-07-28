# Deployment target: pico_arduino.
# Assembles a self-contained Arduino sketch folder by copying the pure-C client
# and the trained model (as nn_model.h) in beside the .ino. After `make prepare`,
# open this folder in the Arduino IDE, pick a Pico/ESP32-class board, and flash.

NAME = fashion_mnist
ROOT = ../../../..

.PHONY: prepare clean
prepare: nn_infer.c nn_infer.h nn_model.h   ## copy client + trained model into the sketch folder

nn_infer.c: $(ROOT)/client/nn_infer.c
	cp $< $@
nn_infer.h: $(ROOT)/client/nn_infer.h
	cp $< $@
nn_model.h: ../../$(NAME)_model.h
	cp $< $@
../../$(NAME)_model.h:
	$(MAKE) -C ../.. model

clean:
	rm -f nn_infer.c nn_infer.h nn_model.h
