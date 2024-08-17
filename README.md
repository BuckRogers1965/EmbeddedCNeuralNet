# EmbeddedCNeuralNet
A Neural Net library designed for use on embedded systems like arduino, esp32, pico and raspberry pi. 

I know, I know, you are going to say, "But there is pytorch, why write your own?" 

For one reason, because it was there and I want to know how it works.  A second reason is that you can't run torch on tiny embedded machines.  

Because this library is aiming to add "classify" functions to these small embedded machines.  I have some ideas for toys, smart thermostats, weather stations, automatic lights and other things that are going to require limited AI to run on very low end hardware.

This is taking advantage of the fact that the training will be ran on desktop machines just for a few hours and then loaded onto embedded hardware to run millions of times one sample at a time. It doesn't even matter if it takes a few seconds to classify something. All you need to classify on the hardware is the forward pass with the activation functions you need to perform that action. And enough memory to hold the neural net, of course. If you choose the right activation functions it can be very efficient and fast. And of course, the smaller the memory, the smaller the neural net has to be.  But I am sure if someone is clever they can make very small networks do amazing things with the proper training. 

I started this effort around the Holidays in 2023, but it wasn't working and I couldn't figure out why.  I poked at it a few times since then without luck. Oh, and I lost 3 months because of the widow maker heart attack and open heart surgery. Two days ago I told myself that I was going to make it work come hell or high water and figured out what was wrong. The input size in the create neural net function was not hooked up to anything. The first layer just set the input size to the same size as the output, instead of to the size of the mnist data set. So it would have never worked with that error. This is how I know AI is not ready to replace actual programmers. 

In the last few days I have made some amazing progress and it is training on the mnist data set, even hitting 90% accuracy a few times.  I know I can do better.  

This is a work in progress. The last two days I got the neural networks training and added in a lot of different optmizers for applying the gradients.  Today I made the library opaque with accessor functions.  I am planning on making major changes in the next few days adding a lot of capability to the backward pass to give people a lot of choices.  

My plan for this net phase is here:  https://mystry-geek.blogspot.com/2024/08/plans-for-making-backward-pass-more.html

This library is also intended to demostrate cutting edge neural network techniques in a strait forward, easy to understand way. It is not optimized for speed, but I want it to be a very complete example of all the techniques for training neural networks. And I want it to be very easy to load in a training set, building a network and train it overnight. 

I am just putting the code out there for anyone to use with the MIT licsense. None of this stuff is hard to do.  If you can use this to train a neuralnet on a different training set, please send me the file to put here as an example to others of how to load and train on that data.  Also put in a feature request if you need a different activation fuction, optimizer, loss function, regularizer, or learning rate function you want added.  Please include example code and links to that technique in the request and give me a few weeks to add it.  Thanks!  

And if anyone can help me come up with a loss function I can use to report training process, I would be eternally grateful.  Right now I am just doing a test run of the first 1000 shuffled results at the end of each epoch and reporting that.


How to build: 

This is how I organize the data and the github project to easily build the project.


I put the mnist data right next to the githug project and rename the directory to mnist_data.
If you have the data in a different place, you have to update mnist.c to load the data correctly. 

<code>
> .
> ├── EmbeddedCNeuralNet
> │   ├── a.out
> │   ├── examples
> │   │   └── mnist.c
> │   ├── library
> │   │   ├── neural_net.c
> │   │   └── neural_net.h
> │   ├── LICENSE
> │   ├── python
> │   │   ├── README.md
> │   │   └── train.2023-07-19.py
> │   └── README.md
> └── mnist_data
>     ├── t10k-images-idx3-ubyte
>     ├── t10k-labels-idx1-ubyte
>    ├── train-images-idx3-ubyte
>    └── train-labels-idx1-ubyte
</code>

 cd EmbeddedCNeuralNet
 gcc library/neural_net.c examples/mnist.c -lm -I. -O6 -lcblas -o embedded_neural_network



This will build the program in the root of the project and if you run the program from that location then it will find the data and run properly


