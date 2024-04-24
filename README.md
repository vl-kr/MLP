# MLP
This project is a feed-forward neural network written in C++. <br>
It's built completely from scratch, without the help of any 3rd-party library for matrix operations, differentiation operations or linear algebra.
<br>
<br>
<br>
The network has achieved the following tesults:

Accuracy on unseen data of 90% on the Fashion MNIST dataset (multi-class classification) in 6 minutes with 8 threads,<br>
config: 
```
OUTPUT_LAYER_SIZE = 10
HIDDEN_LAYERS_NEURON_COUNT = { 128, 128 }
BATCH_SIZE = 200
NORMALIZATION_FACTOR = 255
```

Accuracy on unseen data of 100% on 16-bit XOR dataset (binary classification) in 3 minutes with 1 thread,<br>
config:

```
OUTPUT_LAYER_SIZE = 2
HIDDEN_LAYERS_NEURON_COUNT = { 32, 32, 32, 32 }
BATCH_SIZE = 100
NORMALIZATION_FACTOR = 1
```
<br>
<br>
All of these datasets are located in the data folder.
