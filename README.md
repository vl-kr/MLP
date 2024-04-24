# MLP
This project is a feed-forward neural network written in C++. <br>
It's built completely from scratch, without the help of any 3rd-party library for matrix operations, differentiation operations or linear algebra.
<br>
<br>
The network achieves the following tesults:

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
All of these datasets are located in the data folder.

## Installation
1. Clone repo
2. Compile
   - `cmake src/`
   - `make`
3. Extract datasets
   - From `data.zip` to `MLP/data`, i.e. `MLP/data/fashion_mnist_train_vectors.csv`
4. Run the binary
   - `./main`

The program will start working on the Fashion MNIST dataset with the configuration mentioned above. The dataset and configuration can be changed in `main.cpp` (the hyperparamter constants).

### Example output:
![image](https://github.com/vl-kr/MLP/assets/44015502/c6d7d597-9a1f-4971-8a5b-acb0456a9362)
