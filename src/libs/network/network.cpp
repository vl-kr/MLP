// I'm sorry to everyone who will have to read the content of this file.

#include "network.h"

using namespace std;

double getVariance(size_t n, size_t m, int weightInitMethod) {
	/*
	used before weight initializaion
	n : 				number of neuron in layer k
	m : 				number of neuron in layer k + 1
	weightInitMethod :  either INIT_XAV or INIT_HE
	returns variance
	/*/
	//assert(weightInitMethod == INIT_XAV || weightInitMethod == INIT_HE);
	if (weightInitMethod == INIT_XAV)
		return sqrt(1.0 / n);
	if (weightInitMethod == INIT_HE)
		return sqrt(2.0 / n);
	return 1;
}

void activationFunc(vector<double>& inVect, vector<double>& outVec, int activationFuncType) {
	/*
	applies an activation function to values from inVect, and writes them into outVec
	*/
	//assert(activationFuncType == ACT_ReLU || activationFuncType == ACT_SOFTMAX);
	if (activationFuncType == ACT_ReLU) {
#pragma omp parallel for
		for (size_t i = 0; i < size(inVect); i++) {
			outVec[i] = max(0.0, inVect[i]);
		}
	}
	else if (activationFuncType == ACT_SOFTMAX) {
		double sum = 0;
		for (double num : inVect) {
			sum += exp(num);
		}
		for (size_t i = 0; i < size(inVect); i++) {
			outVec[i] = exp(inVect[i]) / sum;
		}
	}
}

inline double derivativeReLU(double potential) {
	if (potential > 0)
		return 1;
	return 0;
}

vector<vector<double>> computeDeltas(vector<vector<double>>& nonStaticNeuronPotentials, vector<vector<double>>& nonStaticNeuronOutputs, vector<Matrix>& weights, vector<vector<double>>& biases, int activationFuncType, int label) {
	/*
	delta is a derivative of the error function w.r.t the inner potential (dE/dy * o'(inner_potential))
	creates a 2D matrix of deltas; one value for each neuron
	*/
	vector<vector<double>> deltas;
	vector<double> outLayerDeltas(nonStaticNeuronOutputs.back());
	outLayerDeltas[label] -= 1; // delta for the output layer, works only for softmax with cross entropy
	deltas.insert(deltas.begin(), outLayerDeltas);
	for (int layerIndex = nonStaticNeuronOutputs.size() - 2; layerIndex >= 0; layerIndex--) {
		vector<double> layerDeltas(nonStaticNeuronOutputs[layerIndex].size());
#pragma omp parallel for
		for (size_t neuronIndex = 0; neuronIndex < nonStaticNeuronOutputs[layerIndex].size(); neuronIndex++) {
			double sum = 0;
			if (nonStaticNeuronPotentials[layerIndex][neuronIndex] > 0) { // ReLU derivation (1 for x > 0; 0 otherwise), saves time (I think)
				for (size_t nextLayerNeuronIndex = 0; nextLayerNeuronIndex < nonStaticNeuronOutputs[layerIndex + 1].size(); nextLayerNeuronIndex++) {
					sum += deltas[0][nextLayerNeuronIndex] * weights[layerIndex + 1].data[nextLayerNeuronIndex][neuronIndex];
				}
			}
			layerDeltas[neuronIndex] = sum;
		}
		deltas.insert(deltas.begin(), layerDeltas);
	}
	return deltas;
}

void computeWeightChange(vector<Matrix>& weightChangeSum, vector<vector<double>>& biasChangeSum, vector<double>& inputVector, vector<vector<double>>& nonStaticNeuronOutputs, vector<Matrix>& weights, vector<vector<double>>& biases, vector<vector<double>>& deltas) {
	/*
	multiplies deltas with neuron outputs to get dE/dw
	resulting values accumulate over each batch

	weightChangeSum			:		the sum of dE/dw for each training example in a batch
	biasChangeSum			:		weightChangeSum for biases
	inputVector				:		neuron outputs of the input layer
	nonStaticNeuronOutputs	:		neuron outputs of the hidden + output layers
	*/
#pragma omp parallel for
	for (size_t inputNeuronIndexI = 0; inputNeuronIndexI < inputVector.size(); inputNeuronIndexI++) {
		for (size_t outputNeuronIndexJ = 0; outputNeuronIndexJ < weights[0].rows; outputNeuronIndexJ++) {
			weightChangeSum[0].data[outputNeuronIndexJ][inputNeuronIndexI] += deltas[0][outputNeuronIndexJ] * inputVector[inputNeuronIndexI];
		}
	}
#pragma omp parallel for
	for (size_t layerIndex = 1; layerIndex < weights.size(); layerIndex++) {
		for (size_t inputNeuronIndex = 0; inputNeuronIndex < weights[layerIndex].cols; inputNeuronIndex++) {
			for (size_t outputNeuronIndex = 0; outputNeuronIndex < weights[layerIndex].rows; outputNeuronIndex++) {
				weightChangeSum[layerIndex].data[outputNeuronIndex][inputNeuronIndex] += deltas[layerIndex][outputNeuronIndex] * nonStaticNeuronOutputs[layerIndex - 1][inputNeuronIndex];
			}
		}
	}
#pragma omp parallel for
	for (size_t biasLayer = 0; biasLayer < biasChangeSum.size(); biasLayer++) {
		for (size_t biasIndex = 0; biasIndex < biasChangeSum[biasLayer].size(); biasIndex++) {
			biasChangeSum[biasLayer][biasIndex] += deltas[biasLayer][biasIndex];
		}
	}
}

void updateWeights(vector<Matrix>& weightChangeSum, vector<vector<double>>& biasChangeSum, vector<Matrix>& weights, vector<vector<double>>& biases, size_t batchSize, double learningRate, vector<vector<double>>& params, string method) {
	/*
	updates weights and applies some optimizations (most of them are commented out to save time)

	weightChangeSum			:		the sum of dE/dw for each training example in a batch
	biasChangeSum			:		weightChangeSum for biases
	weights					:		weights
	biases					:		biases
	batchSize				:		number of training examples in a batch
	learningRate			: 		learning rate
	params					:		serves as a memory for different optimization algorithms
	method					:		decides which optimization method to use
	*/
	//params[0][4] += 1; // time
	for (size_t layerIndex = 0; layerIndex < weights.size(); layerIndex++) {
		for (size_t inputNeuronIndex = 0; inputNeuronIndex < weights[layerIndex].cols; inputNeuronIndex++) {
			for (size_t outputNeuronIndex = 0; outputNeuronIndex < weights[layerIndex].rows; outputNeuronIndex++) {
				double dw = weightChangeSum[layerIndex].data[outputNeuronIndex][inputNeuronIndex] / batchSize;
				params[0][0] = (0.9 * params[0][0]) + ((0.1) * dw * dw); // RMSProp parameter
				//params[0][1] = -(learningRate * dw) + (0.8 * params[0][1]); // momentum parameter

				//params[0][2] = (0.9 * params[0][2]) + ((0.1) * dw); // adam parameter
				//params[0][3] = (0.999 * params[0][3]) + ((0.001) * dw * dw); // adam parameter

				//double mHat = params[0][2] / 1; //(1 - (pow(0.9, params[0][4]))); 
				//double vHat = params[0][3] / 1; //(1 - (pow(0.999, params[0][4]))); 

				double weightDelta;

				//if(method == "vanilla")
				//	weightDelta = -(learningRate * dw);// vanilla

				//else
				//if (method == "momentum") 
				//	weightDelta = params[0][1]; // momentum

				//else if(method == "RMS")
				weightDelta = -((learningRate / sqrt(params[0][0] + 0.00000001)) * dw); //only RMSProp

				//else if(method == "RMSW")
				//	weightDelta = -(((learningRate / sqrt(params[0][0] + 0.00000001)) * dw) + (learningRate * 0.2 * weights[layerIndex].data[outputNeuronIndex][inputNeuronIndex])); //RMSProp + weight decay

				//else if (method == "ADAM")
				//	weightDelta = -learningRate * (((mHat) / (sqrt(vHat) + 0.00000001)));
				//else if (method == "ADAMW")
				//	weightDelta = -learningRate * (((mHat) / (sqrt(vHat) + 0.00000001)) + 0.01 * weights[layerIndex].data[outputNeuronIndex][inputNeuronIndex]);
				//else {
				//	cout << "!!!!!!!!WRONG METHOD!!!!!!" << endl;
				//	exit(12345);
				//}

				weights[layerIndex].data[outputNeuronIndex][inputNeuronIndex] += weightDelta;

			}
		}
	}

	for (size_t layerIndex = 0; layerIndex < biasChangeSum.size(); layerIndex++) {
		for (size_t biasIndex = 0; biasIndex < biasChangeSum[layerIndex].size(); biasIndex++) {
			double dw = biasChangeSum[layerIndex][biasIndex] / batchSize;
			params[1][0] = (0.9 * params[1][0]) + ((0.1) * dw * dw); // RMSProp parameter
			//params[1][1] = -(learningRate * dw) + (0.9 * params[0][1]); // momentum parameter

			/*params[1][2] = (0.9 * params[1][2]) + ((0.1) * dw);
			params[1][3] = (0.999 * params[1][3]) + ((0.001) * dw * dw);

			double mHat = params[1][2] / (1 - (pow(0.9, params[0][4])));
			double vHat = params[1][3] / (1 - (pow(0.999, params[0][4])));*/

			double weightDelta;

			//if (method == "vanilla")
			//	weightDelta = -(learningRate * dw);// vanilla

			//else 
			//if (method == "momentum")
			//	weightDelta = params[1][1]; // momentum

			//else if (method == "RMS")
			weightDelta = -((learningRate / sqrt(params[1][0])) * dw); //only RMSProp #2

			//else if (method == "RMSW")
			//	weightDelta = -(((learningRate / sqrt(params[1][0])) * dw) + (learningRate * 0.1 * biases[layerIndex][biasIndex])); //RMSProp + weight decay

			//else if (method == "ADAM")
			//	double weightDelta = -learningRate * (((mHat) / (sqrt(vHat) + 0.00000001)));
			//else if (method == "ADAMW")
			//	weightDelta = -learningRate * (((mHat) / (sqrt(vHat) + 0.00000001)) + 0.1 * biases[layerIndex][biasIndex]);

			biases[layerIndex][biasIndex] += weightDelta;
		}
	}

}

double evaluateNetworkError(Matrix& testDataVectors, Matrix& testDataLabels, size_t TESTING_OFFSET, vector<vector<double>>& nonStaticNeuronPotentials, vector<vector<double>>& nonStaticNeuronOutputs, vector<Matrix>& weights, vector<vector<double>>& biases, int activationFuncType) {
	/*
	computes average loss for trainig examples after TESTING_OFFSET

	testDataVectors							:		all training vectors
	testDataLabels							:		all training labels
	TESTING_OFFSET							:		marks the index of the first evaluation example (all following examples are not trained on)
	nonStaticNeuronPotentials				:		inner potenials of neurons in hidden layers + output layer, passed to forwardPass()
	nonStaticNeuronOutputs					:		outputs of neurons in hidden layers + output layer, , passed to forwardPass()
	*/
	//assert(testDataLabels.rows == testDataVectors.rows && testDataVectors.rows > TESTING_OFFSET);
	double errorSum = 0;
	for (size_t i = TESTING_OFFSET; i < testDataVectors.rows; i++) {
		errorSum += forwardPass(testDataVectors.data[i], nonStaticNeuronPotentials, nonStaticNeuronOutputs, weights, biases, activationFuncType, testDataLabels.data[i][0]);
	}
	return -(1.0 / (testDataVectors.rows - TESTING_OFFSET)) * errorSum;
}

double evaluateNetworkAccuracy(Matrix& testDataVectors, Matrix& testDataLabels, size_t TESTING_OFFSET, vector<vector<double>>& nonStaticNeuronPotentials, vector<vector<double>>& nonStaticNeuronOutputs, vector<Matrix>& weights, vector<vector<double>>& biases, int activationFuncType) {
	/*
	computes accuracy for trainig examples after TESTING_OFFSET

	testDataVectors							:		all training vectors
	testDataLabels							:		all training labels
	TESTING_OFFSET							:		marks the index of the first evaluation example (all following examples are not trained on)
	nonStaticNeuronPotentials				:		inner potenials of neurons in hidden layers + output layer, passed to forwardPass()
	nonStaticNeuronOutputs					:		outputs of neurons in hidden layers + output layer, , passed to forwardPass()
	*/
	//assert(testDataLabels.rows == testDataVectors.rows && testDataVectors.rows > TESTING_OFFSET);
	double correct = 0;
	for (size_t i = TESTING_OFFSET; i < testDataVectors.rows; i++) {
		forwardPass(testDataVectors.data[i], nonStaticNeuronPotentials, nonStaticNeuronOutputs, weights, biases, activationFuncType, testDataLabels.data[i][0]);
		if (vecToScalar(nonStaticNeuronOutputs.back()) == testDataLabels.data[i][0])
			correct++;
	}
	return (correct / (testDataVectors.rows - TESTING_OFFSET));
}

double forwardPass(vector<double>& inputNeurons, vector<vector<double>>& nonStaticNeuronPotentials, vector<vector<double>>& nonStaticNeuronOutputs, vector<Matrix>& weights, vector<vector<double>>& biases, int activationFuncType, int label) {
	/*
	performs forward pass for a single example, returns cross entropy loss

	inputNeurons							:		raw input
	nonStaticNeuronPotentials				:		inner potenials of neurons in hidden layers + output layer, passed to forwardPass()
	nonStaticNeuronOutputs					:		outputs of neurons in hidden layers + output layer, passed to forwardPass()
	label									:		label for this training example
	*/
	//assert(label < (int)nonStaticNeuronPotentials[nonStaticNeuronPotentials.size() - 1].size());
	//assert(nonStaticNeuronPotentials.size() >= 2); //at least one hidden layer
	//assert(weights.size() == nonStaticNeuronPotentials.size()); // weights.size() = number of all layers - 1;		 number of all layers = nonStaticNeuronLayers + 1
	for (size_t layerIndex = 0; layerIndex < nonStaticNeuronPotentials.size(); layerIndex++) {
		if (layerIndex == 0) {
			//assert(inputNeurons.size() == weights[0].cols);
#pragma omp parallel for
			for (size_t neuronIndex = 0; neuronIndex < weights[0].rows; neuronIndex++) {
				nonStaticNeuronPotentials[0][neuronIndex] = biases[layerIndex][neuronIndex];
				for (size_t weightIndex = 0; weightIndex < weights[0].cols; weightIndex++) {
					nonStaticNeuronPotentials[0][neuronIndex] += inputNeurons[weightIndex]/255 * weights[0].data[neuronIndex][weightIndex]; // dividing by 255 for normalization
				}
			}
		}
		else {
			//assert(nonStaticNeuronPotentials[layerIndex - 1].size() == weights[layerIndex].cols);
#pragma omp parallel for
			for (size_t neuronIndex = 0; neuronIndex < weights[layerIndex].rows; neuronIndex++) {
				nonStaticNeuronPotentials[layerIndex][neuronIndex] = biases[layerIndex][neuronIndex];
				for (size_t weightIndex = 0; weightIndex < weights[layerIndex].cols; weightIndex++) {
					nonStaticNeuronPotentials[layerIndex][neuronIndex] += nonStaticNeuronOutputs[layerIndex - 1][weightIndex] * weights[layerIndex].data[neuronIndex][weightIndex];
				}
			}
		}
		if (layerIndex == nonStaticNeuronPotentials.size() - 1)
			activationFunc(nonStaticNeuronPotentials[layerIndex], nonStaticNeuronOutputs[layerIndex], ACT_SOFTMAX); //softmax for output layer
		else
			activationFunc(nonStaticNeuronPotentials[layerIndex], nonStaticNeuronOutputs[layerIndex], activationFuncType); //hidden layers
	}
	double n = nonStaticNeuronOutputs[nonStaticNeuronOutputs.size() - 1][label];
	double l = log(n);
	return l; // returns cross entropy
}

vector<Matrix> initWeights(vector<size_t> architecture, int weightInitMethod) { //each number in architecture = number of neurons in an individual layer
	assert(architecture.size() >= 3);
	vector<Matrix> weights;
	for (size_t i = 1; i < architecture.size(); i++) {
		double variance = getVariance(architecture[i - 1], architecture[i], weightInitMethod);
		weights.push_back(Matrix::RandomMatrixSetSize(architecture[i], architecture[i - 1], variance));
	}
	return weights;
}