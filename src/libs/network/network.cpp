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
	*/
	if (weightInitMethod == INIT_XAV)
		return sqrt(1.0 / n);
	if (weightInitMethod == INIT_HE)
		return sqrt(2.0 / n);
	return 1;
}

void activationFunc(const vector<double>& inVect, vector<double>& outVec, int activationFuncType) {
	/*
	applies an activation function to values from inVect, and writes them into outVec
	*/
	if (activationFuncType == ACT_ReLU) {
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

vector<vector<double>> computeDeltas(const vector<vector<double>>& nonStaticNeuronPotentials, const vector<vector<double>>& nonStaticNeuronOutputs, const vector<Matrix>& weights, int label) {
	/*
	delta is a derivative of the error function w.r.t the inner potential (dE/dy * o'(inner_potential))
	creates a 2D matrix of deltas; one value for each neuron

	nonStaticNeuronPotentials				:		inner potenials of neurons in hidden layers + output layer
	nonStaticNeuronOutputs					:		outputs of neurons in hidden layers + output layer
	weights									:		weights between all neurons in adjacent layers
	label									:		label for this training example
	*/
	vector<vector<double>> deltas(nonStaticNeuronOutputs);
	deltas.back()[label] -= 1; // delta for the output layer, works only for softmax with cross entropy

	for (int layerIndex = nonStaticNeuronOutputs.size() - 2; layerIndex >= 0; layerIndex--) { // TODO: compare with the size of something else
		Matrix weightsTransposed = Matrix::Transpose(weights[layerIndex + 1], true);
		deltas[layerIndex] = Matrix::MultiplyMatrixByVector(weightsTransposed, deltas[layerIndex + 1]);
		for (int neuronIndex = 0; neuronIndex < deltas[layerIndex].size(); neuronIndex++) {
			deltas[layerIndex][neuronIndex] = nonStaticNeuronPotentials[layerIndex][neuronIndex] > 0 ? deltas[layerIndex][neuronIndex] : 0; // ReLU derivation (1 for x > 0; 0 otherwise)
		}
	}
	return deltas;
}

void computeWeightChange(vector<Matrix>& weightChangeSum, const vector<double>& inputVector, const vector<vector<double>>& nonStaticNeuronOutputs, const vector<vector<double>>& deltas) {
	/*
	multiplies deltas with neuron outputs to get dE/dw
	resulting values accumulate over each batch

	weightChangeSum							:		the sum of dE/dw for each training example in a batch
	inputVector								:		neuron outputs of the input layer
	nonStaticNeuronOutputs					:		neuron outputs of the hidden + output layers
	deltas									:		deltas from computeDeltas
	*/
	int layerIndex;
#pragma omp parallel for
	for (layerIndex = 0; layerIndex < weightChangeSum.size(); layerIndex++) {
		const vector<double>* neuronOutputs = &inputVector;
		if (layerIndex > 0) {
			neuronOutputs = &nonStaticNeuronOutputs[layerIndex - 1];
		}
		Matrix tempMatrix = Matrix::MultiplyVectors(deltas[layerIndex], *neuronOutputs, true);
		weightChangeSum[layerIndex].AddMatrix(tempMatrix);
	}
}

void updateWeights(const vector<Matrix>& weightChangeSum, vector<Matrix>& weights, size_t batchSize, double learningRate, vector<vector<double>>& params) {
	/*
	updates weights and applies some optimizations (most of them are commented out to save time)

	weightChangeSum							:		the sum of dE/dw for each training example in a batch
	weights									:		weights between all neurons in adjacent layers
	batchSize								:		number of training examples in a batch
	learningRate							: 		learning rate
	params									:		serves as a memory for different optimization algorithms
	*/
	for (size_t layerIndex = 0; layerIndex < weights.size(); layerIndex++) {
		for (size_t outputNeuronIndex = 0; outputNeuronIndex < weights[layerIndex].rows; outputNeuronIndex++) {
			for (size_t inputNeuronIndex = 0; inputNeuronIndex < weights[layerIndex].cols - 1; inputNeuronIndex++) {
				double dw = weightChangeSum[layerIndex].data[outputNeuronIndex][inputNeuronIndex] / batchSize;
				params[0][0] = (0.9 * params[0][0]) + ((0.1) * dw * dw); // RMSProp parameter
				double weightDelta = -((learningRate / sqrt(params[0][0] + 0.00000001)) * dw); //only RMSProp
				weights[layerIndex].data[outputNeuronIndex][inputNeuronIndex] += weightDelta;
			}
			double dwBias = weightChangeSum[layerIndex].data[outputNeuronIndex][weights[layerIndex].cols - 1] / batchSize;
			params[1][0] = (0.9 * params[1][0]) + ((0.1) * dwBias * dwBias);
			double weightDeltaBias = -((learningRate / sqrt(params[1][0])) * dwBias);
			weights[layerIndex].data[outputNeuronIndex][weights[layerIndex].cols - 1] += weightDeltaBias;
		}
	}
}

double evaluateNetworkError(const Matrix& testDataVectors, const Matrix& testDataLabels, size_t TESTING_OFFSET, vector<vector<double>>& nonStaticNeuronPotentials, 
	vector<vector<double>>& nonStaticNeuronOutputs, const vector<Matrix>& weights, int activationFuncType) {
	/*
	computes average loss for trainig examples after TESTING_OFFSET

	testDataVectors							:		all training vectors
	testDataLabels							:		all training labels
	TESTING_OFFSET							:		marks the index of the first evaluation example (all following examples are not trained on)
	nonStaticNeuronPotentials				:		inner potenials of neurons in hidden layers + output layer, passed to forwardPass()
	nonStaticNeuronOutputs					:		outputs of neurons in hidden layers + output layer, , passed to forwardPass()
	weights									:		weights between all neurons in adjacent layers
	activationFuncType						:		activation function of neurons in hidden layers
	*/
	double errorSum = 0;
	for (size_t i = TESTING_OFFSET; i < testDataVectors.rows; i++) {
		int label = testDataLabels.data[i][0];
		forwardPass(testDataVectors.data[i], nonStaticNeuronPotentials, nonStaticNeuronOutputs, weights, activationFuncType);
		double outputOfCorrectNeuron = nonStaticNeuronOutputs[nonStaticNeuronOutputs.size() - 1][label];
		double loss = log(outputOfCorrectNeuron); // cross entropy
		errorSum += loss;
	}
	return -(1.0 / (testDataVectors.rows - TESTING_OFFSET)) * errorSum;
}

double evaluateNetworkAccuracy(const Matrix& testDataVectors, const Matrix& testDataLabels, size_t TESTING_OFFSET,
	vector<vector<double>>& nonStaticNeuronPotentials, vector<vector<double>>& nonStaticNeuronOutputs, const vector<Matrix>& weights, int activationFuncType) {
	/*
	computes accuracy for trainig examples after TESTING_OFFSET

	testDataVectors							:		all training vectors
	testDataLabels							:		all training labels
	TESTING_OFFSET							:		marks the index of the first evaluation example (all following examples are not trained on)
	nonStaticNeuronPotentials				:		inner potenials of neurons in hidden layers + output layer, passed to forwardPass()
	nonStaticNeuronOutputs					:		outputs of neurons in hidden layers + output layer, , passed to forwardPass()
	weights									:		weights between all neurons in adjacent layers
	activationFuncType						:		activation function of neurons in hidden layers
	*/
	double correct = 0;
	for (size_t i = TESTING_OFFSET; i < testDataVectors.rows; i++) {
		int label = testDataLabels.data[i][0];
		forwardPass(testDataVectors.data[i], nonStaticNeuronPotentials, nonStaticNeuronOutputs, weights, activationFuncType);
		if (vecToScalar(nonStaticNeuronOutputs.back()) == label)
			correct++;
	}
	return (correct / (testDataVectors.rows - TESTING_OFFSET));
}

void forwardPass(const vector<double>& inputNeurons, vector<vector<double>>& nonStaticNeuronPotentials,
	vector<vector<double>>& nonStaticNeuronOutputs, const vector<Matrix>& weights, int activationFuncType) {
	/*
	performs forward pass for a single example, returns cross entropy loss

	inputNeurons							:		raw input
	nonStaticNeuronPotentials				:		inner potenials of neurons in hidden layers + output layer
	nonStaticNeuronOutputs					:		outputs of neurons in hidden layers + output layer
	label									:		label for this training example
	weights									:		weights between all neurons in adjacent layers
	activationFuncType						:		activation function of neurons in hidden layers
	*/
	for (size_t layerIndex = 0; layerIndex < nonStaticNeuronPotentials.size(); layerIndex++) {
		const vector<double>* prevLayerOutputs = &inputNeurons;
		if (layerIndex > 0) {
			prevLayerOutputs = &nonStaticNeuronOutputs[layerIndex - 1];
		}
		nonStaticNeuronPotentials[layerIndex] = Matrix::MultiplyMatrixByVector(weights[layerIndex], *prevLayerOutputs, true);

		int activationFunction = layerIndex == nonStaticNeuronPotentials.size() - 1 ? ACT_SOFTMAX : activationFuncType;
		activationFunc(nonStaticNeuronPotentials[layerIndex], nonStaticNeuronOutputs[layerIndex], activationFunction); //softmax for output layer
	}
	return;
}

vector<Matrix> initWeights(vector<size_t> architecture, int weightInitMethod, int initialBias) {
	/*
	initializes biases and weights between all neurons in adjacent layers using the chosen method

	architecture							:		each number in architecture defines number of neurons in an individual layer
	weightInitMethod						:		allows to choose between different weight initialization methods
	initialBias								:		initial bias value for all neurons, default is 0
	*/
	if (architecture.size() < 3) {
		throw invalid_argument("Network must have at least 3 layers (at least one hidden layer)");
	}
	vector<Matrix> weights;
	for (size_t i = 1; i < architecture.size(); i++) {
		double variance = getVariance(architecture[i - 1], architecture[i], weightInitMethod);
		weights.push_back(Matrix::RandomMatrixSetSize(architecture[i], architecture[i - 1] + 1, variance)); // +1 cols for bias
	}
	for (size_t layerIndex = 0; layerIndex < weights.size(); layerIndex++) {
		for (size_t neuronIndex = 0; neuronIndex < weights[layerIndex].rows; neuronIndex++) {
			weights[layerIndex].data[neuronIndex][weights[layerIndex].cols - 1] = initialBias; //set bias
		}
	}
	return weights;
}