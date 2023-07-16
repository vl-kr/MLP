#pragma once

#include "../matrix/matrix.h"
#include "../util/util.h"
#include <tuple> 
#include <vector>
#include <cassert>
#include <cmath>
#include <omp.h>
#include <stdio.h>
#include <numeric>

#define ACT_ReLU 1 //hidden neurons
#define ACT_SOFTMAX 2 //output neurons
#define INIT_XAV 3
#define INIT_HE 4

using namespace std;

class network {
public:
	int weightInitMethod = INIT_HE;
	size_t batchSize = 32;
	size_t evaluationSetSizePercentage = 20;
	size_t epochs = 10;
	const vector<size_t> hiddenLayersNeuronCount = { 10 };
	Matrix trainDataVectors;
	Matrix trainDataLabels;
	size_t trainDataRows;
	size_t trainDataCols;
	static int evaluationOffset;
};

vector<vector<double>> computeDeltas(vector<vector<double>>& nonStaticNeuronPotentials, vector<vector<double>>& nonStaticNeuronOutputs, vector<Matrix>& weights, vector<vector<double>>& biases, int activationFuncType, int label);
void updateWeights(vector<Matrix>& weightChangeSum, vector<vector<double>>& biasChangeSum, vector<Matrix>& weights, vector<vector<double>>& biases, size_t batchSize, double learningRate, vector<vector<double>>& params, string method);
double evaluateNetworkAccuracy(Matrix& testDataVectors, Matrix& testDataLabels, size_t TESTING_OFFSET, vector<vector<double>>& nonStaticNeuronPotentials, vector<vector<double>>& nonStaticNeuronOutputs, vector<Matrix>& weights, vector<vector<double>>& biases, int activationFuncType);
void computeWeightChange(vector<Matrix>& weightChangeSum, vector<vector<double>>& biasChangeSum, vector<double>& inputVector, vector<vector<double>>& nonStaticNeuronOutputs, vector<Matrix>& weights, vector<vector<double>>& biases, vector<vector<double>>& deltas);
double evaluateNetworkError(Matrix& testDataVectors, Matrix& testDataLabels, size_t TESTING_OFFSET, vector<vector<double>>& nonStaticNeuronPotentials, vector<vector<double>>& nonStaticNeuronOutputs,vector<Matrix>& weights, vector<vector<double>>& biases, int activationFuncType);
double forwardPass(vector<double>& inputNeurons, vector<vector<double>>& nonStaticNeuronPotentials, vector<vector<double>>& nonStaticNeuronOutputs, vector<Matrix>& weights, vector<vector<double>>& biases, int activationFuncType, int label = 0);
void activationFunc(vector<double> &inVect, vector<double>& outVec, int activationFuncType);
double getVariance(size_t n, size_t m, int weightInitMethod);
vector<Matrix> initWeights(vector<size_t> architecture, int weightInitMethod);
