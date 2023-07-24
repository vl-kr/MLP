#pragma once

#include "../matrix/matrix.h"
#include "../util/util.h"
#include <vector>
#include <cassert>
#include <cmath>
#include <omp.h>

#define ACT_ReLU 1 //hidden neurons
#define ACT_SOFTMAX 2 //output neurons
#define INIT_XAV 3
#define INIT_HE 4

using namespace std;

void forwardPass(const vector<double>& inputNeurons, vector<vector<double>>& nonStaticNeuronPotentials, vector<vector<double>>& nonStaticNeuronOutputs, const vector<Matrix>& weights, int activationFuncType);
void computeDeltas(const vector<vector<double>>& nonStaticNeuronPotentials, const vector<double>& outputNeuronOutputs, const vector<Matrix>& weights, vector<vector<double>>& deltas, int label);
void computeWeightChange(vector<Matrix>& weightChangeSum, const vector<double>& inputVector, const vector<vector<double>>& nonStaticNeuronOutputs, const vector<vector<double>>& deltas);
void updateWeights(const vector<Matrix>& weightChangeSum, vector<Matrix>& weights, size_t batchSize, double learningRate, vector<vector<double>>& params);

vector<Matrix> initWeights(vector<size_t> architecture, int weightInitMethod, int initialBias = 0);
double getVariance(size_t n, size_t m, int weightInitMethod);

void activationFunc(const vector<double>& inVect, vector<double>& outVec, int activationFuncType);
double evaluateNetworkAccuracy(const Matrix& testDataVectors, const Matrix& testDataLabels, size_t TESTING_OFFSET, vector<vector<double>>& nonStaticNeuronPotentials, vector<vector<double>>& nonStaticNeuronOutputs, const vector<Matrix>& weights, int activationFuncType);
double evaluateNetworkError(const Matrix& testDataVectors, const Matrix& testDataLabels, size_t TESTING_OFFSET, vector<vector<double>>& nonStaticNeuronPotentials, vector<vector<double>>& nonStaticNeuronOutputs, const vector<Matrix>& weights, int activationFuncType);
