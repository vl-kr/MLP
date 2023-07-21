﻿#include "main.h"

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

using namespace std;

//-------------------HYPERPARAMETERS--------------------------

const size_t OUTPUT_LAYER_SIZE = 10;

//------------------------------------------------------------

const vector<size_t> HIDDEN_LAYERS_NEURON_COUNT = { 128,128 };
const size_t BATCH_SIZE = 200;

const int EVALUATION_SET_SIZE_PERCENTAGE = 10;
const double BIAS_INIT_VALUE = 0;
const double WEIGHT_DECAY = 0.01;
const string OPTIMIZER = "RMS"; // TODO: use enums instead of strings

double learningRate = 0.001;
size_t epochs = 70;
size_t threads = 8;

//------------------------------------------------------------

int main(int argc, char** argv)
{
	srand(time(NULL));
	omp_set_num_threads(threads);

	if (argc > 2) { // default values are used if no arguments are passed
		epochs = stoi(argv[1]);
		threads = stoi(argv[2]);
	}

	cout << "Epochs: " << epochs << endl;

	cout << "Architecture: [";
	for (auto val : HIDDEN_LAYERS_NEURON_COUNT) {
		cout << val << ',';
	}
	cout << OUTPUT_LAYER_SIZE << "] ";
	cout << " Learning rate: " << learningRate << " Batch size: " << BATCH_SIZE << " decay: " << WEIGHT_DECAY << endl;

	Matrix trainDataVectors = loadFromCSV("data/fashion_mnist_train_vectors.csv", 255); // dividing by 255 is done to normalize the data
	Matrix trainDataLabels = loadFromCSV("data/fashion_mnist_train_labels.csv");
	Matrix testDataVectors = loadFromCSV("data/fashion_mnist_test_vectors.csv", 255); // dividing by 255 is done to normalize the data
	Matrix testDataLabels = loadFromCSV("data/fashion_mnist_test_labels.csv");

	size_t trainDataRows = trainDataVectors.data.size();
	size_t trainDataCols = trainDataVectors.data[0].size();

	// evaluationOffset marks the index of the first example excluded from training and used for network evaluation
	static size_t evaluationOffset = trainDataRows - (trainDataRows * (((float)EVALUATION_SET_SIZE_PERCENTAGE) / 100)); 

	vector<size_t> layersNeuronCount = HIDDEN_LAYERS_NEURON_COUNT; // adding hidden layers
	layersNeuronCount.insert(layersNeuronCount.begin(), trainDataCols); // adding input layer
	layersNeuronCount.push_back(OUTPUT_LAYER_SIZE); // adding output layer

	vector<vector<double>> nonStaticNeuronLayersPotentials(layersNeuronCount.size() - 1, vector<double>()); // all layers except the input layer
	for (size_t i = 1; i < layersNeuronCount.size(); i++) {
		nonStaticNeuronLayersPotentials[i - 1].resize(layersNeuronCount[i]);
	}

	vector<vector<double>> nonStaticNeuronLayersOutputs(nonStaticNeuronLayersPotentials); // potentials after applying activation function

	vector<Matrix> weights = initWeights(layersNeuronCount, INIT_HE); // initialize weights

	vector<Matrix> weightChangeSum(weights); // will accumulate over each batch

	vector<vector<double>> params(2, vector<double>(5, 0)); // used as a memory for different optimizers

	vector<int> shuffleMap(evaluationOffset); //used to shuffle the training data, maps only to training examples before evaluationOffset
	iota(shuffleMap.begin(), shuffleMap.end(), 0);
	shuffle(shuffleMap.begin(), shuffleMap.end(), random_device());

	double error = 0;
	double accuracy = 0;

	auto runStart = high_resolution_clock::now();
	auto epochTimerStart = high_resolution_clock::now();
	auto epochTimerEnd = high_resolution_clock::now();

	size_t epochCounter = 0;

	while (accuracy < 0.89) {

		cout << "Epoch: " << epochCounter << endl;

		error = evaluateNetworkError(trainDataVectors, trainDataLabels, evaluationOffset, nonStaticNeuronLayersPotentials, nonStaticNeuronLayersOutputs, weights, ACT_ReLU);
		accuracy = evaluateNetworkAccuracy(trainDataVectors, trainDataLabels, evaluationOffset, nonStaticNeuronLayersPotentials, nonStaticNeuronLayersOutputs, weights, ACT_ReLU);

		if (accuracy > 0.86) { // adaptive learning rate
			learningRate = 0.0001;
		}
		else if (accuracy > 0.85) {
			learningRate = 0.0005;
		}

		auto epochLength = duration_cast<milliseconds>(epochTimerEnd - epochTimerStart); // measuring time spent on epoch
		auto runTimeMs = duration_cast<milliseconds>(epochTimerEnd - runStart); // measuring total time spent on training
		double runTimeMins = ((double)runTimeMs.count()) / 60000; // converting to minutes
		epochTimerStart = high_resolution_clock::now();

		cout << "Accuracy: " << accuracy << ", Error: " << error << ", Runtime: " << runTimeMins << " min., ms since last epoch: " << epochLength.count() << endl;

		for (size_t trainingExampleOffset = 0; trainingExampleOffset + BATCH_SIZE <= evaluationOffset; trainingExampleOffset += BATCH_SIZE) {

			for (Matrix& layer : weightChangeSum){ // reset before each batch
				for (vector<double>& v1 : layer.data){
					fill(v1.begin(), v1.end(), 0);
				}
			}
			for (size_t batchNum = 0; batchNum < BATCH_SIZE; batchNum++) {
				vector<double>& trainingExample = trainDataVectors.data[shuffleMap[trainingExampleOffset + batchNum]];
				int label = trainDataLabels.data[shuffleMap[trainingExampleOffset + batchNum]][0];

				forwardPass(trainingExample, nonStaticNeuronLayersPotentials, nonStaticNeuronLayersOutputs, weights, ACT_ReLU);
				vector<vector<double>> deltas = computeDeltas(nonStaticNeuronLayersPotentials, nonStaticNeuronLayersOutputs, weights, label);
				computeWeightChange(weightChangeSum, trainingExample, nonStaticNeuronLayersOutputs, deltas);
			}
			updateWeights(weightChangeSum, weights, BATCH_SIZE, learningRate, params);
		}
		shuffle(shuffleMap.begin(), shuffleMap.end(), random_device());
		epochTimerEnd = high_resolution_clock::now();
		epochCounter += 1;
	}

	vector<double> outLabels;

	for (auto& inputVector : testDataVectors.data) {
		forwardPass(inputVector, nonStaticNeuronLayersPotentials, nonStaticNeuronLayersOutputs, weights, ACT_ReLU);
		outLabels.push_back(vecToScalar(nonStaticNeuronLayersOutputs.back()));
	}

	writeToFile("actualPredictions", outLabels);

	return 0;
}
