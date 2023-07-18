// main.cpp : Defines the entry point for the application.
//


#include "main.h"

#include <chrono>
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

using namespace std;

//-------------------HYPERPARAMETERS--------------------------

#define OUTPUT_LAYER_SIZE 10

//------------------------------------------------------------

vector<size_t> hiddenLayersNeuronCount = { 256,256 };
#define LEARNING_RATE 0.001
#define BATCH_SIZE 200

#define EVALUATION_SET_SIZE_PERCENTAGE 10
#define EPOCHS 70
#define BIAS_INIT_VALUE 0

//------------------------------------------------------------

int main(int argc, char** argv)
{
	srand(time(NULL));

	double learningRate = LEARNING_RATE;
	size_t batchSize = BATCH_SIZE;
	size_t evalSetSizePercent = EVALUATION_SET_SIZE_PERCENTAGE;
	size_t epochs = EPOCHS;
	double biasInitVal = BIAS_INIT_VALUE;
	double weightDecay = 0.01;
	string optimizer = "RMS";

	size_t threads = 16;

	if (argc > 2){
		epochs = stoi(argv[1]);
		threads = stoi(argv[2]);
	}

	omp_set_num_threads(threads);

	cout << "Epochs: " << epochs << endl;

	vector<vector<double>> params(2, vector<double>(5, 0)); // used as a memory for different optimizers

	cout << "Architecture: [";
	for (auto val : hiddenLayersNeuronCount)
		cout << val << ',';
	cout << "] ";
	cout << " Learning rate: " << learningRate << " Batch size: " << batchSize << " decay: " << weightDecay << endl;

	Matrix trainDataVectors = loadFromCSV("data/fashion_mnist_train_vectors.csv");
	Matrix trainDataLabels = loadFromCSV("data/fashion_mnist_train_labels.csv");
	Matrix testDataVectors = loadFromCSV("data/fashion_mnist_test_vectors.csv");
	Matrix testDataLabels = loadFromCSV("data/fashion_mnist_test_labels.csv");

	size_t trainDataRows = trainDataVectors.data.size();
	size_t trainDataCols = trainDataVectors.data[0].size();

	static size_t evaluationOffset = trainDataRows - (trainDataRows * (((float)evalSetSizePercent) / 100)); //marks the index of the first training example used for network evaluation

	vector<size_t> layersNeuronCount = hiddenLayersNeuronCount; // adding hidden layers
	layersNeuronCount.insert(layersNeuronCount.begin(), trainDataCols); // adding input layer
	layersNeuronCount.push_back(OUTPUT_LAYER_SIZE); // adding output layer

	vector<vector<double>> nonStaticNeuronLayersPotentials(layersNeuronCount.size() - 1, vector<double>()); // all layers except the input layer
	for (size_t i = 1; i < layersNeuronCount.size(); i++)
		nonStaticNeuronLayersPotentials[i - 1].resize(layersNeuronCount[i]);

	vector<vector<double>> nonStaticNeuronLayersOutputs(layersNeuronCount.size() - 1, vector<double>()); // potentials after applying activation function
	for (size_t i = 1; i < layersNeuronCount.size(); i++)
		nonStaticNeuronLayersOutputs[i - 1].resize(layersNeuronCount[i]);

	vector<Matrix> weights = initWeights(layersNeuronCount, INIT_HE); // initialize weights

	vector<vector<double>> biases(layersNeuronCount.size() - 1, vector<double>()); //initialize biases
	for (size_t i = 1; i < layersNeuronCount.size(); i++)
		biases[i-1].resize(layersNeuronCount[i], biasInitVal);

	vector<Matrix> weightChangeSum(weights); // accumulates over 1 batch
	vector<vector<double>> biasChangeSum(biases); // accumulates over 1 batch 

	vector<int> shuffleMap(evaluationOffset); //used to shuffle the training data, maps only to training examples before evaluationOffset
	iota(shuffleMap.begin(), shuffleMap.end(), 0);
	shuffle(shuffleMap.begin(), shuffleMap.end(), random_device());

	//double error = 0;
	double accuracy = 0;

	double runTimeD = 0;

	auto runStart = high_resolution_clock::now();
	auto epoch1 = high_resolution_clock::now();
	auto epoch2 = high_resolution_clock::now();

	size_t epochCounter = 0;

	while(runTimeD < 28 && (runTimeD < 23 || epochCounter < epochs || accuracy < 0.883)) {

		cout << "Epoch: " << epochCounter << endl;

		//error = evaluateNetworkError(trainDataVectors, trainDataLabels, evaluationOffset, nonStaticNeuronLayersPotentials, nonStaticNeuronLayersOutputs, weights, biases, ACT_ReLU);
		accuracy = evaluateNetworkAccuracy(trainDataVectors, trainDataLabels, evaluationOffset, nonStaticNeuronLayersPotentials, nonStaticNeuronLayersOutputs, weights, biases, ACT_ReLU);

		if (accuracy > 0.86) { // adaptive learning rate
			learningRate = 0.0001;
		}
		else if (accuracy > 0.85) {
			learningRate = 0.0005;
		}

		auto epoch_int = duration_cast<milliseconds>(epoch2 - epoch1); // getting number of milliseconds
		auto runTime = duration_cast<milliseconds>(epoch2 - runStart);
		epoch1 = high_resolution_clock::now();
		runTimeD = ((double)runTime.count()) / 60000;

		cout << "Accuracy: " << accuracy << ", Error: " << "not calculated" << ", Runtime: " << runTimeD << " min., ms since last epoch: " << epoch_int.count() << endl;

		for (size_t trainingExampleOffset = 0; trainingExampleOffset + batchSize <= evaluationOffset; trainingExampleOffset += batchSize) {

			for (Matrix& layer : weightChangeSum) // reset before each batch
				for (vector<double>& v1 : layer.data)
					fill(v1.begin(), v1.end(), 0);

			for (vector<double>& v1 : biasChangeSum) // reset before each batch
				fill(v1.begin(), v1.end(), 0);

			for (size_t batchNum = 0; batchNum < batchSize; batchNum++) {
				vector<double>& trainingExample = trainDataVectors.data[shuffleMap[trainingExampleOffset + batchNum]];
				int label = trainDataLabels.data[shuffleMap[trainingExampleOffset + batchNum]][0];

				forwardPass(trainingExample, nonStaticNeuronLayersPotentials, nonStaticNeuronLayersOutputs, weights, biases, ACT_ReLU, label);
				vector<vector<double>> deltas = computeDeltas(nonStaticNeuronLayersPotentials, nonStaticNeuronLayersOutputs, weights, biases, ACT_ReLU, label);
				computeWeightChange(weightChangeSum, biasChangeSum, trainingExample, nonStaticNeuronLayersOutputs, weights, biases, deltas);
			}
			updateWeights(weightChangeSum, biasChangeSum, weights, biases, batchSize, learningRate, params, optimizer);
		}
		shuffle(shuffleMap.begin(), shuffleMap.end(), random_device());
		epoch2 = high_resolution_clock::now();
		epochCounter += 1;
	}

	for (Matrix& layer : weightChangeSum) // reset before test evaluation
		for (vector<double>& v1 : layer.data)
			fill(v1.begin(), v1.end(), 0);

	for (vector<double>& v1 : biasChangeSum)
		fill(v1.begin(), v1.end(), 0);

	vector<double> outLabels;

	for (auto& inputVector : testDataVectors.data) {
		forwardPass(inputVector, nonStaticNeuronLayersPotentials, nonStaticNeuronLayersOutputs, weights, biases, ACT_ReLU, 0);
		outLabels.push_back(vecToScalar(nonStaticNeuronLayersOutputs.back()));
	}

	writeToFile("actualPredictions", outLabels);

	return 0;
}
